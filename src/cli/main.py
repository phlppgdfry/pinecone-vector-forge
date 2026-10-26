"""Pinecone Vector Forge - Professional Terminal CLI (TUI)."""

import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from ..core.config import settings
from ..core.pinecone_client import get_pinecone_manager
from ..patterns.hybrid_search import HybridSearchEngine
from ..patterns.metadata_filtering import MetadataFilterBuilder
from ..patterns.reranking import TwoStageReranker
from ..seed.ingest import run_ingestion_pipeline

app = typer.Typer(
    name="vectorforge",
    help="⚡ Pinecone Vector Forge: High-Performance Vector Intelligence CLI",
    add_completion=False
)
console = Console()


@app.command()
def status():
    """Display live Pinecone connectivity, CLI status, and index telemetry."""
    pc = get_pinecone_manager()
    telemetry = pc.get_telemetry()
    cli_info = pc.get_cli_status()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]🌲 Pinecone Vector Forge Telemetry[/bold cyan]\n"
        "[dim]Serverless Vector Intelligence Engine[/dim]",
        border_style="cyan"
    ))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Cloud & Region", f"{settings.pinecone_cloud} / {settings.pinecone_region}")
    table.add_row("Index Name", telemetry.index_name)
    table.add_row("Dimension", str(telemetry.dimension))
    table.add_row("Metric", telemetry.metric)
    table.add_row("Connection Status", f"[green]{telemetry.status}[/green]" if telemetry.connected_to_pinecone else f"[yellow]{telemetry.status}[/yellow]")
    table.add_row("Total Vectors", str(telemetry.total_vector_count))

    ns_str = ", ".join(f"{k}: {v}" for k, v in telemetry.namespaces.items()) if telemetry.namespaces else "None"
    table.add_row("Namespaces", ns_str)
    table.add_row("Pinecone CLI (pc)", "Installed & Ready" if cli_info.get("cli_installed") else "Not detected")

    console.print(table)
    console.print()


@app.command()
def seed(namespace: str = typer.Option("default", "--namespace", "-n", help="Target namespace")):
    """Ingest curated knowledge base into Pinecone index."""
    run_ingestion_pipeline(namespace=namespace)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query string"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    alpha: float = typer.Option(0.7, "--alpha", "-a", help="Hybrid balance: 1.0=Dense, 0.0=Sparse"),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Pinecone namespace"),
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Metadata category filter"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank", help="Apply two-stage cross-encoder reranking")
):
    """Execute hybrid vector search with metadata filtering & reranking."""
    engine = HybridSearchEngine()
    reranker = TwoStageReranker()

    filter_dict = None
    if category:
        filter_dict = MetadataFilterBuilder().category(category).build()

    console.print(f"\n[dim]Searching for:[/dim] [bold cyan]'{query}'[/bold cyan] [dim](namespace: {namespace}, alpha: {alpha})[/dim]\n")

    hits = engine.search(
        query=query,
        top_k=top_k * 2 if rerank else top_k,
        alpha=alpha,
        namespace=namespace,
        filter_dict=filter_dict
    )

    if rerank and hits:
        hits = reranker.rerank(query=query, candidates=hits, top_k=top_k)
    else:
        hits = hits[:top_k]

    if not hits:
        console.print("[yellow]No matching vectors found. Have you seeded the index? Run: vectorforge seed[/yellow]\n")
        return

    table = Table(title=f"Search Results ({len(hits)} hits)", show_header=True, header_style="bold green")
    table.add_column("Rank", justify="center", style="dim", width=6)
    table.add_column("Score", justify="right", style="bold yellow", width=10)
    table.add_column("Document Title", style="bold white", width=30)
    table.add_column("Category", style="cyan", width=20)
    table.add_column("Text Excerpt", style="dim")

    for i, hit in enumerate(hits, 1):
        table.add_row(
            f"#{i}",
            f"{hit.score:.4f}",
            hit.title,
            hit.category,
            (hit.text[:120] + "...") if len(hit.text) > 120 else hit.text
        )

    console.print(table)
    console.print()


@app.command()
def evaluate(
    alpha: float = typer.Option(0.70, "--alpha", "-a", help="Hybrid balance (0.0 to 1.0)"),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Pinecone namespace"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Evaluation Top-K window")
):
    """Run quantitative evaluation suite (MRR@K, Recall@K, Precision@K)."""
    from ..patterns.evaluation import RetrievalEvaluator

    console.print()
    console.print(Panel.fit(
        "[bold cyan]📊 Pinecone Vector Forge - Retrieval Benchmark Suite[/bold cyan]\n"
        f"[dim]Evaluating over golden IR test suite (alpha={alpha}, Top-K={top_k})[/dim]",
        border_style="cyan"
    ))

    evaluator = RetrievalEvaluator(top_k=top_k)
    pc = get_pinecone_manager()
    if pc.get_telemetry().total_vector_count == 0:
        run_ingestion_pipeline(namespace=namespace)
    res = evaluator.run_benchmark(alpha=alpha, namespace=namespace)

    table = Table(title="Information Retrieval (IR) Benchmark Metrics", show_header=True, header_style="bold green")
    table.add_column("Metric", style="bold")
    table.add_column("Score", style="cyan")
    table.add_column("Industry Target", style="dim")

    table.add_row("Mean Reciprocal Rank (MRR@K)", f"{res.mrr_at_k:.4f}", "> 0.70")
    table.add_row("Recall@K (Hit Rate)", f"{res.recall_at_k * 100:.1f}%", "> 80.0%")
    table.add_row("Precision@K", f"{res.precision_at_k * 100:.1f}%", "> 50.0%")
    table.add_row("Avg Query Latency", f"{res.avg_latency_ms:.2f} ms", "< 25.0 ms")
    table.add_row("Evaluated Queries", str(res.total_eval_queries), "7 Golden Pairs")

    console.print(table)
    console.print()


@app.command()
def demo():
    """Run an automated showcase tour demonstrating hybrid retrieval and filtering."""
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🚀 Pinecone Vector Forge - Showcase Tour[/bold cyan]\n"
        "[white]Demonstrating Hybrid Search, Metadata Filtering & Multi-Tenancy[/white]",
        border_style="cyan"
    ))

    # 1. Seed
    console.print("\n[bold]Step 1: Populating Index with Knowledge Corpus[/bold]")
    run_ingestion_pipeline(namespace="default")

    # 2. Semantic query
    console.print("\n[bold]Step 2: Semantic Dense Query (alpha=1.0)[/bold]")
    search(query="How does serverless scaling handle idle costs?", top_k=3, alpha=1.0, namespace="default", category=None, rerank=True)

    # 3. Hybrid query with category filter
    console.print("\n[bold]Step 3: Filtered Hybrid Query (category='security_compliance', alpha=0.7)[/bold]")
    search(query="Strict multi-tenancy and data isolation", top_k=2, alpha=0.7, namespace="default", category="security_compliance", rerank=True)

    console.print("[bold green]✔ Showcase tour completed successfully![/bold green]\n")


def main():
    app()


if __name__ == "__main__":
    main()

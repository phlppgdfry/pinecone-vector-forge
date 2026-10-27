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


@app.command()
def query_plan(
    query: str = typer.Argument(..., help="Natural language inquiry with implied filters")
):
    """Deconstruct natural language query into semantic vector + Pinecone metadata filter."""
    from ..patterns.self_query import SelfQueryDeconstructor

    console.print()
    console.print(Panel.fit(
        "[bold cyan]🧠 Pinecone Vector Forge - Self-Query Deconstructor[/bold cyan]\n"
        "[dim]Translating free text to structured Pinecone filter payloads[/dim]",
        border_style="cyan"
    ))

    parser = SelfQueryDeconstructor()
    plan = parser.deconstruct(query)

    table = Table(title="Deconstructed Query Plan", show_header=True, header_style="bold green")
    table.add_column("Component", style="bold")
    table.add_column("Parsed Value", style="cyan")

    table.add_row("Raw User Query", plan.raw_query)
    table.add_row("Semantic Dense String", plan.semantic_query)
    table.add_row("Extracted Category", plan.detected_category or "None (Global Search)")
    table.add_row("Extracted Priority", plan.detected_priority or "None")
    table.add_row("Min Quality Threshold", f"{plan.detected_min_quality}" if plan.detected_min_quality else "None")
    table.add_row("Optimal Alpha Balance", f"{plan.suggested_alpha:.2f} ({int(plan.suggested_alpha*100)}% Dense)")
    table.add_row("Compiled Pinecone Filter", str(plan.filter_dict) if plan.filter_dict else "None (Unfiltered)")

    console.print(table)
    console.print()


@app.command()
def heatmap():
    """Analyze vector corpus similarity matrix, redundancy, and cluster centroids."""
    from ..patterns.similarity_matrix import SimilarityMatrixAnalyzer

    console.print()
    console.print(Panel.fit(
        "[bold cyan]🔥 Pinecone Vector Forge - Corpus Similarity & Redundancy Matrix[/bold cyan]\n"
        "[dim]Auditing pairwise cosine distances & knowledge centroids[/dim]",
        border_style="cyan"
    ))

    analyzer = SimilarityMatrixAnalyzer()
    report = analyzer.compute_matrix()

    info_table = Table(show_header=True, header_style="bold magenta")
    info_table.add_column("Corpus Metric", style="bold")
    info_table.add_column("Value", style="cyan")

    info_table.add_row("Average Pairwise Similarity", f"{report.average_corpus_similarity:.4f}")
    info_table.add_row("Most Similar Pair", f"{report.most_similar_pair[0]} ↔ {report.most_similar_pair[1]} (Score: {report.most_similar_pair[2]})")
    info_table.add_row("Most Distinct Pair", f"{report.most_distinct_pair[0]} ↔ {report.most_distinct_pair[1]} (Score: {report.most_distinct_pair[2]})")
    info_table.add_row("Top Knowledge Centroids", ", ".join(report.cluster_centroids[:2]))

    console.print(info_table)
    console.print()


@app.command()
def hyde(
    query: str = typer.Argument(..., help="Query to expand with Hypothetical Document Embedding"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Results count")
):
    """Execute search with Hypothetical Document Embeddings (HyDE)."""
    from ..patterns.hyde import HyDESearchEngine

    console.print(f"\n[dim]Executing HyDE search for:[/dim] [bold cyan]'{query}'[/bold cyan]\n")
    engine = HyDESearchEngine()
    hits = engine.search(query=query, top_k=top_k)

    table = Table(title=f"HyDE Results ({len(hits)} hits)", show_header=True, header_style="bold green")
    table.add_column("Rank", justify="center", style="dim", width=6)
    table.add_column("Score", justify="right", style="bold yellow", width=10)
    table.add_column("Title", style="bold white", width=30)
    table.add_column("Category", style="cyan")

    for i, h in enumerate(hits, 1):
        table.add_row(f"#{i}", f"{h.score:.4f}", h.title, h.category)

    console.print(table)
    console.print()


@app.command()
def assistant(
    question: str = typer.Argument(..., help="Technical inquiry to ask Pinecone Assistant"),
    upload: Optional[str] = typer.Option(None, "--upload", "-u", help="Optional file path to index into assistant first")
):
    """Chat directly with managed Pinecone Assistant with verified document citations."""
    from ..patterns.pinecone_assistant import PineconeAssistantManager

    console.print()
    console.print(Panel.fit(
        "[bold cyan]🤖 Pinecone Assistant - Autonomous Question-Answering[/bold cyan]\n"
        "[dim]Agentic RAG with file parsing & citation verification[/dim]",
        border_style="cyan"
    ))

    mgr = PineconeAssistantManager()

    if upload:
        console.print(f"[yellow]Uploading '{upload}' to Pinecone Assistant...[/yellow]")
        up_res = mgr.upload_file(upload)
        console.print(f"[green]✔ File status: {up_res.get('status')}[/green]\n")

    console.print(f"[dim]Asking:[/dim] [bold white]'{question}'[/bold white]...\n")
    resp = mgr.chat(question)

    console.print(Panel(
        resp.answer,
        title=f"[bold green]Assistant Answer ({'Live Pinecone Cloud' if resp.is_live_assistant else 'Local Fallback'})[/bold green]",
        border_style="green"
    ))

    if resp.citations:
        cit_table = Table(title="Document References & Grounded Citations", show_header=True, header_style="bold cyan")
        cit_table.add_column("Source File", style="bold")
        cit_table.add_column("Reference Details", style="dim")
        for c in resp.citations:
            cit_table.add_row(str(c.get("file", "File")), str(c))
        console.print(cit_table)
    console.print()


@app.command(name="chat")
def interactive_chat():
    """Start an interactive conversational REPL with Pinecone Assistant."""
    from rich.prompt import Prompt
    from ..patterns.pinecone_assistant import PineconeAssistantManager

    console.print()
    console.print(Panel.fit(
        "[bold cyan]💬 Pinecone Vector Forge - Interactive Assistant REPL[/bold cyan]\n"
        "[dim]Direct terminal session with grounded citations & autonomous RAG\n"
        "Commands: [green]/files[/green] (list knowledge), [yellow]/upload <path>[/yellow] (index doc), [red]/exit[/red] (quit)[/dim]",
        border_style="cyan"
    ))

    mgr = PineconeAssistantManager()
    info = mgr.get_or_create_assistant()
    status_label = "🟢 Live Pinecone Cloud" if info.get("status") == "Ready" and mgr.client.is_connected else "🟡 Local Fallback Mode"
    console.print(f"[bold]Session Status:[/bold] {status_label} | [bold]Assistant:[/bold] [cyan]{mgr.assistant_name}[/cyan]\n")

    files = mgr.list_files()
    if files:
        console.print(f"[dim]Indexed knowledge base files: {', '.join(f.get('name', 'file') for f in files)}[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold green]vectorforge >[/bold green]")
            if not user_input.strip():
                continue

            cmd = user_input.strip()
            if cmd.lower() in ("/exit", "exit", "quit", ":q"):
                console.print("[dim]Closing interactive session. Happy searching![/dim]\n")
                break
            elif cmd.lower() == "/files":
                current_files = mgr.list_files()
                console.print(f"[cyan]Files in assistant:[/cyan] {current_files}\n")
                continue
            elif cmd.lower().startswith("/upload "):
                filepath = cmd.split(" ", 1)[1].strip()
                console.print(f"[yellow]Uploading '{filepath}' to assistant...[/yellow]")
                res = mgr.upload_file(filepath)
                console.print(f"[green]✔ Upload result: {res.get('status')}[/green]\n")
                continue
            elif cmd.lower() == "/clear":
                console.clear()
                continue

            console.print("[dim]Thinking...[/dim]")
            resp = mgr.chat(cmd)

            console.print(Panel(
                resp.answer,
                title=f"[bold green]Assistant Answer ({'Live Pinecone Cloud' if resp.is_live_assistant else 'Local Fallback'})[/bold green]",
                border_style="green"
            ))

            if resp.citations:
                cit_table = Table(show_header=True, header_style="bold cyan")
                cit_table.add_column("Cited Source File", style="bold")
                cit_table.add_column("Citation Reference", style="dim")
                for c in resp.citations:
                    cit_table.add_row(str(c.get("file", "File")), str(c.get("text", c)))
                console.print(cit_table)
            console.print()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Interrupted. Goodbye![/dim]\n")
            break


def main():
    app()


if __name__ == "__main__":
    main()

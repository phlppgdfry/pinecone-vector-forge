"""High-Throughput Batch Ingestion & Vector Pipeline."""

import sys
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from ..core.config import settings
from ..core.pinecone_client import get_pinecone_manager
from ..patterns.contextual_chunking import HierarchicalChunker
from .sample_data import KNOWLEDGE_BASE_DOCUMENTS

console = Console()


def run_ingestion_pipeline(namespace: str = "default") -> Dict[str, Any]:
    """Runs end-to-end ingestion: document chunking, embedding, and vector upsert."""
    console.rule("[bold cyan]Pinecone Vector Forge - Ingestion Pipeline[/bold cyan]")

    pc = get_pinecone_manager()
    chunker = HierarchicalChunker(chunk_size=350, overlap=50)

    all_records = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_chunk = progress.add_task("[yellow]Chunking & Vectorizing Documents...", total=len(KNOWLEDGE_BASE_DOCUMENTS))

        for doc in KNOWLEDGE_BASE_DOCUMENTS:
            records = chunker.process_and_vectorize(doc)
            all_records.extend(records)
            progress.advance(task_chunk)

        task_upsert = progress.add_task(f"[green]Upserting {len(all_records)} vectors into Pinecone [{namespace}]...", total=1)
        res = pc.upsert_vectors(records=all_records, namespace=namespace)
        progress.advance(task_upsert)

    # Summary table
    table = Table(title="[bold green]Ingestion Summary[/bold green]", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Total Parent Documents", str(len(KNOWLEDGE_BASE_DOCUMENTS)))
    table.add_row("Total Vector Chunks", str(len(all_records)))
    table.add_row("Target Namespace", namespace)
    table.add_row("Target Index", settings.pinecone_index_name)
    table.add_row("Connected to Pinecone", "✅ LIVE" if res.get("connected_to_pinecone") else "⚡ LOCAL SIMULATION")
    table.add_row("Latency", f"{res.get('latency_ms', 0)} ms")

    console.print(table)
    return res


if __name__ == "__main__":
    ns = sys.argv[1] if len(sys.argv) > 1 else settings.pinecone_namespace
    run_ingestion_pipeline(namespace=ns)

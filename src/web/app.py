"""FastAPI Web Application & REST API for Pinecone Vector Forge."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from ..core.config import settings
from ..core.models import QueryPayload, SearchResponse, IndexTelemetry, Document
from ..core.pinecone_client import get_pinecone_manager
from ..patterns.hybrid_search import HybridSearchEngine
from ..patterns.metadata_filtering import MetadataFilterBuilder
from ..patterns.reranking import TwoStageReranker
from ..patterns.contextual_chunking import HierarchicalChunker
from ..patterns.self_query import SelfQueryDeconstructor
from ..patterns.hyde import HyDESearchEngine
from ..patterns.similarity_matrix import SimilarityMatrixAnalyzer
from ..patterns.pinecone_assistant import PineconeAssistantManager
from ..seed.ingest import run_ingestion_pipeline
from ..seed.sample_data import KNOWLEDGE_BASE_DOCUMENTS


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Populate initial vectors if empty
    pc = get_pinecone_manager()
    telem = pc.get_telemetry()
    if telem.total_vector_count == 0:
        run_ingestion_pipeline(namespace="default")
    yield


app = FastAPI(
    title="Pinecone Vector Forge API",
    description="High-Performance Hybrid Vector Search & RAG Intelligence Engine",
    version="1.0.0",
    lifespan=lifespan
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the primary UI dashboard."""
    index_file = STATIC_DIR / "index.html"
    return FileResponse(index_file)


@app.get("/api/status", response_model=IndexTelemetry)
async def get_status():
    """Get live index telemetry and Pinecone connection health."""
    pc = get_pinecone_manager()
    return pc.get_telemetry()


@app.get("/api/cli")
async def get_cli_details():
    """Retrieve Pinecone CLI ('pc') installation status and context."""
    pc = get_pinecone_manager()
    return pc.get_cli_status()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.7
    namespace: str = "default"
    category: Optional[str] = None
    rerank: bool = True


@app.post("/api/search", response_model=SearchResponse)
async def execute_search(req: SearchRequest):
    """Executes hybrid vector search with metadata filtering & two-stage rerank."""
    import time
    start = time.perf_counter()

    engine = HybridSearchEngine()
    reranker = TwoStageReranker()
    pc = get_pinecone_manager()

    filter_dict = None
    if req.category and req.category != "all":
        filter_dict = MetadataFilterBuilder().category(req.category).build()

    # Step 1: Hybrid / Dense ANN
    fetch_k = req.top_k * 2 if req.rerank else req.top_k
    hits = engine.search(
        query=req.query,
        top_k=fetch_k,
        alpha=req.alpha,
        namespace=req.namespace,
        filter_dict=filter_dict
    )

    # Step 2: Rerank
    mode = "hybrid" if 0.0 < req.alpha < 1.0 else ("dense" if req.alpha >= 1.0 else "sparse")
    if req.rerank and hits:
        hits = reranker.rerank(query=req.query, candidates=hits, top_k=req.top_k)
        mode += " + reranked"
    else:
        hits = hits[:req.top_k]

    latency_ms = (time.perf_counter() - start) * 1000

    return SearchResponse(
        query=req.query,
        hits=hits,
        total_hits=len(hits),
        namespace=req.namespace,
        retrieval_mode=mode,
        latency_ms=round(latency_ms, 2),
        index_connected=pc.is_connected,
        used_mock=not pc.is_connected
    )


class IngestCustomRequest(BaseModel):
    title: str
    content: str
    category: str = "general"
    tags: List[str] = []
    namespace: str = "default"


@app.post("/api/ingest")
async def ingest_document(req: IngestCustomRequest):
    """Chunk, vectorize, and upsert a custom document into Pinecone."""
    import uuid
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    doc = Document(
        id=doc_id,
        title=req.title,
        content=req.content,
        category=req.category,
        tags=req.tags,
        author="Interactive User"
    )

    chunker = HierarchicalChunker(chunk_size=350, overlap=50)
    records = chunker.process_and_vectorize(doc)

    pc = get_pinecone_manager()
    res = pc.upsert_vectors(records=records, namespace=req.namespace)
    return {
        "status": "success",
        "document_id": doc_id,
        "chunks_created": len(records),
        "namespace": req.namespace,
        "pinecone_response": res
    }


@app.post("/api/seed")
async def trigger_seed(namespace: str = "default"):
    """Reseed knowledge base."""
    res = run_ingestion_pipeline(namespace=namespace)
    return res


@app.get("/api/heatmap")
async def get_similarity_heatmap():
    """Retrieve pairwise cosine similarity matrix across corpus."""
    analyzer = SimilarityMatrixAnalyzer()
    report = analyzer.compute_matrix()
    return {
        "titles": report.document_titles,
        "ids": report.document_ids,
        "matrix": report.matrix,
        "average_similarity": report.average_corpus_similarity,
        "most_similar_pair": report.most_similar_pair,
        "most_distinct_pair": report.most_distinct_pair,
        "centroids": report.cluster_centroids
    }


class QueryPlanRequest(BaseModel):
    query: str


@app.post("/api/query-plan")
async def generate_query_plan(req: QueryPlanRequest):
    """Deconstruct natural language request into vector text and Pinecone metadata filter."""
    parser = SelfQueryDeconstructor()
    plan = parser.deconstruct(req.query)
    return {
        "raw_query": plan.raw_query,
        "semantic_query": plan.semantic_query,
        "filter_dict": plan.filter_dict,
        "suggested_alpha": plan.suggested_alpha,
        "detected_category": plan.detected_category,
        "detected_priority": plan.detected_priority,
        "detected_quality": plan.detected_min_quality
    }


@app.post("/api/hyde")
async def execute_hyde_search(req: SearchRequest):
    """Executes search using Hypothetical Document Embeddings."""
    engine = HyDESearchEngine()
    hits = engine.search(query=req.query, top_k=req.top_k, namespace=req.namespace)
    return {
        "query": req.query,
        "hits": hits,
        "total_hits": len(hits),
        "mode": "HyDE (Hypothetical Document Expansion)"
    }


class AssistantChatRequest(BaseModel):
    query: str


@app.post("/api/assistant/chat")
async def assistant_chat(req: AssistantChatRequest):
    """Chat with managed Pinecone Assistant."""
    mgr = PineconeAssistantManager()
    resp = mgr.chat(req.query)
    return {
        "query": resp.query,
        "answer": resp.answer,
        "citations": resp.citations,
        "assistant_name": resp.assistant_name,
        "is_live_assistant": resp.is_live_assistant
    }


@app.get("/api/assistant/files")
async def assistant_list_files():
    """List documents indexed in Pinecone Assistant."""
    mgr = PineconeAssistantManager()
    return mgr.list_files()


@app.get("/api/metrics")
async def get_metrics():
    """Expose Prometheus/OpenTelemetry style vector engine metrics."""
    pc = get_pinecone_manager()
    telem = pc.get_telemetry()
    lines = [
        "# HELP vectorforge_total_vectors Total number of indexed vectors",
        "# TYPE vectorforge_total_vectors gauge",
        f"vectorforge_total_vectors {telem.total_vector_count}",
        "# HELP vectorforge_dimension Dimensionality of the active index",
        "# TYPE vectorforge_dimension gauge",
        f"vectorforge_dimension {telem.dimension}",
        "# HELP vectorforge_connected Connection health to Pinecone cloud (1=connected, 0=offline)",
        "# TYPE vectorforge_connected gauge",
        f"vectorforge_connected {1 if telem.connected_to_pinecone else 0}",
    ]
    for ns, count in telem.namespaces.items():
        lines.append(f'vectorforge_namespace_vectors{{namespace="{ns}"}} {count}')
    return HTMLResponse("\n".join(lines), media_type="text/plain; version=0.0.4")


def run():
    import uvicorn
    uvicorn.run("src.web.app:app", host=settings.web_host, port=settings.web_port, reload=settings.debug)


if __name__ == "__main__":
    run()

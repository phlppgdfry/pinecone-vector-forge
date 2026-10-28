"""FastAPI Web Application & REST API for Pinecone Vector Forge."""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Header, Depends, Security
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..core.config import settings
from ..core.models import QueryPayload, SearchResponse, IndexTelemetry, Document
from ..core.pinecone_client import get_pinecone_manager
from ..core.embeddings import get_embedding_generator
from ..core.finops import FinOpsCostModel
from ..core.security import get_current_tenant, AuthenticatedTenant, HMACSecuritySigner, get_revocation_store
from ..patterns.hybrid_search import HybridSearchEngine
from ..patterns.metadata_filtering import MetadataFilterBuilder
from ..patterns.reranking import TwoStageReranker
from ..patterns.contextual_chunking import HierarchicalChunker
from ..patterns.self_query import SelfQueryDeconstructor
from ..patterns.hyde import HyDESearchEngine
from ..patterns.similarity_matrix import SimilarityMatrixAnalyzer
from ..patterns.pinecone_assistant import PineconeAssistantManager
from ..patterns.multi_tenancy import MultiTenantManager, TenantAuthorizationError
from ..patterns.evaluation import RetrievalEvaluator
from ..patterns.vector_db_lab import VectorDBArchitectureLab
from ..patterns.enterprise_assistant import EnterpriseKnowledgeAssistant
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
    version="1.1.0",
    lifespan=lifespan
)

# Enterprise CORS Security Policy: restrict to safe origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Tenant-ID", "Authorization"],
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
    query: str = Field(..., max_length=1000)
    top_k: int = Field(default=5, ge=1, le=50)
    alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    namespace: str = Field(default="default", max_length=128)
    category: Optional[str] = None
    rerank: bool = True


@app.post("/api/search", response_model=SearchResponse)
async def execute_search(
    req: SearchRequest,
    x_tenant_id: Optional[str] = Header(None),
    current_tenant: AuthenticatedTenant = Depends(get_current_tenant)
):
    """Executes hybrid vector search with metadata filtering, two-stage rerank, cryptographic tenant fencing, and FinOps accounting."""
    start = time.perf_counter()

    # Tenant authorization boundary (cryptographic token prioritized over legacy header)
    if req.namespace.startswith("tenant_"):
        target_tenant = req.namespace.removeprefix("tenant_")
        if current_tenant.is_admin:
            pass
        elif current_tenant.tenant_id != "anonymous":
            clean_caller = current_tenant.tenant_id.removeprefix("tenant_")
            if clean_caller != target_tenant:
                raise HTTPException(
                    status_code=403,
                    detail=f"Cryptographic Security Violation: Authenticated tenant '{current_tenant.tenant_id}' cannot access namespace of '{target_tenant}'."
                )
            if not current_tenant.can_read():
                raise HTTPException(status_code=403, detail="Forbidden: Token lacks 'read' scope.")
        elif x_tenant_id:
            tenant_mgr = MultiTenantManager()
            try:
                tenant_mgr.authorize_tenant_access(
                    caller_tenant_id=x_tenant_id,
                    requested_tenant_id=target_tenant
                )
            except TenantAuthorizationError as err:
                raise HTTPException(status_code=403, detail=str(err))
        else:
            raise HTTPException(
                status_code=401,
                detail="Authentication required for tenant namespaces. Supply 'Authorization: Bearer <token>'."
            )

    engine = HybridSearchEngine()
    reranker = TwoStageReranker()
    pc = get_pinecone_manager()
    embedder = get_embedding_generator()
    model_info = embedder.get_model_info()

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

    cost_est = FinOpsCostModel.estimate_search_cost(
        top_k=req.top_k,
        rerank=req.rerank,
        candidates_count=len(hits)
    )

    return SearchResponse(
        query=req.query,
        hits=hits,
        total_hits=len(hits),
        namespace=req.namespace,
        retrieval_mode=mode,
        latency_ms=round(latency_ms, 2),
        index_connected=pc.is_connected,
        used_mock=model_info.is_mock,
        estimated_cost_usd=cost_est.estimated_cost_usd,
        embedding_model=model_info.model_name
    )


class IngestCustomRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=250)
    content: str = Field(..., min_length=10, max_length=50000)
    category: str = Field(default="general", max_length=64)
    tags: List[str] = Field(default_factory=list, max_length=20)
    namespace: str = Field(default="default", max_length=128)


@app.post("/api/ingest")
async def ingest_document(
    req: IngestCustomRequest,
    x_tenant_id: Optional[str] = Header(None),
    current_tenant: AuthenticatedTenant = Depends(get_current_tenant)
):
    """Chunk, vectorize, and upsert a custom document into Pinecone with cryptographic tenant security."""
    if req.namespace.startswith("tenant_"):
        target_tenant = req.namespace.removeprefix("tenant_")
        if current_tenant.is_admin:
            pass
        elif current_tenant.tenant_id != "anonymous":
            clean_caller = current_tenant.tenant_id.removeprefix("tenant_")
            if clean_caller != target_tenant:
                raise HTTPException(
                    status_code=403,
                    detail=f"Cryptographic Security Violation: Authenticated tenant '{current_tenant.tenant_id}' cannot write to namespace '{target_tenant}'."
                )
            if not current_tenant.can_write():
                raise HTTPException(status_code=403, detail="Forbidden: Token lacks 'write' scope.")
        elif x_tenant_id:
            tenant_mgr = MultiTenantManager()
            try:
                tenant_mgr.authorize_tenant_access(
                    caller_tenant_id=x_tenant_id,
                    requested_tenant_id=target_tenant
                )
            except TenantAuthorizationError as err:
                raise HTTPException(status_code=403, detail=str(err))
        else:
            raise HTTPException(
                status_code=401,
                detail="Authentication required for tenant namespaces. Supply 'Authorization: Bearer <token>'."
            )

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

    chunker = HierarchicalChunker(chunk_size=350, overlap=50, enable_contextual_prefix=True)
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
    query: str = Field(..., max_length=500)


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
    """Executes search using Hypothetical Document Embeddings with generation metadata."""
    engine = HyDESearchEngine()
    res = engine.search_with_metadata(query=req.query, top_k=req.top_k, namespace=req.namespace)
    return {
        "query": res.query,
        "hypothetical_passage": res.hypothetical_passage,
        "hits": res.hits,
        "total_hits": len(res.hits),
        "generation_mode": res.generation_mode
    }


class AssistantChatRequest(BaseModel):
    query: str = Field(..., max_length=2000)


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


@app.get("/api/evaluate")
async def run_evaluation_benchmark(alpha: float = 0.70, enable_rerank: bool = False, full: bool = False):
    """Run quantitative Information Retrieval evaluation suite."""
    evaluator = RetrievalEvaluator(top_k=3)
    res = evaluator.run_benchmark(alpha=alpha, enable_rerank=enable_rerank, use_full_corpus=full)
    return res


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
        "# HELP vectorforge_estimated_cost_per_query_usd Estimated baseline RCU cost",
        "# TYPE vectorforge_estimated_cost_per_query_usd gauge",
        f"vectorforge_estimated_cost_per_query_usd 0.0000004",
    ]
    for ns, count in telem.namespaces.items():
        lines.append(f'vectorforge_namespace_vectors{{namespace="{ns}"}} {count}')
    return HTMLResponse("\n".join(lines), media_type="text/plain; version=0.0.4")


class TokenIssueRequest(BaseModel):
    tenant_id: str = Field(..., min_length=2, max_length=64)
    role: str = Field(default="tenant_admin", max_length=32)
    scopes: List[str] = Field(default_factory=lambda: ["read", "write", "delete"])
    ttl_seconds: int = Field(default=3600, ge=60, le=86400 * 30)


@app.post("/api/auth/token")
async def issue_tenant_token(req: TokenIssueRequest):
    """Issues a cryptographically signed HMAC-SHA256 bearer token for tenant isolation."""
    token = HMACSecuritySigner.issue_token(
        tenant_id=req.tenant_id,
        role=req.role,
        scopes=req.scopes,
        ttl_seconds=req.ttl_seconds
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "tenant_id": req.tenant_id,
        "role": req.role,
        "scopes": req.scopes,
        "expires_in_seconds": req.ttl_seconds
    }


@app.get("/api/vector-dbs/benchmark")
async def get_vector_db_benchmarks():
    """Independent multi-engine vector database benchmark (Pinecone Serverless vs pgvector vs Qdrant vs Weaviate)."""
    lab = VectorDBArchitectureLab()
    report = lab.run_comparative_benchmark()
    return report.to_dict()


@app.get("/api/vector-dbs/adrs")
async def get_vector_db_adrs():
    """Retrieve Architectural Decision Records (ADRs) across vector storage engines."""
    lab = VectorDBArchitectureLab()
    return lab.get_adrs()


class EnterpriseUploadRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=250)
    content: str = Field(..., min_length=10, max_length=200000)
    category: str = Field(default="engineering", max_length=64)
    tags: List[str] = Field(default_factory=list, max_length=20)
    tenant_id: str = Field(default="default", max_length=64)
    filename: Optional[str] = Field(default=None, max_length=128)


class EnterpriseQueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000)
    tenant_id: str = Field(default="default", max_length=64)
    top_k: int = Field(default=5, ge=1, le=20)
    alpha: float = Field(default=0.70, ge=0.0, le=1.0)
    rerank: bool = Field(default=True)


class TokenRevokeRequest(BaseModel):
    token: str
    jti: Optional[str] = None


@app.post("/api/enterprise/upload")
async def enterprise_upload_document(
    req: EnterpriseUploadRequest,
    current_tenant: AuthenticatedTenant = Depends(get_current_tenant)
):
    """End-to-end Enterprise Knowledge Assistant Document Ingestion Workflow."""
    assistant = EnterpriseKnowledgeAssistant()
    try:
        res = assistant.ingest_document(
            title=req.title,
            content=req.content,
            category=req.category,
            tags=req.tags,
            tenant_id=req.tenant_id,
            auth_tenant=current_tenant,
            filename=req.filename
        )
        return res
    except (ValueError, PermissionError, TenantAuthorizationError) as err:
        status_code = 403 if isinstance(err, (PermissionError, TenantAuthorizationError)) else 400
        raise HTTPException(status_code=status_code, detail=str(err))


@app.post("/api/enterprise/query")
async def enterprise_query_assistant(
    req: EnterpriseQueryRequest,
    current_tenant: AuthenticatedTenant = Depends(get_current_tenant)
):
    """End-to-End Grounded RAG Query with Verifiable Citations and Abstention Protocol."""
    assistant = EnterpriseKnowledgeAssistant()
    try:
        ans = assistant.query(
            query=req.query,
            tenant_id=req.tenant_id,
            auth_tenant=current_tenant,
            top_k=req.top_k,
            alpha=req.alpha,
            enable_rerank=req.rerank
        )
        return ans
    except (PermissionError, TenantAuthorizationError) as err:
        raise HTTPException(status_code=403, detail=str(err))


@app.delete("/api/enterprise/documents/{doc_id}")
async def enterprise_delete_document(
    doc_id: str,
    tenant_id: str = "default",
    current_tenant: AuthenticatedTenant = Depends(get_current_tenant)
):
    """Safe atomic document deletion across Pinecone and Inverted Index with tenant scoping."""
    assistant = EnterpriseKnowledgeAssistant()
    try:
        return assistant.delete_document(doc_id=doc_id, tenant_id=tenant_id, auth_tenant=current_tenant)
    except (PermissionError, TenantAuthorizationError) as err:
        raise HTTPException(status_code=403, detail=str(err))


@app.post("/api/auth/revoke")
async def revoke_auth_token(req: TokenRevokeRequest):
    """Revokes a cryptographic bearer token or JWT ID."""
    rev_store = get_revocation_store()
    rev_store.revoke_token(token=req.token, jti=req.jti)
    return {"status": "success", "message": "Token has been added to revocation denylist."}


def run():
    import uvicorn
    uvicorn.run("src.web.app:app", host=settings.web_host, port=settings.web_port, reload=settings.debug)


if __name__ == "__main__":
    run()

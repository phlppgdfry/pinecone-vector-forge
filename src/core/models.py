"""Strongly typed Domain and Vector Models for Pinecone Vector Forge."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


def _now_utc():
    return datetime.now(timezone.utc)


class Document(BaseModel):
    """Raw ingestion document representation."""
    id: str
    title: str
    content: str
    category: str = "general"
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = "Anonymous"
    created_at: datetime = Field(default_factory=_now_utc)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Semantic partition of a parent document."""
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int = 0
    parent_title: str
    category: str
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorRecord(BaseModel):
    """Payload prepared for Pinecone vector upsert."""
    id: str
    values: List[float]
    sparse_values: Optional[Dict[str, List[Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    """Single vector match result."""
    id: str
    score: float
    text: str
    title: str
    category: str
    tags: List[str] = Field(default_factory=list)
    document_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryPayload(BaseModel):
    """Search request specification."""
    query: str
    top_k: int = 5
    namespace: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    filter: Optional[Dict[str, Any]] = None
    alpha: float = 0.7  # 1.0 = 100% dense, 0.0 = 100% sparse
    rerank: bool = True


class SearchResponse(BaseModel):
    """Standardized search response with telemetry."""
    query: str
    hits: List[SearchHit]
    total_hits: int
    namespace: str
    retrieval_mode: str  # "dense", "hybrid", "reranked"
    latency_ms: float
    index_connected: bool
    used_mock: bool = False


class IndexTelemetry(BaseModel):
    """Live index statistics and health metrics."""
    index_name: str
    status: str
    dimension: int
    metric: str
    total_vector_count: int
    namespaces: Dict[str, int] = Field(default_factory=dict)
    connected_to_pinecone: bool
    is_serverless: bool = True

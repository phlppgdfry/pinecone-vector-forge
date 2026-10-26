"""Core infrastructure: Configuration, Pinecone Client, Models, and Embeddings."""

from .config import settings, Settings
from .models import Document, Chunk, VectorRecord, SearchHit, QueryPayload, SearchResponse
from .embeddings import get_embedding_generator, EmbeddingGenerator
from .pinecone_client import PineconeManager, get_pinecone_manager

__all__ = [
    "settings",
    "Settings",
    "Document",
    "Chunk",
    "VectorRecord",
    "SearchHit",
    "QueryPayload",
    "SearchResponse",
    "get_embedding_generator",
    "EmbeddingGenerator",
    "PineconeManager",
    "get_pinecone_manager",
]

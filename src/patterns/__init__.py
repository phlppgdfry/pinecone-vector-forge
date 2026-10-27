"""Pinecone Production Architecture Patterns & Vernuftige Snippets."""

from .hybrid_search import HybridSearchEngine
from .multi_tenancy import MultiTenantManager
from .metadata_filtering import MetadataFilterBuilder
from .contextual_chunking import HierarchicalChunker
from .reranking import TwoStageReranker
from .rag_agent import GroundedRAGAgent, GroundedResponse
from .evaluation import RetrievalEvaluator, GOLDEN_DATASET
from .self_query import SelfQueryDeconstructor, DeconstructedQuery
from .hyde import HyDESearchEngine
from .similarity_matrix import SimilarityMatrixAnalyzer, SimilarityMatrixReport

__all__ = [
    "HybridSearchEngine",
    "MultiTenantManager",
    "MetadataFilterBuilder",
    "HierarchicalChunker",
    "TwoStageReranker",
    "GroundedRAGAgent",
    "GroundedResponse",
    "RetrievalEvaluator",
    "GOLDEN_DATASET",
    "SelfQueryDeconstructor",
    "DeconstructedQuery",
    "HyDESearchEngine",
    "SimilarityMatrixAnalyzer",
    "SimilarityMatrixReport",
]

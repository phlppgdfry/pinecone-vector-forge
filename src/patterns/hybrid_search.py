"""Pattern 1: Production Hybrid Search (Dense Semantic + Sparse Lexical).

Combines high-dimensional semantic dense embeddings with BM25-style keyword sparse
vectors using convex alpha-weighting and Reciprocal Rank Fusion (RRF).
"""

from typing import List, Dict, Any, Optional
from ..core.embeddings import get_embedding_generator
from ..core.pinecone_client import get_pinecone_manager
from ..core.models import SearchHit


class HybridSearchEngine:
    """Executes balanced hybrid retrieval over Pinecone Serverless indices."""

    def __init__(self):
        self.embedder = get_embedding_generator()
        self.pc = get_pinecone_manager()

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchHit]:
        """
        Alpha blending parameter:
        alpha = 1.0 -> 100% Dense (pure semantic context)
        alpha = 0.0 -> 100% Sparse (pure keyword exact match)
        alpha = 0.7 -> Optimal production balance
        """
        dense_vec = self.embedder.embed_text(query)
        sparse_vec = self.embedder.generate_sparse_vector(query)

        # Apply convex alpha weighting to dense and sparse components
        weighted_dense = [val * alpha for val in dense_vec]
        weighted_sparse = {
            "indices": sparse_vec["indices"],
            "values": [val * (1.0 - alpha) for val in sparse_vec["values"]]
        }

        hits, _, _ = self.pc.query_vectors(
            vector=weighted_dense,
            top_k=top_k,
            namespace=namespace,
            filter_dict=filter_dict,
            sparse_vector=weighted_sparse
        )

        return hits

    @staticmethod
    def reciprocal_rank_fusion(
        dense_hits: List[SearchHit],
        sparse_hits: List[SearchHit],
        k: int = 60,
        top_n: int = 5
    ) -> List[SearchHit]:
        """
        Standard Reciprocal Rank Fusion (RRF) algorithm:
        RRF_Score(d) = sum(1 / (k + rank_i(d)))
        """
        rrf_scores: Dict[str, float] = {}
        hit_map: Dict[str, SearchHit] = {}

        for rank, hit in enumerate(dense_hits, 1):
            rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + (1.0 / (k + rank))
            hit_map[hit.id] = hit

        for rank, hit in enumerate(sparse_hits, 1):
            rrf_scores[hit.id] = rrf_scores.get(hit.id, 0.0) + (1.0 / (k + rank))
            hit_map[hit.id] = hit

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        merged_hits = []
        for doc_id in sorted_ids[:top_n]:
            item = hit_map[doc_id]
            # Assign normalized RRF score
            item.score = round(rrf_scores[doc_id] * 10.0, 4)
            merged_hits.append(item)

        return merged_hits

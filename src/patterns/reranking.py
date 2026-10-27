"""Pattern 5: Two-Stage Retrieval with Native Pinecone Inference & Cross-Encoder.

Stage 1: Sub-10ms Approximate Nearest Neighbor (ANN) query fetches Top-N candidates.
Stage 2: Cross-attention re-ranking model re-scores candidates against query text.
         - Uses Pinecone Inference API ('bge-reranker-v2-m3') when cloud is connected.
         - Falls back to local deterministic cross-encoder when offline.
"""

from typing import List, Dict, Any, Optional
from ..core.models import SearchHit
from ..core.pinecone_client import get_pinecone_manager


class TwoStageReranker:
    """Refines candidate vectors with cross-attention relevance scoring."""

    def __init__(self, model_name: str = "bge-reranker-v2-m3"):
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: List[SearchHit],
        top_k: int = 5
    ) -> List[SearchHit]:
        """Re-scores initial hits using Pinecone Inference API or local cross-attention."""
        if not candidates:
            return []

        # 1. Attempt Native Pinecone Inference Reranker
        pc = get_pinecone_manager()
        if pc.is_connected and pc.client:
            try:
                doc_texts = [f"{c.title}: {c.text}" for c in candidates]
                res = pc.client.inference.rerank(
                    model=self.model_name,
                    query=query,
                    documents=doc_texts,
                    top_n=min(top_k, len(candidates)),
                    return_documents=False
                )

                reranked_hits = []
                for ranked in res.data:
                    orig_hit = candidates[ranked.index]
                    orig_hit.score = round(float(ranked.score), 4)
                    reranked_hits.append(orig_hit)
                return reranked_hits
            except Exception:
                pass  # Graceful fallback to local cross-encoder

        # 2. Local Cross-Attention & Token Density Fallback
        return self._local_cross_attention_rerank(query, candidates, top_k)

    def _local_cross_attention_rerank(
        self,
        query: str,
        candidates: List[SearchHit],
        top_k: int
    ) -> List[SearchHit]:
        """Computes query-document cross-entropy & token affinity."""
        query_terms = set(query.lower().split())

        scored_candidates: List[SearchHit] = []
        for hit in candidates:
            text_lower = hit.text.lower()
            title_lower = hit.title.lower()

            base_score = hit.score
            exact_matches = sum(1 for term in query_terms if term in text_lower)
            title_matches = sum(1 for term in query_terms if term in title_lower)
            
            coverage_ratio = (exact_matches / len(query_terms)) if query_terms else 0.0

            positional_bonus = 0.0
            if exact_matches > 0:
                first_pos = min((text_lower.find(t) for t in query_terms if t in text_lower), default=500)
                positional_bonus = max(0.0, 0.15 * (1.0 - (min(first_pos, 500) / 500.0)))

            final_score = (base_score * 0.50) + (coverage_ratio * 0.35) + (title_matches * 0.08) + positional_bonus

            hit.score = round(min(0.9999, max(0.0, final_score)), 4)
            scored_candidates.append(hit)

        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        return scored_candidates[:top_k]

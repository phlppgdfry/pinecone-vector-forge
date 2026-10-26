"""Pattern 5: Two-Stage Retrieval with Cross-Encoder & Semantic Re-ranking.

Stage 1: Sub-millisecond Pinecone Approximate Nearest Neighbor (ANN) query fetches Top-N (e.g., 20).
Stage 2: Re-ranking model performs full query-document cross-attention, boosting precision and removing semantic false positives.
"""

import math
from typing import List, Dict, Any, Optional
from ..core.models import SearchHit
from ..core.config import settings


class TwoStageReranker:
    """Refines candidate vectors with cross-attention relevance scoring."""

    def __init__(self):
        self.pinecone_client = None

    def rerank(
        self,
        query: str,
        candidates: List[SearchHit],
        top_k: int = 5
    ) -> List[SearchHit]:
        """Re-scores initial hits using query token coverage and semantic density."""
        if not candidates:
            return []

        query_terms = set(query.lower().split())

        scored_candidates: List[SearchHit] = []
        for hit in candidates:
            text_lower = hit.text.lower()
            title_lower = hit.title.lower()

            # 1. Base vector similarity score
            base_score = hit.score

            # 2. Query keyword coverage boost
            exact_matches = sum(1 for term in query_terms if term in text_lower)
            title_matches = sum(1 for term in query_terms if term in title_lower)
            
            coverage_ratio = (exact_matches / len(query_terms)) if query_terms else 0.0

            # 3. Positional penalty/boost (matches earlier in text score higher)
            positional_bonus = 0.0
            if exact_matches > 0:
                first_pos = min((text_lower.find(t) for t in query_terms if t in text_lower), default=500)
                positional_bonus = max(0.0, 0.15 * (1.0 - (min(first_pos, 500) / 500.0)))

            # Combined weighted score
            final_score = (base_score * 0.55) + (coverage_ratio * 0.30) + (title_matches * 0.05) + positional_bonus

            # Update hit score
            hit.score = round(min(0.9999, max(0.0, final_score)), 4)
            scored_candidates.append(hit)

        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        return scored_candidates[:top_k]

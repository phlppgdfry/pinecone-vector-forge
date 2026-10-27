"""Pattern 1: Production Hybrid Search (Dense Semantic + Sparse Lexical).

Combines high-dimensional semantic dense embeddings with BM25-style keyword sparse
vectors using calibrated MinMax score blending and Reciprocal Rank Fusion (RRF).
Solves the cosine scale-invariance problem on Pinecone Serverless cosine indexes.
"""

import math
import re
from typing import List, Dict, Any, Optional
from ..core.embeddings import get_embedding_generator
from ..core.pinecone_client import get_pinecone_manager
from ..core.models import SearchHit


class HybridSearchEngine:
    """Executes mathematically sound hybrid retrieval over Pinecone Serverless indices."""

    def __init__(self):
        self.embedder = get_embedding_generator()
        self.pc = get_pinecone_manager()

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None,
        fusion_method: str = "score_blend"  # "score_blend" or "rrf"
    ) -> List[SearchHit]:
        """
        Alpha blending parameter:
        alpha = 1.0 -> 100% Dense (pure semantic context)
        alpha = 0.0 -> 100% Sparse (pure keyword exact match)
        alpha = 0.7 -> Optimal production balance

        Uses two-pass retrieval to support cosine indexes where sparse vectors
        cannot be directly submitted in a single Pinecone query call.
        """
        fetch_k = max(top_k * 3, 10)

        # 1. Dense retrieval pass
        dense_vec = self.embedder.embed_text(query)
        dense_hits, _, _ = self.pc.query_vectors(
            vector=dense_vec,
            top_k=fetch_k,
            namespace=namespace,
            filter_dict=filter_dict
        )

        if alpha >= 0.99:
            return dense_hits[:top_k]

        # 2. Sparse lexical pass (BM25 term frequency over candidate corpus)
        sparse_hits = self._compute_lexical_hits(query, dense_hits, namespace, filter_dict, top_k=fetch_k)

        if alpha <= 0.01:
            return sparse_hits[:top_k]

        # 3. Fusion
        if fusion_method == "rrf":
            return self.reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, top_n=top_k)

        return self._score_blended_fusion(dense_hits, sparse_hits, alpha=alpha, top_k=top_k)

    def _compute_lexical_hits(
        self,
        query: str,
        initial_hits: List[SearchHit],
        namespace: str,
        filter_dict: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[SearchHit]:
        """Computes BM25-style lexical scores over documents."""
        query_terms = re.findall(r"\w+", query.lower())
        if not query_terms:
            return initial_hits[:top_k]

        # Pool hits from local store or initial candidate list
        candidate_map: Dict[str, SearchHit] = {h.id: h for h in initial_hits}

        # Also pull any available documents in the local namespace store for broad lexical match
        local_ns = self.pc._local_store.get(namespace, {})
        for rec_id, rec in local_ns.items():
            if rec_id not in candidate_map:
                meta = rec.get("metadata", {})
                if filter_dict and not self.pc._matches_filter(meta, filter_dict):
                    continue
                candidate_map[rec_id] = SearchHit(
                    id=rec_id,
                    score=0.0,
                    text=meta.get("text", ""),
                    title=meta.get("parent_title", meta.get("title", "Untitled")),
                    category=meta.get("category", "general"),
                    tags=meta.get("tags", []),
                    document_id=meta.get("document_id"),
                    metadata=meta
                )

        scored: List[SearchHit] = []
        for hit in candidate_map.values():
            text_tokens = re.findall(r"\w+", (hit.title + " " + hit.text).lower())
            if not text_tokens:
                continue

            doc_len = len(text_tokens)
            score = 0.0
            for term in query_terms:
                tf = text_tokens.count(term)
                if tf > 0:
                    # BM25 term saturation formula: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_dl)))
                    # Using k1=1.5, b=0.75, avg_dl=50
                    bm25_tf = (tf * 2.5) / (tf + 1.5 * (0.25 + 0.75 * (doc_len / 50.0)))
                    score += bm25_tf

            hit_copy = SearchHit(
                id=hit.id,
                score=round(score, 4),
                text=hit.text,
                title=hit.title,
                category=hit.category,
                tags=hit.tags,
                document_id=hit.document_id,
                metadata=hit.metadata
            )
            scored.append(hit_copy)

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def _score_blended_fusion(
        self,
        dense_hits: List[SearchHit],
        sparse_hits: List[SearchHit],
        alpha: float,
        top_k: int
    ) -> List[SearchHit]:
        """
        Calibrated MinMax normalized convex combination:
        S_blended = alpha * norm(S_dense) + (1 - alpha) * norm(S_sparse)
        """
        dense_map = {h.id: h.score for h in dense_hits}
        sparse_map = {h.id: h.score for h in sparse_hits}

        all_ids = set(dense_map.keys()) | set(sparse_map.keys())
        hit_objects = {h.id: h for h in (dense_hits + sparse_hits)}

        d_vals = list(dense_map.values())
        s_vals = list(sparse_map.values())

        d_min = min(d_vals) if d_vals else 0.0
        d_max = max(d_vals) if d_vals else 1.0
        d_range = (d_max - d_min) if (d_max - d_min) > 1e-6 else 1.0

        s_min = min(s_vals) if s_vals else 0.0
        s_max = max(s_vals) if s_vals else 1.0
        s_range = (s_max - s_min) if (s_max - s_min) > 1e-6 else 1.0

        blended_hits: List[SearchHit] = []
        for doc_id in all_ids:
            d_raw = dense_map.get(doc_id, 0.0)
            s_raw = sparse_map.get(doc_id, 0.0)

            # Saturated lexical score: s / (s + 2.0) maps [0, inf) -> [0, 1)
            s_norm = s_raw / (s_raw + 2.0) if s_raw > 0 else 0.0
            d_norm = max(0.0, min(1.0, d_raw))

            # Calibrated convex blend preserving absolute similarity scale for guardrails
            blended_score = (alpha * d_norm) + ((1.0 - alpha) * s_norm)

            base_hit = hit_objects[doc_id]
            blended_hits.append(
                SearchHit(
                    id=base_hit.id,
                    score=round(float(blended_score), 4),
                    text=base_hit.text,
                    title=base_hit.title,
                    category=base_hit.category,
                    tags=base_hit.tags,
                    document_id=base_hit.document_id,
                    metadata=base_hit.metadata
                )
            )

        blended_hits.sort(key=lambda x: x.score, reverse=True)
        return blended_hits[:top_k]

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
            merged_hits.append(
                SearchHit(
                    id=item.id,
                    score=round(rrf_scores[doc_id] * 10.0, 4),
                    text=item.text,
                    title=item.title,
                    category=item.category,
                    tags=item.tags,
                    document_id=item.document_id,
                    metadata=item.metadata
                )
            )

        return merged_hits

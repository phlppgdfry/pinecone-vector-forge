"""Pattern 1: Production Hybrid Search (Independent Dense Semantic + Sparse Lexical Inverted Index).

Combines high-dimensional semantic dense embeddings with an independent Okapi BM25
inverted index to guarantee zero candidate-selection bias:
- Pass 1: Pinecone Serverless ANN query fetches Top-N dense candidates.
- Pass 2: Inverted BM25 index fetches Top-N lexical candidates directly from inverted postings.
- Fusion: Merges both independent candidate streams via calibrated score blending or
  Reciprocal Rank Fusion (RRF, k=60).
"""

import math
import re
from typing import List, Dict, Any, Optional
from ..core.embeddings import get_embedding_generator
from ..core.pinecone_client import get_pinecone_manager
from ..core.inverted_index import get_inverted_index_registry
from ..core.models import SearchHit


class HybridSearchEngine:
    """Executes independent two-pass hybrid retrieval over Pinecone Serverless and Inverted Index."""

    def __init__(self):
        self.embedder = get_embedding_generator()
        self.pc = get_pinecone_manager()
        self.inv_registry = get_inverted_index_registry()

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None,
        fusion_method: str = "score_blend",  # "score_blend" or "rrf"
        fusion_strategy: Optional[str] = None
    ) -> List[SearchHit]:
        """
        Alpha blending parameter:
        alpha = 1.0 -> 100% Dense (pure semantic context)
        alpha = 0.0 -> 100% Sparse (pure keyword exact match)
        alpha = 0.7 -> Optimal production balance

        Solves candidate-selection bias: Pass 1 (Dense) and Pass 2 (Sparse)
        are queried independently from separate index structures and merged.
        """
        active_fusion = fusion_strategy or fusion_method
        fetch_k = max(top_k * 3, 10)

        # Pass 1: Independent Dense ANN retrieval pass
        dense_hits: List[SearchHit] = []
        if alpha > 0.01:
            dense_vec = self.embedder.embed_text(query)
            dense_hits, _, _ = self.pc.query_vectors(
                vector=dense_vec,
                top_k=fetch_k,
                namespace=namespace,
                filter_dict=filter_dict
            )

        if alpha >= 0.99:
            return dense_hits[:top_k]

        # Pass 2: Independent Sparse Okapi BM25 retrieval pass from inverted postings
        sparse_hits = self.inv_registry.search_lexical(
            query=query,
            top_k=fetch_k,
            namespace=namespace,
            filter_dict=filter_dict
        )

        # Fallback if inverted index was empty for this namespace: populate from local store or dense hits
        if not sparse_hits and dense_hits:
            sparse_hits = self._compute_lexical_hits_fallback(query, dense_hits, namespace, filter_dict, top_k=fetch_k)

        if alpha <= 0.01:
            return sparse_hits[:top_k]

        # Pass 3: Fusion of Independent Streams
        if active_fusion == "rrf":
            return self.reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, top_n=top_k)

        return self._score_blended_fusion(dense_hits, sparse_hits, alpha=alpha, top_k=top_k)

    def _compute_lexical_hits_fallback(
        self,
        query: str,
        initial_hits: List[SearchHit],
        namespace: str,
        filter_dict: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[SearchHit]:
        """Fallback lexical computation if inverted index is uninitialized."""
        query_terms = re.findall(r"\w+", query.lower())
        if not query_terms:
            return initial_hits[:top_k]

        candidate_map: Dict[str, SearchHit] = {h.id: h for h in initial_hits}
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
        Calibrated convex combination of independent dense and sparse hits.
        S_blended = alpha * S_dense + (1 - alpha) * (S_sparse / (S_sparse + 2.0))
        """
        dense_map = {h.id: h.score for h in dense_hits}
        sparse_map = {h.id: h.score for h in sparse_hits}

        all_ids = set(dense_map.keys()) | set(sparse_map.keys())
        hit_objects = {h.id: h for h in (dense_hits + sparse_hits)}

        blended_hits: List[SearchHit] = []
        for doc_id in all_ids:
            d_raw = dense_map.get(doc_id, 0.0)
            s_raw = sparse_map.get(doc_id, 0.0)

            # Saturated lexical score: s / (s + 2.0) maps [0, inf) -> [0, 1)
            s_norm = s_raw / (s_raw + 2.0) if s_raw > 0 else 0.0
            d_norm = max(0.0, min(1.0, d_raw))

            # Calibrated convex blend preserving absolute similarity scale
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
        Standard Reciprocal Rank Fusion (RRF) over independent dense and sparse streams:
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

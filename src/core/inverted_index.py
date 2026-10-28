"""Inverted Index & Okapi BM25 Engine for Independent Lexical Retrieval.

Provides genuine inverted index postings (term -> [(doc_id, tf), ...]), corpus-level
document frequency (DF), Inverse Document Frequency (IDF), and document length statistics.
Guarantees zero candidate-selection bias in hybrid retrieval: documents matching rare
lexical keywords or error codes that have near-zero dense similarity are independently
retrieved and fused with dense candidates.
"""

import math
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from .models import SearchHit


@dataclass
class InvertedPosting:
    doc_id: str
    term_freq: int


class NamespaceInvertedIndex:
    """Okapi BM25 Inverted Index scoped to a single namespace partition."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        # Inverted index: term -> list of InvertedPosting
        self.postings: Dict[str, List[InvertedPosting]] = defaultdict(list)
        # Document catalog: doc_id -> metadata dictionary
        self.docs: Dict[str, Dict[str, Any]] = {}
        # Document length: doc_id -> token count
        self.doc_lengths: Dict[str, int] = {}
        # Document frequency: term -> number of docs containing term
        self.doc_frequencies: Dict[str, int] = defaultdict(int)
        self.total_docs: int = 0
        self.total_tokens: int = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercases and extracts alphanumeric word and code tokens."""
        # Supports error codes like ERR_CONN_REFUSED_8091 or CVE-2024-9128
        return [tok for tok in re.findall(r"[a-z0-9_\-]+", text.lower()) if len(tok) > 1]

    def add_document(self, doc_id: str, title: str, text: str, category: str, tags: List[str], metadata: Dict[str, Any]) -> None:
        """Indexes or replaces a document in the inverted index."""
        if doc_id in self.docs:
            self.remove_document(doc_id)

        tokens = self.tokenize(f"{title} {title} {text} {' '.join(tags)}")
        doc_len = len(tokens)
        if doc_len == 0:
            return

        self.docs[doc_id] = {
            "title": title,
            "text": text,
            "category": category,
            "tags": tags,
            **metadata
        }
        self.doc_lengths[doc_id] = doc_len
        self.total_docs += 1
        self.total_tokens += doc_len

        # Count term frequencies within this document
        tf_map: Dict[str, int] = defaultdict(int)
        for term in tokens:
            tf_map[term] += 1

        for term, tf in tf_map.items():
            self.postings[term].append(InvertedPosting(doc_id=doc_id, term_freq=tf))
            self.doc_frequencies[term] += 1

    def remove_document(self, doc_id: str) -> bool:
        """Removes a document and updates corpus DF statistics."""
        if doc_id not in self.docs:
            return False

        old_len = self.doc_lengths.pop(doc_id, 0)
        self.docs.pop(doc_id, None)
        self.total_docs = max(0, self.total_docs - 1)
        self.total_tokens = max(0, self.total_tokens - old_len)

        for term, post_list in list(self.postings.items()):
            new_list = [p for p in post_list if p.doc_id != doc_id]
            if len(new_list) != len(post_list):
                self.doc_frequencies[term] = max(0, self.doc_frequencies[term] - 1)
                if not new_list:
                    del self.postings[term]
                else:
                    self.postings[term] = new_list
        return True

    def search_bm25(self, query: str, top_k: int = 10, filter_dict: Optional[Dict[str, Any]] = None) -> List[SearchHit]:
        """
        Executes Okapi BM25 ranking across all postings.
        Formula:
        IDF(t) = ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
        BM25(D, Q) = sum( IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (|D| / avgdl))) )
        """
        if self.total_docs == 0:
            return []

        avg_dl = max(1.0, self.total_tokens / self.total_docs)
        query_terms = self.tokenize(query)
        if not query_terms:
            return []

        # Accumulator for document scores: doc_id -> score
        scores: Dict[str, float] = defaultdict(float)

        for term in query_terms:
            postings = self.postings.get(term)
            if not postings:
                continue

            n_t = self.doc_frequencies.get(term, 0)
            # Standard Robertson-Spärck Jones IDF
            idf = math.log(1.0 + (self.total_docs - n_t + 0.5) / (n_t + 0.5))

            for post in postings:
                doc_len = self.doc_lengths.get(post.doc_id, avg_dl)
                tf = post.term_freq
                # BM25 term weight
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_dl))
                term_score = idf * (num / denom)
                scores[post.doc_id] += term_score

        # Convert to SearchHit objects
        hits: List[SearchHit] = []
        for doc_id, score in scores.items():
            meta = self.docs[doc_id]
            # Apply metadata filter if specified
            if filter_dict and not self._matches_filter(meta, filter_dict):
                continue

            hits.append(
                SearchHit(
                    id=doc_id,
                    score=round(float(score), 4),
                    text=meta.get("text", ""),
                    title=meta.get("title", "Untitled"),
                    category=meta.get("category", "general"),
                    tags=meta.get("tags", []),
                    document_id=meta.get("document_id", doc_id),
                    metadata=meta
                )
            )

        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:top_k]

    @staticmethod
    def _matches_filter(metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        for field, condition in filter_dict.items():
            val = metadata.get(field)
            if isinstance(condition, dict):
                for op, target in condition.items():
                    if op == "$eq" and val != target:
                        return False
                    if op == "$in" and (val not in target if not isinstance(val, list) else not any(x in target for x in val)):
                        return False
                    if op == "$gte" and (val is None or val < target):
                        return False
                    if op == "$lte" and (val is None or val > target):
                        return False
            else:
                if val != condition:
                    return False
        return True


class CorpusInvertedIndexRegistry:
    """Singleton managing inverted indexes partitioned across namespaces."""

    def __init__(self):
        self._namespaces: Dict[str, NamespaceInvertedIndex] = defaultdict(NamespaceInvertedIndex)

    def get_namespace_index(self, namespace: str = "default") -> NamespaceInvertedIndex:
        return self._namespaces[namespace]

    def add_document(self, namespace: str, doc_id: str, title: str, text: str, category: str, tags: List[str], metadata: Dict[str, Any]) -> None:
        idx = self.get_namespace_index(namespace)
        idx.add_document(doc_id, title, text, category, tags, metadata)

    def search_lexical(self, query: str, top_k: int = 10, namespace: str = "default", filter_dict: Optional[Dict[str, Any]] = None) -> List[SearchHit]:
        idx = self.get_namespace_index(namespace)
        return idx.search_bm25(query=query, top_k=top_k, filter_dict=filter_dict)


_index_registry: Optional[CorpusInvertedIndexRegistry] = None


def get_inverted_index_registry() -> CorpusInvertedIndexRegistry:
    global _index_registry
    if _index_registry is None:
        _index_registry = CorpusInvertedIndexRegistry()
    return _index_registry

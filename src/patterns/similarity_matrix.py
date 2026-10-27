"""Pattern 10: Vector Similarity Matrix & Semantic Dispersion Analyzer.

Computes pairwise cosine distances across the vector corpus to audit:
1. Knowledge redundancy & near-duplicate clusters.
2. Under-represented semantic topic gaps.
3. Central knowledge hubs (cluster centroids).
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from ..seed.sample_data import KNOWLEDGE_BASE_DOCUMENTS
from ..core.embeddings import get_embedding_generator


@dataclass
class SimilarityMatrixReport:
    document_titles: List[str]
    document_ids: List[str]
    matrix: List[List[float]]
    average_corpus_similarity: float
    most_similar_pair: Tuple[str, str, float]
    most_distinct_pair: Tuple[str, str, float]
    cluster_centroids: List[str]


class SimilarityMatrixAnalyzer:
    """Analyzes geometric properties of the vector index."""

    def __init__(self):
        self.embedder = get_embedding_generator()

    def compute_matrix(self) -> SimilarityMatrixReport:
        """Generates full pairwise cosine similarity matrix over documents."""
        docs = KNOWLEDGE_BASE_DOCUMENTS
        titles = [d.title for d in docs]
        doc_ids = [d.id for d in docs]

        texts = [d.content for d in docs]
        vectors = np.array(self.embedder.embed_batch(texts), dtype=np.float32)

        # Normalize rows
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = vectors / norms

        # Cosine matrix = V * V.T
        cos_matrix = np.dot(normalized, normalized.T)
        matrix_list = [[round(float(val), 4) for val in row] for row in cos_matrix]

        # Find most similar and most distinct non-identical pairs
        n = len(docs)
        max_sim = -1.0
        min_sim = 2.0
        max_pair = ("", "", 0.0)
        min_pair = ("", "", 1.0)
        all_off_diag = []

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(cos_matrix[i, j])
                all_off_diag.append(sim)
                if sim > max_sim:
                    max_sim = sim
                    max_pair = (titles[i], titles[j], round(sim, 4))
                if sim < min_sim:
                    min_sim = sim
                    min_pair = (titles[i], titles[j], round(sim, 4))

        avg_similarity = float(np.mean(all_off_diag)) if all_off_diag else 0.0

        # Cluster centroids: documents with highest mean similarity to all others
        mean_similarities = [float(np.mean(cos_matrix[i])) for i in range(n)]
        sorted_indices = sorted(range(n), key=lambda idx: mean_similarities[idx], reverse=True)
        centroids = [titles[i] for i in sorted_indices[:3]]

        return SimilarityMatrixReport(
            document_titles=titles,
            document_ids=doc_ids,
            matrix=matrix_list,
            average_corpus_similarity=round(avg_similarity, 4),
            most_similar_pair=max_pair,
            most_distinct_pair=min_pair,
            cluster_centroids=centroids
        )

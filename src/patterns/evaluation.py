"""Pattern 7: Quantitative Retrieval Evaluation & Benchmarking Engine.

Measures Mean Reciprocal Rank (MRR@K), Recall@K, Precision@K, and Latency
against benchmark gold standards.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .hybrid_search import HybridSearchEngine


@dataclass
class EvalBenchmarkResult:
    mrr_at_k: float
    recall_at_k: float
    precision_at_k: float
    avg_latency_ms: float
    total_eval_queries: int


# Golden evaluation dataset (Query -> Expected Relevant Document IDs)
GOLDEN_DATASET: List[Tuple[str, List[str]]] = [
    ("How does serverless scaling handle idle costs?", ["doc_pinecone_serverless_01"]),
    ("Sparse-dense hybrid search with BM25", ["doc_hybrid_search_02"]),
    ("RAG hallucination prevention and parent-document chunking", ["doc_rag_hallucination_03"]),
    ("Zero-leakage multi-tenancy and namespace isolation", ["doc_multi_tenant_04"]),
    ("Two-stage cross-encoder retrieval pipeline", ["doc_two_stage_rerank_05"]),
    ("Automating Pinecone CLI deployments in CI/CD", ["doc_cli_automation_06"]),
    ("Cosine versus dot product vector distance metrics", ["doc_vector_metrics_07"]),
]


class RetrievalEvaluator:
    """Computes academic-standard IR (Information Retrieval) evaluation metrics."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.engine = HybridSearchEngine()

    def run_benchmark(self, alpha: float = 0.70, namespace: str = "default") -> EvalBenchmarkResult:
        """Executes golden queries and computes MRR, Recall, and Precision."""
        import time

        reciprocal_ranks = []
        recalls = []
        precisions = []
        latencies = []

        for query, expected_ids in GOLDEN_DATASET:
            t0 = time.perf_counter()
            hits = self.engine.search(query=query, top_k=self.top_k, alpha=alpha, namespace=namespace)
            latency = (time.perf_counter() - t0) * 1000
            latencies.append(latency)

            retrieved_doc_ids = [h.document_id for h in hits if h.document_id]

            # 1. Reciprocal Rank (RR)
            rr = 0.0
            for rank, doc_id in enumerate(retrieved_doc_ids, 1):
                if doc_id in expected_ids:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            # 2. Recall@K
            true_positives = sum(1 for d in retrieved_doc_ids if d in expected_ids)
            recall = true_positives / len(expected_ids) if expected_ids else 0.0
            recalls.append(recall)

            # 3. Precision@K
            precision = true_positives / len(hits) if hits else 0.0
            precisions.append(precision)

        avg_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return EvalBenchmarkResult(
            mrr_at_k=round(avg_mrr, 4),
            recall_at_k=round(avg_recall, 4),
            precision_at_k=round(avg_precision, 4),
            avg_latency_ms=round(avg_latency, 2),
            total_eval_queries=len(GOLDEN_DATASET)
        )

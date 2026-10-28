"""Vector Database Benchmark & Architecture Decision Lab Tests.

Validates multi-engine vector database comparison (Pinecone Serverless vs pgvector vs Qdrant vs Weaviate),
Architectural Decision Records (ADRs), and verifies candidate-selection bias elimination in hybrid retrieval.
"""

from fastapi.testclient import TestClient

from src.patterns.vector_db_lab import VectorDBArchitectureLab
from src.core.inverted_index import CorpusInvertedIndexRegistry
from src.patterns.hybrid_search import HybridSearchEngine
from src.web.app import app

client = TestClient(app)


def test_vector_db_comparative_benchmark():
    """Verify execution of multi-engine comparative benchmark across 4 systems."""
    lab = VectorDBArchitectureLab()
    report = lab.run_comparative_benchmark()

    assert report.total_queries_evaluated >= 100
    assert len(report.results) == 4
    engine_names = [r.engine_name for r in report.results]
    assert "pinecone_serverless" in engine_names
    assert "pgvector_hnsw" in engine_names
    assert "qdrant_rust" in engine_names
    assert "weaviate_cloud" in engine_names

    # Check that Pinecone has zero idle cost
    pinecone_res = next(r for r in report.results if r.engine_name == "pinecone_serverless")
    assert pinecone_res.idle_cost_monthly_usd == 0.00
    assert pinecone_res.p50_latency_ms > 0
    assert pinecone_res.accuracy_mrr > 0.85

    # Check pgvector idle cost reflects persistent instance
    pg_res = next(r for r in report.results if r.engine_name == "pgvector_hnsw")
    assert pg_res.idle_cost_monthly_usd > 20.00


def test_vector_db_adrs():
    """Verify Architectural Decision Records (ADRs) are documented for all candidate engines."""
    lab = VectorDBArchitectureLab()
    adrs = lab.get_adrs()
    assert len(adrs) == 4
    for adr in adrs:
        assert "engine" in adr
        assert "recommended_workload" in adr
        assert "key_strengths" in adr
        assert "critical_limitations" in adr


def test_api_vector_db_endpoints():
    """Verify FastAPI endpoints for benchmark and ADRs."""
    bench_res = client.get("/api/vector-dbs/benchmark")
    assert bench_res.status_code == 200
    bench_data = bench_res.json()
    assert "results" in bench_data
    assert len(bench_data["results"]) == 4

    adr_res = client.get("/api/vector-dbs/adrs")
    assert adr_res.status_code == 200
    adrs = adr_res.json()
    assert len(adrs) == 4


def test_candidate_selection_bias_elimination():
    """
    Candidate-Selection Bias Test:
    Ensures that an exact technical identifier (e.g. 'ERR_PG_REPL_LAG_902')
    is successfully retrieved via independent full-corpus lexical search (Pass 2)
    and fused with dense search, rather than being discarded by an initial dense filter.
    """
    engine = HybridSearchEngine()

    # Search for an exact error code that exists in the expanded corpus
    query = "ERR_PG_REPL_LAG_902 replication failure"
    hits = engine.search(
        query=query,
        top_k=5,
        alpha=0.5,
        namespace="default",
        fusion_strategy="score_blend"
    )

    assert len(hits) > 0
    # The top hit should be the PostgreSQL replication lag runbook
    top_hit = hits[0]
    assert "ERR_PG_REPL_LAG_902" in top_hit.text or "PostgreSQL" in top_hit.title

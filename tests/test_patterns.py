"""Test architecture patterns: filtering, chunking, reranking, and multi-tenancy."""

from src.core.models import Document, SearchHit
from src.patterns.metadata_filtering import MetadataFilterBuilder
from src.patterns.contextual_chunking import HierarchicalChunker
from src.patterns.reranking import TwoStageReranker
from src.patterns.multi_tenancy import MultiTenantManager
from src.patterns.rag_agent import GroundedRAGAgent
from src.patterns.evaluation import RetrievalEvaluator


def test_metadata_filter_builder():
    builder = MetadataFilterBuilder()
    flt = builder.category("security_compliance").gte("quality_score", 0.9).build()
    
    assert "$and" in flt
    assert {"category": {"$eq": "security_compliance"}} in flt["$and"]
    assert {"quality_score": {"$gte": 0.9}} in flt["$and"]


def test_hierarchical_chunker():
    chunker = HierarchicalChunker(chunk_size=50, overlap=10, enable_contextual_prefix=True)
    doc = Document(
        id="doc_test_1",
        title="Scaling Vector Indexes",
        category="engineering",
        tags=["pinecone", "scaling"],
        content=" ".join(["word" for _ in range(120)])
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    assert chunks[0].document_id == "doc_test_1"
    assert chunks[0].parent_title == "Scaling Vector Indexes"

    records = chunker.process_and_vectorize(doc)
    assert len(records) == len(chunks)
    assert len(records[0].values) == 1536
    assert records[0].metadata["parent_title"] == "Scaling Vector Indexes"


def test_two_stage_reranker():
    reranker = TwoStageReranker()
    candidates = [
        SearchHit(
            id="1", score=0.6, text="Unrelated text about gardening and plants",
            title="Gardening", category="general"
        ),
        SearchHit(
            id="2", score=0.55, text="Serverless scaling handles zero idle cost efficiently",
            title="Pinecone Serverless", category="vector_architecture"
        )
    ]
    reranked = reranker.rerank(query="serverless idle cost", candidates=candidates, top_k=2)
    # The matching hit should be boosted to rank #1
    assert reranked[0].id == "2"
    assert reranked[0].score > reranked[1].score


def test_multi_tenant_namespace_sanitization():
    mgr = MultiTenantManager()
    ns = mgr.get_tenant_namespace("Acme Corp / EMEA #1")
    assert ns == "tenant_acme_corp___emea__1"


def test_grounded_rag_agent():
    from src.seed.ingest import run_ingestion_pipeline
    run_ingestion_pipeline()
    agent = GroundedRAGAgent(confidence_threshold=0.1)
    res = agent.answer_query("How does serverless scaling handle idle costs?")
    assert res.query == "How does serverless scaling handle idle costs?"
    assert len(res.citations) > 0
    assert res.passed_guardrail is True
    assert "Pinecone" in res.synthesized_answer


def test_retrieval_evaluator_benchmark():
    from src.seed.ingest import run_ingestion_pipeline
    run_ingestion_pipeline()
    evaluator = RetrievalEvaluator(top_k=3)
    results = evaluator.run_benchmark(alpha=0.7)
    assert results.total_eval_queries >= 7
    assert results.mrr_at_k > 0.0
    assert results.recall_at_k > 0.0
    assert results.precision_at_k > 0.0
    assert results.ndcg_at_k > 0.0


def test_self_query_deconstructor():
    from src.patterns.self_query import SelfQueryDeconstructor
    parser = SelfQueryDeconstructor()
    plan = parser.deconstruct("Show critical security architecture with quality above 90%")
    assert plan.detected_priority == "critical"
    assert plan.detected_min_quality == 0.9
    assert plan.filter_dict is not None
    assert "$and" in plan.filter_dict or "quality_score" in plan.filter_dict


def test_hyde_engine():
    from src.patterns.hyde import HyDESearchEngine
    engine = HyDESearchEngine()
    passage = engine.generate_hypothetical_passage("How does serverless scaling handle idle costs?")
    assert len(passage) > 50
    assert "serverless" in passage.lower()


def test_similarity_matrix_analyzer():
    from src.patterns.similarity_matrix import SimilarityMatrixAnalyzer
    analyzer = SimilarityMatrixAnalyzer()
    report = analyzer.compute_matrix()
    assert len(report.matrix) >= 7
    assert len(report.matrix[0]) >= 7
    # Diagonal should be 1.0 (self-similarity)
    assert abs(report.matrix[0][0] - 1.0) < 1e-3
    assert len(report.cluster_centroids) > 0


def test_pinecone_assistant_manager():
    from src.patterns.pinecone_assistant import PineconeAssistantManager
    mgr = PineconeAssistantManager()
    info = mgr.get_or_create_assistant()
    assert "name" in info
    assert info["name"] == "vector-forge-assistant"
    files = mgr.list_files()
    assert isinstance(files, list)


def test_vector_lifecycle_and_deletion():
    from src.core.pinecone_client import get_pinecone_manager
    from src.core.models import VectorRecord
    mgr = get_pinecone_manager()

    rec = VectorRecord(
        id="test_del_123",
        values=[0.1] * 1536,
        metadata={"category": "temporary_test"}
    )
    upsert_res = mgr.upsert_vectors([rec], namespace="test_cleanup_ns")
    assert upsert_res["upserted_count"] == 1

    del_res = mgr.delete_vectors(ids=["test_del_123"], namespace="test_cleanup_ns")
    assert del_res["status"].startswith("success")

    wipe_res = mgr.delete_vectors(delete_all=True, namespace="test_cleanup_ns")
    assert wipe_res["status"].startswith("success")


def test_tenant_authorization_boundary():
    import pytest
    from src.patterns.multi_tenancy import MultiTenantManager, TenantAuthorizationError
    mgr = MultiTenantManager()

    # Legitimate access
    mgr.authorize_tenant_access(caller_tenant_id="tenant_alpha", requested_tenant_id="tenant_alpha")

    # Unauthorized access must raise TenantAuthorizationError
    with pytest.raises(TenantAuthorizationError):
        mgr.authorize_tenant_access(caller_tenant_id="tenant_intruder", requested_tenant_id="tenant_alpha")

    # Admin override works
    mgr.authorize_tenant_access(caller_tenant_id="admin_service", requested_tenant_id="tenant_alpha", is_admin=True)


def test_finops_cost_accounting():
    from src.core.finops import FinOpsCostModel
    est = FinOpsCostModel.estimate_search_cost(top_k=5, rerank=True, candidates_count=10)
    assert est.estimated_rcu == 1
    assert est.estimated_cost_usd > 0.0
    assert "rcu_cost_usd" in est.cost_breakdown
    assert "rerank_cost_usd" in est.cost_breakdown


def test_rag_entity_grounding_and_abstention():
    from src.patterns.rag_agent import GroundedRAGAgent
    agent = GroundedRAGAgent(confidence_threshold=0.25)

    # Completely out-of-domain inquiry should trigger abstention
    res = agent.answer_query("Recipe for sourdough bread with 70% hydration")
    assert res.passed_guardrail is False
    assert "abstain" in res.synthesized_answer.lower()
    assert res.grounding_status in ("abstained_low_confidence", "abstained_out_of_domain")


def test_full_corpus_retrieval_benchmark():
    from src.patterns.evaluation import RetrievalEvaluator
    evaluator = RetrievalEvaluator(top_k=3)
    res = evaluator.run_benchmark(alpha=0.70, enable_rerank=False, use_full_corpus=True)
    assert res.total_eval_queries >= 25
    assert res.abstention_accuracy > 0.50
    assert res.mrr_at_k > 0.0

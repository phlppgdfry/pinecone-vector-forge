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
    chunker = HierarchicalChunker(chunk_size=50, overlap=10)
    doc = Document(
        id="doc_test_1",
        title="Scaling Vector Indexes",
        category="engineering",
        content=" ".join(["word" for _ in range(120)])
    )
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    assert chunks[0].document_id == "doc_test_1"
    assert chunks[0].parent_title == "Scaling Vector Indexes"


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
    assert results.total_eval_queries == 7
    assert results.mrr_at_k > 0.0
    assert results.recall_at_k > 0.0

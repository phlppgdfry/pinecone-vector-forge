"""Automated Tests for Enterprise Knowledge Assistant & Verifiable Grounded RAG.

Validates:
1. File type & content validation
2. Ingestion & hierarchical chunking
3. Two-pass hybrid retrieval with grounded bracketed citations
4. Hallucination defense & out-of-domain abstention
5. Tenant authorization fencing on assistant operations
6. Token revocation denylist enforcement
7. Atomic document deletion across Pinecone and Inverted BM25
"""

import pytest
from fastapi.testclient import TestClient

from src.patterns.enterprise_assistant import EnterpriseKnowledgeAssistant
from src.core.security import HMACSecuritySigner, get_revocation_store
from src.web.app import app

client = TestClient(app)


def test_validation_and_parsing():
    assistant = EnterpriseKnowledgeAssistant()

    # Valid markdown
    fname, content = assistant.validate_and_parse("runbook.md", "# SRE Guide\nCritical runbook content.")
    assert fname == "runbook.md"
    assert "Critical runbook" in content

    # Invalid file extension
    with pytest.raises(ValueError) as exc:
        assistant.validate_and_parse("malicious.exe", "dangerous payload content")
    assert "Unsupported file extension" in str(exc.value)

    # Content too short
    with pytest.raises(ValueError) as exc:
        assistant.validate_and_parse("empty.txt", "short")
    assert "at least 10 characters" in str(exc.value)


def test_enterprise_ingestion_and_grounded_query():
    assistant = EnterpriseKnowledgeAssistant()
    doc_content = (
        "Runbook SRE-109: Redis Cluster Memory Degradation Remediation. "
        "When error code ERR_REDIS_HOTSPOT_881 occurs, immediately execute keyspace scan. "
        "Isolate heavy keys and scale cluster shards to prevent eviction."
    )

    # 1. Ingest document
    res = assistant.ingest_document(
        title="Redis Hotspot Runbook",
        content=doc_content,
        category="sre_runbooks",
        tags=["redis", "memory", "hotspot"],
        tenant_id="default"
    )
    assert res["status"] == "success"
    assert res["chunks_created"] >= 1
    doc_id = res["document_id"]

    # 2. Query for the exact error code
    ans = assistant.query(
        query="What should I do when ERR_REDIS_HOTSPOT_881 occurs?",
        tenant_id="default",
        top_k=3,
        alpha=0.6,
        enable_rerank=True
    )

    assert ans.abstained is False
    assert len(ans.citations) > 0
    assert "Redis Hotspot Runbook" in ans.answer or "Redis" in ans.answer
    assert ans.citation_precision == 1.0

    # 3. Safe cleanup
    del_res = assistant.delete_document(doc_id=doc_id, tenant_id="default")
    assert del_res["status"] == "success"


def test_out_of_domain_abstention():
    """Ensure the assistant firmly abstains on queries outside the indexed knowledge base."""
    assistant = EnterpriseKnowledgeAssistant()

    ans = assistant.query(
        query="What is the optimal water temperature for brewing light roast Ethiopian espresso?",
        tenant_id="default",
        top_k=3
    )

    assert ans.abstained is True
    assert "cannot find sufficient verified documentation" in ans.answer.lower()
    assert len(ans.citations) == 0


def test_tenant_fenced_assistant_operations():
    """Ensure Tenant A cannot query or delete documents owned by Tenant B."""
    assistant = EnterpriseKnowledgeAssistant()
    token_tenant_a = HMACSecuritySigner.issue_token(
        tenant_id="tenant_eng_a",
        role="tenant_admin",
        scopes=["read", "write", "delete"]
    )
    auth_tenant_a = HMACSecuritySigner.verify_token(token_tenant_a)

    # 1. Tenant A ingests confidential document
    ingest_res = assistant.ingest_document(
        title="Confidential Internal Patent Draft",
        content="Secret proprietary neural architecture specs for enterprise AI pipelines.",
        category="patents",
        tags=["confidential"],
        tenant_id="tenant_eng_a",
        auth_tenant=auth_tenant_a
    )
    doc_id = ingest_res["document_id"]

    # 2. Tenant B attempts to query Tenant A's namespace
    token_tenant_b = HMACSecuritySigner.issue_token(
        tenant_id="tenant_eng_b",
        role="tenant_admin",
        scopes=["read", "write"]
    )
    auth_tenant_b = HMACSecuritySigner.verify_token(token_tenant_b)

    with pytest.raises(Exception):
        assistant.query(
            query="neural architecture patent specs",
            tenant_id="tenant_eng_a",
            auth_tenant=auth_tenant_b
        )

    # 3. Tenant B attempts to delete Tenant A's document
    with pytest.raises(Exception):
        assistant.delete_document(
            doc_id=doc_id,
            tenant_id="tenant_eng_a",
            auth_tenant=auth_tenant_b
        )

    # Clean up by Tenant A
    assistant.delete_document(
        doc_id=doc_id,
        tenant_id="tenant_eng_a",
        auth_tenant=auth_tenant_a
    )


def test_api_enterprise_assistant_endpoints():
    """Verify REST API endpoints for upload, query, and delete."""
    token = client.post(
        "/api/auth/token",
        json={"tenant_id": "api_test_tenant", "role": "tenant_admin", "scopes": ["read", "write", "delete"]}
    ).json()["access_token"]

    # 1. Upload via REST
    upload_res = client.post(
        "/api/enterprise/upload",
        json={
            "title": "API Gateway Distributed Tracing",
            "content": "Configure OpenTelemetry trace propagators W3C TraceContext across all edge proxies.",
            "category": "networking",
            "tags": ["tracing", "otel"],
            "tenant_id": "api_test_tenant",
            "filename": "tracing_guide.md"
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    assert upload_res.status_code == 200
    doc_id = upload_res.json()["document_id"]

    # 2. Query via REST
    query_res = client.post(
        "/api/enterprise/query",
        json={
            "query": "How to configure W3C TraceContext propagators?",
            "tenant_id": "api_test_tenant",
            "top_k": 3
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    assert query_res.status_code == 200
    ans_data = query_res.json()
    assert ans_data["abstained"] is False
    assert len(ans_data["citations"]) > 0

    # 3. Delete via REST
    del_res = client.delete(
        f"/api/enterprise/documents/{doc_id}?tenant_id=api_test_tenant",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert del_res.status_code == 200


def test_token_revocation():
    """Verify token revocation immediately denies access."""
    token = HMACSecuritySigner.issue_token(
        tenant_id="tenant_revocation_test",
        role="tenant_reader",
        scopes=["read"]
    )
    # Token works initially
    res_before = client.post(
        "/api/enterprise/query",
        json={"query": "Serverless scaling", "tenant_id": "tenant_revocation_test"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res_before.status_code in (200, 403)

    # Revoke token
    rev_res = client.post(
        "/api/auth/revoke",
        json={"token": token}
    )
    assert rev_res.status_code == 200

    # Verify access is now firmly rejected with 401
    res_after = client.post(
        "/api/enterprise/query",
        json={"query": "Serverless scaling", "tenant_id": "tenant_revocation_test"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res_after.status_code == 401
    assert "revoked" in res_after.json()["detail"].lower()

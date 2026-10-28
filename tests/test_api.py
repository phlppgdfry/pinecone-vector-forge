"""Test FastAPI REST endpoints."""

from fastapi.testclient import TestClient
from src.web.app import app

client = TestClient(app)


def test_status_endpoint():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "index_name" in data
    assert "dimension" in data
    assert data["dimension"] == 1536


def test_cli_endpoint():
    res = client.get("/api/cli")
    assert res.status_code == 200
    data = res.json()
    assert "cli_installed" in data


def test_search_endpoint():
    payload = {
        "query": "Serverless vector indexing",
        "top_k": 3,
        "alpha": 0.7,
        "namespace": "default",
        "category": None,
        "rerank": True
    }
    res = client.post("/api/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "hits" in data
    assert "latency_ms" in data
    assert data["total_hits"] <= 3


def test_custom_ingestion_endpoint():
    payload = {
        "title": "Continuous Vector Audits",
        "content": "Automating continuous vector quality audits prevents embedding drift over time.",
        "category": "devops_automation",
        "tags": ["ci", "testing", "vector"],
        "namespace": "default"
    }
    res = client.post("/api/ingest", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["chunks_created"] >= 1


def test_heatmap_endpoint():
    res = client.get("/api/heatmap")
    assert res.status_code == 200
    data = res.json()
    assert "matrix" in data
    assert "centroids" in data
    assert len(data["matrix"]) >= 7


def test_query_plan_endpoint():
    res = client.post("/api/query-plan", json={"query": "Find critical RAG docs"})
    assert res.status_code == 200
    data = res.json()
    assert data["detected_priority"] == "critical"


def test_hyde_endpoint():
    res = client.post("/api/hyde", json={"query": "Serverless idle scaling", "top_k": 2})
    assert res.status_code == 200
    data = res.json()
    assert "hits" in data


def test_metrics_endpoint():
    res = client.get("/api/metrics")
    assert res.status_code == 200
    assert "vectorforge_total_vectors" in res.text


def test_assistant_chat_endpoint():
    res = client.post("/api/assistant/chat", json={"query": "How does serverless scaling handle idle costs?"})
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert len(data["answer"]) > 0

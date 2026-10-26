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

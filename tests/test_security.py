"""Adversarial and Cryptographic Multi-Tenant Security Tests.

Validates that tenant isolation is cryptographically enforced via HMAC-SHA256,
preventing header spoofing, cross-tenant token leakage, unauthorized writes,
token tampering, and expired session reuse.
"""

import time
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.core.security import HMACSecuritySigner, AuthenticatedTenant
from src.web.app import app

client = TestClient(app)


def test_hmac_token_issuance_and_verification():
    """Verify standard token creation, signature validation, and payload extraction."""
    token = HMACSecuritySigner.issue_token(
        tenant_id="tenant_payments",
        role="tenant_admin",
        scopes=["read", "write"],
        ttl_seconds=1200
    )
    assert token.startswith("vf1.")
    
    tenant = HMACSecuritySigner.verify_token(token)
    assert tenant.tenant_id == "tenant_payments"
    assert tenant.role == "tenant_admin"
    assert tenant.can_read() is True
    assert tenant.can_write() is True
    assert tenant.can_delete() is False  # not in scopes


def test_tampered_signature_rejection():
    """Ensure any modification to payload or signature immediately triggers 401 Unauthorized."""
    token = HMACSecuritySigner.issue_token(
        tenant_id="tenant_billing",
        role="tenant_reader",
        scopes=["read"],
        ttl_seconds=3600
    )
    parts = token.split(".")
    # Tamper with signature
    tampered_sig = parts[0] + "." + parts[1] + "." + ("0" * len(parts[2]))
    with pytest.raises(HTTPException) as exc:
        HMACSecuritySigner.verify_token(tampered_sig)
    assert exc.value.status_code == 401

    # Tamper with payload (flip a byte)
    tampered_payload = parts[0] + ".e30=" + "." + parts[2]
    with pytest.raises(HTTPException) as exc:
        HMACSecuritySigner.verify_token(tampered_payload)
    assert exc.value.status_code == 401


def test_expired_token_rejection():
    """Ensure expired tokens are firmly rejected."""
    expired_token = HMACSecuritySigner.issue_token(
        tenant_id="tenant_ephemeral",
        role="tenant_reader",
        scopes=["read"],
        ttl_seconds=-5  # expired 5 seconds ago
    )
    with pytest.raises(HTTPException) as exc:
        HMACSecuritySigner.verify_token(expired_token)
    assert exc.value.status_code == 401
    assert "expired" in exc.value.detail.lower()


def test_cross_tenant_token_fencing_via_api():
    """Ensure a bearer token for tenant_alpha cannot query tenant_beta."""
    token_alpha = HMACSecuritySigner.issue_token(
        tenant_id="tenant_alpha",
        role="tenant_reader",
        scopes=["read"],
        ttl_seconds=3600
    )

    # 1. Attempt unauthorized cross-tenant query
    unauth_res = client.post(
        "/api/search",
        json={
            "query": "Database replication failover",
            "top_k": 3,
            "namespace": "tenant_beta"
        },
        headers={"Authorization": f"Bearer {token_alpha}"}
    )
    assert unauth_res.status_code == 403
    assert "Cryptographic Security Violation" in unauth_res.json()["detail"]

    # 2. Legitimate query to own tenant
    auth_res = client.post(
        "/api/search",
        json={
            "query": "Database replication failover",
            "top_k": 3,
            "namespace": "tenant_alpha"
        },
        headers={"Authorization": f"Bearer {token_alpha}"}
    )
    assert auth_res.status_code == 200
    assert "hits" in auth_res.json()


def test_insufficient_scope_rejection_on_ingest():
    """Ensure a read-only token cannot perform document ingestion mutations."""
    read_only_token = HMACSecuritySigner.issue_token(
        tenant_id="tenant_read_only",
        role="tenant_reader",
        scopes=["read"],  # No write scope
        ttl_seconds=3600
    )

    res = client.post(
        "/api/ingest",
        json={
            "title": "Malicious Injection Document",
            "content": "Trying to write to tenant namespace with read-only token.",
            "category": "security",
            "tags": ["test"],
            "namespace": "tenant_read_only"
        },
        headers={"Authorization": f"Bearer {read_only_token}"}
    )
    assert res.status_code == 403
    assert "Token lacks 'write' scope" in res.json()["detail"]


def test_token_issue_endpoint():
    """Verify REST endpoint for programmatic tenant token acquisition."""
    res = client.post(
        "/api/auth/token",
        json={
            "tenant_id": "enterprise_customer_42",
            "role": "tenant_admin",
            "scopes": ["read", "write"],
            "ttl_seconds": 1800
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["tenant_id"] == "enterprise_customer_42"
    assert data["role"] == "tenant_admin"
    assert data["token_type"] == "bearer"

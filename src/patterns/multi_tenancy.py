"""Pattern 2: Multi-Tenant Enterprise Namespace Partitioning & Authorization.

Enforces logical data isolation across departments, projects, or B2B SaaS tenants
using native Pinecone serverless namespaces paired with application-level authorization.

Important Architectural Clarification:
Pinecone Serverless namespaces provide LOGICAL partition boundaries inside an index.
They decouple tenants via metadata keys, but execute on shared multi-tenant cloud storage
and stateless compute workers. Application-level authorization is mandatory to prevent
cross-tenant leakage.
"""

from typing import List, Dict, Any, Optional
from ..core.pinecone_client import get_pinecone_manager
from ..core.models import VectorRecord, SearchHit


class TenantAuthorizationError(PermissionError):
    """Raised when an unauthorized entity attempts cross-tenant retrieval or mutation."""
    pass


class MultiTenantManager:
    """Orchestrates tenant-scoped operations, isolation fences, and namespace quotas."""

    def __init__(self):
        self.pc = get_pinecone_manager()

    def sanitize_tenant_id(self, tenant_id: str) -> str:
        """Sanitizes tenant identifier to prevent path traversal or wildcard injection."""
        clean = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tenant_id.lower()).strip("_")
        if not clean:
            raise ValueError("Tenant ID must contain alphanumeric characters.")
        return clean

    def get_tenant_namespace(self, tenant_id: str) -> str:
        """Format sanitized tenant identifier into safe namespace."""
        clean = self.sanitize_tenant_id(tenant_id)
        return f"tenant_{clean}"

    def authorize_tenant_access(self, caller_tenant_id: str, requested_tenant_id: str, is_admin: bool = False) -> None:
        """Validates that caller is authorized to access requested tenant namespace."""
        if is_admin:
            return

        caller_clean = self.sanitize_tenant_id(caller_tenant_id)
        requested_clean = self.sanitize_tenant_id(requested_tenant_id)

        if caller_clean != requested_clean:
            raise TenantAuthorizationError(
                f"Security Violation: Caller '{caller_clean}' is not authorized to access namespace of tenant '{requested_clean}'."
            )

    def upsert_for_tenant(
        self,
        tenant_id: str,
        records: List[VectorRecord],
        caller_tenant_id: Optional[str] = None,
        is_admin: bool = False,
        metadata_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Upsert vectors strictly fenced into tenant's private namespace."""
        if caller_tenant_id:
            self.authorize_tenant_access(caller_tenant_id, tenant_id, is_admin)

        namespace = self.get_tenant_namespace(tenant_id)
        
        # Stamp tenant ownership into payload metadata for defense-in-depth
        for r in records:
            r.metadata["tenant_id"] = tenant_id
            if metadata_override:
                r.metadata.update(metadata_override)

        return self.pc.upsert_vectors(records=records, namespace=namespace)

    def query_for_tenant(
        self,
        tenant_id: str,
        vector: List[float],
        caller_tenant_id: Optional[str] = None,
        is_admin: bool = False,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchHit]:
        """Execute vector query guaranteed isolated to tenant namespace with defense-in-depth."""
        if caller_tenant_id:
            self.authorize_tenant_access(caller_tenant_id, tenant_id, is_admin)

        namespace = self.get_tenant_namespace(tenant_id)

        # Enforce tenant fencing in filter as defense-in-depth
        strict_filter = filter_dict.copy() if filter_dict else {}
        strict_filter["tenant_id"] = tenant_id

        hits, _, _ = self.pc.query_vectors(
            vector=vector,
            top_k=top_k,
            namespace=namespace,
            filter_dict=strict_filter
        )
        return hits

    def delete_tenant_data(
        self,
        tenant_id: str,
        caller_tenant_id: Optional[str] = None,
        is_admin: bool = False
    ) -> Dict[str, Any]:
        """Completely purges a tenant's private namespace."""
        if caller_tenant_id:
            self.authorize_tenant_access(caller_tenant_id, tenant_id, is_admin)

        namespace = self.get_tenant_namespace(tenant_id)
        return self.pc.delete_vectors(delete_all=True, namespace=namespace)

    def get_tenant_stats(self) -> Dict[str, int]:
        """Returns vector counts broken down by tenant namespace."""
        telemetry = self.pc.get_telemetry()
        tenant_counts = {}
        for ns, count in telemetry.namespaces.items():
            if ns.startswith("tenant_"):
                tenant_id = ns.replace("tenant_", "")
                tenant_counts[tenant_id] = count
        return tenant_counts

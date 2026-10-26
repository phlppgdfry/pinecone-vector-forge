"""Pattern 2: Multi-Tenant Enterprise Namespace Partitioning.

Enforces zero-leakage data isolation across departments, projects, or B2B SaaS tenants
using native Pinecone serverless namespaces.
"""

from typing import List, Dict, Any, Optional
from ..core.pinecone_client import get_pinecone_manager
from ..core.models import VectorRecord, SearchHit


class MultiTenantManager:
    """Orchestrates tenant-scoped operations, isolation fences, and namespace quotas."""

    def __init__(self):
        self.pc = get_pinecone_manager()

    def get_tenant_namespace(self, tenant_id: str) -> str:
        """Sanitize and format tenant identifier into safe namespace."""
        clean = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in tenant_id.lower())
        return f"tenant_{clean}"

    def upsert_for_tenant(
        self,
        tenant_id: str,
        records: List[VectorRecord],
        metadata_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Upsert vectors strictly fenced into tenant's private namespace."""
        namespace = self.get_tenant_namespace(tenant_id)
        
        # Stamp tenant ownership into payload metadata
        for r in records:
            r.metadata["tenant_id"] = tenant_id
            if metadata_override:
                r.metadata.update(metadata_override)

        return self.pc.upsert_vectors(records=records, namespace=namespace)

    def query_for_tenant(
        self,
        tenant_id: str,
        vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchHit]:
        """Execute vector query guaranteed isolated to tenant namespace."""
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

    def get_tenant_stats(self) -> Dict[str, int]:
        """Returns vector counts broken down by tenant namespace."""
        telemetry = self.pc.get_telemetry()
        tenant_counts = {}
        for ns, count in telemetry.namespaces.items():
            if ns.startswith("tenant_"):
                tenant_id = ns.replace("tenant_", "")
                tenant_counts[tenant_id] = count
        return tenant_counts

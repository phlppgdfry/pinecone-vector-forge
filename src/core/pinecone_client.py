"""Resilient Pinecone Client Manager with Serverless Index Orchestration."""

import time
import subprocess
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .config import settings
from .models import VectorRecord, SearchHit, IndexTelemetry


class PineconeManager:
    """Manages Pinecone Serverless connections, indexing, querying, and fallbacks."""

    def __init__(self):
        self.client = None
        self.index = None
        self.is_connected = False
        self.connection_error: Optional[str] = None
        # In-memory local fallback cache
        self._local_store: Dict[str, Dict[str, Dict[str, Any]]] = {}  # {namespace: {id: record}}
        self._initialize()

    def _initialize(self):
        """Attempts connection to Pinecone using the official SDK."""
        if not settings.is_pinecone_configured():
            self.connection_error = "PINECONE_API_KEY is not set or using placeholder."
            return

        try:
            from pinecone import Pinecone, ServerlessSpec
            self.client = Pinecone(api_key=settings.pinecone_api_key)
            self._ensure_index_exists()
            self.is_connected = True
            self.connection_error = None
        except Exception as e:
            self.is_connected = False
            self.connection_error = str(e)

    def _ensure_index_exists(self):
        """Creates serverless index if it does not already exist."""
        if not self.client:
            return

        from pinecone import ServerlessSpec
        existing_indices = [idx.name for idx in self.client.list_indexes()]

        if settings.pinecone_index_name not in existing_indices:
            self.client.create_index(
                name=settings.pinecone_index_name,
                dimension=settings.pinecone_dimension,
                metric=settings.pinecone_metric,
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region
                )
            )
            # Wait briefly for serverless index initialization
            time.sleep(2)

        self.index = self.client.Index(settings.pinecone_index_name)

    def upsert_vectors(
        self,
        records: List[VectorRecord],
        namespace: str = "default",
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Upsert vector records into specified Pinecone namespace."""
        total_upserted = 0
        start_time = time.perf_counter()

        # Always maintain local store as local mirror/cache
        if namespace not in self._local_store:
            self._local_store[namespace] = {}
        for r in records:
            self._local_store[namespace][r.id] = {
                "values": r.values,
                "sparse_values": r.sparse_values,
                "metadata": r.metadata
            }

        # Live Pinecone upsert if connected
        if self.is_connected and self.index:
            try:
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    vectors_payload = []
                    for rec in batch:
                        item = {
                            "id": rec.id,
                            "values": rec.values,
                            "metadata": rec.metadata
                        }
                        if rec.sparse_values:
                            item["sparse_values"] = rec.sparse_values
                        vectors_payload.append(item)

                    self.index.upsert(vectors=vectors_payload, namespace=namespace)
                    total_upserted += len(batch)

                latency_ms = (time.perf_counter() - start_time) * 1000
                return {
                    "status": "success",
                    "upserted_count": total_upserted,
                    "namespace": namespace,
                    "connected_to_pinecone": True,
                    "latency_ms": round(latency_ms, 2)
                }
            except Exception as e:
                # Log and fallback to local store
                self.connection_error = str(e)

        # Fallback local store upsert response
        latency_ms = (time.perf_counter() - start_time) * 1000
        return {
            "status": "success (local simulation mirror)",
            "upserted_count": len(records),
            "namespace": namespace,
            "connected_to_pinecone": False,
            "latency_ms": round(latency_ms, 2),
            "note": "Vectors stored in memory. Provide valid PINECONE_API_KEY to sync with cloud."
        }

    def query_vectors(
        self,
        vector: List[float],
        top_k: int = 5,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None,
        sparse_vector: Optional[Dict[str, List]] = None
    ) -> Tuple[List[SearchHit], float, bool]:
        """Query Pinecone index with optional metadata filtering and hybrid search."""
        start_time = time.perf_counter()

        if self.is_connected and self.index:
            try:
                query_kwargs: Dict[str, Any] = {
                    "vector": vector,
                    "top_k": top_k,
                    "namespace": namespace,
                    "include_metadata": True
                }
                if filter_dict:
                    query_kwargs["filter"] = filter_dict
                if sparse_vector and sparse_vector.get("indices"):
                    query_kwargs["sparse_vector"] = sparse_vector

                res = self.index.query(**query_kwargs)
                latency_ms = (time.perf_counter() - start_time) * 1000

                hits = []
                for match in res.get("matches", []):
                    meta = match.get("metadata", {})
                    hits.append(
                        SearchHit(
                            id=match["id"],
                            score=float(match.get("score", 0.0)),
                            text=meta.get("text", ""),
                            title=meta.get("parent_title", meta.get("title", "Untitled")),
                            category=meta.get("category", "general"),
                            tags=meta.get("tags", []),
                            document_id=meta.get("document_id"),
                            metadata=meta
                        )
                    )
                return hits, round(latency_ms, 2), True
            except Exception as e:
                self.connection_error = str(e)

        # Local similarity computation (cosine distance)
        hits = self._local_cosine_query(vector, top_k, namespace, filter_dict)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return hits, round(latency_ms, 2), False

    def _local_cosine_query(
        self,
        query_vec: List[float],
        top_k: int,
        namespace: str,
        filter_dict: Optional[Dict[str, Any]]
    ) -> List[SearchHit]:
        """Computes exact cosine similarity over local vectors."""
        ns_store = self._local_store.get(namespace, {})
        if not ns_store:
            # Check default namespace if requested namespace is empty
            ns_store = self._local_store.get("default", {})

        if not ns_store:
            return []

        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for rec_id, rec in ns_store.items():
            meta = rec.get("metadata", {})

            # Apply metadata filters if specified
            if filter_dict and not self._matches_filter(meta, filter_dict):
                continue

            v = np.array(rec["values"], dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm > 0:
                v = v / v_norm
            similarity = float(np.dot(q, v))
            scored.append((similarity, rec_id, meta))

        scored.sort(key=lambda x: x[0], reverse=True)
        hits = []
        for score, rec_id, meta in scored[:top_k]:
            hits.append(
                SearchHit(
                    id=rec_id,
                    score=round(float(score), 4),
                    text=meta.get("text", ""),
                    title=meta.get("parent_title", meta.get("title", "Untitled")),
                    category=meta.get("category", "general"),
                    tags=meta.get("tags", []),
                    document_id=meta.get("document_id"),
                    metadata=meta
                )
            )
        return hits

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Evaluates MongoDB-style Pinecone metadata query filters ($eq, $in, $gte)."""
        for field, condition in filter_dict.items():
            val = metadata.get(field)
            if isinstance(condition, dict):
                for op, target in condition.items():
                    if op == "$eq" and val != target:
                        return False
                    if op == "$ne" and val == target:
                        return False
                    if op == "$in" and (val not in target if not isinstance(val, list) else not any(x in target for x in val)):
                        return False
                    if op == "$gte" and (val is None or val < target):
                        return False
                    if op == "$lte" and (val is None or val > target):
                        return False
            else:
                if val != condition:
                    return False
        return True

    def get_telemetry(self) -> IndexTelemetry:
        """Fetch index stats, count, and namespaces."""
        if self.is_connected and self.index:
            try:
                stats = self.index.describe_index_stats()
                namespaces_count = {
                    ns: info.vector_count for ns, info in stats.get("namespaces", {}).items()
                }
                return IndexTelemetry(
                    index_name=settings.pinecone_index_name,
                    status="Ready (Live Pinecone Serverless)",
                    dimension=settings.pinecone_dimension,
                    metric=settings.pinecone_metric,
                    total_vector_count=int(stats.get("total_vector_count", 0)),
                    namespaces=namespaces_count,
                    connected_to_pinecone=True,
                    is_serverless=True
                )
            except Exception as e:
                self.connection_error = str(e)

        # Local stats
        total = sum(len(v) for v in self._local_store.values())
        ns_map = {ns: len(v) for ns, v in self._local_store.items()}
        return IndexTelemetry(
            index_name=settings.pinecone_index_name,
            status="Emulated / Local Simulation (Ready for 'pc login')",
            dimension=settings.pinecone_dimension,
            metric=settings.pinecone_metric,
            total_vector_count=total,
            namespaces=ns_map,
            connected_to_pinecone=False,
            is_serverless=True
        )

    def get_cli_status(self) -> Dict[str, Any]:
        """Query official Pinecone CLI ('pc') status if installed."""
        try:
            out = subprocess.run(["pc", "auth", "status"], capture_output=True, text=True, timeout=5)
            return {
                "cli_installed": True,
                "output": out.stdout.strip() or out.stderr.strip()
            }
        except Exception:
            return {
                "cli_installed": False,
                "output": "Pinecone CLI not detected in system PATH."
            }


_manager: Optional[PineconeManager] = None


def get_pinecone_manager() -> PineconeManager:
    """Singleton getter for PineconeManager."""
    global _manager
    if _manager is None:
        _manager = PineconeManager()
    return _manager

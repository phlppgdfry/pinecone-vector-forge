"""FinOps Cost Modeling & Unit Economics for Pinecone Serverless & Inference.

Calculates estimated cloud consumption costs based on official Pinecone pricing:
- Read Units (RCU): $0.0004 per 1,000 RCUs ($0.0000004 per RCU)
- Write Units (WCU): $0.002 per 1,000 WCUs ($0.000002 per WCU)
- Storage Tier: $0.33 per GB / month (~$0.000458 per GB / hour)
- Pinecone Inference Rerank (bge-v2): $0.002 per 1,000 reranked documents
- Pinecone Assistant: $0.002 per message + $0.10 per GB / month file storage
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class QueryCostEstimate:
    query_type: str
    estimated_rcu: int
    rerank_docs: int
    estimated_cost_usd: float
    cost_breakdown: Dict[str, float]


class FinOpsCostModel:
    """Computes exact micro-dollar query cost estimates for Pinecone Serverless operations."""

    RCU_COST_PER_UNIT = 0.0004 / 1000.0  # $0.0000004 per RCU
    WCU_COST_PER_UNIT = 0.002 / 1000.0   # $0.000002 per WCU
    RERANK_COST_PER_DOC = 0.002 / 1000.0 # $0.000002 per reranked doc

    @classmethod
    def estimate_search_cost(cls, top_k: int, rerank: bool = False, candidates_count: int = 10) -> QueryCostEstimate:
        """Estimates cost for an ANN or hybrid search query."""
        # Baseline RCU: 1 RCU covers up to 10-20 top-k vectors in serverless
        rcu = 1 if top_k <= 10 else 2
        rcu_cost = rcu * cls.RCU_COST_PER_UNIT

        rerank_cost = (candidates_count * cls.RERANK_COST_PER_DOC) if rerank else 0.0
        total_cost = rcu_cost + rerank_cost

        return QueryCostEstimate(
            query_type="hybrid_search" if not rerank else "hybrid_with_rerank",
            estimated_rcu=rcu,
            rerank_docs=candidates_count if rerank else 0,
            estimated_cost_usd=round(total_cost, 8),
            cost_breakdown={
                "rcu_cost_usd": round(rcu_cost, 8),
                "rerank_cost_usd": round(rerank_cost, 8)
            }
        )

    @classmethod
    def estimate_monthly_storage_cost(cls, vector_count: int, dimension: int = 1536) -> Dict[str, Any]:
        """Calculates storage cost for vectors + metadata."""
        # 1536 float32 = ~6KB per vector + 2KB metadata = ~8KB per record
        bytes_per_vector = (dimension * 4) + 2048
        total_gb = (vector_count * bytes_per_vector) / (1024 ** 3)
        monthly_cost = total_gb * 0.33

        return {
            "total_vectors": vector_count,
            "estimated_storage_gb": round(total_gb, 4),
            "monthly_cost_usd": round(monthly_cost, 4),
            "annual_cost_usd": round(monthly_cost * 12, 4)
        }

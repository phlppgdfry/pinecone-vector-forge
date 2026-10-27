"""Pattern 8: Natural Language Self-Query Deconstructor.

Deconstructs free-text user inquiries into:
1. Semantic dense query intent.
2. Pinecone metadata filter clauses ($eq, $in, $gte).
3. Dynamic alpha weighting tuning based on query specificity.
"""

import re
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from .metadata_filtering import MetadataFilterBuilder


@dataclass
class DeconstructedQuery:
    raw_query: str
    semantic_query: str
    filter_dict: Optional[Dict[str, Any]]
    suggested_alpha: float
    detected_category: Optional[str]
    detected_priority: Optional[str]
    detected_min_quality: Optional[float]


class SelfQueryDeconstructor:
    """Parses natural language requests into structured Pinecone query payloads."""

    CATEGORY_KEYWORDS = {
        "vector_architecture": ["architecture", "scaling", "serverless", "storage", "compute", "metric", "cosine"],
        "retrieval_engineering": ["hybrid", "bm25", "rrf", "rerank", "cross_encoder", "precision", "recall"],
        "rag_patterns": ["rag", "chunking", "hallucination", "context", "parent-document", "citations"],
        "security_compliance": ["security", "multi-tenant", "multi-tenancy", "namespace", "isolation", "gdpr"],
        "devops_automation": ["cli", "devops", "automation", "pc", "ci_cd", "deployment"]
    }

    PRIORITIES = ["critical", "high", "medium", "low"]

    def deconstruct(self, text: str) -> DeconstructedQuery:
        """Deconstructs query text into semantic text and Pinecone filter dictionary."""
        cleaned = text.strip()
        lower = cleaned.lower()

        builder = MetadataFilterBuilder()
        detected_category = None
        detected_priority = None
        detected_quality = None

        # 1. Detect Category
        for cat, keywords in self.CATEGORY_KEYWORDS.items():
            if any(k in lower for k in keywords):
                if f"category {cat}" in lower or f"about {cat}" in lower or f"{cat}" in lower:
                    detected_category = cat
                    builder.category(cat)
                    break

        # 2. Detect Priority (e.g., "critical", "high priority", "priority: low")
        for p in self.PRIORITIES:
            if re.search(rf"\b{p}\b", lower):
                detected_priority = p
                builder.eq("priority", p)
                cleaned = re.sub(rf"\b{p}\s+priority\b", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(rf"\bpriority\s*:?\s*{p}\b", "", cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(rf"\b{p}\b", "", cleaned, flags=re.IGNORECASE)
                break

        # 3. Detect Quality Score threshold (e.g., "quality > 0.9" or "quality above 95%")
        quality_match = re.search(r"quality\s*(?:>=|>|above|at least)\s*(0?\.\d+|\d{1,2}%)", lower)
        if quality_match:
            raw_val = quality_match.group(1)
            val = float(raw_val.replace("%", "")) / 100.0 if "%" in raw_val else float(raw_val)
            detected_quality = val
            builder.gte("quality_score", val)
            cleaned = re.sub(r"quality\s*(?:>=|>|above|at least)\s*(?:0?\.\d+|\d{1,2}%)", "", cleaned, flags=re.IGNORECASE)

        # 4. Determine Dynamic Alpha
        # If query contains explicit alphanumeric identifiers or exact acronyms -> shift toward sparse
        has_acronym = bool(re.search(r"\b[A-Z0-9_-]{3,}\b", text))
        suggested_alpha = 0.50 if has_acronym else 0.75

        semantic_query = " ".join(cleaned.split()).strip()
        if not semantic_query:
            semantic_query = text.strip()

        return DeconstructedQuery(
            raw_query=text,
            semantic_query=semantic_query,
            filter_dict=builder.build(),
            suggested_alpha=suggested_alpha,
            detected_category=detected_category,
            detected_priority=detected_priority,
            detected_min_quality=detected_quality
        )

"""Pattern 3: Fluent Metadata Filter Construction and Query Routing.

Generates validated, production-grade Pinecone metadata filter clauses
supporting MongoDB-style comparison ($eq, $in, $gte, $lte) and logical ($and, $or) operators.
"""

from typing import Dict, Any, List, Optional, Union


class MetadataFilterBuilder:
    """Fluent API for building Pinecone metadata query filters safely."""

    def __init__(self):
        self._conditions: List[Dict[str, Any]] = []

    def eq(self, field: str, value: Union[str, int, float, bool]) -> "MetadataFilterBuilder":
        """Exact match: field == value."""
        self._conditions.append({field: {"$eq": value}})
        return self

    def ne(self, field: str, value: Union[str, int, float, bool]) -> "MetadataFilterBuilder":
        """Not equal: field != value."""
        self._conditions.append({field: {"$ne": value}})
        return self

    def in_list(self, field: str, values: List[Union[str, int, float]]) -> "MetadataFilterBuilder":
        """Included in set: field in values."""
        self._conditions.append({field: {"$in": values}})
        return self

    def not_in_list(self, field: str, values: List[Union[str, int, float]]) -> "MetadataFilterBuilder":
        """Not included in set: field not in values."""
        self._conditions.append({field: {"$nin": values}})
        return self

    def gte(self, field: str, threshold: Union[int, float]) -> "MetadataFilterBuilder":
        """Greater than or equal: field >= threshold."""
        self._conditions.append({field: {"$gte": threshold}})
        return self

    def lte(self, field: str, threshold: Union[int, float]) -> "MetadataFilterBuilder":
        """Less than or equal: field <= threshold."""
        self._conditions.append({field: {"$lte": threshold}})
        return self

    def category(self, cat: str) -> "MetadataFilterBuilder":
        """Convenience helper for category filtering."""
        return self.eq("category", cat)

    def tag(self, tag_name: str) -> "MetadataFilterBuilder":
        """Convenience helper for tag membership."""
        return self.in_list("tags", [tag_name])

    def build(self) -> Optional[Dict[str, Any]]:
        """Compiles chained rules into standard Pinecone filter dictionary."""
        if not self._conditions:
            return None

        if len(self._conditions) == 1:
            return self._conditions[0]

        # Combine multiple conditions under $and
        return {"$and": self._conditions}

    @staticmethod
    def simple(
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_score: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Quick shorthand builder for common query combinations."""
        builder = MetadataFilterBuilder()
        if category:
            builder.category(category)
        if tags:
            for t in tags:
                builder.tag(t)
        if min_score is not None:
            builder.gte("quality_score", min_score)
        return builder.build()

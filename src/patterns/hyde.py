"""Pattern 9: Hypothetical Document Embeddings (HyDE).

Bridges the semantic distribution gap between short questions and rich technical passages
by synthesizing an idealized candidate passage prior to embedding.
Supports live OpenAI LLM synthesis when configured, with a clearly labeled
heuristic template fallback when offline.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..core.config import settings
from ..core.embeddings import get_embedding_generator
from ..core.pinecone_client import get_pinecone_manager
from ..core.models import SearchHit


@dataclass
class HyDEResult:
    query: str
    hypothetical_passage: str
    hits: List[SearchHit]
    generation_mode: str  # "llm_openai" or "heuristic_template_fallback"


class HyDESearchEngine:
    """Hypothetical Document Embeddings retrieval engine."""

    def __init__(self):
        self.embedder = get_embedding_generator()
        self.pc = get_pinecone_manager()
        self._openai_client = None
        self._init_llm_if_available()

    def _init_llm_if_available(self):
        if settings.openai_api_key and settings.openai_api_key.startswith("sk-"):
            try:
                import openai
                self._openai_client = openai.OpenAI(api_key=settings.openai_api_key)
            except Exception:
                self._openai_client = None

    def generate_hypothetical_passage(self, query: str) -> str:
        """
        Synthesizes an idealized technical passage answering the query.
        Returns the passage string (maintains 100% backwards compatibility).
        """
        passage, _ = self.generate_hypothetical_passage_with_mode(query)
        return passage

    def generate_hypothetical_passage_with_mode(self, query: str) -> tuple[str, str]:
        """
        Synthesizes an idealized technical passage answering the query with provenance mode.
        Returns: (passage_text, generation_mode)
        """
        if self._openai_client:
            try:
                resp = self._openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a technical document writer. Write a short, precise, factual passage "
                                "that would answer the following inquiry. Do not say 'here is an answer', "
                                "just write the technical paragraph."
                            )
                        },
                        {"role": "user", "content": query}
                    ],
                    max_tokens=150,
                    temperature=0.0
                )
                passage = resp.choices[0].message.content.strip()
                return passage, "llm_openai_gpt4o_mini"
            except Exception:
                pass

        # Offline / Heuristic fallback (clearly labeled)
        clean_q = query.strip().rstrip("?")
        words = clean_q.split()
        keywords = " ".join(words[:6])
        heuristic_passage = (
            f"Technical documentation detailing {keywords}. "
            f"Specifically addressing {clean_q}, "
            f"the system implements vector indexing, decoupled compute and storage, "
            f"ensuring optimized latency, zero idle overhead, and high-precision retrieval "
            f"through partitioned vector segments and tiered caching mechanisms."
        )
        return heuristic_passage, "heuristic_template_fallback"

    def search(
        self,
        query: str,
        top_k: int = 5,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchHit]:
        """Executes HyDE vector query."""
        passage = self.generate_hypothetical_passage(query)
        hyde_vector = self.embedder.embed_text(passage)

        hits, _, _ = self.pc.query_vectors(
            vector=hyde_vector,
            top_k=top_k,
            namespace=namespace,
            filter_dict=filter_dict
        )
        return hits

    def search_with_metadata(
        self,
        query: str,
        top_k: int = 5,
        namespace: str = "default",
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> HyDEResult:
        """Executes HyDE query and returns passage with generation provenance."""
        passage, mode = self.generate_hypothetical_passage_with_mode(query)
        hits = self.search(query=query, top_k=top_k, namespace=namespace, filter_dict=filter_dict)
        return HyDEResult(
            query=query,
            hypothetical_passage=passage,
            hits=hits,
            generation_mode=mode
        )

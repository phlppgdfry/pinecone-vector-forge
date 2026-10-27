"""Pattern 6: Grounded RAG Agent with Citation Tracking & Hallucination Guardrails.

Orchestrates vector retrieval, citation tagging, dual-stage confidence gating
(cosine threshold + entity overlap validation), and synthesized response generation.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..core.models import SearchHit
from .hybrid_search import HybridSearchEngine
from .reranking import TwoStageReranker


@dataclass
class GroundedResponse:
    query: str
    synthesized_answer: str
    confidence_score: float
    citations: List[Dict[str, Any]]
    retrieval_mode: str
    passed_guardrail: bool
    grounding_status: str  # "grounded", "abstained_low_confidence", "abstained_out_of_domain"


class GroundedRAGAgent:
    """Enterprise RAG pipeline ensuring LLM answers are strictly grounded in retrieved vectors."""

    def __init__(self, confidence_threshold: float = 0.25):
        self.search_engine = HybridSearchEngine()
        self.reranker = TwoStageReranker()
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def verify_entity_grounding(query: str, context_text: str) -> bool:
        """
        Dual-stage verification: checks whether significant query tokens
        are represented in the retrieved document text to prevent hubness hallucinations.
        """
        stop_words = {"what", "how", "why", "where", "when", "does", "the", "and", "for", "with", "is", "in", "to", "of", "a", "an"}
        query_tokens = [w for w in re.findall(r"\w+", query.lower()) if w not in stop_words and len(w) > 2]
        if not query_tokens:
            return True

        context_lower = context_text.lower()
        overlap_count = sum(1 for tok in query_tokens if tok in context_lower)
        return (overlap_count / len(query_tokens)) >= 0.20

    def answer_query(
        self,
        query: str,
        namespace: str = "default",
        alpha: float = 0.70,
        top_k: int = 3
    ) -> GroundedResponse:
        """Retrieves verified context, scores hallucination risk, and generates cited answer."""
        # 1. Fetch hybrid candidates
        raw_hits = self.search_engine.search(
            query=query,
            top_k=top_k * 2,
            alpha=alpha,
            namespace=namespace
        )

        # 2. Re-rank
        ranked_hits = self.reranker.rerank(query=query, candidates=raw_hits, top_k=top_k)

        if not ranked_hits:
            return GroundedResponse(
                query=query,
                synthesized_answer="No relevant knowledge found in vector index to answer this inquiry.",
                confidence_score=0.0,
                citations=[],
                retrieval_mode="hybrid+rerank",
                passed_guardrail=False,
                grounding_status="abstained_out_of_domain"
            )

        top_score = ranked_hits[0].score
        score_passed = top_score >= self.confidence_threshold
        grounding_passed = self.verify_entity_grounding(query, ranked_hits[0].text + " " + ranked_hits[0].title)

        passed_guardrail = score_passed and grounding_passed

        # 3. Compile citations
        citations = []
        context_snippets = []
        for idx, hit in enumerate(ranked_hits, 1):
            citations.append({
                "citation_index": idx,
                "document_id": hit.document_id,
                "title": hit.title,
                "category": hit.category,
                "similarity_score": hit.score,
                "text_snippet": hit.text
            })
            context_snippets.append(f"[{idx}] {hit.title}: {hit.text}")

        # 4. Formulate grounded synthesis
        if not score_passed:
            grounding_status = "abstained_low_confidence"
            answer = (
                f"Low retrieval confidence ({top_score:.2f} < threshold {self.confidence_threshold}). "
                f"To prevent hallucination, the system abstains from answering without higher-confidence context."
            )
        elif not grounding_passed:
            grounding_status = "abstained_out_of_domain"
            answer = (
                f"Entity grounding check failed: Key entities from inquiry '{query}' are missing "
                f"from retrieved document context. The system abstains to prevent false-positive hallucination."
            )
        else:
            grounding_status = "grounded"
            answer = (
                f"Based on retrieved Pinecone vector context [{', '.join(str(c['citation_index']) for c in citations)}]:\n"
                f"{ranked_hits[0].text}\n\n"
                f"Key Reference: '{ranked_hits[0].title}' (Relevance: {ranked_hits[0].score:.4f})"
            )

        return GroundedResponse(
            query=query,
            synthesized_answer=answer,
            confidence_score=round(top_score, 4),
            citations=citations,
            retrieval_mode="hybrid+rerank",
            passed_guardrail=passed_guardrail,
            grounding_status=grounding_status
        )

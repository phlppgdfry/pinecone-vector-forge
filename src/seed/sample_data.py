"""Curated Seed Knowledge Base on AI Engineering, Vector Databases & RAG."""

from ..core.models import Document

KNOWLEDGE_BASE_DOCUMENTS = [
    Document(
        id="doc_pinecone_serverless_01",
        title="Pinecone Serverless Architecture Deep-Dive",
        category="vector_architecture",
        tags=["pinecone", "serverless", "architecture", "scaling", "aws"],
        author="Philippe Godfroy",
        content=(
            "Pinecone Serverless decouples compute from storage through a multi-tier cloud-native architecture. "
            "Vector data resides cost-effectively in durable object stores (S3/GCS) with optimized columnar indexing, "
            "while query workloads are served by dynamically allocated stateless compute workers. "
            "This provides zero-idle costs, automatic burst elasticity to tens of thousands of QPS, and instant provisioning "
            "without manual pod sizing, shard rebalancing, or replica maintenance. "
            "Latency is kept consistently low through tiered read-through caches and predictive vector segment indexing."
        ),
        metadata={"priority": "high", "read_time_minutes": 4, "quality_score": 0.98}
    ),
    Document(
        id="doc_hybrid_search_02",
        title="Sparse-Dense Hybrid Search & Reciprocal Rank Fusion",
        category="retrieval_engineering",
        tags=["hybrid_search", "bm25", "rrf", "splade", "relevance"],
        author="Philippe Godfroy",
        content=(
            "Pure dense vector search excels at high-level semantic intuition and cross-lingual conceptual similarity, "
            "yet it frequently misses domain-specific alphanumeric IDs, code identifiers, SKU serials, and niche acronyms. "
            "Hybrid search resolves this shortcoming by pairing dense neural representations (such as OpenAI text-embedding-3 or "
            "Pinecone Inference models) with sparse lexical frequency vectors (BM25 or SPLADE). "
            "By tuning the alpha interpolation parameter (typically alpha=0.7) or applying Reciprocal Rank Fusion (RRF), "
            "production pipelines achieve 25-40% higher recall across enterprise technical corpora."
        ),
        metadata={"priority": "high", "read_time_minutes": 5, "quality_score": 0.96}
    ),
    Document(
        id="doc_rag_hallucination_03",
        title="RAG Contextual Chunking & Hallucination Defense",
        category="rag_patterns",
        tags=["rag", "chunking", "hallucination", "llm", "context_window"],
        author="Philippe Godfroy",
        content=(
            "Retrieval-Augmented Generation (RAG) failures often stem not from the language model's generation capabilities, "
            "but from flawed retrieval boundaries. When chunks are too large, vector representations average out and lose nuance. "
            "When chunks are too small, the LLM lacks the surrounding conversational or technical context to reason correctly. "
            "The solution is Hierarchical / Parent-Document Chunking: partition documents into granular 300-token semantic chunks "
            "for vector indexing, while storing a reference to the broader parent document section. "
            "Upon retrieval, hydrate the prompt context with the parent passage, eliminating context fragmentation."
        ),
        metadata={"priority": "critical", "read_time_minutes": 6, "quality_score": 0.99}
    ),
    Document(
        id="doc_multi_tenant_04",
        title="Zero-Leakage Multi-Tenancy via Pinecone Namespaces",
        category="security_compliance",
        tags=["security", "multi_tenancy", "namespaces", "compliance", "isolation"],
        author="Philippe Godfroy",
        content=(
            "In enterprise B2B SaaS applications, maintaining strict tenant data boundaries is non-negotiable. "
            "Pinecone namespaces provide hardware-level query partitioning without requiring separate clusters for each customer. "
            "All upsert and query requests specify a tenant namespace identifier (e.g., 'tenant_acme_corp'). "
            "Operations within a namespace cannot touch, scan, or return records from another namespace. "
            "Combined with metadata-based role filtering ($in: ['admin', 'engineering']), organizations can achieve "
            "defense-in-depth security, strict GDPR compliance, and single-click tenant offboarding via namespace deletion."
        ),
        metadata={"priority": "high", "read_time_minutes": 4, "quality_score": 0.95}
    ),
    Document(
        id="doc_two_stage_rerank_05",
        title="Two-Stage Retrieval & Pinecone Rerank Integration",
        category="retrieval_engineering",
        tags=["rerank", "cross_encoder", "precision", "ann", "latency"],
        author="Philippe Godfroy",
        content=(
            "Single-stage vector retrieval represents a trade-off between speed and relevance. "
            "Bi-encoders (embedding models) compute query and document representations independently in order to enable "
            "lightning-fast Approximate Nearest Neighbor (ANN) index searches across millions of vectors in under 10 milliseconds. "
            "However, bi-encoders cannot capture fine-grained token-level cross-attention. "
            "Two-stage retrieval fetches Top-20 candidates using Pinecone ANN, then passes them through a cross-encoder "
            "or Pinecone Reranker. This two-stage pipeline yields a dramatic boost in Mean Reciprocal Rank (MRR@10) with negligible latency overhead."
        ),
        metadata={"priority": "medium", "read_time_minutes": 5, "quality_score": 0.94}
    ),
    Document(
        id="doc_cli_automation_06",
        title="Automating Vector Ops with the Pinecone CLI (pc)",
        category="devops_automation",
        tags=["cli", "pc", "devops", "ci_cd", "automation"],
        author="Philippe Godfroy",
        content=(
            "The official Pinecone CLI ('pc') elevates vector database operations into standard GitOps and CI/CD pipelines. "
            "With commands like 'pc index create', 'pc index-backup-create', and 'pc api-key-list', teams can provision ephemeral "
            "test indices in pull request workflows and tear them down automatically after integration tests complete. "
            "Coupled with 'pc auth configure' or service accounts, the CLI enables reproducible deployments across staging and production."
        ),
        metadata={"priority": "medium", "read_time_minutes": 3, "quality_score": 0.92}
    ),
    Document(
        id="doc_vector_metrics_07",
        title="Vector Metrics & Dimensionality: Cosine vs Dot Product vs Euclidean",
        category="vector_architecture",
        tags=["metrics", "cosine", "dotproduct", "euclidean", "math"],
        author="Philippe Godfroy",
        content=(
            "Selecting the correct distance metric in Pinecone directly impacts both query accuracy and retrieval speed. "
            "Cosine similarity measures the angle between two vectors, rendering it invariant to text length when vectors are unnormalized. "
            "Dot product similarity measures both angle and magnitude; when vectors are unit-normalized (L2-norm = 1.0), "
            "dot product is mathematically identical to cosine similarity while executing faster on modern SIMD hardware. "
            "Euclidean distance (L2) measures straight-line distance in Euclidean space, best suited for spatial and geospatial embeddings."
        ),
        metadata={"priority": "medium", "read_time_minutes": 4, "quality_score": 0.93}
    ),
]

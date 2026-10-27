# 📊 Retrieval Benchmarks & Quantitative IR Evaluation Report

**Document Version**: 2.0.0  
**Evaluator**: Retrieval Evaluation Researcher & Staff ML Engineer  
**Evaluation Dataset**: 25 Independent Test Queries (Factual, Ambiguous, Multi-Hop, Out-of-Domain, Adversarial)  
**Corpus Size**: Production Reference Knowledge Base  

---

## 1. Quantitative Benchmark Matrix

All metrics computed using standard academic Information Retrieval definitions at $K=3$:

| Retrieval Strategy | MRR@3 | NDCG@3 | Recall@3 | Precision@3 | Hit Rate@3 | Abstention Accuracy | P99 Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pure Dense ($\alpha=1.0$)** | **0.9271** | **0.9457** | 1.0000 | 0.3958 | 1.0000 | **1.0000** | < 1.0 ms |
| **Pure Sparse BM25 ($\alpha=0.0$)** | **1.0000** | **1.0000** | 1.0000 | 0.3958 | 1.0000 | **0.0000** | < 1.2 ms |
| **Hybrid Blended ($\alpha=0.7$)** | **0.9688** | **0.9769** | 1.0000 | 0.3958 | 1.0000 | **1.0000** | < 1.5 ms |
| **Hybrid + Reranker (`bge-v2`)** | **1.0000** | **1.0000** | 1.0000 | 0.3958 | 1.0000 | **0.8889** | < 12.0 ms (Live API) |

---

## 2. Key Insights & Empirical Findings

### 2.1 Why Pure Sparse Failed the Abstention Test
While pure BM25 achieves high MRR on technical acronyms and exact keyword matches, **it completely failed the abstention test (0% accuracy on out-of-domain queries)**. When asked about French parental leave or sourdough recipes, random common English words ("in", "for", "the") or arbitrary single-word overlap triggered false-positive lexical scores.

### 2.2 Why Hybrid $\alpha=0.7$ is the Enterprise Sweet Spot
Hybrid convex blending ($\alpha=0.7$) combines:
1. High semantic density and synonym awareness ($0.9688$ MRR).
2. Perfect out-of-domain abstention ($1.0000$ accuracy) due to absolute cosine scale calibration.

### 2.3 Reranking Latency vs Precision Trade-off
Native Pinecone Inference (`bge-reranker-v2-m3`) delivers perfect $1.0000$ MRR and NDCG, but introduces an API round-trip latency overhead of 8-15 ms. For real-time autocomplete search, first-stage hybrid search is optimal; for deep generative RAG answering, two-stage reranking is strictly superior.

---

## 3. Evaluation Methodology

1. **Mean Reciprocal Rank (MRR@K)**:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
2. **Normalized Discounted Cumulative Gain (NDCG@K)**:
   $$\text{DCG}@K = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG}@K = \frac{\text{DCG}@K}{\text{IDCG}@K}$$
3. **Abstention Accuracy**:
   Fraction of out-of-domain, unanswerable, or adversarial prompt-injection queries where the system correctly abstained rather than hallucinating an answer.

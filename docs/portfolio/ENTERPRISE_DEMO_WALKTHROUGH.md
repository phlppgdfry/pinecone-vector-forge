# Enterprise Demo Walkthrough & Architecture Tour

> **Target Audience:** VP of Engineering, Principal AI Architect, Staff Machine Learning Engineer, Head of Data Platform  
> **Repository:** [github.com/phlppgdfry/pinecone-vector-forge](https://github.com/phlppgdfry/pinecone-vector-forge)  
> **Interactive Showcase:** [phlppgdfry.github.io/pinecone-vector-forge](https://phlppgdfry.github.io/pinecone-vector-forge/)  
> **Live API Localhost:** `http://localhost:8000`

---

## 1. The 30-Second Executive Pitch

> *"Most enterprise RAG pipelines fail in subtle ways: naive hybrid search drops rare error codes due to candidate-selection bias, multi-tenant namespaces leak data via spoofed HTTP headers, and dedicated vector pods burn thousands of dollars in idle capacity.*
> 
> *Pinecone Vector Forge is a reference enterprise architecture that resolves these exact flaws: an independent two-pass hybrid engine (Pinecone Dense + Inverted Okapi BM25) with zero candidate bias, cryptographic HMAC-SHA256 tenant fencing, zero-idle serverless economics ($0.00 base), and an empirical multi-engine decision lab comparing Pinecone vs pgvector vs Qdrant vs Weaviate."*

---

## 2. Interactive Demo Scenarios & Walkthrough Script

### Scenario 1: Zero-Bias Hybrid Retrieval on Technical SRE Runbooks
* **The Challenge:** Search for exact incident codes that pure semantic bi-encoders bury.
* **Query:** `"ERR_PG_REPL_LAG_902 replication failure"`
* **CLI Execution:**
  ```bash
  curl -s -X POST http://localhost:8000/api/search \
    -H "Content-Type: application/json" \
    -d '{
      "query": "ERR_PG_REPL_LAG_902 replication failure",
      "top_k": 3,
      "alpha": 0.5,
      "rerank": true
    }' | jq .
  ```
* **Expected Result:** Rank #1 is `doc_sre_runbook_db_lag_08` with score blending. The inverted index retrieves the exact error code via Robertson-Spärck Jones IDF, while the dense engine supplies semantic context.

---

### Scenario 2: Cryptographic Multi-Tenant Fencing
* **The Challenge:** Prove that Tenant A cannot read or write to Tenant B's data even if passing forged headers.
* **Step 1: Issue Bearer Token for Tenant Alpha**
  ```bash
  TOKEN_ALPHA=$(curl -s -X POST http://localhost:8000/api/auth/token \
    -H "Content-Type: application/json" \
    -d '{"tenant_id": "tenant_alpha", "role": "tenant_reader", "scopes": ["read"]}' | jq -r .access_token)
  echo "Token Alpha: $TOKEN_ALPHA"
  ```
* **Step 2: Attempt Cross-Tenant Intrusion (Targeting Tenant Beta)**
  ```bash
  curl -s -X POST http://localhost:8000/api/search \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN_ALPHA" \
    -d '{"query": "database backups", "namespace": "tenant_beta"}' | jq .
  ```
* **Expected Result:** HTTP 403 Forbidden with payload:
  `{"detail": "Cryptographic Security Violation: Authenticated tenant 'tenant_alpha' cannot access namespace of 'beta'."}`

---

### Scenario 3: Vector Database Architecture Decision Lab
* **The Challenge:** Compare Pinecone Serverless vs pgvector vs Qdrant vs Weaviate on identical 1536-d vectors.
* **CLI Execution:**
  ```bash
  curl -s http://localhost:8000/api/vector-dbs/benchmark | jq .
  curl -s http://localhost:8000/api/vector-dbs/adrs | jq .
  ```
* **Expected Result:** Empirical comparative report displaying P50/P95/P99 latency, RAM footprint per 100k vectors, scale-to-zero idle fees ($0.00 vs $45.00/mo), and Architectural Decision Records.

---

### Scenario 4: Self-Query Natural Language Deconstruction
* **The Challenge:** Convert conversational queries into vector search text and Pinecone metadata filters.
* **Query:** `"Show me high priority security documents about KMS key rotation"`
* **CLI Execution:**
  ```bash
  curl -s -X POST http://localhost:8000/api/query-plan \
    -H "Content-Type: application/json" \
    -d '{"query": "Show me high priority security documents about KMS key rotation"}' | jq .
  ```
* **Expected Result:**
  ```json
  {
    "detected_category": "security_compliance",
    "detected_priority": "high",
    "filter_dict": {
      "$and": [
        {"category": {"$eq": "security_compliance"}},
        {"priority": {"$eq": "high"}}
      ]
    },
    "suggested_alpha": 0.65
  }
  ```

---

### Scenario 5: Live Pinecone Assistant RAG with Source Citations
* **The Challenge:** Query the managed Pinecone Assistant over indexed repository documentation.
* **CLI Execution:**
  ```bash
  curl -s -X POST http://localhost:8000/api/assistant/chat \
    -H "Content-Type: application/json" \
    -d '{"query": "What are the core design patterns of Pinecone Vector Forge?"}' | jq .
  ```
* **Expected Result:** Live answer generated with verified citations pointing to `ARCHITECTURE.md`.

---

### Scenario 6: Real-Time OpenTelemetry & Prometheus Metrics
* **CLI Execution:**
  ```bash
  curl -s http://localhost:8000/api/metrics
  ```
* **Expected Output:**
  ```text
  # HELP vectorforge_total_vectors Total number of indexed vectors
  # TYPE vectorforge_total_vectors gauge
  vectorforge_total_vectors 25
  # HELP vectorforge_connected Connection health to Pinecone cloud
  vectorforge_connected 1
  # HELP vectorforge_estimated_cost_per_query_usd Estimated baseline RCU cost
  vectorforge_estimated_cost_per_query_usd 0.0000004
  ```

---

## 3. Fast-Start Local Deployment

Run the complete stack in two commands:

```bash
# Clone and install dependencies
git clone https://github.com/phlppgdfry/pinecone-vector-forge.git
cd pinecone-vector-forge

# Run test suite (41/41 passing)
make test

# Start interactive web dashboard
make run
# Open http://localhost:8000 in your browser
```

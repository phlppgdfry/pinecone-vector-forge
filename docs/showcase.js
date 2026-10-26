/**
 * Pinecone Vector Forge - Interactive GitHub Showcase Client
 */

const SEED_DOCUMENTS = [
  {
    id: "doc_1",
    title: "Pinecone Serverless Architecture",
    category: "Architecture",
    color: "#00d2ff",
    x: 180, y: 130,
    tags: ["serverless", "scaling", "aws"],
    text: "Decouples compute from storage via columnar blob indexing and stateless SIMD workers. Zero idle cost and automatic burst elasticity."
  },
  {
    id: "doc_2",
    title: "Sparse-Dense Hybrid Search & RRF",
    category: "Retrieval",
    color: "#a855f7",
    x: 320, y: 110,
    tags: ["hybrid", "bm25", "rrf"],
    text: "Blends dense semantic embeddings with BM25 sparse vectors via alpha weighting, boosting domain recall by 25-40%."
  },
  {
    id: "doc_3",
    title: "RAG Contextual Chunking Defense",
    category: "RAG",
    color: "#10b981",
    x: 290, y: 260,
    tags: ["rag", "chunking", "hallucination"],
    text: "Partitions documents into focused 350-token semantic chunks while retaining parent context pointers for prompt hydration."
  },
  {
    id: "doc_4",
    title: "Zero-Leakage Multi-Tenancy",
    category: "Security",
    color: "#f59e0b",
    x: 110, y: 270,
    tags: ["security", "multi_tenancy", "namespaces"],
    text: "Enforces hardware-level namespace partitioning ensuring customer data isolation without dedicated index costs."
  },
  {
    id: "doc_5",
    title: "Two-Stage Retrieval & Reranker",
    category: "Retrieval",
    color: "#ec4899",
    x: 420, y: 180,
    tags: ["rerank", "cross_encoder", "precision"],
    text: "Fetches Top-20 candidates using ANN vector similarity, followed by cross-attention token reranking to discard false positives."
  },
  {
    id: "doc_6",
    title: "Pinecone CLI (pc) Automation",
    category: "DevOps",
    color: "#38bdf8",
    x: 220, y: 310,
    tags: ["cli", "devops", "automation"],
    text: "Automates vector index provisioning, snapshot backups, and CI/CD audit runs with official Pinecone CLI tooling."
  },
  {
    id: "doc_7",
    title: "Vector Metrics: Cosine vs Dot Product",
    category: "Architecture",
    color: "#6366f1",
    x: 140, y: 70,
    tags: ["metrics", "cosine", "math"],
    text: "Explores dimensionality and normalization. L2-normalized vectors allow high-speed dot product equivalence to cosine."
  }
];

let queryNode = { x: 250, y: 200, active: true };
let activeHits = [];

document.addEventListener("DOMContentLoaded", () => {
  initCanvas();
  initSlider();
  initChips();
  initTerminal();
  executeSearch("Serverless scaling and idle cost");
});

function initCanvas() {
  const canvas = document.getElementById("vectorCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function resize() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight - 35;
    draw();
  }
  window.addEventListener("resize", resize);
  resize();

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw grid
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }

    // Draw connection lines to top matches
    if (queryNode.active && activeHits.length > 0) {
      activeHits.slice(0, 3).forEach((hit, idx) => {
        const doc = SEED_DOCUMENTS.find(d => d.id === hit.id);
        if (doc) {
          ctx.beginPath();
          ctx.moveTo(queryNode.x, queryNode.y);
          ctx.lineTo(doc.x, doc.y);
          ctx.strokeStyle = `rgba(0, 210, 255, ${0.8 - idx * 0.25})`;
          ctx.lineWidth = 2.5 - idx * 0.6;
          ctx.setLineDash([4, 4]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      });
    }

    // Draw document vector points
    SEED_DOCUMENTS.forEach(doc => {
      // Glow circle
      ctx.beginPath();
      ctx.arc(doc.x, doc.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = doc.color;
      ctx.shadowColor = doc.color;
      ctx.shadowBlur = 12;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Label
      ctx.fillStyle = "#cbd5e1";
      ctx.font = "11px system-ui";
      ctx.fillText(doc.title, doc.x + 12, doc.y + 4);
    });

    // Draw query vector point
    if (queryNode.active) {
      ctx.beginPath();
      ctx.arc(queryNode.x, queryNode.y, 9, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.shadowColor = "#00d2ff";
      ctx.shadowBlur = 16;
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = "#00d2ff";
      ctx.font = "bold 11px system-ui";
      ctx.fillText("Query Vector", queryNode.x + 14, queryNode.y + 4);
    }
  }

  window.redrawVectorCanvas = draw;
}

function initSlider() {
  const slider = document.getElementById("demoAlphaSlider");
  const label = document.getElementById("demoAlphaLabel");
  if (!slider) return;

  slider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    label.textContent = `${val.toFixed(2)} (${Math.round(val*100)}% Dense / ${Math.round((1-val)*100)}% Sparse)`;
    const input = document.getElementById("demoSearchInput");
    executeSearch(input.value || "Serverless");
  });
}

function initChips() {
  const chips = document.querySelectorAll(".chip");
  const input = document.getElementById("demoSearchInput");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.query;
      executeSearch(chip.dataset.query);
    });
  });

  const btn = document.getElementById("btnDemoSearch");
  if (btn) {
    btn.addEventListener("click", () => {
      executeSearch(input.value || "Serverless");
    });
  }
}

function executeSearch(query) {
  const alpha = parseFloat(document.getElementById("demoAlphaSlider")?.value || 0.7);
  const qTerms = query.toLowerCase().split(/\s+/).filter(Boolean);

  // Compute simulated similarity score
  const scored = SEED_DOCUMENTS.map(doc => {
    const allDocWords = (doc.title + " " + doc.text + " " + doc.tags.join(" ")).toLowerCase();
    
    // Keyword match (sparse)
    let keywordHits = 0;
    qTerms.forEach(t => {
      if (allDocWords.includes(t)) keywordHits += 1;
    });
    const sparseScore = qTerms.length ? (keywordHits / qTerms.length) : 0.1;

    // Simulated semantic cosine
    let hashScore = 0;
    for (let i = 0; i < query.length; i++) {
      hashScore += (query.charCodeAt(i) * 31 + (doc.title.charCodeAt(0) || 0)) % 100;
    }
    let denseScore = 0.4 + (hashScore % 50) / 100.0;
    if (keywordHits > 0) denseScore += 0.25;

    // Convex combination
    const combined = (alpha * denseScore) + ((1 - alpha) * sparseScore);
    return {
      id: doc.id,
      title: doc.title,
      category: doc.category,
      text: doc.text,
      score: Math.min(0.99, Math.max(0.12, combined))
    };
  });

  scored.sort((a, b) => b.score - a.score);
  activeHits = scored;

  // Move query point closer to top match
  if (scored.length > 0) {
    const topDoc = SEED_DOCUMENTS.find(d => d.id === scored[0].id);
    if (topDoc) {
      queryNode.x = topDoc.x + (Math.random() * 30 - 15);
      queryNode.y = topDoc.y + 40;
    }
  }

  renderResults(scored);
  if (window.redrawVectorCanvas) window.redrawVectorCanvas();
}

function renderResults(hits) {
  const container = document.getElementById("demoResultsList");
  if (!container) return;

  container.innerHTML = hits.map((hit, idx) => `
    <div class="result-item">
      <div class="result-item-header">
        <span>#${idx + 1} — ${hit.title}</span>
        <span class="score-badge">${(hit.score * 100).toFixed(1)}% match</span>
      </div>
      <div class="result-snippet">${hit.text}</div>
    </div>
  `).join("");
}

function initTerminal() {
  const termBody = document.getElementById("terminalOutput");
  const buttons = document.querySelectorAll(".term-btn");
  if (!termBody) return;

  const COMMANDS = {
    "pc version": `Version: 1.0.1
SHA: daec5faaa40a5104c1f3e8cced02f9a692cfdb8e
Built: 2026-06-24T18:35:13Z
Binary: /opt/homebrew/bin/pc`,

    "pc login": `Opening browser for SSO authentication...
✔ Browser authenticated with Pinecone account: philippe.godfroy@hotmail.com
Target Project: 'default-production' (aws/us-east-1)
Session token cached in ~/.pinecone/token`,

    "pc index list": `NAME                     DIMENSION   METRIC   SPEC_CLOUD   SPEC_REGION   STATUS
vector-forge-knowledge   1536        cosine   aws          us-east-1     Ready
customer-support-kb      1536        cosine   aws          us-east-1     Ready`,

    "pc index describe": `Index: vector-forge-knowledge
Dimension: 1536
Metric: cosine
Host: vector-forge-knowledge-c49b1a.svc.us-east-1.pinecone.io
Cloud: aws
Region: us-east-1
Status: Ready (Serverless)
Total Vectors: 7
Namespaces:
  - default: 7 records
  - tenant_acme: 0 records`,

    "pc index record search": `Querying index 'vector-forge-knowledge' (Top-K=3, namespace=default)...

RANK  ID                        SCORE     TITLE
#1    doc_pinecone_serverless   0.9412    Pinecone Serverless Architecture Deep-Dive
#2    doc_hybrid_search         0.8845    Sparse-Dense Hybrid Search & RRF
#3    doc_rag_hallucination     0.8210    RAG Contextual Chunking & Hallucination Defense`
  };

  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      const cmd = btn.dataset.cmd;
      termBody.textContent = `$ ${cmd}\n\n${COMMANDS[cmd] || "Command executed."}`;
    });
  });

  // Initial command
  termBody.textContent = `$ pc index describe --name vector-forge-knowledge\n\n${COMMANDS["pc index describe"]}`;
}

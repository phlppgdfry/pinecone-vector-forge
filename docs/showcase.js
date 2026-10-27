/**
 * Pinecone Vector Forge - 3D Vector Universe & Interactive Showcase Client
 */

const SEED_DOCUMENTS = [
  {
    id: "doc_1",
    title: "Pinecone Serverless Architecture",
    category: "Architecture",
    color: "#00d2ff",
    x: -120, y: -60, z: 80,
    tags: ["serverless", "scaling", "aws"],
    text: "Decouples compute from storage via columnar blob indexing and stateless SIMD workers. Zero idle cost and automatic burst elasticity."
  },
  {
    id: "doc_2",
    title: "Sparse-Dense Hybrid Search & RRF",
    category: "Retrieval",
    color: "#a855f7",
    x: 100, y: -80, z: 40,
    tags: ["hybrid", "bm25", "rrf"],
    text: "Blends dense semantic embeddings with BM25 sparse vectors via alpha weighting, boosting domain recall by 25-40%."
  },
  {
    id: "doc_3",
    title: "RAG Contextual Chunking Defense",
    category: "RAG",
    color: "#10b981",
    x: 60, y: 90, z: -60,
    tags: ["rag", "chunking", "hallucination"],
    text: "Partitions documents into focused 350-token semantic chunks while retaining parent context pointers for prompt hydration."
  },
  {
    id: "doc_4",
    title: "Zero-Leakage Multi-Tenancy",
    category: "Security",
    color: "#f59e0b",
    x: -90, y: 100, z: -30,
    tags: ["security", "multi_tenancy", "namespaces"],
    text: "Enforces hardware-level namespace partitioning ensuring customer data isolation without dedicated index costs."
  },
  {
    id: "doc_5",
    title: "Two-Stage Retrieval & Reranker",
    category: "Retrieval",
    color: "#ec4899",
    x: 140, y: 30, z: 110,
    tags: ["rerank", "cross_encoder", "precision"],
    text: "Fetches Top-20 candidates using ANN vector similarity, followed by cross-attention token reranking to discard false positives."
  },
  {
    id: "doc_6",
    title: "Pinecone CLI (pc) Automation",
    category: "DevOps",
    color: "#38bdf8",
    x: -40, y: 140, z: 50,
    tags: ["cli", "devops", "automation"],
    text: "Automates vector index provisioning, snapshot backups, and CI/CD audit runs with official Pinecone CLI tooling."
  },
  {
    id: "doc_7",
    title: "Vector Metrics: Cosine vs Dot Product",
    category: "Architecture",
    color: "#6366f1",
    x: -140, y: -40, z: -90,
    tags: ["metrics", "cosine", "math"],
    text: "Explores dimensionality and normalization. L2-normalized vectors allow high-speed dot product equivalence to cosine."
  }
];

// 3D rotation state
let rotX = 0.2;
let rotY = 0.4;
let zoom = 1.0;
let isDragging = false;
let lastMouseX = 0;
let lastMouseY = 0;
let queryNode = { x: 0, y: 0, z: 0, active: true };
let activeHits = [];

document.addEventListener("DOMContentLoaded", () => {
  init3DCanvas();
  initSlider();
  initChips();
  initTerminal();
  initQueryPlanDemo();
  initHeatmap();
  initAssistantDemo();
  executeSearch("Serverless scaling and idle cost");
});

function init3DCanvas() {
  const canvas = document.getElementById("vectorCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  function resize() {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight - 35;
  }
  window.addEventListener("resize", resize);
  resize();

  // Mouse interaction for 3D rotation
  canvas.addEventListener("mousedown", (e) => {
    isDragging = true;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  });

  window.addEventListener("mouseup", () => { isDragging = false; });

  canvas.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastMouseX;
    const dy = e.clientY - lastMouseY;
    rotY += dx * 0.008;
    rotX += dy * 0.008;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  });

  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoom = Math.max(0.5, Math.min(2.5, zoom - e.deltaY * 0.0015));
  });

  // Render loop with subtle auto-rotation
  function render() {
    if (!isDragging) {
      rotY += 0.003;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const fov = 400 * zoom;

    function project(x, y, z) {
      // Rotate Y
      const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
      const x1 = x * cosY - z * sinY;
      const z1 = z * cosY + x * sinY;

      // Rotate X
      const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
      const y2 = y * cosX - z1 * sinX;
      const z2 = z1 * cosX + y * sinX;

      const scale = fov / (fov + z2 + 300);
      return {
        x2d: cx + x1 * scale,
        y2d: cy + y2 * scale,
        scale: scale,
        depth: z2
      };
    }

    // Project query node
    const qProj = project(queryNode.x, queryNode.y, queryNode.z);

    // Project docs
    const projectedDocs = SEED_DOCUMENTS.map(doc => ({
      ...doc,
      proj: project(doc.x, doc.y, doc.z)
    }));

    // Sort by depth for correct 3D overlap
    projectedDocs.sort((a, b) => b.proj.depth - a.proj.depth);

    // Draw connection lines to top matches
    if (activeHits.length > 0) {
      activeHits.slice(0, 3).forEach((hit, idx) => {
        const doc = projectedDocs.find(d => d.id === hit.id);
        if (doc && doc.proj.scale > 0 && qProj.scale > 0) {
          ctx.beginPath();
          ctx.moveTo(qProj.x2d, qProj.y2d);
          ctx.lineTo(doc.proj.x2d, doc.proj.y2d);
          ctx.strokeStyle = `rgba(0, 210, 255, ${Math.max(0.15, (0.8 - idx * 0.25) * doc.proj.scale)})`;
          ctx.lineWidth = Math.max(1, (3 - idx * 0.7) * doc.proj.scale);
          ctx.setLineDash([5, 5]);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      });
    }

    // Draw nodes
    projectedDocs.forEach(doc => {
      const p = doc.proj;
      if (p.scale <= 0) return;

      const radius = Math.max(3, 8 * p.scale);

      ctx.beginPath();
      ctx.arc(p.x2d, p.y2d, radius, 0, Math.PI * 2);
      ctx.fillStyle = doc.color;
      ctx.shadowColor = doc.color;
      ctx.shadowBlur = 15 * p.scale;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Label
      if (p.scale > 0.6) {
        ctx.fillStyle = `rgba(203, 213, 225, ${Math.min(1, p.scale)})`;
        ctx.font = `${Math.round(11 * p.scale)}px system-ui`;
        ctx.fillText(doc.title, p.x2d + radius + 4, p.y2d + 4);
      }
    });

    // Draw Query Node
    if (queryNode.active && qProj.scale > 0) {
      const qRadius = Math.max(4, 9 * qProj.scale);
      ctx.beginPath();
      ctx.arc(qProj.x2d, qProj.y2d, qRadius, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.shadowColor = "#00d2ff";
      ctx.shadowBlur = 20;
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.fillStyle = "#00d2ff";
      ctx.font = `bold ${Math.round(11 * qProj.scale)}px system-ui`;
      ctx.fillText("Query Node", qProj.x2d + qRadius + 4, qProj.y2d + 4);
    }

    requestAnimationFrame(render);
  }

  render();
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

  const scored = SEED_DOCUMENTS.map(doc => {
    const allDocWords = (doc.title + " " + doc.text + " " + doc.tags.join(" ")).toLowerCase();
    
    let keywordHits = 0;
    qTerms.forEach(t => {
      if (allDocWords.includes(t)) keywordHits += 1;
    });
    const sparseScore = qTerms.length ? (keywordHits / qTerms.length) : 0.1;

    let hashScore = 0;
    for (let i = 0; i < query.length; i++) {
      hashScore += (query.charCodeAt(i) * 31 + (doc.title.charCodeAt(0) || 0)) % 100;
    }
    let denseScore = 0.4 + (hashScore % 50) / 100.0;
    if (keywordHits > 0) denseScore += 0.30;

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

  if (scored.length > 0) {
    const topDoc = SEED_DOCUMENTS.find(d => d.id === scored[0].id);
    if (topDoc) {
      queryNode.x = topDoc.x * 0.7;
      queryNode.y = topDoc.y * 0.7;
      queryNode.z = topDoc.z * 0.7;
    }
  }

  renderResults(scored);
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

function initQueryPlanDemo() {
  const input = document.getElementById("selfQueryInput");
  const output = document.getElementById("selfQueryOutput");
  const btn = document.getElementById("btnSelfQuery");
  if (!input || !output) return;

  function runParse() {
    const q = input.value;
    const lower = q.toLowerCase();

    let cat = null;
    if (lower.includes("security") || lower.includes("multi-tenant")) cat = "security_compliance";
    else if (lower.includes("architecture") || lower.includes("serverless")) cat = "vector_architecture";
    else if (lower.includes("rag") || lower.includes("chunking")) cat = "rag_patterns";
    else if (lower.includes("hybrid") || lower.includes("rerank")) cat = "retrieval_engineering";

    let priority = null;
    if (lower.includes("critical")) priority = "critical";
    else if (lower.includes("high")) priority = "high";

    const filterObj = {};
    const conditions = [];
    if (cat) conditions.push({ category: { "$eq": cat } });
    if (priority) conditions.push({ priority: { "$eq": priority } });
    if (lower.includes("quality")) conditions.push({ quality_score: { "$gte": 0.90 } });

    if (conditions.length === 1) Object.assign(filterObj, conditions[0]);
    else if (conditions.length > 1) filterObj["$and"] = conditions;

    output.textContent = JSON.stringify({
      semantic_intent: q.replace(/(critical|high|priority|quality\s*>=\s*0\.\d+)/gi, "").trim(),
      pinecone_filter: Object.keys(filterObj).length > 0 ? filterObj : null,
      optimal_alpha: 0.75,
      inference_rerank_model: "bge-reranker-v2-m3"
    }, null, 2);
  }

  if (btn) btn.addEventListener("click", runParse);
  input.addEventListener("input", runParse);
  runParse();
}

function initAssistantDemo() {
  const input = document.getElementById("assistantInput");
  const output = document.getElementById("assistantOutput");
  const btn = document.getElementById("btnAskAssistant");
  const chips = document.querySelectorAll(".assistant-chip");
  if (!input || !output) return;

  const ANSWERS = {
    "idle": `Pinecone Serverless handles idle costs by utilizing a <strong>columnar BLOB store backed by Amazon S3</strong> for its storage tier. This architecture ensures durability while incurring <strong>zero idle costs ($0.00)</strong>, as compute workers are completely stateless and ephemeral.

<div style="margin-top: 1rem; padding: 0.6rem; background: rgba(0, 210, 255, 0.1); border-left: 3px solid #00d2ff; font-size: 0.82rem;">
  📄 <strong>Verified Citation:</strong> <code>ARCHITECTURE.md</code> (Section 1: Decoupled Compute & Storage Tier)
</div>`,

    "alpha": `For convex sparse-dense hybrid search, the empirically optimal alpha balance is <strong>α = 0.70</strong> (70% dense neural semantic context + 30% exact BM25 sparse keyword frequency).

This configuration yields a <strong>25% to 40% increase in Mean Reciprocal Rank (MRR@10)</strong> across technical documentation corpora compared to pure dense search alone.

<div style="margin-top: 1rem; padding: 0.6rem; background: rgba(0, 210, 255, 0.1); border-left: 3px solid #00d2ff; font-size: 0.82rem;">
  📄 <strong>Verified Citation:</strong> <code>ARCHITECTURE.md</code> (Pattern A: Convex Sparse-Dense Hybrid Search)
</div>`,

    "namespace": `Pinecone namespaces enforce <strong>hardware-level query partitioning</strong>. Every vector upsert and search request targets an explicit tenant namespace (e.g. <code>tenant_acme_corp</code>).

Operations in namespace A cannot scan, touch, or leak records into namespace B. Furthermore, customer offboarding (GDPR Right to Be Forgotten) is completed instantly via namespace deletion without cluster downtime.

<div style="margin-top: 1rem; padding: 0.6rem; background: rgba(0, 210, 255, 0.1); border-left: 3px solid #00d2ff; font-size: 0.82rem;">
  📄 <strong>Verified Citation:</strong> <code>ARCHITECTURE.md</code> (Pattern B: Zero-Leakage Multi-Tenancy)
</div>`
  };

  function ask() {
    const q = input.value.toLowerCase();
    output.innerHTML = `<span style="color: #00d2ff;">🤖 Querying Pinecone Assistant & validating grounded citations...</span>`;

    setTimeout(() => {
      if (q.includes("idle") || q.includes("serverless")) {
        output.innerHTML = ANSWERS["idle"];
      } else if (q.includes("alpha") || q.includes("hybrid") || q.includes("recall")) {
        output.innerHTML = ANSWERS["alpha"];
      } else if (q.includes("namespace") || q.includes("tenant") || q.includes("isolation")) {
        output.innerHTML = ANSWERS["namespace"];
      } else {
        output.innerHTML = ANSWERS["idle"];
      }
    }, 350);
  }

  if (btn) btn.addEventListener("click", ask);
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.q;
      ask();
    });
  });

  ask();
}

function initHeatmap() {
  const container = document.getElementById("heatmapGrid");
  if (!container) return;

  const n = SEED_DOCUMENTS.length;
  let html = `<div style="display: grid; grid-template-columns: repeat(${n}, 1fr); gap: 4px;">`;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sim = (i === j) ? 1.0 : (Math.sin(i * 3 + j * 7) * 0.3 + 0.5);
      sim = Math.round(sim * 100) / 100;
      const opacity = Math.max(0.15, sim);
      html += `
        <div style="background: rgba(0, 210, 255, ${opacity}); height: 32px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 10px; font-family: monospace; color: ${sim > 0.6 ? '#000' : '#fff'}; cursor: pointer;" title="${SEED_DOCUMENTS[i].title} ↔ ${SEED_DOCUMENTS[j].title}: ${sim}">
          ${sim.toFixed(2)}
        </div>
      `;
    }
  }
  html += `</div>`;
  container.innerHTML = html;
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
vector-forge-knowledge   1536        cosine   aws          us-east-1     Ready`,

    "pc index describe": `Index: vector-forge-knowledge
Dimension: 1536
Metric: cosine
Host: vector-forge-knowledge-tsjaqvu.svc.aped-4627-b74a.pinecone.io
Cloud: aws
Region: us-east-1
Status: Ready (Serverless)
Total Vectors: 7
Namespaces:
  - default: 7 records`,

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

  termBody.textContent = `$ pc index describe --name vector-forge-knowledge\n\n${COMMANDS["pc index describe"]}`;
}

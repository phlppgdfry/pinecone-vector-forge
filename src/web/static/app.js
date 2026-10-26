/**
 * Pinecone Vector Forge - Dashboard Client Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initAlphaSlider();
  initQueryChips();
  fetchTelemetry();
  bindSearchForm();
  bindIngestForm();

  // Trigger initial search
  performSearch("Serverless scaling and idle costs", 0.7);
});

function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".tab-panel");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      panels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add("active");
    });
  });
}

function initAlphaSlider() {
  const slider = document.getElementById("alphaSlider");
  const valLabel = document.getElementById("alphaValue");

  slider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    let desc = `${val.toFixed(2)} `;
    if (val === 1.0) desc += "(100% Dense Semantic)";
    else if (val === 0.0) desc += "(100% Sparse Lexical)";
    else desc += `(Hybrid: ${Math.round(val*100)}% Dense / ${Math.round((1-val)*100)}% Sparse)`;
    valLabel.textContent = desc;
  });
}

function initQueryChips() {
  const chips = document.querySelectorAll(".chip");
  const input = document.getElementById("searchInput");

  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.query;
      performSearch(chip.dataset.query);
    });
  });
}

async function fetchTelemetry() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();

    document.getElementById("statTotalVectors").textContent = data.total_vector_count;
    document.getElementById("statDimension").textContent = data.dimension;
    document.getElementById("statMetric").textContent = data.metric.toUpperCase();
    
    const nsCount = Object.keys(data.namespaces).length;
    document.getElementById("statNamespaces").textContent = `${nsCount} Active`;

    const statusBadge = document.getElementById("connectionStatus");
    if (data.connected_to_pinecone) {
      statusBadge.className = "badge-status";
      statusBadge.innerHTML = `<span class="pulse-dot"></span> Live Pinecone Serverless (${data.index_name})`;
    } else {
      statusBadge.className = "badge-status warning";
      statusBadge.innerHTML = `<span class="pulse-dot"></span> Local Simulation Mirror (Run 'pc login' to sync)`;
    }
  } catch (err) {
    console.error("Telemetry fetch failed", err);
  }
}

function bindSearchForm() {
  const btn = document.getElementById("btnSearch");
  const input = document.getElementById("searchInput");

  btn.addEventListener("click", () => {
    if (input.value.trim()) {
      performSearch(input.value.trim());
    }
  });

  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      performSearch(input.value.trim());
    }
  });
}

async function performSearch(queryText, forceAlpha = null) {
  const resultsContainer = document.getElementById("searchResults");
  const metaContainer = document.getElementById("searchMeta");
  const alphaVal = forceAlpha !== null ? forceAlpha : parseFloat(document.getElementById("alphaSlider").value);
  const categoryVal = document.getElementById("categorySelect").value;
  const rerankVal = document.getElementById("rerankToggle").checked;

  resultsContainer.innerHTML = `<div style="text-align:center; padding: 2rem; color: var(--text-muted);">⚡ Scanning vector index & computing similarity...</div>`;

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: queryText,
        top_k: 5,
        alpha: alphaVal,
        namespace: "default",
        category: categoryVal,
        rerank: rerankVal
      })
    });

    const data = await res.json();
    renderResults(data);
  } catch (err) {
    resultsContainer.innerHTML = `<div style="color: #ef4444; padding: 1rem;">Query execution failed: ${err.message}</div>`;
  }
}

function renderResults(data) {
  const container = document.getElementById("searchResults");
  const metaContainer = document.getElementById("searchMeta");

  metaContainer.innerHTML = `
    <span>Returned <strong>${data.total_hits} hits</strong> in <strong>${data.latency_ms} ms</strong></span>
    <span>Mode: <strong style="color: #00d2ff;">${data.retrieval_mode}</strong></span>
  `;

  if (!data.hits || data.hits.length === 0) {
    container.innerHTML = `<div style="text-align: center; padding: 3rem; color: var(--text-muted);">No matching vectors found. Try broadening the query or adjusting filters.</div>`;
    return;
  }

  let html = "";
  data.hits.forEach((hit, idx) => {
    const scorePct = Math.min(100, Math.max(0, Math.round(hit.score * 100)));
    const tagsHtml = (hit.tags || []).map(t => `<span class="tag-pill">#${t}</span>`).join(" ");

    html += `
      <div class="hit-card">
        <div class="hit-top">
          <div class="hit-title">#${idx + 1} — ${hit.title}</div>
          <div class="score-badge">Score: ${hit.score.toFixed(4)}</div>
        </div>
        <div class="score-bar-bg">
          <div class="score-bar-fill" style="width: ${scorePct}%;"></div>
        </div>
        <div class="hit-text">${hit.text}</div>
        <div class="hit-meta">
          <span style="color: #00d2ff; font-weight: 600;">[${hit.category}]</span>
          ${tagsHtml}
        </div>
      </div>
    `;
  });

  container.innerHTML = html;
}

function bindIngestForm() {
  const btn = document.getElementById("btnIngest");
  const statusEl = document.getElementById("ingestStatus");

  btn.addEventListener("click", async () => {
    const title = document.getElementById("ingestTitle").value.trim();
    const category = document.getElementById("ingestCategory").value.trim();
    const tagsStr = document.getElementById("ingestTags").value.trim();
    const content = document.getElementById("ingestContent").value.trim();

    if (!title || !content) {
      alert("Please provide both a title and document content.");
      return;
    }

    btn.disabled = true;
    btn.textContent = "Vectorizing & Upserting...";
    statusEl.innerHTML = `<span style="color: #00d2ff;">Processing chunks and computing embeddings...</span>`;

    const tags = tagsStr ? tagsStr.split(",").map(t => t.trim()) : [];

    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title,
          category: category || "custom",
          tags,
          content,
          namespace: "default"
        })
      });

      const data = await res.json();
      statusEl.innerHTML = `<span style="color: #10b981; font-weight: 600;">✔ Successfully indexed ${data.chunks_created} vector chunk(s) into Pinecone!</span>`;
      fetchTelemetry();
      
      // Clear content
      document.getElementById("ingestContent").value = "";
    } catch (err) {
      statusEl.innerHTML = `<span style="color: #ef4444;">Failed to upsert: ${err.message}</span>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Vectorize & Upsert to Pinecone";
    }
  });
}

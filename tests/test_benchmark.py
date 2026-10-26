"""Latency and Throughput Micro-benchmarks for Pinecone Vector Forge."""

import time
import numpy as np
from src.core.embeddings import get_embedding_generator
from src.patterns.hybrid_search import HybridSearchEngine


def test_embedding_generation_latency_benchmark():
    embedder = get_embedding_generator()
    text = "Benchmarking sub-millisecond vector indexing performance in Pinecone Serverless."
    
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = embedder.embed_text(text)
        times.append((time.perf_counter() - t0) * 1000)

    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)

    # Embedding latency should be rapid
    assert p50 < 10.0, f"p50 latency too high: {p50}ms"
    assert p95 < 25.0, f"p95 latency too high: {p95}ms"


def test_hybrid_search_latency_benchmark():
    engine = HybridSearchEngine()
    times = []
    
    for _ in range(30):
        t0 = time.perf_counter()
        hits = engine.search(query="serverless idle cost scaling", top_k=5, alpha=0.7)
        times.append((time.perf_counter() - t0) * 1000)

    p50 = np.percentile(times, 50)
    assert p50 < 15.0, f"p50 query latency too high: {p50}ms"

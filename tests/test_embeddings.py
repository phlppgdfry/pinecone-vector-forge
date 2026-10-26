"""Test embedding generation and dimensionality."""

import numpy as np
from src.core.embeddings import get_embedding_generator


def test_dense_embedding_dimension():
    embedder = get_embedding_generator()
    vec = embedder.embed_text("Pinecone Serverless Architecture")
    assert len(vec) == 1536
    # Verify L2 normalization (norm should be ~1.0)
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-4


def test_sparse_embedding_generation():
    embedder = get_embedding_generator()
    sparse = embedder.generate_sparse_vector("multi-tenant vector namespace isolation")
    assert "indices" in sparse
    assert "values" in sparse
    assert len(sparse["indices"]) > 0
    assert len(sparse["indices"]) == len(sparse["values"])


def test_batch_embedding_consistency():
    embedder = get_embedding_generator()
    texts = ["Hybrid search", "BM25 keyword retrieval", "Two-stage reranking"]
    batch = embedder.embed_batch(texts)
    assert len(batch) == 3
    for v in batch:
        assert len(v) == 1536

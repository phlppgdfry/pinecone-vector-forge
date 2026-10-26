"""Test configuration and environment resolver."""

from src.core.config import Settings


def test_default_settings():
    s = Settings()
    assert s.pinecone_cloud == "aws"
    assert s.pinecone_dimension == 1536
    assert s.pinecone_metric == "cosine"
    assert s.pinecone_index_name == "vector-forge-knowledge"


def test_pinecone_configured_detection():
    s_empty = Settings(pinecone_api_key=None)
    assert not s_empty.is_pinecone_configured()

    s_placeholder = Settings(pinecone_api_key="your_pinecone_api_key_here")
    assert not s_placeholder.is_pinecone_configured()

    s_valid = Settings(pinecone_api_key="mock_test_key_valid_123")
    assert s_valid.is_pinecone_configured()

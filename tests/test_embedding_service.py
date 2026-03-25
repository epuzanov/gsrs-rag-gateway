"""Tests for EmbeddingService SSL verification behavior."""
from app.services.embedding import EmbeddingService


def test_embedding_service_verifies_ssl_by_default(monkeypatch):
    seen = {}

    class FakeClient:
        def __init__(self, *, timeout, verify):
            seen["timeout"] = timeout
            seen["verify"] = verify

    monkeypatch.setattr("app.services.embedding.httpx.Client", FakeClient)

    service = EmbeddingService(api_key="test-key")
    client = service.client

    assert client is not None
    assert seen["timeout"] == 60.0
    assert seen["verify"] is True


def test_embedding_service_can_disable_ssl_verification(monkeypatch):
    seen = {}

    class FakeClient:
        def __init__(self, *, timeout, verify):
            seen["timeout"] = timeout
            seen["verify"] = verify

    monkeypatch.setattr("app.services.embedding.httpx.Client", FakeClient)

    service = EmbeddingService(api_key="test-key", verify_ssl=False)
    client = service.client

    assert client is not None
    assert seen["timeout"] == 60.0
    assert seen["verify"] is False

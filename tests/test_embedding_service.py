"""Tests for EmbeddingService behavior."""
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


def test_embed_uses_full_openai_endpoint_and_parses_data(monkeypatch):
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

    class FakeClient:
        def __init__(self, *, timeout, verify):
            pass

        def post(self, url, headers, json):
            seen["url"] = url
            seen["headers"] = headers
            seen["json"] = json
            return FakeResponse()

    monkeypatch.setattr("app.services.embedding.httpx.Client", FakeClient)

    service = EmbeddingService(
        api_key="test-key",
        url="https://api.openai.com/v1/embeddings",
    )

    result = service.embed("hello")

    assert result == [0.1, 0.2]
    assert seen["url"] == "https://api.openai.com/v1/embeddings"
    assert seen["json"]["input"] == "hello"
    assert seen["json"]["encoding_format"] == "float"


def test_embed_batch_supports_ollama_embeddings_response(monkeypatch):
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

    class FakeClient:
        def __init__(self, *, timeout, verify):
            pass

        def post(self, url, headers, json):
            seen["url"] = url
            seen["headers"] = headers
            seen["json"] = json
            return FakeResponse()

    monkeypatch.setattr("app.services.embedding.httpx.Client", FakeClient)

    service = EmbeddingService(
        api_key="ollama",
        url="http://localhost:11434/api/embed",
    )

    result = service.embed_batch(["hello", "world"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert seen["url"] == "http://localhost:11434/api/embed"
    assert seen["json"] == {"model": "text-embedding-3-small", "input": ["hello", "world"]}


def test_embed_supports_flat_ollama_embedding_response(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embeddings": [0.5, 0.6]}

    class FakeClient:
        def __init__(self, *, timeout, verify):
            pass

        def post(self, url, headers, json):
            return FakeResponse()

    monkeypatch.setattr("app.services.embedding.httpx.Client", FakeClient)

    service = EmbeddingService(api_key="ollama", url="http://localhost:11434/api/embed")

    assert service.embed("hello") == [0.5, 0.6]

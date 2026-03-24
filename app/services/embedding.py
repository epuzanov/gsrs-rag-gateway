"""
GSRS RAG Gateway - OpenAI Embeddings Service

Simple OpenAI-compatible embeddings service.
"""
from typing import List
import httpx
from app.config import settings


class EmbeddingService:
    """
    OpenAI-compatible embeddings service.

    Works with:
    - OpenAI API (api.openai.com)
    - Azure OpenAI
    - Any OpenAI-compatible API (vLLM, Ollama, etc.)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dimension: int = 1536
    ):
        """
        Initialize OpenAI embeddings service.

        Args:
            api_key: API key for authentication
            model: Model name (default: text-embedding-3-small)
            base_url: API base URL (default: https://api.openai.com/v1)
            dimension: Embedding dimension (default: 1536)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.dimension = dimension
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=60.0)
        return self._client

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        url = f"{self.base_url}/embeddings"

        response = self.client.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "input": text,
                "encoding_format": "float"
            }
        )
        response.raise_for_status()
        data = response.json()

        return data["data"][0]["embedding"]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            url = f"{self.base_url}/embeddings"

            response = self.client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "input": batch,
                    "encoding_format": "float"
                }
            )
            response.raise_for_status()
            data = response.json()

            # Sort by index to ensure correct order
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            embeddings.extend([item["embedding"] for item in sorted_data])

        return embeddings

    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "provider": "openai",
            "model": self.model,
            "dimension": self.dimension,
            "base_url": self.base_url
        }

    def close(self):
        """Close HTTP client."""
        if self._client:
            self._client.close()

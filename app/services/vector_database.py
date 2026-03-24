"""
GSRS RAG Gateway - Vector Database Service
Unified service layer for vector database operations.
"""
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

from app.db.base import VectorDatabase
from app.db.factory import create_vector_database
from app.models import VectorDocument


class VectorDatabaseService:
    """
    Service layer for vector database operations.

    Provides a unified interface regardless of the underlying backend.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the vector database service.

        Args:
            database_url: Database URL (scheme determines backend)
            **kwargs: Backend-specific arguments
        """
        self._database_url = database_url
        self._db: Optional[VectorDatabase] = None
        self._kwargs = kwargs

    @property
    def db(self) -> VectorDatabase:
        """Lazy-load the database connection."""
        if self._db is None:
            self._db = create_vector_database(self._database_url, **self._kwargs)
            self._db.connect()
        return self._db

    def initialize(self, dimension: int = 384) -> None:
        """
        Initialize the database (create tables/collections).

        Args:
            dimension: Embedding dimension
        """
        self.db.initialize(dimension=dimension)

    def upsert_chunks(self, chunks: List[VectorDocument], embeddings: List[List[float]]) -> int:
        """
        Insert or update chunks with their embeddings.

        Args:
            chunks: List of VectorDocument objects
            embeddings: List of embedding vectors

        Returns:
            Number of chunks inserted/updated
        """
        from uuid import UUID

        documents = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk.set_embedding(embedding)
            documents.append(chunk)

        return self.db.upsert_documents(documents)

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[VectorDocument, float]]:
        """
        Search for similar documents.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filters: Additional filters

        Returns:
            List of (document, score) tuples
        """
        results = self.db.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters or {}
        )

        return [(r.document, r.score) for r in results]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[VectorDocument]:
        """
        Get a chunk by its chunk ID.

        Args:
            chunk_id: The chunk ID (e.g., "root_uuid:12345678-...")

        Returns:
            The document or None
        """
        return self.db.get_document(chunk_id)

    def get_chunks_by_substance(
        self,
        substance_uuid: UUID,
        limit: Optional[int] = None
    ) -> List[VectorDocument]:
        """
        Get all chunks for a substance.

        Args:
            substance_uuid: Substance UUID
            limit: Optional limit

        Returns:
            List of documents
        """
        return self.db.get_documents_by_substance(substance_uuid, limit)

    def delete_chunks_by_substance(self, substance_uuid: UUID) -> int:
        """
        Delete all chunks for a substance.

        Args:
            substance_uuid: Substance UUID

        Returns:
            Number of deleted chunks
        """
        return self.db.delete_documents_by_substance(substance_uuid)

    def delete_all(self) -> None:
        """Delete all documents."""
        self.db.delete_all()

    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        return self.db.get_statistics()

    def close(self) -> None:
        """Close database connection."""
        if self._db:
            self._db.disconnect()
            self._db = None

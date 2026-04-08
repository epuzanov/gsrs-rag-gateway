"""
GSRS RAG Gateway - Vector Database Abstract Interface
"""
from typing import Any, List, Dict, Optional
from uuid import UUID, uuid4

from sqlalchemy import String, Text, DateTime, JSON, Index
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func as sql_func
from pgvector.sqlalchemy import Vector, HALFVEC as HalfVec

from app.config import settings


class Base(DeclarativeBase):
    pass


class VectorDocument(Base):
    """
    Represents a chunk of a GSRS Substance document.
    Each chunk corresponds to a specific section in the substance JSON.

    Compatible with gsrs.services.ai SubstanceChunker(class_=VectorDocument).chunks(substance)
    output format.
    """
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(primary_key=True, default=lambda: str(uuid4()))

    # Unique chunk identifier from gsrs.model (e.g., "root_uuid:12345678-...")
    chunk_id: Mapped[str] = mapped_column(String(512), nullable=False, unique=True, index=True)

    # Document ID (substance UUID)
    document_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False, index=True)

    # Section name (e.g., "root", "names", "codes", "structure", "references")
    section: Mapped[str] = mapped_column(String(256), nullable=False, index=True)

    # Source URL/name (system-generated, from gsrs.model)
    source_url: Mapped[Optional[str]] = mapped_column(String(256), nullable=True, index=True)

    # The actual text content of the chunk
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Vector embedding (uses HalfVec for dimensions > 2000)
    embedding: Mapped[List[float]] = mapped_column(
        HalfVec(settings.embedding_dimension) if settings.embedding_dimension > 2000 else Vector(settings.embedding_dimension),
        nullable=False,
        index=True,
    )

    # Metadata containing all element attributes from gsrs.model:
    # - canonical_name: preferred substance name
    # - chunk_type: type of chunk (overview, name, code, etc.)
    # - hierarchy: parent context information
    # - additional gsrs.model metadata fields
    metadata_json: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, nullable=False, default=dict)

    # Timestamps
    created_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=sql_func.now())
    updated_at: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=sql_func.now(), onupdate=sql_func.now())

    __table_args__ = (
        Index('idx_document_id', 'document_id'),
        Index('idx_section', 'section'),
        Index('idx_source_url', 'source_url'),
        Index(
            'idx_embedding_hnsw',
            'embedding',
            postgresql_using='hnsw',
            postgresql_with={'m': 16, 'ef_construction': 64},
            postgresql_ops={'embedding': "halfvec_cosine_ops" if settings.embedding_dimension > 2000 else "vector_cosine_ops"},
        ),
    )

    def __init__(self, *args: Any, **kwargs: Any):
        """Accept `metadata` constructor input and store it in `metadata_json`."""
        if "metadata" in kwargs:
            kwargs["metadata_json"] = kwargs.pop("metadata")
        super().__init__(*args, **kwargs)

    def values(self):
        return {
            "document_id": self.document_id,
            "section": self.section,
            "source_url": self.source_url,
            "text": self.text,
            "embedding": self.embedding,
            "metadata_json": self.metadata_json,
        }

    def set_embedding(self, embedding: List[float]) -> None:
        """Set the embedding vector."""
        self.embedding = embedding

    def __repr__(self):
        return f"<VectorDocument(chunk_id={self.chunk_id}, section={self.section})>"


class DBQueryResult:
    """Represents a query result with similarity score."""
    document: Any  # VectorDocument from app.models
    score: float

    def __init__(self, document: Any, score: float):
        self.document = document
        self.score = score


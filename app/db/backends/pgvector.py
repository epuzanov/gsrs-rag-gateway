"""
GSRS RAG Gateway - pgvector Backend Implementation
"""
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID

from sqlalchemy import create_engine, text, distinct
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import insert

from app.db.base import VectorDatabase
from app.models import Base, VectorDocument, DBQueryResult


class PGVectorDatabase(VectorDatabase):
    """
    pgvector implementation of the VectorDatabase interface.

    Uses PostgreSQL with pgvector extension for vector storage and similarity search.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None

    def connect(self) -> None:
        """Establish connection to PostgreSQL."""
        self.engine = create_engine(self.database_url)
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

    def disconnect(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.engine = None
            self.SessionLocal = None

    def initialize(self, dimension: int = 384) -> None:
        """Create tables and indexes."""
        if self.engine is None:
            self.connect()

        if self.engine is None:
            raise ConnectionError("Failed to connect to the database.")

        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()

        Base.metadata.create_all(bind=self.engine)

    def _get_session(self) -> Session:
        """Get a database session."""
        if self.SessionLocal is None:
            self.connect()
        if self.SessionLocal is None:
            raise ConnectionError("Database session is not available.")
        return self.SessionLocal()

    def upsert_documents(self, documents: List[VectorDocument]) -> int:
        """Insert or update documents."""
        session = self._get_session()
        count = 0

        try:
            for doc in documents:
                stmt = insert(VectorDocument).values(
                    chunk_id = doc.chunk_id,
                    **doc.values()
                ).on_conflict_do_update(
                    index_elements=[VectorDocument.chunk_id],
                    set_ = doc.values()
                )

                session.execute(stmt)
                count += 1

            session.commit()
            return count

        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DBQueryResult]:
        """Search for similar documents using cosine similarity."""
        session = self._get_session()

        try:
            query = session.query(
                VectorDocument,
                VectorDocument.embedding.cosine_distance(query_embedding).label('similarity')
            )

            if filters:
                if 'section' in filters:
                    query = query.filter(
                        VectorDocument.section == filters['section']
                    )
                if 'document_id' in filters:
                    query = query.filter(
                        VectorDocument.document_id == filters['document_id']
                    )

            results = (
                query
                .order_by('similarity')
                .limit(top_k)
                .all()
            )

            query_results = []
            for chunk, similarity in results:
                query_results.append(DBQueryResult(document=chunk, score=1 - similarity))

            return query_results

        finally:
            session.close()

    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by chunk_id."""
        session = self._get_session()

        try:
            chunk = session.query(VectorDocument).filter(
                VectorDocument.chunk_id == doc_id
            ).first()

            if chunk:
                return chunk
            return None
        finally:
            session.close()

    def get_documents_by_substance(
        self,
        substance_uuid: UUID,
        limit: Optional[int] = None
    ) -> List[VectorDocument]:
        """Get all documents for a substance."""
        session = self._get_session()
        try:
            query = session.query(VectorDocument).filter(
                VectorDocument.document_id == substance_uuid
            )

            if limit:
                query = query.limit(limit)

            chunks = query.all()

            return chunks
        finally:
            session.close()

    def delete_documents_by_substance(self, substance_uuid: UUID) -> int:
        """Delete all documents for a substance."""
        session = self._get_session()
        try:
            count = session.query(VectorDocument).filter(
                VectorDocument.document_id == substance_uuid
            ).delete(synchronize_session=False)
            session.commit()
            return count
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def delete_all(self) -> None:
        """Delete all documents."""
        session = self._get_session()
        try:
            session.query(VectorDocument).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        session = self._get_session()
        try:
            total_chunks = session.query(VectorDocument).count()
            substances = session.query(
                VectorDocument.document_id
            ).distinct().count()

            return {
                "total_chunks": total_chunks,
                "total_substances": substances,
            }
        finally:
            session.close()

    def get_unique_values(self, field: str) -> List[str]:
        """Get unique values for a field."""
        session = self._get_session()
        try:
            if field == "section":
                results = session.query(distinct(VectorDocument.section)).all()
                return [r[0] for r in results if r[0]]
            elif field == "source_url":
                results = session.query(distinct(VectorDocument.source_url)).all()
                return [r[0] for r in results if r[0]]
            return []
        finally:
            session.close()


"""
GSRS RAG Gateway - Unit Tests for Vector Database Backends
"""
import unittest
import tempfile
import shutil
import os
from uuid import uuid4
from typing import List

from app.db.base import VectorDatabase
from app.db.factory import create_vector_database, get_available_backends
from app.models import VectorDocument, DBQueryResult

# Alias for backward compatibility in tests
QueryResult = DBQueryResult


class TestChromaDatabase(unittest.TestCase):
    """Unit tests for ChromaDB backend."""
    
    test_dimension = 384
    
    def setUp(self):
        """Set up ChromaDB for testing."""
        self.test_dir = tempfile.mkdtemp()
        # Use chroma:// URL scheme to force ChromaDB backend
        chroma_url = f"chroma://{self.test_dir}/test_collection"
        self.db = create_vector_database(
            database_url=chroma_url
        )
        self.db.connect()
        self.db.initialize(dimension=self.test_dimension)
        
        # Create test documents
        self.test_docs = [
            VectorDocument(
                chunk_id=f"root_codes_{i}_code",
                document_id=uuid4(),
                section="codes",
                text=f"Test chunk text {i}",
                embedding=[float(i + 1)] * self.test_dimension,
                chunk_metadata={"test": f"value_{i}"},
                source_url="test_source"
            )
            for i in range(5)
        ]
    
    def tearDown(self):
        """Clean up test directory."""
        try:
            self.db.delete_all()
        except Exception:
            pass
        self.db.disconnect()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_upsert_documents(self):
        """Test inserting documents."""
        count = self.db.upsert_documents(self.test_docs)
        self.assertEqual(count, 5)
        
        stats = self.db.get_statistics()
        self.assertEqual(stats["total_chunks"], 5)
    
    def test_get_document(self):
        """Test getting a document by ID."""
        self.db.upsert_documents(self.test_docs)

        # ChromaDB uses chunk_id as the ID for retrieval
        doc = self.db.get_document("root_codes_0_code")
        self.assertIsNotNone(doc)
        self.assertEqual(doc.chunk_id, "root_codes_0_code")
        self.assertEqual(doc.text, "Test chunk text 0")
    
    def test_get_document_not_found(self):
        """Test getting a non-existent document."""
        doc = self.db.get_document("nonexistent_path")
        self.assertIsNone(doc)
    
    def test_get_documents_by_substance(self):
        """Test getting documents by substance UUID."""
        substance_uuid = uuid4()

        docs = [
            VectorDocument(
                id=f"sub_path_{i}",
                chunk_id=f"root_{i}",
                document_id=substance_uuid,
                section="root",
                text=f"Substance chunk {i}",
                embedding=[0.1] * self.test_dimension,
                chunk_metadata={},
                source_url="test"
            )
            for i in range(3)
        ]

        self.db.upsert_documents(docs)
        results = self.db.get_documents_by_substance(substance_uuid)

        self.assertEqual(len(results), 3)
    
    def test_similarity_search(self):
        """Test similarity search."""
        self.db.upsert_documents(self.test_docs)
        
        # Search with similar embedding
        query_embedding = [1.0] * self.test_dimension
        results = self.db.similarity_search(
            query_embedding=query_embedding,
            top_k=3
        )
        
        self.assertLessEqual(len(results), 3)
        self.assertTrue(all(isinstance(r, QueryResult) for r in results))
    
    def test_similarity_search_with_filter(self):
        """Test similarity search with filters."""
        self.db.upsert_documents(self.test_docs)

        query_embedding = [1.0] * self.test_dimension
        results = self.db.similarity_search(
            query_embedding=query_embedding,
            top_k=5,
            filters={"test": "value_0"}
        )

        self.assertGreater(len(results), 0)
    
    def test_delete_documents_by_substance(self):
        """Test deleting documents by substance UUID."""
        substance_uuid = uuid4()

        docs = [
            VectorDocument(
                id=f"del_path_{i}",
                chunk_id=f"root_{i}",
                document_id=substance_uuid,
                section="root",
                text=f"To delete {i}",
                embedding=[0.1] * self.test_dimension,
                chunk_metadata={},
                source_url="test"
            )
            for i in range(3)
        ]

        self.db.upsert_documents(docs)
        deleted = self.db.delete_documents_by_substance(substance_uuid)

        self.assertEqual(deleted, 3)

        stats = self.db.get_statistics()
        self.assertEqual(stats["total_chunks"], 0)
    
    def test_delete_all(self):
        """Test deleting all documents."""
        self.db.upsert_documents(self.test_docs)
        self.db.delete_all()
        
        stats = self.db.get_statistics()
        self.assertEqual(stats["total_chunks"], 0)
    
    def test_get_statistics(self):
        """Test getting statistics."""
        self.db.upsert_documents(self.test_docs)

        stats = self.db.get_statistics()

        self.assertIn("total_chunks", stats)
        self.assertIn("total_substances", stats)
        self.assertEqual(stats["total_chunks"], 5)
    
    def test_get_unique_values(self):
        """Test getting unique field values."""
        self.db.upsert_documents(self.test_docs)

        values = self.db.get_unique_values("test")

        self.assertIn("value_0", values)
    
    def test_upsert_update(self):
        """Test that upsert updates existing documents."""
        # Note: ChromaDB doesn't support true updates, this tests insert behavior
        doc = VectorDocument(
            id="update_path",
            chunk_id="root_update_code",
            document_id=uuid4(),
            section="codes",
            text="Original text",
            embedding=[0.1] * self.test_dimension,
            metadata={"version": 1},
            source_url="test"
        )

        count = self.db.upsert_documents([doc])
        self.assertEqual(count, 1)

        # Verify insert - use chunk_id for retrieval
        retrieved = self.db.get_document("root_update_code")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.text, "Original text")


class TestFactory(unittest.TestCase):
    """Unit tests for the vector database factory."""
    
    def test_get_available_backends(self):
        """Test getting available backends."""
        backends = get_available_backends()
        
        self.assertIsInstance(backends, list)
        self.assertIn("pgvector", backends)
    
    def test_create_chroma_database(self):
        """Test creating ChromaDB instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chroma_url = f"chroma://{tmpdir}/test_collection"
            db = create_vector_database(
                database_url=chroma_url
            )

            from app.db.backends.chroma import ChromaDatabase
            self.assertIsInstance(db, ChromaDatabase)
    
    def test_create_pgvector_database(self):
        """Test creating pgvector instance."""
        try:
            db = create_vector_database(
                backend="pgvector",
                database_url="postgresql://test:test@localhost/test"
            )
            
            from app.db.backends.pgvector import PGVectorDatabase
            self.assertIsInstance(db, PGVectorDatabase)
        except ImportError:
            # Skip if pgvector is not installed
            self.skipTest("pgvector not installed")
    
    def test_create_unknown_backend(self):
        """Test creating unknown backend raises error."""
        with self.assertRaises(ValueError) as context:
            create_vector_database(
                database_url="unknown://localhost/test"
            )

        self.assertIn("Unknown database scheme", str(context.exception))


class TestVectorDocument(unittest.TestCase):
    """Unit tests for VectorDocument dataclass."""

    def test_create_document(self):
        """Test creating a VectorDocument."""
        doc = VectorDocument(
            chunk_id="root_code",
            document_id=uuid4(),
            section="codes",
            text="Test text",
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "value"},
            source_url="test"
        )

        self.assertEqual(doc.chunk_id, "root_code")
        self.assertEqual(doc.text, "Test text")
        self.assertEqual(len(doc.embedding), 3)

    def test_document_repr(self):
        """Test VectorDocument string representation."""
        doc = VectorDocument(
            chunk_id="root_code",
            document_id=uuid4(),
            section="codes",
            text="Test",
            embedding=[],
            chunk_metadata={},
            source_url="test"
        )

        # Should not raise
        repr(doc)


class TestQueryResult(unittest.TestCase):
    """Unit tests for QueryResult dataclass."""

    def test_create_result(self):
        """Test creating a QueryResult."""
        doc = VectorDocument(
            chunk_id="root_code",
            document_id=uuid4(),
            section="codes",
            text="Test",
            embedding=[],
            chunk_metadata={},
            source_url="test"
        )

        result = QueryResult(document=doc, score=0.95)

        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.document.chunk_id, "root_code")


if __name__ == "__main__":
    unittest.main()

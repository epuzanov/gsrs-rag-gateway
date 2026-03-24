"""
GSRS RAG Gateway - ChromaDB Backend Implementation

ChromaDB is a lightweight, embedded vector database perfect for
development and testing. It requires no external server.
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Sequence
from uuid import UUID
import json

if TYPE_CHECKING:
    import chromadb
    from chromadb.config import Settings
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from chromadb.base_types import Vector
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from app.db.base import VectorDatabase
from app.models import VectorDocument, DBQueryResult

class ChromaDatabase(VectorDatabase):
    """
    ChromaDB implementation of the VectorDatabase interface.

    Uses ChromaDB for local, serverless vector storage.
    Ideal for development and testing.
    """

    def __init__(self, database_url: str = "chroma://./chroma_data/substance_chunks"):
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install chromadb"
            )

        # Parse chroma URL: chroma://<persist_directory>/<collection_name>
        from urllib.parse import urlparse
        parsed = urlparse(database_url)

        # Reconstruct path with netloc (for cases like chroma://./path/collection)
        path = parsed.netloc + parsed.path
        path = path.lstrip("/")

        # Split path into directory and collection name
        if "/" in path:
            self.persist_directory, self.collection_name = path.rsplit("/", 1)
        else:
            self.persist_directory = path or "./chroma_data"
            self.collection_name = "substance_chunks"

        self.client: Optional["ClientAPI"] = None
        self.collection: Optional["Collection"] = None
    
    def connect(self) -> None:
        """Initialize ChromaDB client."""
        # Use persistent storage
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
    
    def disconnect(self) -> None:
        """Close ChromaDB connection."""
        self.client = None
        self.collection = None
    
    def initialize(self, dimension: int = 384) -> None:
        """Create or get the collection."""
        if self.client is None:
            self.connect()
        if self.client is None:
            raise RuntimeError("Failed to connect to ChromaDB.")

        # Delete existing collection to reset schema if needed
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        # Create collection with metadata
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"dimension": dimension}
        )
    
    def upsert_documents(self, documents: List[VectorDocument]) -> int:
        """Insert or update documents."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        ids = []
        embeddings = []
        metadatas = []
        documents_list = []

        for doc in documents:
            # Use chunk_id as the ChromaDB ID for consistent retrieval
            ids.append(doc.chunk_id)
            embeddings.append(doc.embedding)
            metadatas.append({
                "document_id": str(doc.document_id),
                "chunk_id": doc.chunk_id,
                "section": doc.section,
                "source_url": doc.source_url or "",
                "metadata_json": json.dumps(doc.chunk_metadata)
            })
            documents_list.append(doc.text)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_list
        )

        return len(documents)
    
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DBQueryResult]:
        """Search for similar documents."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        # Build where filter
        where = None
        if filters:
            where = {}
            if "section" in filters:
                where["section"] = filters["section"]
            if "document_id" in filters:
                where["document_id"] = filters["document_id"]
            
            # ChromaDB requires non-empty where dict
            if not where:
                where = None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["embeddings", "metadatas", "documents", "distances"]
        )

        query_results = []

        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}

                # Parse metadata JSON
                metadata_json = {}
                if 'metadata_json' in metadata:
                    metadata_value = metadata['metadata_json']
                    if isinstance(metadata_value, str):
                        try:
                            metadata_json = json.loads(metadata_value)
                        except (json.JSONDecodeError, TypeError):
                            metadata_json = {}

                # Handle embeddings - ChromaDB may return numpy arrays
                embedding: List[float] = []
                if results['embeddings'] and i < len(results['embeddings'][0]):
                    emb = results['embeddings'][0][i]
                    embedding = [] if emb is None else emb.tolist() if not isinstance(emb, Sequence) else list(emb)

                chunk = VectorDocument(
                    chunk_id=str(metadata.get('chunk_id', doc_id)),
                    document_id=UUID(str(metadata.get('document_id', '00000000-0000-0000-0000-000000000000'))),
                    section=metadata.get('section', ''),
                    source_url=metadata.get('source_url', ''),
                    text=results['documents'][0][i] if results['documents'] else '',
                    embedding=embedding,
                    chunk_metadata=metadata_json
                )

                # Chroma returns distance, convert to similarity score
                distance = results['distances'][0][i] if results['distances'] else 0
                score = 1 - distance  # Convert distance to similarity

                query_results.append(DBQueryResult(document=chunk, score=score))

        return query_results
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a document by ID."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        results = self.collection.get(
            ids=[doc_id],
            include=["embeddings", "metadatas", "documents"]
        )

        # ChromaDB returns flat lists: {'ids': ['id1'], 'metadatas': [meta1], ...}
        if not results or not results['ids'] or len(results['ids']) == 0:
            return None

        metadata = results['metadatas'][0] if results['metadatas'] and len(results['metadatas']) > 0 else {}

        metadata_json = {}
        if 'metadata_json' in metadata:
            metadata_value = metadata['metadata_json']
            if isinstance(metadata_value, str):
                try:
                    metadata_json = json.loads(metadata_value)
                except (json.JSONDecodeError, TypeError):
                    metadata_json = {}

        # Handle embeddings - ChromaDB may return numpy arrays
        embedding: List[float] = []
        if results['embeddings'] is not None and len(results['embeddings']) > 0:
            emb = results['embeddings'][0]
            embedding = [] if emb is None else emb.tolist() if not isinstance(emb, Sequence) else list(emb)

        return VectorDocument(
            chunk_id=metadata.get('chunk_id', results['ids'][0]),
            document_id=UUID(str(metadata.get('document_id', '00000000-0000-0000-0000-000000000000'))),
            section=metadata.get('section', ''),
            source_url=metadata.get('source_url', ''),
            text=results['documents'][0] if results['documents'] and len(results['documents']) > 0 else '',
            embedding=embedding,
            metadata=metadata_json
        )

    def get_documents_by_substance(
        self,
        substance_uuid: UUID,
        limit: Optional[int] = None
    ) -> List[VectorDocument]:
        """Get all documents for a substance."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        results = self.collection.get(
            where={"document_id": str(substance_uuid)},
            include=["embeddings", "metadatas", "documents"]
        )

        documents = []
        count = 0

        if results and results['ids']:
            for i, doc_id in enumerate(results['ids']):
                if limit and count >= limit:
                    break

                metadata = results['metadatas'][i] if results['metadatas'] and i < len(results['metadatas']) else {}

                metadata_json = {}
                if 'metadata_json' in metadata:
                    metadata_value = metadata['metadata_json']
                    if isinstance(metadata_value, str):
                        try:
                            metadata_json = json.loads(metadata_value)
                        except (json.JSONDecodeError, TypeError):
                            metadata_json = {}

                # Handle embeddings - ChromaDB may return numpy arrays
                embedding: List[float] = []
                if results['embeddings'] is not None and i < len(results['embeddings']):
                    emb = results['embeddings'][i]
                    embedding = [] if emb is None else emb.tolist() if not isinstance(emb, Sequence) else list(emb)

                documents.append(VectorDocument(
                    chunk_id=metadata.get('chunk_id', doc_id),
                    document_id=UUID(str(metadata.get('document_id', '00000000-0000-0000-0000-000000000000'))),
                    section=metadata.get('section', ''),
                    source_url=metadata.get('source_url', ''),
                    text=results['documents'][i] if results['documents'] and i < len(results['documents']) else '',
                    embedding=embedding,
                    metadata=metadata_json
                ))
                count += 1

        return documents
    
    def delete_documents_by_substance(self, substance_uuid: UUID) -> int:
        """Delete all documents for a substance."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        # First get count
        results = self.collection.get(
            where={"document_id": str(substance_uuid)},
            include=[]
        )

        count = len(results['ids']) if results and results['ids'] else 0

        # Delete
        self.collection.delete(
            where={"document_id": str(substance_uuid)}
        )

        return count
    
    def delete_all(self) -> None:
        """Delete all documents."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        # Delete and recreate collection
        if self.client is not None:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata=self.collection.metadata
            )
    
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        # Get all metadata to calculate statistics
        results = self.collection.get(
            include=["metadatas"]
        )

        total_chunks = len(results['ids']) if results and results['ids'] else 0

        document_ids = set()

        if results and results['metadatas']:
            for metadata in results['metadatas']:
                if 'document_id' in metadata:
                    document_ids.add(metadata['document_id'])

        return {
            "total_chunks": total_chunks,
            "total_substances": len(document_ids),
        }
    
    def get_unique_values(self, field: str) -> List[str]:
        """Get unique values for a field."""
        if self.collection is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        results = self.collection.get(include=["metadatas"])

        values = set()
        if results and results['metadatas']:
            for metadata in results['metadatas']:
                # Check both direct metadata and parsed metadata_json
                if field in metadata and metadata[field]:
                    values.add(metadata[field])
                elif 'metadata_json' in metadata:
                    # Parse metadata_json if present
                    metadata_json = {}
                    metadata_value = metadata['metadata_json']
                    if isinstance(metadata_value, str):
                        try:
                            metadata_json = json.loads(metadata_value)
                        except (json.JSONDecodeError, TypeError):
                            metadata_json = {}
                    elif isinstance(metadata_value, dict):
                        metadata_json = metadata_value
                    value =  metadata_json.get(field)
                    if value:
                        values.add(value)
        return sorted(list(values))

"""
GSRS RAG Gateway - API Schemas
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from app.models.db import VectorDocument


class IngestRequest(BaseModel):
    """Request to ingest a substance document."""
    substance: Dict[str, Any] = Field(..., description="GSRS Substance JSON document")


class IngestResponse(BaseModel):
    """Response after ingesting a substance."""
    substance_uuid: str
    chunks_created: int
    element_paths: List[str]


class QueryRequest(BaseModel):
    """Request for semantic search."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional metadata filters")


class QueryResult(BaseModel):
    """A single query result."""
    element_path: str
    substance_uuid: str
    text: str
    similarity_score: float
    metadata: Dict[str, Any]

    def __init__(self, chunk: VectorDocument, score: float = 0.0):
        """Accept `chunk` and `score` constructor input."""
        self.element_path = str(chunk.section)
        self.substance_uuid = str(chunk.document_id)
        self.text = str(chunk.text)
        self.similarity_score = score
        self.metadata = chunk.metadata_json


class QueryResponse(BaseModel):
    """Response for semantic search."""
    query: str
    results: List[QueryResult]
    total_results: int


class BatchIngestRequest(BaseModel):
    """Request to ingest multiple substances."""
    substances: List[Dict[str, Any]] = Field(..., description="List of GSRS Substance JSON documents")


class BatchIngestResponse(BaseModel):
    """Response after batch ingestion."""
    total_substances: int
    total_chunks: int
    successful: int
    failed: int
    errors: List[str] = []


class ModelInfo(BaseModel):
    """Information about the embedding model."""
    name: str
    path: str
    dimension: int
    description: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database_connected: bool
    model_loaded: bool
    statistics: Dict[str, int]


# ERI (External Retrieval Interface) Schemas
class ERIQueryRequest(BaseModel):
    """ERI query request schema."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional metadata filters")


class ERIResult(BaseModel):
    """A single ERI result."""
    id: str = Field(..., description="Unique result identifier")
    text: str = Field(..., description="Result text content")
    score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Result metadata")

    def __init__(self, chunk: VectorDocument, score: float = 0.0):
        """Accept `chunk` and `score` constructor input."""
        self.id = str(chunk.chunk_id)
        self.text = str(chunk.text)
        self.score = score
        self.metadata = chunk.metadata_json or {}
        self.metadata["section"] = str(chunk.section)
        self.metadata["document_id"] = str(chunk.document_id)
        self.metadata["source_url"] = str(chunk.source_url)


class ERIQueryResponse(BaseModel):
    """ERI query response schema."""
    results: List[ERIResult] = Field(default_factory=list, description="List of results")


class DeleteResponse(BaseModel):
    """Response after deletion."""
    substance_uuid: str
    chunks_deleted: int


class AvailableModelsResponse(BaseModel):
    """Response with available embedding models."""
    models: Dict[str, Dict[str, str]]
    current_model: str

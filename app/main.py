"""
GSRS RAG Gateway - FastAPI Application
"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from uuid import UUID
import logging
import secrets

from app.config import settings
from app.services import VectorDatabaseService, ChunkerService, EmbeddingService
from app.models import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, QueryResult,
    BatchIngestRequest, BatchIngestResponse,
    ModelInfo, HealthResponse,
    DeleteResponse, AvailableModelsResponse,
    ERIQueryRequest, ERIQueryResponse, ERIResult
 )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GSRS RAG Gateway",
    description="Retrieval-Augmented Generation Gateway for GSRS Substances using pgvector",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
vector_db = VectorDatabaseService()
embedding_service = EmbeddingService(
    api_key=settings.embedding_api_key,
    model=settings.embedding_model,
    base_url=settings.embedding_base_url,
    dimension=settings.embedding_dimension,
    verify_ssl=settings.embedding_verify_ssl,
)
chunker = ChunkerService()

# Basic authentication
security = HTTPBasic()


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify HTTP Basic credentials."""
    correct_username = secrets.compare_digest(credentials.username, settings.api_username)
    correct_password = secrets.compare_digest(credentials.password, settings.api_password)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.on_event("startup")
async def startup_event():
    """Initialize database and models on startup."""
    logger.info("Initializing vector database...")
    vector_db.initialize(dimension=settings.embedding_dimension)
    logger.info(f"Loaded embedding model: {embedding_service.get_model_info()}")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check the health status of the service."""
    try:
        stats = vector_db.get_statistics()
        db_connected = True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        db_connected = False
        stats = {"total_chunks": 0, "total_substances": 0}

    try:
        model_info = embedding_service.get_model_info()
        model_loaded = True
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        model_loaded = False

    return HealthResponse(
        status="healthy" if db_connected and model_loaded else "unhealthy",
        database_connected=db_connected,
        model_loaded=model_loaded,
        statistics=stats
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_substance(request: IngestRequest, username: str = Depends(verify_credentials)):
    """
    Ingest a single GSRS Substance document.

    The substance is chunked based on element paths,
    embeddings are generated, and stored in pgvector.

    Requires authentication.
    """
    try:
        # Chunk the substance
        chunks = chunker.chunk_substance(request.substance)

        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from substance")

        # Generate embeddings
        texts = [str(chunk.text) for chunk in chunks]
        embeddings = embedding_service.embed_batch(texts)

        # Store in database
        count = vector_db.upsert_chunks(chunks, embeddings)

        substance_uuid = request.substance.get("uuid", "unknown")
        sections = [str(chunk.section) for chunk in chunks]

        logger.info(f"Ingested substance {substance_uuid}: {count} chunks")

        return IngestResponse(
            substance_uuid=substance_uuid,
            chunks_created=count,
            element_paths=sections
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/batch", response_model=BatchIngestResponse, tags=["Ingestion"])
async def ingest_batch(request: BatchIngestRequest, username: str = Depends(verify_credentials)):
    """
    Ingest multiple GSRS Substance documents in batch.

    Requires authentication.
    """
    successful = 0
    failed = 0
    total_chunks = 0
    errors = []

    for idx, substance in enumerate(request.substances):
        try:
            chunks = chunker.chunk_substance(substance)

            if chunks:
                texts = [str(chunk.text) for chunk in chunks]
                embeddings = embedding_service.embed_batch(texts)
                count = vector_db.upsert_chunks(chunks, embeddings)
                total_chunks += count
                successful += 1
            else:
                failed += 1
                errors.append(f"Substance {idx}: No chunks generated")

        except Exception as e:
            failed += 1
            errors.append(f"Substance {idx}: {str(e)}")
            logger.error(f"Batch ingestion failed for substance {idx}: {e}")

    return BatchIngestResponse(
        total_substances=len(request.substances),
        total_chunks=total_chunks,
        successful=successful,
        failed=failed,
        errors=errors
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Perform semantic search on substance chunks.

    Returns the most relevant chunks based on vector similarity.
    """
    try:
        # Generate query embedding
        query_embedding = embedding_service.embed(request.query)

        # Search database
        results = vector_db.similarity_search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            filters=request.filters
        )

        query_results = [
            QueryResult(
                element_path=str(chunk.section),
                substance_uuid=str(chunk.document_id),
                text=str(chunk.text),
                similarity_score=score,
                metadata={k: str(v) for k, v in chunk.chunk_metadata.items()} if chunk.chunk_metadata else {}
            )
            for chunk, score in results
        ]

        return QueryResponse(
            query=request.query,
            results=query_results,
            total_results=len(query_results)
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/eri/query", response_model=ERIQueryResponse, tags=["ERI"])
async def eri_query(request: ERIQueryRequest):
    """
    ERI (External Retrieval Interface) endpoint for Open WebUI integration.

    This endpoint follows the ERI protocol for external retrieval systems,
    providing a standardized interface for RAG applications.

    Args:
        request: ERI query request with query text and optional filters

    Returns:
        ERI-formatted response with results
    """
    try:
        # Generate query embedding
        query_embedding = embedding_service.embed(request.query)

        # Search database
        results = vector_db.similarity_search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            filters=request.filters
        )

        # Transform to ERI format
        eri_results = [
            ERIResult(
                id=str(chunk.chunk_id),
                text=str(chunk.text),
                score=score,
                metadata={
                    "section": str(chunk.section),
                    "document_id": str(chunk.document_id),
                    "source_url": str(chunk.source_url),
                    **({k: str(v) for k, v in chunk.chunk_metadata.items()} if chunk.chunk_metadata else {})
                }
            )
            for chunk, score in results
        ]

        return ERIQueryResponse(results=eri_results)

    except Exception as e:
        logger.error(f"ERI query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/substances/{substance_uuid}", response_model=DeleteResponse, tags=["Management"])
async def delete_substance(substance_uuid: UUID, username: str = Depends(verify_credentials)):
    """
    Delete all chunks for a specific substance.

    Requires authentication.
    """
    try:
        count = vector_db.delete_chunks_by_substance(substance_uuid)
        logger.info(f"Deleted substance {substance_uuid} by user {username}")
        return DeleteResponse(
            substance_uuid=str(substance_uuid),
            chunks_deleted=count
        )
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=AvailableModelsResponse, tags=["Configuration"])
async def get_available_models():
    """Get information about the current embedding model."""
    # Return current model info as a single-item dict for compatibility
    current = embedding_service.get_model_info()
    return AvailableModelsResponse(
        models={"current": current},
        current_model=current["model"]
    )


@app.get("/models/current", response_model=ModelInfo, tags=["Configuration"])
async def get_current_model():
    """Get information about the current embedding model."""
    return embedding_service.get_model_info()


@app.get("/statistics", tags=["Management"])
async def get_statistics():
    """Get database statistics."""
    return vector_db.get_statistics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

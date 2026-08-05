from fastapi import FastAPI
from app.core.config import settings
from app.api import documents
from app.api import query

app = FastAPI(
    title="Agentic RAG API",
    description="A production-ready RAG backend exposing AI models as scalable RESTful web services.",
    version="1.0.0"
)
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
app.include_router(query.router, prefix="/api/v1", tags=["Queries"])

@app.get("/health", tags=["System"])
async def health_check():
    """
    Check the health of the API and verify environment configurations.
    """
    return {
        "status": "healthy",
        "model_in_use": settings.model_name,
        "embedding_model": settings.embedding_model,
        "vector_store": settings.vector_store_path
    }
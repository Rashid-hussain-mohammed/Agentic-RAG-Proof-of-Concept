import time
import logging
from fastapi import FastAPI, Request
from app.api import documents, query

# 1. Set up the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentic_rag")

app = FastAPI(title="Agentic RAG API")

# 2. Add the Observability Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Process the request
    response = await call_next(request)
    
    # Calculate time and log it
    process_time = time.time() - start_time
    logger.info(f"Path: {request.url.path} | Method: {request.method} | Status: {response.status_code} | Time: {process_time:.4f}s")
    
    # Add the processing time to the response headers so the frontend can see it
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

app.include_router(documents.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
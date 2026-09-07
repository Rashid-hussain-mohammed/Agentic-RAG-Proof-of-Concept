from fastapi.testclient import TestClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.main import app
from app.core.config import get_hardware_device, settings

client = TestClient(app)

def test_api_is_responsive():
    """Verify the FastAPI application boots and the auto-docs are accessible."""
    response = client.get("/docs")
    assert response.status_code == 200

def test_hardware_device_detection():
    """Ensure the dynamic tensor routing correctly identifies a valid hardware backend."""
    device = get_hardware_device()
    assert device in ["mps", "cuda", "cpu"]

def test_text_chunking_length():
    """Validate that the recursive text splitter strictly enforces the configured ceiling."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    sample_text = "A" * 2500
    chunks = splitter.split_text(sample_text)
    
    assert len(chunks) >= 3
    # split_text returns strings, so we check len(chunk) directly
    assert all(len(chunk) <= settings.chunk_size for chunk in chunks)

def test_text_chunking_overlap():
    """Verify that document chunks overlap correctly to prevent data fragmentation."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    sample_text = "A" * (settings.chunk_size - 50) + "B" * (settings.chunk_overlap + 50)
    chunks = splitter.split_text(sample_text)
    
    assert len(chunks) > 1
    assert "B" in chunks[0]
    assert "B" in chunks[1]

def test_query_endpoint_rejects_empty_payloads():
    """Ensure the Pydantic schemas reject empty strings."""
    response = client.post("/api/v1/query", json={"question": ""})
    assert response.status_code in [400, 422]

def test_query_endpoint_rejects_malformed_json():
    """Ensure the API catches improperly formatted requests before hitting the LangGraph node."""
    response = client.post("/api/v1/query", content="not valid json")
    assert response.status_code == 422
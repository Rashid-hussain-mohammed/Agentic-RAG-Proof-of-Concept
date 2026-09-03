from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.main import app

# This creates a mock version of your server for testing
client = TestClient(app)

def test_api_health():
    """
    Test that the FastAPI server rejects invalid payload structures
    before ever reaching the AI agent.
    """
    # Sending an empty JSON payload instead of {"question": "..."}
    response = client.post("/api/v1/query", json={})
    
    # FastAPI should automatically reject this with a 422 Unprocessable Entity
    assert response.status_code == 422

def test_chunking_math():
    """
    Test that the RecursiveCharacterTextSplitter is accurately applying
    the chunk size and overlap math.
    """
    # Create a dummy document of exactly 1200 characters
    dummy_text = "A" * 1200
    doc = [Document(page_content=dummy_text)]
    
    # Split with 500 size and 50 overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(doc)
    
    # Chunk 1: 0 to 500 (Length: 500)
    # Chunk 2: 450 to 950 (Length: 500)
    # Chunk 3: 900 to 1200 (Length: 300)
    
    assert len(chunks) == 3, "Should create exactly 3 chunks"
    assert len(chunks[0].page_content) == 500, "First chunk should be exactly 500 characters"
    assert len(chunks[2].page_content) == 300, "Final chunk should be exactly 300 characters"
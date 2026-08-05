from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os

# Import the new ingestion service
from app.rag.ingestor import process_and_index_document

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document to be indexed into the RAG vector store.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Trigger the document processing and indexing
        total_chunks = process_and_index_document(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    return {
        "filename": file.filename, 
        "status": "Successfully uploaded and indexed",
        "chunks_created": total_chunks,
        "vector_store": "updated"
    }
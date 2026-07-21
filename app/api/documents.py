from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document to be indexed into the RAG vector store.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    
    # Save the uploaded file to the local disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # TODO: Bring in your ingest.py logic here to split and embed the document
    
    return {
        "filename": file.filename, 
        "status": "Successfully uploaded and saved to disk",
        "next_steps": "Pending vectorization"
    }
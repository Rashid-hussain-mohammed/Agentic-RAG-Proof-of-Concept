from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.rag.retriever import get_relevant_context

router = APIRouter()

# Define what the incoming JSON request should look like
class QueryRequest(BaseModel):
    question: str

@router.post("/query")
async def ask_question(request: QueryRequest):
    """
    Submit a question to retrieve relevant context from the uploaded documents.
    """
    try:
        # Fetch the context from our Chroma database
        context = get_relevant_context(request.question)
        
        # TODO: Pass this context to Ollama/Claude via LangGraph
        
        return {
            "question": request.question,
            "retrieved_context": context,
            "status": "Context retrieved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")
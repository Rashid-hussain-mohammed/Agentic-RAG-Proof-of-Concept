import traceback
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.agent import app_agent

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

@router.post("/query")
async def ask_question(request: QueryRequest):
    try:
        initial_state = {
            "question": request.question,
            "context": "",
            "answer": "",
            "generation_ready": "",
            "loop_count": 0
        }
        
        result = app_agent.invoke(initial_state)
        
        return {
            "original_question": request.question,
            "final_question_used": result.get("question"),
            "answer": result.get("answer"),
            "retrieval_loops": result.get("loop_count")
        }
    except Exception as e:
        traceback.print_exc()  # <--- Prints the exact line and error to your terminal
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
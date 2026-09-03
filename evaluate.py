import json
from app.services.agent import app_agent
from app.core.config import settings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# 1. THE BLUEPRINTS (Setup)
# We set up Llama3 to act as a strict teacher grading a test
grader_llm = ChatOllama(model=settings.model_name, temperature=0)

grading_prompt = PromptTemplate(
    template=(
        "You are an expert grader. Compare the Agent's Answer to the Expected Answer. "
        "Does the Agent's Answer contain the same core information and meaning as the Expected Answer? "
        "Respond with exactly one word: 'PASS' or 'FAIL'. Do not include any other text.\n\n"
        "Expected Answer: {expected}\n"
        "Agent's Answer: {agent}\n"
        "Grade:"
    ),
    input_variables=["expected", "agent"]
)

# We connect the prompt blueprint to the LLM walkie-talkie
grader_chain = grading_prompt | grader_llm

def run_evaluation():
    # 2. LOAD THE TEST DATA
    print("Loading test dataset...")
    with open("data/eval_dataset.json", "r") as f:
        dataset = json.load(f)
    
    total_score = 0
    
    print("\n--- STARTING EVALUATION ---")
    # 3. LOOP THROUGH EACH QUESTION
    for i, item in enumerate(dataset):
        print(f"\nEvaluating Question {i+1}: {item['question']}")
        
        # Action A: Ask your RAG Agent the question
        initial_state = {
            "question": item['question'],
            "context": "", 
            "answer": "", 
            "generation_ready": "", 
            "loop_count": 0
        }
        result = app_agent.invoke(initial_state)
        agent_answer = result.get("answer")
        
        # Action B: Give both answers to the Llama3 Grader
        grade_result = grader_chain.invoke({
            "expected": item['expected_answer'],
            "agent": agent_answer
        }).content.strip().upper()
        
        # Action C: Tally the score
        if "PASS" in grade_result:
            total_score += 1
            print("✅ Result: PASS")
        else:
            print("❌ Result: FAIL")
            print(f"   Expected: {item['expected_answer']}")
            print(f"   Agent Said: {agent_answer}")
            
    print(f"\n--- EVALUATION COMPLETE ---")
    print(f"Final Score: {total_score} / {len(dataset)}")

if __name__ == "__main__":
    run_evaluation()
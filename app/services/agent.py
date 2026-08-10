from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from app.rag.retriever import get_relevant_context
from app.core.config import settings

class AgentState(TypedDict):
    question: str
    context: str
    answer: str
    generation_ready: str
    loop_count: int

llm = ChatOllama(model=settings.model_name, temperature=0)

def retrieve_node(state: AgentState):
    question = state["question"]
    context = get_relevant_context(question)
    return {"context": context, "loop_count": state.get("loop_count", 0) + 1}

def evaluate_context_node(state: AgentState):
    question = state["question"]
    context = state["context"]
    
    prompt = PromptTemplate(
        template="Does the following context contain sufficient information to answer the question? Answer exactly 'yes' or 'no'.\nContext: {context}\nQuestion: {question}",
        input_variables=["context", "question"]
    )
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": question}).content.strip().lower()
    
    if "yes" in result:
        return {"generation_ready": "yes"}
    else:
        return {"generation_ready": "no"}

def rewrite_query_node(state: AgentState):
    question = state["question"]
    prompt = PromptTemplate(
        template="Rewrite the following question to be more specific for a vector database search.\nQuestion: {question}\nRewritten Question:",
        input_variables=["question"]
    )
    chain = prompt | llm
    rewritten = chain.invoke({"question": question}).content.strip()
    return {"question": rewritten}

def generate_node(state: AgentState):
    question = state["question"]
    context = state["context"]
    
    prompt = PromptTemplate(
        template="Answer the question based only on the context provided. If the context does not contain the answer, state that you do not know.\nContext: {context}\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": question}).content
    return {"answer": answer}

def route_evaluation(state: AgentState):
    if state["generation_ready"] == "yes":
        return "generate"
    elif state["loop_count"] >= 3:
        return "generate"
    else:
        return "rewrite"

workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("evaluate", evaluate_context_node)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "evaluate")
workflow.add_conditional_edges(
    "evaluate",
    route_evaluation,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

app_agent = workflow.compile()
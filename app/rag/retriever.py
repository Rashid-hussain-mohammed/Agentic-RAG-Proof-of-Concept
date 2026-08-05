from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

def get_relevant_context(query: str):
    """
    Searches the Chroma vector store for chunks relevant to the user's query.
    """
    # Re-initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    
    # Connect to the existing database
    vector_store = Chroma(
        persist_directory=settings.vector_store_path, 
        embedding_function=embeddings
    )
    
    # Retrieve the top 3 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    
    # Format the output
    context = "\n\n".join([doc.page_content for doc in docs])
    return context
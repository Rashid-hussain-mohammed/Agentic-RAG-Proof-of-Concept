import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings

def process_and_index_document(file_path: str):
    """
    Loads a PDF, chunks the text, and stores the embeddings in ChromaDB.
    """
    # 1. Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Split the document into chunks using your configurable settings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Initialize the embedding model
    # Note: Ensure sentence-transformers is installed if using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    # 4. Store the chunks in the Chroma vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.vector_store_path
    )
    vector_store.persist()
    
    return len(chunks)
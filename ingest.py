import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    # 1. Load the Document (Phase 2)
    pdf_path = "./data/sample.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find {pdf_path}. Please add a PDF to the data folder.")
        return

    print(f"Loading document: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 2. Chunk the Document (Phase 2)
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Matching the exact size from your README
        chunk_overlap=100, # 10% overlap to keep context between chunks
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Successfully split document into {len(chunks)} chunks.")

    # 3. Create Embeddings & Build Vector DB (Phase 3)
    print("Initializing Local Embedding Model (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Building and saving ChromaDB Vector Store...")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./vector_store"
    )

    print("✅ Ingestion Complete! Your vector database is saved locally in ./vector_store")

if __name__ == "__main__":
    main()
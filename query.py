from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    # Helper function to combine our retrieved chunks into one big string
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    print("Loading your local vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        persist_directory="./vector_store", 
        embedding_function=embeddings
    )

    # 1. Create the retriever interface
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("Connecting to local Ollama instance (Llama 3)...")
    llm = Ollama(model="llama3")

    # 2. Define the Prompt Template
    template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know.

Context:
{context}

Question: {input}
"""
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Build the Core LCEL RAG Chain 
    # This pipes the formatted context and user input into the prompt, then to Llama 3, then parses to a string
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Wrap it in a Parallel Runnable so we can return the Source Documents alongside the answer
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    print("\n🚀 Pure LCEL RAG system ready! Ask a question about your PDF (type 'quit' to exit):")
    while True:
        user_input = input("\nUser Question: ")
        if user_input.lower() == 'quit':
            break
            
        if not user_input.strip():
            continue

        print("Searching database and generating answer...")
        
        # Trigger the pipeline
        response = rag_chain_with_source.invoke(user_input)
        
        print("\n🤖 AI Answer:")
        print(response["answer"])
        
        print("\n📚 Sources Used:")
        for doc in response["context"]:
            print(f"- Page {doc.metadata.get('page', 'Unknown')}: {doc.page_content[:100].strip()}...")

if __name__ == "__main__":
    main()
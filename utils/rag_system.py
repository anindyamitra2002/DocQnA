from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from utils.document_processor import load_and_split_pdfs
from utils.vector_store import create_vector_store, get_retriever
from langchain_core.runnables import RunnablePassthrough
from typing import Any

def initialize_llm(model_name="llama3", temperature=0.8, num_predict=256):
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        num_predict=num_predict
    )

def process_documents(uploaded_files, k_value, chunk_length, chunk_overlap, embedding_model_id, embed_device):
    documents = load_and_split_pdfs(uploaded_files, chunk_length, chunk_overlap)
    vector_store = create_vector_store(documents, embedding_model_id, embed_device)
    retriever = get_retriever(vector_store, k=k_value)
    return retriever



def get_answer(llm: ChatOllama, retriever: Any, query: str) -> str:
    try:
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Use the following context to answer the question. 
        If you cannot find the answer in the context, say "I cannot find the answer in the provided context."

        Context: {context}
        Question: {question}

        Answer: """)

        document_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt,
            document_variable_name="context",
        )

        chain = RunnablePassthrough.assign(
            context=lambda x: retriever.invoke(x["question"])
        ) | document_chain

        response = chain.invoke({"question": query})
        
        return response if isinstance(response, str) else str(response)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error details: {error_details}")
        return f"An error occurred while generating the answer: {str(e)}"
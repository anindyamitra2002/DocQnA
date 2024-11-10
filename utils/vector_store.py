from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

def create_vector_store(documents, model_name="BAAI/bge-small-en"):
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = Chroma(
        collection_name="rag_collection",
        embedding_function=embedding_model,
        persist_directory="./chroma_langchain_db"
    )
    uuids = [str(uuid4()) for _ in range(len(documents))]
    print("creating embedding vector")
    vector_store.add_documents(documents=documents, ids=uuids)
    return vector_store

def get_retriever(vector_store, k=3):
    if not vector_store:
        raise ValueError("Vector store not initialized. Please create vector store first.")
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": k * 2}
    )


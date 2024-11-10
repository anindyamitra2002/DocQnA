import streamlit as st
import shutil
import os
from utils.rag_system import initialize_llm, process_documents, get_answer
from utils.config import Config

# Dictionary mapping display names to Ollama model IDs
LLM_MODELS = {
    "Llama 3.2 - 3B": "llama3.2",
    "Llama 3.2 - 1B": "llama3.2:1b",
    "Llama 3.1 - 8B": "llama3.1",
    "Llama 3.1 - 70B": "llama3.1:70b",
    "Llama 3.1 - 405B": "llama3.1:405b",
    "Phi 3 Mini - 3.8B": "phi3",
    "Phi 3 Medium - 14B": "phi3:medium",
    "Gemma 2 - 2B": "gemma2:2b",
    "Gemma 2 - 9B": "gemma2",
    "Gemma 2 - 27B": "gemma2:27b",
    "Mistral - 7B": "mistral",
    "Moondream 2 - 1.4B": "moondream",
    "Neural Chat - 7B": "neural-chat",
    "Starling - 7B": "starling-lm",
    "Code Llama - 7B": "codellama",
    "Llama 2 Uncensored - 7B": "llama2-uncensored",
    "LLaVA - 7B": "llava",
    "Solar - 10.7B": "solar"
}

# Dictionary mapping display names to HuggingFace embedding model IDs
EMBEDDING_MODELS = {
    # Most Popular Models
    "BGE Large English v1.5 (1024d)": "BAAI/bge-large-en-v1.5",
    "Multilingual E5 Large (1024d)": "intfloat/multilingual-e5-large",
    "Jina Embeddings v2 Base EN (768d)": "jinaai/jina-embeddings-v2-base-en",
    "MXBAi Embed Large v1 (1024d)": "mixedbread-ai/mxbai-embed-large-v1",
    "E5 Mistral 7B Instruct (1024d)": "intfloat/e5-mistral-7b-instruct",
    "Jina Embeddings v3 (768d)": "jinaai/jina-embeddings-v3",
    
    # Medium-Sized Models
    "BGE Base English v1.5 (768d)": "BAAI/bge-base-en-v1.5",
    "BCE Embedding Base v1 (768d)": "maidalun1020/bce-embedding-base_v1",
    "BGE Reranker Large (768d)": "BAAI/bge-reranker-large",
    "BGE Small English v1.5 (384d)": "BAAI/bge-small-en-v1.5",
    "SFR Embedding Mistral (768d)": "Salesforce/SFR-Embedding-Mistral",
    
    # Specialized Models
    "MiniCPM Embedding (768d)": "openbmb/MiniCPM-Embedding",
    "Multilingual E5 Large Instruct (1024d)": "intfloat/multilingual-e5-large-instruct",
    "Jina CLIP v1 (512d)": "jinaai/jina-clip-v1",
    "UAE Large V1 (1024d)": "WhereIsAI/UAE-Large-V1",
    "NV Embed v2 (768d)": "nvidia/NV-Embed-v2",
    
    # Multilingual Models
    "Jina Embeddings v2 Base ZH (768d)": "jinaai/jina-embeddings-v2-base-zh",
    "Jina Embeddings v2 Base DE (768d)": "jinaai/jina-embeddings-v2-base-de",
    
    # Lightweight Models
    "Jina Embeddings v2 Small EN (384d)": "jinaai/jina-embeddings-v2-small-en",
    "BGE Small EN (384d)": "BAAI/bge-small-en",
    "GTE Small (384d)": "Supabase/gte-small",
    
    # Legacy Models but Still Used
    "all-MiniLM-L6-v2 (384d)": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2 (768d)": "sentence-transformers/all-mpnet-base-v2",
    "e5-large-v2 (1024d)": "intfloat/e5-large-v2",
    "instructor-large (768d)": "hkunlp/instructor-large"
}

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "config_locked" not in st.session_state:
        st.session_state.config_locked = False
    if "current_config" not in st.session_state:
        st.session_state.current_config = {
            "model_name": list(LLM_MODELS.keys())[1],
            "embedding_model": list(EMBEDDING_MODELS.keys())[9],
            "temperature": Config.DEFAULT_TEMPERATURE,
            "chunk_length": Config.DEFAULT_CHUNK_SIZE,
            "chunk_overlap": Config.DEFAULT_CHUNK_OVERLAP,
            "k_value": Config.DEFAULT_K_VALUE
        }

def has_new_documents(uploaded_files):
    """Check if there are any new documents that haven't been processed"""
    if not uploaded_files:
        return False
    current_files = {f.name for f in uploaded_files}
    processed_files = st.session_state.processed_files
    return bool(current_files - processed_files)

def process_docs(uploaded_files, config):
    """Process documents and update session state"""
    if not uploaded_files:
        st.warning("Please upload some documents to begin.")
        st.session_state.retriever = None
        st.session_state.llm = None
        return False

    current_files = {f.name for f in uploaded_files}
    
    with st.spinner("Processing documents..."):
        try:
            ollama_model_id = LLM_MODELS[config["model_name"]]
            llm = initialize_llm(ollama_model_id, config["temperature"])
            st.session_state.llm = llm

            embedding_model_id = EMBEDDING_MODELS[config["embedding_model"]]
            retriever = process_documents(
                uploaded_files,
                config["k_value"],
                config["chunk_length"],
                config["chunk_overlap"],
                embedding_model_id,
                Config.EMBEDDING_DEVICE
            )
            st.session_state.retriever = retriever
            st.session_state.processed_files = current_files
            st.session_state.config_locked = True  # Lock configuration after successful processing
            
            st.success(f"Processed {len(current_files)} documents successfully!")
            return True
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return False

def reset_session():
    """Reset the session and clean up vector database"""
    # Clear the vector database directory
    vector_db_path = "vector_db"  # Update this path according to your setup
    if os.path.exists(vector_db_path):
        shutil.rmtree(vector_db_path)
    
    # Reset all session state variables
    st.session_state.clear()
    st.rerun()

def main():
    st.title("RAG System with Dynamic Document Processing")
    
    initialize_session_state()

    with st.sidebar:
        st.header("Model Configuration")
        
        # Configuration inputs - disabled when locked
        config = st.session_state.current_config
        
        config["model_name"] = st.selectbox(
            "Select LLM Model",
            options=list(LLM_MODELS.keys()),
            index=list(LLM_MODELS.keys()).index(config["model_name"]),
            disabled=st.session_state.config_locked
        )
        
        config["embedding_model"] = st.selectbox(
            "Select Embedding Model",
            options=list(EMBEDDING_MODELS.keys()),
            index=list(EMBEDDING_MODELS.keys()).index(config["embedding_model"]),
            disabled=st.session_state.config_locked
        )

        config["temperature"] = st.slider(
            "Temperature",
            0.0,
            1.0,
            config["temperature"],
            disabled=st.session_state.config_locked
        )

        st.header("Chunking Configuration")
        config["chunk_length"] = st.slider(
            "Chunk Length",
            100,
            2000,
            config["chunk_length"],
            step=50,
            disabled=st.session_state.config_locked
        )
        
        config["chunk_overlap"] = st.slider(
            "Chunk Overlap",
            0,
            500,
            config["chunk_overlap"],
            step=10,
            disabled=st.session_state.config_locked
        )
        
        config["k_value"] = st.slider(
            "Number of relevant chunks (k)",
            1,
            20,
            config["k_value"],
            disabled=st.session_state.config_locked
        )

        st.header("Document Management")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            key="document_upload"
        )

        if uploaded_files:
            st.write("Uploaded documents:")
            for file in uploaded_files:
                st.write(f"- {file.name}")
            
            if has_new_documents(uploaded_files):
                st.warning("New documents detected! Please process them before chatting.")
                if st.button("Process Documents"):
                    success = process_docs(uploaded_files, config)
                    if success:
                        st.rerun()

        if st.button("Reset Session"):
            reset_session()

        if st.session_state.config_locked:
            st.info("Configuration is locked. Reset session to modify settings.")

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Enter your query here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not uploaded_files:
            st.error("Please upload some documents first.")
        elif has_new_documents(uploaded_files):
            st.error("Please process the new documents before chatting.")
        elif not hasattr(st.session_state, 'retriever') or st.session_state.retriever is None:
            st.error("Please process the documents first.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    try:
                        answer = get_answer(st.session_state.llm, st.session_state.retriever, prompt)
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_message = f"Error generating answer: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()
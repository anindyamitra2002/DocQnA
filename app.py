import streamlit as st
from utils.rag_system import initialize_llm, process_documents, get_answer
from utils.config import Config
import json

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "document_processing_needed" not in st.session_state:
        st.session_state.document_processing_needed = False

def has_new_documents(uploaded_files):
    """Check if there are any new documents that haven't been processed"""
    if not uploaded_files:
        return False
    current_files = {f.name for f in uploaded_files}
    processed_files = st.session_state.processed_files
    return bool(current_files - processed_files)

def process_documents_if_needed(uploaded_files, model_name, temperature, k_value):
    """Process documents if there are new ones and update session state"""
    if not uploaded_files:
        st.warning("Please upload some documents to begin.")
        st.session_state.retriever = None
        st.session_state.llm = None
        return False

    current_files = {f.name for f in uploaded_files}
    
    # Check if we need to process new documents
    if has_new_documents(uploaded_files):
        with st.spinner("Processing new documents..."):
            try:
                # Initialize LLM
                llm = initialize_llm(model_name, temperature)
                st.session_state.llm = llm

                # Process all documents (including previously processed ones)
                retriever = process_documents(uploaded_files, k_value)
                st.session_state.retriever = retriever

                # Update processed files set
                st.session_state.processed_files = current_files
                
                st.success(f"Processed {len(current_files)} documents successfully!")
                return True
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                return False
    return True

def main():
    st.title("RAG System with Dynamic Document Processing")
    
    # Initialize session state
    initialize_session_state()

    # Sidebar for configuration and document upload
    with st.sidebar:
        st.header("Configuration")
        model_name = st.selectbox(
            "Select LLM Model",
            Config.AVAILABLE_MODELS,
            index=Config.AVAILABLE_MODELS.index(Config.DEFAULT_MODEL_NAME)
        )
        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            Config.DEFAULT_TEMPERATURE
        )
        k_value = st.slider(
            "Number of relevant chunks (k)",
            1,
            10,
            Config.DEFAULT_K_VALUE
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
                st.warning("New documents detected! They will be processed when you send your next message.")
                st.session_state.document_processing_needed = True

        # Add a button to clear chat history
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Enter your query here"):
        # Process any new documents before handling the query
        if st.session_state.document_processing_needed:
            success = process_documents_if_needed(uploaded_files, model_name, temperature, k_value)
            if success:
                st.session_state.document_processing_needed = False
            else:
                st.error("Failed to process new documents. Please try uploading them again.")
                return

        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        if not uploaded_files:
            st.error("Please upload some documents first.")
        elif not hasattr(st.session_state, 'retriever') or st.session_state.retriever is None:
            st.error("Please wait for document processing to complete.")
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
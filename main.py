# app/streamlit_app.py
import streamlit as st
from utils.rag_system import initialize_llm, process_documents, get_answer
from utils.config import Config

def run_streamlit_app():
    st.title("RAG System with Local Ollama")
    # Sidebar for configuration and uploaded files
    uploaded_files = []
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

        st.header("Uploaded Documents")
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

    # Initialize retriever only if not already in session state
    if uploaded_files and "retriever" not in st.session_state:
        with st.spinner("Processing documents..."):
            llm = initialize_llm(model_name, temperature)
            st.session_state.llm = llm
            retriever = process_documents(uploaded_files, k_value)
            st.session_state.retriever = retriever  # Store in session state
        st.success("Documents processed successfully!")


    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input and generate response
    if prompt := st.chat_input("Enter your query here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check if retriever exists in session state
        if "retriever" in st.session_state:
            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    answer = get_answer(st.session_state.retriever, st.session_state.llm, prompt)
                    st.write(answer)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.error("Please upload and process documents first.")

if __name__ == "__main__":
    run_streamlit_app()

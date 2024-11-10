import streamlit as st
from utils.rag_system import initialize_llm, process_documents, get_answer
from utils.config import Config

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

# Initialize system components when documents are uploaded
if uploaded_files and "retriever" not in st.session_state:
    with st.spinner("Processing documents..."):
        try:
            llm = initialize_llm(model_name, temperature)
            st.session_state.llm = llm
            retriever = process_documents(uploaded_files, k_value)
            st.session_state.retriever = retriever
            st.success("Documents processed successfully!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Enter your query here"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "retriever" not in st.session_state or "llm" not in st.session_state:
        st.error("Please upload and process documents first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                # try:
                # Note the corrected argument order here
                print("prompt: ", prompt)
                answer = get_answer(st.session_state.llm, st.session_state.retriever, prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                # except Exception as e:
                #     error_message = f"Error generating answer: {str(e)}"
                #     st.error(error_message)
                #     st.session_state.messages.append({"role": "assistant", "content": error_message})
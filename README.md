# DocQnA with Ollama

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system using Streamlit as the user interface, LangChain for document processing and retrieval, and Ollama for conversational AI responses. The system allows users to upload PDF documents, process them to create a vector store for retrieval, and then answer queries based on the document content using a Large Language Model (LLM). 

This RAG system is especially useful for contexts where domain-specific knowledge stored in documents needs to be accessed dynamically through natural language queries.

## Content

- [System Workflow](#system-workflow)
- [Script Descriptions](#script-descriptions)
- [Tech Stack](#tech-stack)
- [Ollama Installation](#ollama-installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Conclusion](#conclusion)

## System Workflow

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/RAG_workflow.png" alt="RAG Workflow" width="600">
</div>

1. **Upload Documents**: Users upload PDF documents via Streamlit, which checks for any new documents and flags them for processing.
2. **Document Processing**:
   - Documents are loaded, split into chunks using LangChain's `RecursiveCharacterTextSplitter`.
   - Chunks are embedded with a Hugging Face model (`BAAI/bge-small-en`) and stored in a vector store using Chroma.
3. **Retriever Initialization**: A retriever instance is created from the vector store, allowing similarity-based search on document chunks.
4. **Question-Answering**:
   - The user inputs a query, which is processed through a retrieval chain.
   - Relevant document chunks are retrieved and fed into an Ollama-based LLM for contextual answers.

## Script Descriptions

- **`app.py`**: The main application file that initializes the RAG system and manages user interactions, serving as the entry point for deploying the retrieval-augmented generation interface.
- **`document_processor.py`**: Handles loading and splitting of PDF documents into manageable chunks for efficient retrieval.
- **`vector_store.py`**: Creates and maintains the vector store, supporting document embedding and retrieval operations.
- **`rag_system.py`**: Initializes the language model and orchestrates the generation of responses based on retrieved document data.
- **`config.py`**: Contains configuration settings, such as model parameters and retrieval options, essential for customizing the RAG system.
- **`test.py`**: A dedicated test script that debug the RAG pipeline, including document processing, vector storage, retrieval setup, and answer generation. It also keeps track in the log file.


## Tech Stack

- **Streamlit**: Web interface for document upload, configuration, and interaction.
- **LangChain**: Document loading, text splitting, vectorization, and retrieval workflows.
- **Ollama**: Language model API for conversational responses.
- **Chroma**: Vector store to store and retrieve document embeddings.
- **Hugging Face**: Embedding model for document vectorization.

## Ollama Installation

Ollama is the LLM used in this project. To install it:

1. Visit the [Ollama Official Repo](https://github.com/ollama/ollama) and download the installer for your OS (Linux/WSL Recommended).
2. Follow the installation instructions provided for your operating system.

After installation, Run Ollama model from your environment by running:

```bash
ollama run <model_name>
```
More ollama models are provided in the official Ollama Repository.

## Usage

To run the RAG system:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/anindyamitra2002/DocQnA.git
    cd DocQnA
    ```

2. **Install Dependencies**:
    Ensure you have Python 3.10+ installed. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. **Start the Streamlit Application**:
    ```bash
    streamlit run app.py
    ```

4. **Upload PDF Documents**:
    - Use the sidebar to upload documents. The system will process and store them in a vector store.
    - Configure LLM model options and document retrieval parameters in the sidebar.

5. **Query the System**:
    - Enter a query in the chat input box, and the system will retrieve relevant document chunks and generate an answer based on the uploaded documents.
  
## Future Work

- **Improving Retrieval Accuracy:**  Incorporate hybrid search and dynamic indexing to improve retrieval precision and ensure real-time data relevance.

- **Contextual Understanding:**  Use contextual embeddings and chunk sequencing to enhance the relevance and coherence of retrieved information.

- **Handling Complex Queries:**  Enable multi-document reasoning and filtering techniques to manage complex queries and reduce irrelevant data distractions.

- **Evaluation Frameworks:**  Develop robust evaluation metrics and establish benchmarks to measure and compare system performance effectively.

- **User Interaction and Feedback Loops:**  Integrate interactive learning and feedback mechanisms to refine responses based on user interactions over time.

- **Agentic RAG:**  Add agentic capabilities to autonomously determine retrieval based on user intent, improving response relevance.

- **Graph RAG:**  Utilize graph-based structures to map data relationships, enhancing retrieval by connecting related information.

## Conclusion

This RAG system leverages document embeddings, similarity search, and LLM-driven responses to build a robust question-answering tool for document-based knowledge. Future expansions can enhance the capabilities for a variety of applications, from customer support to research.

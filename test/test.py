import logging
import os
import sys
from pathlib import Path
import tempfile

# Add parent directory to path to import utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.document_processor import load_and_split_pdfs
from utils.vector_store import create_vector_store, get_retriever
from utils.rag_system import initialize_llm, get_answer
from utils.config import Config

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_rag_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TestPDFFile:
    """Mock PDF file object to simulate Streamlit's uploaded file"""
    def __init__(self, file_path):
        self.name = os.path.basename(file_path)
        self._file_path = file_path

    def getbuffer(self):
        with open(self._file_path, 'rb') as f:
            return f.read()

def test_rag_system():
    try:
        logger.info("Starting RAG system test")

        # Test configuration
        test_pdf_path = "DocQnA/sample/goog-10-k-2023 (1).pdf"  # Put your PDF in the same directory as test.py
        test_query = "What is the total revenue for Google Search?"  # Replace with your test query

        logger.info(f"Using test PDF: {test_pdf_path}")
        logger.info(f"Using test query: {test_query}")

        # Verify PDF file exists
        if not os.path.exists(test_pdf_path):
            logger.error(f"Test PDF file not found at {test_pdf_path}")
            return False

        # Test document processing
        logger.info("Testing document processing...")
        pdf_file = TestPDFFile(test_pdf_path)
        documents = load_and_split_pdfs([pdf_file])
        logger.info(f"Successfully split PDF into {len(documents)} chunks")
        
        for i, doc in enumerate(documents):
            logger.debug(f"Chunk {i+1} content preview: {doc.page_content[:200]}...")

        # Test vector store creation
        logger.info("Testing vector store creation...")
        vector_store = create_vector_store(documents)
        logger.info("Vector store created successfully")

        # Test retriever
        logger.info("Testing retriever...")
        retriever = get_retriever(vector_store, k=Config.DEFAULT_K_VALUE)
        logger.info(f"Retriever created successfully with k={Config.DEFAULT_K_VALUE}")

        # Test LLM initialization
        logger.info("Testing LLM initialization...")
        llm = initialize_llm(
            model_name=Config.DEFAULT_MODEL_NAME,
            temperature=Config.DEFAULT_TEMPERATURE
        )
        logger.info(f"LLM initialized successfully with model={Config.DEFAULT_MODEL_NAME}, temp={Config.DEFAULT_TEMPERATURE}")

        # Test complete RAG pipeline
        logger.info("Testing complete RAG pipeline with query...")
        logger.debug(f"Test query: {test_query}")
        
        # Get answer
        logger.info("Generating answer...")
        answer = get_answer(llm, retriever, test_query)
        logger.info("Successfully generated answer")
        logger.debug(f"Generated answer: {answer}")

        # Clean up
        if os.path.exists(Config.VECTOR_STORE_PATH):
            logger.info("Cleaning up vector store...")
            import shutil
            shutil.rmtree(Config.VECTOR_STORE_PATH)

        logger.info("RAG system test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    test_result = test_rag_system()
    sys.exit(0 if test_result else 1)
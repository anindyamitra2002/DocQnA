# utils/config.py
class Config:
    # Model configurations
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_MODEL_NAME = "llama3.2:1b"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_K_VALUE = 5
    
    # Vector store configurations
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"
    VECTOR_STORE_PATH = "./chroma_langchain_db"
    
    # Available LLM models
    AVAILABLE_MODELS = ["llama3.2:1b", "mistral", "codellama"]
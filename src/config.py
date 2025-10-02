import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys - loaded from environment variables for security
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    """
    Application-wide settings managed by Pydantic.
    Reads from environment variables or uses default values.
    """
    # --- File Paths ---
    # Use absolute paths to ensure robustness
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH: str = os.path.join(BASE_DIR, 'data/kg.db')
    FAISS_INDEX_PATH: str = os.path.join(BASE_DIR, 'data/processed/dense_index.faiss')
    FAISS_ID_MAP_PATH: str = os.path.join(BASE_DIR, 'data/processed/faiss_id_map.json')
    VERBALIZED_KG_PATH: str = os.path.join(BASE_DIR, 'data/processed/verbalized_kg.jsonl')

    # --- Model Configuration ---
    EMBEDDING_MODEL: str = 'BAAI/bge-large-en-v1.5'
    QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages: "
    
    # --- LLM Configuration ---
    LLM_PROVIDER: str = "groq"  # Options: "openai", "groq"
    LLM_MODEL: str = "llama-3.1-8b-instant"  # Groq model name
    LLM_TEMPERATURE: float = 0.2
    LLM_MAX_TOKENS: int = 2048

    # --- API Configuration ---
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "HelixRAGxMem"

    class Config:
        case_sensitive = True

# Instantiate settings to be imported by other modules
settings = Settings()

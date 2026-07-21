from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings and environment variables.
    Pydantic automatically reads these from the .env file.
    """
    # LLM and Embedding Settings
    model_name: str = "ollama"
    embedding_model: str = "all-MiniLM"
    
    # RAG Configuration
    chunk_size: int = 500
    chunk_overlap: int = 50
    vector_store_path: str = "./vector_store"

    # Pydantic configuration to load the .env file
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Instantiate the settings object to be imported across the application
settings = Settings()
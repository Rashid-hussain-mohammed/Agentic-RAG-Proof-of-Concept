import platform
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch

def get_hardware_device() -> str:
    """Dynamically detect the best available hardware accelerator."""
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon (Mac M4 Air)
    elif torch.cuda.is_available():
        return "cuda" # Windows/Linux with supported GPU
    else:
        return "cpu"  # Fallback

class Settings(BaseSettings):
    # LLM and Embedding Settings
    model_name: str = "llama3"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # RAG Configuration
    chunk_size: int = 500
    chunk_overlap: int = 50
    vector_store_path: str = "./vector_store"
    
    # Automatically set the device
    device: str = get_hardware_device()

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
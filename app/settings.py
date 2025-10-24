from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    TRANSFORMERS_OFFLINE: bool = False
    BATCH_SIZE: int = 32
    WORKERS: int = 1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
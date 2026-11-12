from functools import lru_cache
from typing import Literal
 
from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["openai", "azure"] = "openai"
    #llm_provider: Literal["openai", "azure"] = "azure"

    openai_api_key: SecretStr | None = None
    chat_model: str = "gpt-6-luna"
    embedding_model: str = "text-embedding-3-small"

    azure_openai_api_key: SecretStr | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str | None = None
    azure_chat_deployment: str | None = None
    azure_embedding_deployment: str | None = None

    pdf_dir: str = "data/pdfs"
    chroma_dir: str = "chroma_db"
    collection_name: str = "artigos"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 4

    @model_validator(mode="after")
    def check_credentials(self) -> "Settings":

        if self.llm_provider == "openai" and self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY é obrigatória quando LLM_PROVIDER=openai")

        if self.llm_provider == "azure":
            required = {
                "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
                "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
                "AZURE_OPENAI_API_VERSION": self.azure_openai_api_version,
                "AZURE_CHAT_DEPLOYMENT": self.azure_chat_deployment,
                "AZURE_EMBEDDING_DEPLOYMENT": self.azure_embedding_deployment,
            }

            missing = [name for name, value in required.items() if not value]

            if missing:
                raise ValueError(f"Variáveis faltando para Azure: {', '.join(missing)}")


        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP precisa ser menor que CHUNK_SIZE")

        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


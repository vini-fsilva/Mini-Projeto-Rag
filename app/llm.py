from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
 
from app.config import Settings

def get_embeddings(settings: Settings) -> Embeddings:

    if settings.llm_provider == "azure":
        return AzureOpenAIEmbeddings(
            azure_deployment=settings.azure_embedding_deployment,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

def get_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "azure":
        return AzureChatOpenAI(
            azure_deployment=settings.azure_chat_deployment,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )

    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
    )


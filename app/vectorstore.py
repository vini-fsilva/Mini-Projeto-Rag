from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.llm import get_embeddings

def get_vectorstore(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings or get_embeddings(settings),
        persist_directory=settings.chroma_dir,
    )
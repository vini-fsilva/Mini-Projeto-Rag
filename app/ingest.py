from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import Settings
from app.vectorstore import get_vectorstore


def load_pdfs(pdf_dir: str) -> list[Document]:

    documents: list[Document] = []

    for path in sorted(Path(pdf_dir).glob("*.pdf")):
        reader = PdfReader(path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name, "page": page_number},
                    )
                )

    return documents

def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )

    return splitter.split_documents(documents)

def build_index(settings: Settings, embeddings: Embeddings | None = None) -> int:
    documents = load_pdfs(settings.pdf_dir)
    if not documents:
        raise ValueError(f"Nenhum PDF com texto encontrado em '{settings.pdf_dir}'")

    chunks = split_documents(documents, settings.chunk_size, settings.chunk_overlap)

    vectorstore = get_vectorstore(settings, embeddings)

    vectorstore.reset_collection()

    ids = [
        f"{c.metadata['source']}:p{c.metadata['page']}:{c.metadata['start_index']}"
        for c in chunks
    ]

    vectorstore.add_documents(chunks, ids=ids)

    return len(chunks)
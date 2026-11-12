from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import Settings
from app.llm import get_chat_model
from app.vectorstore import get_vectorstore

SYSTEM_PROMPT = """Você é um assistente que responde perguntas sobre artigos científicos.
 
Regras:
- Use APENAS as informações do contexto abaixo.
- Se a resposta não estiver no contexto, diga que não encontrou essa informação nos artigos.
- Cite as fontes no formato [arquivo, p. X].
- Responda no mesmo idioma da pergunta.
 
Contexto:
{context}"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)

NOT_FOUND_MESSAGE = "Não encontrei informações sobre isso nos artigos indexados."

@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict] = field(default_factory=list)


def format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{d.metadata['source']}, p. {d.metadata['page']}]\n{d.page_content}"
        for d in docs
    )

def extract_sources(docs: list[Document]) -> list[dict]:
    seen: set[tuple[str, int]] = set()
    sources: list[dict] = []
    for d in docs:
        key = (d.metadata["source"], d.metadata["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"arquivo": key[0], "pagina": key[1]})
    return sources

class RAGService:
    def __init__(self, vectorstore: VectorStore, llm: BaseChatModel, top_k: int = 4):

        self.vectorstore = vectorstore
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        self.chain = PROMPT | llm | StrOutputParser() #LangChain Expression Language (LCEL)

    def count_chunks(self) -> int:
        return len(self.vectorstore.get(include=[])["ids"])

    def ask(self, question: str) -> RAGAnswer:
        docs = self.retriever.invoke(question)
        if not docs:
            return RAGAnswer(answer=NOT_FOUND_MESSAGE)

        answer = self.chain.invoke({"context": format_docs(docs), "question": question})
        return RAGAnswer(answer=answer, sources=extract_sources(docs))

def build_rag_service(settings: Settings) -> RAGService:

    return RAGService(
        vectorstore=get_vectorstore(settings),
        llm=get_chat_model(settings),
        top_k=settings.top_k,
    )
import logging
from contextlib import asynccontextmanager
 
from fastapi import Depends, FastAPI, HTTPException, Request
from openai import OpenAIError
 
from app.config import get_settings
from app.rag import RAGService, build_rag_service
from app.schemas import Fonte, HealthResponse, PerguntaRequest, PerguntaResponse
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.rag = build_rag_service(settings)
    logger.info(
        "RAG pronto: provedor=%s, chunks no índice=%d",
        settings.llm_provider,
        app.state.rag.count_chunks(),
    )

    yield

app = FastAPI(
    title="Assistente RAG - Artigos Científicos",
    version="1.0.0",
    lifespan=lifespan,
)

def get_rag(request: Request) -> RAGService:
    return request.app.state.rag

@app.get("/health", response_model=HealthResponse)
def health(rag: RAGService = Depends(get_rag)) -> HealthResponse:
    total = rag.count_chunks()
    return HealthResponse(status="ok", indice_carregado=total > 0, total_chunks=total)

@app.post("/perguntar", response_model=PerguntaResponse)
def perguntar(
    body: PerguntaRequest, rag: RAGService = Depends(get_rag)
) -> PerguntaResponse:

    if rag.count_chunks() == 0:
        raise HTTPException(
            status_code=503,
            detail="Índice vazio. Rode a ingestão: python -m scripts.ingest",
        )

    try:
        result = rag.ask(body.pergunta)
    except OpenAIError:
        logger.exception("Erro ao chamar o provedor de LLM")
        raise HTTPException(
            status_code=502, detail="Falha ao consultar o provedor de LLM"
        )

    return PerguntaResponse(
        resposta=result.answer,
        fontes=[Fonte(**s) for s in result.sources],
    )
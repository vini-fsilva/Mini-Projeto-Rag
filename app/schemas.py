from pydantic import BaseModel, ConfigDict, Field

class PerguntaRequest(BaseModel):

    model_config = ConfigDict(str_strip_whitespace=True)

    pergunta: str = Field(
        min_length=3,
        max_length=1000,
        description="Pergunta sobre os artigos indexados",
        examples=["Qual metodologia foi usada no artigo sobre NER?"],
    )


class Fonte(BaseModel):
    arquivo: str
    pagina: int

class PerguntaResponse(BaseModel):
    resposta: str
    fontes: list[Fonte]

class HealthResponse(BaseModel):
    status: str
    indice_carregado: bool
    total_chunks: int
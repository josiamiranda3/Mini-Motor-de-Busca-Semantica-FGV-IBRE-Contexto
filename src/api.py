
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request

from src.busca import MotorBuscaSemantica
from src.busca_bm25 import MotorBuscaBM25
from src.busca_hibrida import MotorBuscaHibrida


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Inicializa os motores uma única vez ao iniciar a API."""

    print("Inicializando motor de busca semântica...")
    motor_semantico = MotorBuscaSemantica()

    print("Inicializando motor BM25...")
    motor_bm25 = MotorBuscaBM25()

    app.state.motor_semantico = motor_semantico
    app.state.motor_bm25 = motor_bm25
    app.state.motor_hibrido = MotorBuscaHibrida(
        motor_semantico=motor_semantico,
        motor_bm25=motor_bm25,
    )

    app.state.total_noticias = len(motor_semantico.noticias)

    print(
        f"API pronta! {app.state.total_noticias} notícias indexadas."
    )

    yield


app = FastAPI(
    title="Mini Motor de Busca Semântica",
    description=(
        "API de pesquisa de notícias com busca semântica, "
        "BM25 e combinação híbrida usando Reciprocal Rank Fusion."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", tags=["Geral"])
def inicio():
    """Apresenta informações básicas sobre a API."""

    return {
        "projeto": "Mini Motor de Busca Semântica — FGV IBRE Contexto",
        "status": "online",
        "documentacao": "/docs",
        "metodos_disponiveis": ["hibrida", "semantica", "bm25"],
    }


@app.get("/health", tags=["Geral"])
def health(request: Request):
    """Verifica se a API foi inicializada corretamente."""

    return {
        "status": "ok",
        "noticias_indexadas": request.app.state.total_noticias,
    }


@app.get("/buscar", tags=["Busca"])
def buscar(
    request: Request,
    consulta: str = Query(
        ...,
        min_length=1,
        max_length=300,
        description="Texto que será utilizado para pesquisar notícias.",
        examples=["inflação e taxa Selic"],
    ),
    metodo: Literal["hibrida", "semantica", "bm25"] = Query(
        "hibrida",
        description="Método de pesquisa utilizado.",
    ),
    top_k: int = Query(
        5,
        ge=1,
        le=19,
        description="Quantidade máxima de resultados retornados.",
    ),
):
    """Pesquisa notícias utilizando o método selecionado."""

    consulta = consulta.strip()

    if not consulta:
        raise HTTPException(
            status_code=400,
            detail="A consulta não pode conter apenas espaços.",
        )

    motores = {
        "hibrida": request.app.state.motor_hibrido,
        "semantica": request.app.state.motor_semantico,
        "bm25": request.app.state.motor_bm25,
    }

    try:
        resultados = motores[metodo].buscar(
            consulta=consulta,
            top_k=top_k,
        )
    except ValueError as erro:
        raise HTTPException(
            status_code=400,
            detail=str(erro),
        ) from erro

    return {
        "consulta": consulta,
        "metodo": metodo,
        "total": len(resultados),
        "resultados": resultados,
    }

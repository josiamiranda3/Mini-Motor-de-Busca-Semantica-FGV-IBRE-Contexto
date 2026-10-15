
import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="module")
def client():
    """Reutiliza o mesmo cliente durante os testes deste módulo."""

    with TestClient(app) as test_client:
        yield test_client


def test_rota_inicial(client):
    """Verifica se a rota principal está funcionando."""

    resposta = client.get("/")

    assert resposta.status_code == 200

    dados = resposta.json()

    assert dados["status"] == "online"
    assert "documentacao" in dados
    assert "hibrida" in dados["metodos_disponiveis"]


def test_health(client):
    """Verifica o estado da API e as notícias carregadas."""

    resposta = client.get("/health")

    assert resposta.status_code == 200

    dados = resposta.json()

    assert dados["status"] == "ok"
    assert dados["noticias_indexadas"] == 19


def test_busca_hibrida(client):
    """Verifica a busca híbrida."""

    resposta = client.get(
        "/buscar",
        params={
            "consulta": "inflação e taxa Selic",
            "metodo": "hibrida",
            "top_k": 5,
        },
    )

    assert resposta.status_code == 200

    dados = resposta.json()

    assert dados["metodo"] == "hibrida"
    assert dados["total"] > 0
    assert len(dados["resultados"]) <= 5
    assert "titulo" in dados["resultados"][0]


def test_busca_semantica(client):
    """Verifica a busca semântica."""

    resposta = client.get(
        "/buscar",
        params={
            "consulta": "inflação e taxa Selic",
            "metodo": "semantica",
            "top_k": 5,
        },
    )

    assert resposta.status_code == 200

    dados = resposta.json()

    assert dados["metodo"] == "semantica"
    assert dados["total"] > 0


def test_busca_bm25(client):
    """Verifica a busca lexical BM25."""

    resposta = client.get(
        "/buscar",
        params={
            "consulta": "inflação e taxa Selic",
            "metodo": "bm25",
            "top_k": 5,
        },
    )

    assert resposta.status_code == 200

    dados = resposta.json()

    assert dados["metodo"] == "bm25"
    assert dados["total"] > 0


def test_busca_sem_consulta(client):
    """Verifica se a API rejeita uma consulta ausente."""

    resposta = client.get("/buscar")

    assert resposta.status_code == 422


def test_busca_com_metodo_invalido(client):
    """Verifica se a API rejeita um método desconhecido."""

    resposta = client.get(
        "/buscar",
        params={
            "consulta": "inflação",
            "metodo": "invalido",
        },
    )

    assert resposta.status_code == 422


def test_busca_com_top_k_invalido(client):
    """Verifica se a API rejeita uma quantidade inválida."""

    resposta = client.get(
        "/buscar",
        params={
            "consulta": "inflação",
            "top_k": 0,
        },
    )

    assert resposta.status_code == 422


def test_busca_com_consulta_vazia(client):
    """Verifica se a API rejeita uma consulta vazia."""

    resposta = client.get(
        "/buscar",
        params={"consulta": "   "},
    )

    assert resposta.status_code == 400

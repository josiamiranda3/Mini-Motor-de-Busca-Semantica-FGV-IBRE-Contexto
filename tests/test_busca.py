import json

import numpy as np
import pytest

from src.busca import MotorBuscaSemantica
from src.embeddings import MODELO, calcular_fingerprint


class ModeloFalso:
    """Simula o modelo de embeddings sem carregar uma IA real."""

    def encode(self, textos, **kwargs):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def criar_motor_temporario(tmp_path, monkeypatch):
    caminho_dados = tmp_path / "noticias.json"
    caminho_embeddings = tmp_path / "embeddings.npy"
    caminho_metadados = tmp_path / "embeddings_metadata.json"

    noticias = [
        {
            "id": 1,
            "titulo": "Inflação brasileira",
            "texto_limpo": "A inflação aumentou no período analisado.",
            "data": "2025-01-01",
            "fonte": "IBGE",
        },
        {
            "id": 2,
            "titulo": "Mercado de trabalho",
            "texto_limpo": "O emprego apresentou crescimento.",
            "data": "2025-01-02",
            "fonte": "IBGE",
        },
    ]

    caminho_dados.write_text(
        json.dumps(noticias, ensure_ascii=False),
        encoding="utf-8",
    )

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.save(caminho_embeddings, embeddings)

    textos = [
        f'{noticia["titulo"]}. {noticia["texto_limpo"]}'
        for noticia in noticias
    ]

    metadados = {
        "modelo": MODELO,
        "quantidade_noticias": len(noticias),
        "dimensao_embeddings": embeddings.shape[1],
        "fingerprint": calcular_fingerprint(textos),
    }

    caminho_metadados.write_text(
        json.dumps(metadados, ensure_ascii=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.busca.SentenceTransformer",
        lambda modelo: ModeloFalso(),
    )

    return MotorBuscaSemantica(caminho_dados, caminho_embeddings)


def test_busca_retorna_resultados_ordenados(tmp_path, monkeypatch):
    motor = criar_motor_temporario(tmp_path, monkeypatch)

    resultados = motor.buscar("inflação", top_k=2)

    assert len(resultados) == 2
    assert resultados[0]["id"] == 1
    assert resultados[0]["similaridade"] > resultados[1]["similaridade"]


def test_busca_limita_quantidade_de_resultados(tmp_path, monkeypatch):
    motor = criar_motor_temporario(tmp_path, monkeypatch)

    resultados = motor.buscar("economia", top_k=1)

    assert len(resultados) == 1


@pytest.mark.parametrize("consulta", ["", "   ", None, 123])
def test_busca_rejeita_consulta_invalida(
    tmp_path, monkeypatch, consulta
):
    motor = criar_motor_temporario(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        motor.buscar(consulta)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_busca_rejeita_top_k_invalido(
    tmp_path, monkeypatch, top_k
):
    motor = criar_motor_temporario(tmp_path, monkeypatch)

    with pytest.raises(ValueError):
        motor.buscar("inflação", top_k=top_k)


def test_busca_rejeita_arquivo_de_dados_inexistente(tmp_path):
    caminho_embeddings = tmp_path / "embeddings.npy"
    np.save(caminho_embeddings, np.array([[1.0, 0.0]]))

    with pytest.raises(FileNotFoundError):
        MotorBuscaSemantica(
            tmp_path / "inexistente.json",
            caminho_embeddings,
        )


def test_busca_rejeita_metadados_inexistentes(
    tmp_path, monkeypatch
):
    caminho_dados = tmp_path / "noticias.json"
    caminho_embeddings = tmp_path / "embeddings.npy"

    noticias = [
        {
            "id": 1,
            "titulo": "Notícia",
            "texto_limpo": "Texto de exemplo para teste.",
            "data": "2025-01-01",
            "fonte": "IBGE",
        }
    ]

    caminho_dados.write_text(
        json.dumps(noticias, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(caminho_embeddings, np.array([[1.0, 0.0]]))

    monkeypatch.setattr(
        "src.busca.SentenceTransformer",
        lambda modelo: ModeloFalso(),
    )

    with pytest.raises(FileNotFoundError, match="Metadados"):
        MotorBuscaSemantica(caminho_dados, caminho_embeddings)


def test_busca_rejeita_quantidade_incorreta_de_embeddings(
    tmp_path, monkeypatch
):
    caminho_dados = tmp_path / "noticias.json"
    caminho_embeddings = tmp_path / "embeddings.npy"
    caminho_metadados = tmp_path / "embeddings_metadata.json"

    noticias = [
        {
            "id": 1,
            "titulo": "Notícia",
            "texto_limpo": "Texto de exemplo para teste.",
            "data": "2025-01-01",
            "fonte": "IBGE",
        }
    ]

    caminho_dados.write_text(
        json.dumps(noticias, ensure_ascii=False),
        encoding="utf-8",
    )

    np.save(
        caminho_embeddings,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    textos = [f'{noticias[0]["titulo"]}. {noticias[0]["texto_limpo"]}']

    caminho_metadados.write_text(
        json.dumps(
            {
                "modelo": MODELO,
                "quantidade_noticias": 1,
                "dimensao_embeddings": 2,
                "fingerprint": calcular_fingerprint(textos),
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.busca.SentenceTransformer",
        lambda modelo: ModeloFalso(),
    )

    with pytest.raises(ValueError, match="quantidade de embeddings"):
        MotorBuscaSemantica(caminho_dados, caminho_embeddings)


def test_busca_rejeita_textos_desatualizados(tmp_path, monkeypatch):
    criar_motor_temporario(tmp_path, monkeypatch)

    caminho_dados = tmp_path / "noticias.json"
    noticias = json.loads(caminho_dados.read_text(encoding="utf-8"))
    noticias[0]["texto_limpo"] = "Texto atualizado para teste."

    caminho_dados.write_text(
        json.dumps(noticias, ensure_ascii=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="foram alterados"):
        MotorBuscaSemantica(
            caminho_dados,
            tmp_path / "embeddings.npy",
        )


def test_busca_rejeita_modelo_incompativel(tmp_path, monkeypatch):
    criar_motor_temporario(tmp_path, monkeypatch)

    caminho_metadados = tmp_path / "embeddings_metadata.json"
    metadados = json.loads(
        caminho_metadados.read_text(encoding="utf-8")
    )
    metadados["modelo"] = "modelo-incompativel"

    caminho_metadados.write_text(
        json.dumps(metadados),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="modelo registrado"):
        MotorBuscaSemantica(
            tmp_path / "noticias.json",
            tmp_path / "embeddings.npy",
        )
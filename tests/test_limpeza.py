
import json

import pytest

from src.limpeza import limpar_texto, gerar_dados_limpos


def test_limpar_html_urls_e_espacos():
    texto = (
        "<p>Inflação&nbsp; aumentou.</p> "
        "https://exemplo.com/noticia"
    )

    resultado = limpar_texto(texto)

    assert resultado == "Inflação aumentou."


def test_limpar_texto_vazio():
    assert limpar_texto("") == ""
    assert limpar_texto(None) == ""


def test_remover_dominio_residual_sem_protocolo():
    texto = (
        "Dados completos disponíveis em ibge.gov.br ."
    )

    resultado = limpar_texto(texto)

    assert "ibge.gov.br" not in resultado
    assert resultado == "Dados completos disponíveis em."


def test_corrigir_espaco_apos_pontuacao():
    texto = (
        "O comitê decidiu,por unanimidade; "
        "a decisão foi mantida:sem alterações."
    )

    resultado = limpar_texto(texto)

    assert resultado == (
        "O comitê decidiu, por unanimidade; "
        "a decisão foi mantida: sem alterações."
    )


def test_preservar_numeros_decimais_e_percentuais():
    texto = (
        "O IPCA ficou em 0,12%, "
        "e a taxa alcançou 13,75% ao ano."
    )

    resultado = limpar_texto(texto)

    assert "0,12%" in resultado
    assert "13,75%" in resultado
    assert resultado == (
        "O IPCA ficou em 0,12%, "
        "e a taxa alcançou 13,75% ao ano."
    )


def test_decodificar_entidades_html():
    texto = (
        "O &Iacute;ndice de Pre&ccedil;os "
        "ao Consumidor Amplo."
    )

    resultado = limpar_texto(texto)

    assert resultado == (
        "O Índice de Preços ao Consumidor Amplo."
    )


def test_remover_textos_curto_e_duplicado(tmp_path):
    entrada = tmp_path / "noticias.json"
    saida = tmp_path / "noticias_limpas.json"

    texto = (
        "A economia brasileira apresentou crescimento "
        "durante o período analisado."
    )

    noticias = [
        {
            "id": 1,
            "titulo": "Crescimento econômico",
            "texto": texto,
            "data": "2023-09-01",
            "fonte": "IBGE",
        },
        {
            "id": 2,
            "titulo": "Crescimento econômico",
            "texto": texto,
            "data": "2023-09-01",
            "fonte": "IBGE",
        },
        {
            "id": 3,
            "titulo": "Nota curta",
            "texto": "Selic.",
            "data": "2023-09-01",
            "fonte": "IBGE",
        },
    ]

    entrada.write_text(
        json.dumps(noticias, ensure_ascii=False),
        encoding="utf-8",
    )

    gerar_dados_limpos(
        caminho_entrada=entrada,
        caminho_saida=saida,
    )

    resultado = json.loads(saida.read_text(encoding="utf-8"))

    assert len(resultado) == 1
    assert resultado[0]["id"] == 1
    assert resultado[0]["titulo"] == "Crescimento econômico"


def test_arquivo_de_entrada_inexistente(tmp_path):
    entrada = tmp_path / "nao_existe.json"
    saida = tmp_path / "saida.json"

    with pytest.raises(FileNotFoundError):
        gerar_dados_limpos(entrada, saida)


def test_entrada_sem_noticias_validas(tmp_path):
    entrada = tmp_path / "noticias.json"
    saida = tmp_path / "saida.json"

    entrada.write_text(
        json.dumps([
            {
                "id": 1,
                "titulo": "Nota curta",
                "texto": "Selic.",
                "data": "2023-09-01",
                "fonte": "IBGE",
            }
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Nenhuma notícia válida"):
        gerar_dados_limpos(entrada, saida)

    assert not saida.exists()

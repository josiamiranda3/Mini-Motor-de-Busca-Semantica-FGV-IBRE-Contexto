
import html
import json
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup


def limpar_texto(texto: str) -> str:
    """Remove HTML e ruídos comuns de textos de notícias."""
    if not isinstance(texto, str):
        return ""

    texto = html.unescape(texto)

    soup = BeautifulSoup(texto, "html.parser")
    texto = soup.get_text(separator=" ")

    # Remove metadados de publicação, preservando datas no corpo
    # da notícia que possam ter valor informativo.
    texto = re.sub(
        r"\bPublicado em:\s*\d{1,2}/\d{1,2}/\d{4}"
        r"(?:\s*(?:às|as)?\s*\d{1,2}h\d{0,2})?",
        " ",
        texto,
        flags=re.IGNORECASE
    )

    # Remove URLs.
    texto = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        texto,
        flags=re.IGNORECASE
    )

    # Normaliza espaços e caracteres de espaço invisíveis.
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def normalizar_para_comparacao(texto: str) -> str:
    """Gera uma versão normalizada para identificar duplicatas."""
    texto = unicodedata.normalize("NFKC", texto)
    texto = texto.casefold()
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def gerar_dados_limpos(
    caminho_entrada="dados/noticias_brutas.json",
    caminho_saida="dados/noticias_limpas.json",
    min_caracteres=40
):
    caminho_entrada = Path(caminho_entrada)
    caminho_saida = Path(caminho_saida)

    if min_caracteres < 0:
        raise ValueError("min_caracteres não pode ser negativo.")

    if not caminho_entrada.is_file():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {caminho_entrada}"
        )

    with caminho_entrada.open("r", encoding="utf-8") as arquivo:
        noticias = json.load(arquivo)

    if not isinstance(noticias, list):
        raise ValueError(
            "O arquivo de entrada deve conter uma lista JSON."
        )

    dados_limpos = []
    chaves_vistas = set()

    invalidas = 0
    curtas = 0
    duplicadas = 0

    for noticia in noticias:
        if not isinstance(noticia, dict):
            invalidas += 1
            continue

        # Campos essenciais para a identificação e o conteúdo.
        if "id" not in noticia or not noticia.get("titulo"):
            invalidas += 1
            continue

        titulo = limpar_texto(noticia.get("titulo"))
        texto_limpo = limpar_texto(noticia.get("texto"))

        if not titulo:
            invalidas += 1
            continue

        # Descarta conteúdo insuficiente para busca semântica.
        if len(texto_limpo) < min_caracteres:
            curtas += 1
            continue

        # Detecta duplicatas usando título e corpo normalizados.
        chave = (
            normalizar_para_comparacao(titulo),
            normalizar_para_comparacao(texto_limpo)
        )

        if chave in chaves_vistas:
            duplicadas += 1
            continue

        chaves_vistas.add(chave)

        dados_limpos.append({
            "id": noticia["id"],
            "titulo": titulo,
            "texto_limpo": texto_limpo,
            "data": noticia.get("data", ""),
            "fonte": noticia.get("fonte", "")
        })

    # Não sobrescreve o arquivo de saída se nenhum registro passar
    # pela validação.
    if not dados_limpos:
        raise ValueError(
            "Nenhuma notícia válida restou após a limpeza. "
            "Revise os dados de entrada e o limite de caracteres."
        )

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    with caminho_saida.open("w", encoding="utf-8") as arquivo:
        json.dump(
            dados_limpos,
            arquivo,
            ensure_ascii=False,
            indent=2
        )

    print("\n=== Relatório da limpeza ===")
    print(f"Registros recebidos: {len(noticias)}")
    print(f"Registros inválidos: {invalidas}")
    print(f"Textos muito curtos: {curtas}")
    print(f"Duplicatas removidas: {duplicadas}")
    print(f"Notícias salvas: {len(dados_limpos)}")
    print(f"Arquivo gerado: {caminho_saida}")


if __name__ == "__main__":
    gerar_dados_limpos()

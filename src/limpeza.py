import json
import re
import html
from bs4 import BeautifulSoup
from pathlib import Path


def limpar_texto(texto: str) -> str:
    if not texto:
        return ""

    texto = html.unescape(texto)

    soup = BeautifulSoup(texto, "html.parser")
    texto = soup.get_text(separator=" ")

    texto = re.sub(r"Publicado em:\s*\d{2}/\d{2}/\d{4}.*?(\s|$)", " ", texto)
    texto = re.sub(r"\d{2}/\d{2}/\d{4}\s*[-—|]?\s*\d{0,2}h?\d{0,2}?", " ", texto)
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)

    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def gerar_dados_limpos(
    caminho_entrada="dados/noticias_brutas.json",
    caminho_saida="dados/noticias_limpas.json"
):
    caminho_entrada = Path(caminho_entrada)
    caminho_saida = Path(caminho_saida)

    with open(caminho_entrada, "r", encoding="utf-8") as arquivo:
        noticias = json.load(arquivo)

    dados_limpos = []

    for noticia in noticias:
        texto_limpo = limpar_texto(noticia["texto"])

        dados_limpos.append({
            "id": noticia["id"],
            "titulo": noticia["titulo"],
            "texto_limpo": texto_limpo,
            "data": noticia["data"],
            "fonte": noticia["fonte"]
        })

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    with open(caminho_saida, "w", encoding="utf-8") as arquivo:
        json.dump(dados_limpos, arquivo, ensure_ascii=False, indent=2)

    print(f"Dados limpos salvos em: {caminho_saida}")


if __name__ == "__main__":
    gerar_dados_limpos()
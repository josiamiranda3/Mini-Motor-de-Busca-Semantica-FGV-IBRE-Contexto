import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


MODELO = "paraphrase-multilingual-MiniLM-L12-v2"


def calcular_fingerprint(textos):
    """Gera uma identificação do conteúdo e da ordem dos textos."""
    conteudo = json.dumps(
        textos,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        conteudo.encode("utf-8")
    ).hexdigest()


def gerar_embeddings(
    caminho_dados="dados/noticias_limpas.json",
    caminho_saida="dados/embeddings.npy",
):
    caminho_dados = Path(caminho_dados)
    caminho_saida = Path(caminho_saida)

    if not caminho_dados.is_file():
        raise FileNotFoundError(
            f"Arquivo de notícias não encontrado: {caminho_dados}"
        )

    with caminho_dados.open("r", encoding="utf-8") as arquivo:
        noticias = json.load(arquivo)

    if not isinstance(noticias, list) or not noticias:
        raise ValueError(
            "O arquivo de notícias deve conter uma lista JSON não vazia."
        )

    textos = []

    for indice, noticia in enumerate(noticias):
        if not isinstance(noticia, dict):
            raise ValueError(
                f"Notícia inválida na posição {indice}."
            )

        titulo = noticia.get("titulo")
        texto_limpo = noticia.get("texto_limpo")

        if (
            not isinstance(titulo, str)
            or not titulo.strip()
            or not isinstance(texto_limpo, str)
            or not texto_limpo.strip()
        ):
            raise ValueError(
                f"Notícia na posição {indice} possui título ou "
                "texto_limpo inválido."
            )

        textos.append(f"{titulo}. {texto_limpo}")

    print(f"Gerando embeddings para {len(textos)} notícias...")

    modelo = SentenceTransformer(MODELO)

    embeddings = modelo.encode(
        textos,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    caminho_saida.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(caminho_saida, embeddings)

    metadados = {
        "modelo": MODELO,
        "quantidade_noticias": len(textos),
        "dimensao_embeddings": int(embeddings.shape[1]),
        "fingerprint": calcular_fingerprint(textos),
    }

    caminho_metadados = caminho_saida.with_name(
        "embeddings_metadata.json"
    )

    with caminho_metadados.open("w", encoding="utf-8") as arquivo:
        json.dump(
            metadados,
            arquivo,
            ensure_ascii=False,
            indent=4,
        )
        arquivo.write("\n")

    print(f"Embeddings salvos em: {caminho_saida}")
    print(f"Metadados salvos em: {caminho_metadados}")


if __name__ == "__main__":
    gerar_embeddings()
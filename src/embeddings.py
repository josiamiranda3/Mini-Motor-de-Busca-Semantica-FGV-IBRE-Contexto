import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


MODELO = "paraphrase-multilingual-MiniLM-L12-v2"


def gerar_embeddings(
    caminho_dados="dados/noticias_limpas.json",
    caminho_saida="dados/embeddings.npy"
):
    with open(caminho_dados, "r", encoding="utf-8") as arquivo:
        noticias = json.load(arquivo)

    textos = [
        noticia["titulo"] + ". " + noticia["texto_limpo"]
        for noticia in noticias
    ]

    modelo = SentenceTransformer(MODELO)

    embeddings = modelo.encode(
        textos,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    Path(caminho_saida).parent.mkdir(parents=True, exist_ok=True)
    np.save(caminho_saida, embeddings)

    print(f"Embeddings salvos em: {caminho_saida}")


if __name__ == "__main__":
    gerar_embeddings()
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODELO = "paraphrase-multilingual-MiniLM-L12-v2"


class MotorBuscaSemantica:
    def __init__(
        self,
        caminho_dados="dados/noticias_limpas.json",
        caminho_embeddings="dados/embeddings.npy"
    ):
        with open(caminho_dados, "r", encoding="utf-8") as arquivo:
            self.noticias = json.load(arquivo)

        self.embeddings = np.load(caminho_embeddings)
        self.modelo = SentenceTransformer(MODELO)

    def buscar(self, consulta, top_k=5):
        embedding_consulta = self.modelo.encode(
            [consulta],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similaridades = cosine_similarity(
            embedding_consulta,
            self.embeddings
        )[0]

        indices = np.argsort(similaridades)[::-1][:top_k]

        resultados = []

        for indice in indices:
            noticia = self.noticias[indice]

            resultados.append({
                "id": noticia["id"],
                "titulo": noticia["titulo"],
                "fonte": noticia["fonte"],
                "data": noticia["data"],
                "similaridade": round(float(similaridades[indice]), 4),
                "trecho": noticia["texto_limpo"][:300] + "..."
            })

        return resultados
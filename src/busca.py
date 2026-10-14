
import json
from pathlib import Path

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
        caminho_dados = Path(caminho_dados)
        caminho_embeddings = Path(caminho_embeddings)

        if not caminho_dados.is_file():
            raise FileNotFoundError(
                f"Arquivo de notícias não encontrado: {caminho_dados}"
            )

        if not caminho_embeddings.is_file():
            raise FileNotFoundError(
                f"Arquivo de embeddings não encontrado: {caminho_embeddings}. "
                "Execute primeiro o pipeline de geração de embeddings."
            )

        with caminho_dados.open("r", encoding="utf-8") as arquivo:
            self.noticias = json.load(arquivo)

        if not isinstance(self.noticias, list):
            raise ValueError(
                "O arquivo de notícias deve conter uma lista JSON."
            )

        if not self.noticias:
            raise ValueError(
                "Não existem notícias disponíveis para a busca."
            )

        for indice, noticia in enumerate(self.noticias):
            if not isinstance(noticia, dict):
                raise ValueError(
                    f"Notícia inválida na posição {indice}."
                )

            campos_obrigatorios = (
                "id", "titulo", "texto_limpo", "data", "fonte"
            )

            campos_ausentes = [
                campo
                for campo in campos_obrigatorios
                if campo not in noticia
            ]

            if campos_ausentes:
                raise ValueError(
                    f"Notícia na posição {indice} sem os campos: "
                    f"{', '.join(campos_ausentes)}"
                )

        self.embeddings = np.load(
            caminho_embeddings,
            allow_pickle=False
        )

        if self.embeddings.ndim != 2:
            raise ValueError(
                "Os embeddings devem formar uma matriz bidimensional."
            )

        if self.embeddings.shape[0] != len(self.noticias):
            raise ValueError(
                "A quantidade de embeddings não corresponde à quantidade "
                "de notícias. Gere novamente os dados e os embeddings."
            )

        if self.embeddings.shape[1] == 0:
            raise ValueError("Os embeddings não possuem dimensões.")

        if not np.isfinite(self.embeddings).all():
            raise ValueError(
                "Os embeddings contêm valores inválidos (NaN ou infinito)."
            )

        self.modelo = SentenceTransformer(MODELO)

    def buscar(self, consulta, top_k=5):
        if not isinstance(consulta, str) or not consulta.strip():
            raise ValueError(
                "A consulta deve ser um texto não vazio."
            )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise ValueError(
                "top_k deve ser um número inteiro maior que zero."
            )

        embedding_consulta = self.modelo.encode(
            [consulta.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if embedding_consulta.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                "A dimensão do embedding da consulta não corresponde "
                "à dimensão dos embeddings indexados. "
                "Verifique se foi utilizado o mesmo modelo."
            )

        similaridades = cosine_similarity(
            embedding_consulta,
            self.embeddings
        )[0]

        quantidade = min(top_k, len(self.noticias))
        indices = np.argsort(similaridades)[::-1][:quantidade]

        resultados = []

        for indice in indices:
            noticia = self.noticias[indice]
            texto = noticia["texto_limpo"].strip()
            trecho = texto[:300]

            if len(texto) > 300:
                trecho += "..."

            resultados.append({
                "id": noticia["id"],
                "titulo": noticia["titulo"],
                "fonte": noticia["fonte"],
                "data": noticia["data"],
                "similaridade": round(
                    float(similaridades[indice]), 4
                ),
                "trecho": trecho
            })

        return resultados

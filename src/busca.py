import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings import calcular_fingerprint


MODELO = "paraphrase-multilingual-MiniLM-L12-v2"


class MotorBuscaSemantica:
    def __init__(
        self,
        caminho_dados="dados/noticias_limpas.json",
        caminho_embeddings="dados/embeddings.npy",
    ):
        caminho_dados = Path(caminho_dados)
        caminho_embeddings = Path(caminho_embeddings)
        caminho_metadados = caminho_embeddings.with_name(
            "embeddings_metadata.json"
        )

        if not caminho_dados.is_file():
            raise FileNotFoundError(
                f"Arquivo de notícias não encontrado: {caminho_dados}"
            )

        if not caminho_embeddings.is_file():
            raise FileNotFoundError(
                f"Arquivo de embeddings não encontrado: "
                f"{caminho_embeddings}. Execute primeiro "
                "'python -m src.embeddings'."
            )

        if not caminho_metadados.is_file():
            raise FileNotFoundError(
                f"Metadados dos embeddings não encontrados: "
                f"{caminho_metadados}. Gere novamente os embeddings "
                "com 'python -m src.embeddings'."
            )

        try:
            with caminho_dados.open("r", encoding="utf-8") as arquivo:
                noticias = json.load(arquivo)
        except json.JSONDecodeError as erro:
            raise ValueError(
                f"Arquivo de notícias contém JSON inválido: {caminho_dados}"
            ) from erro

        if not isinstance(noticias, list):
            raise ValueError(
                "O arquivo de notícias deve conter uma lista JSON."
            )

        if not noticias:
            raise ValueError(
                "Não existem notícias disponíveis para a busca."
            )

        campos_obrigatorios = (
            "id",
            "titulo",
            "texto_limpo",
            "data",
            "fonte",
        )

        textos = []

        for indice, noticia in enumerate(noticias):
            if not isinstance(noticia, dict):
                raise ValueError(
                    f"Notícia inválida na posição {indice}."
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

            titulo = noticia["titulo"]
            texto_limpo = noticia["texto_limpo"]

            if (
                not isinstance(titulo, str)
                or not isinstance(texto_limpo, str)
                or not titulo.strip()
                or not texto_limpo.strip()
            ):
                raise ValueError(
                    f"Notícia na posição {indice} possui título "
                    "ou texto_limpo inválido."
                )

            textos.append(f"{titulo}. {texto_limpo}")

        try:
            with caminho_metadados.open(
                "r", encoding="utf-8"
            ) as arquivo:
                metadados = json.load(arquivo)
        except json.JSONDecodeError as erro:
            raise ValueError(
                f"Arquivo de metadados contém JSON inválido: "
                f"{caminho_metadados}. Execute 'python -m src.embeddings'."
            ) from erro

        if not isinstance(metadados, dict):
            raise ValueError(
                "O arquivo de metadados deve conter um objeto JSON."
            )

        campos_metadados = (
            "modelo",
            "quantidade_noticias",
            "dimensao_embeddings",
            "fingerprint",
        )

        campos_ausentes = [
            campo
            for campo in campos_metadados
            if campo not in metadados
        ]

        if campos_ausentes:
            raise ValueError(
                "Metadados incompletos. Campos ausentes: "
                f"{', '.join(campos_ausentes)}. "
                "Execute 'python -m src.embeddings'."
            )

        if metadados["modelo"] != MODELO:
            raise ValueError(
                "O modelo registrado nos metadados é incompatível. "
                "Execute 'python -m src.embeddings' para regenerar "
                "os embeddings com o modelo configurado."
            )

        if metadados["quantidade_noticias"] != len(noticias):
            raise ValueError(
                "A quantidade de notícias difere da registrada nos "
                "metadados. Execute 'python -m src.embeddings'."
            )

        fingerprint_atual = calcular_fingerprint(textos)

        if metadados["fingerprint"] != fingerprint_atual:
            raise ValueError(
                "Os textos das notícias foram alterados ou sua ordem "
                "mudou desde a geração dos embeddings. Execute "
                "'python -m src.embeddings' para atualizá-los."
            )

        try:
            self.embeddings = np.load(
                caminho_embeddings,
                allow_pickle=False,
            )
        except (OSError, ValueError) as erro:
            raise ValueError(
                f"Não foi possível carregar os embeddings: "
                f"{caminho_embeddings}. Gere novamente com "
                "'python -m src.embeddings'."
            ) from erro

        if self.embeddings.ndim != 2:
            raise ValueError(
                "Os embeddings devem formar uma matriz bidimensional."
            )

        if self.embeddings.shape[0] != len(noticias):
            raise ValueError(
                "A quantidade de embeddings não corresponde à quantidade "
                "de notícias. Gere novamente os dados e os embeddings."
            )

        if self.embeddings.shape[1] == 0:
            raise ValueError("Os embeddings não possuem dimensões.")

        if (
            metadados["dimensao_embeddings"]
            != self.embeddings.shape[1]
        ):
            raise ValueError(
                "A dimensão dos embeddings difere da registrada nos "
                "metadados. Execute 'python -m src.embeddings'."
            )

        if not np.isfinite(self.embeddings).all():
            raise ValueError(
                "Os embeddings contêm valores inválidos (NaN ou infinito)."
            )

        self.noticias = noticias
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
            normalize_embeddings=True,
        )

        if embedding_consulta.shape[1] != self.embeddings.shape[1]:
            raise ValueError(
                "A dimensão do embedding da consulta não corresponde "
                "à dimensão dos embeddings indexados. "
                "Verifique se foi utilizado o mesmo modelo."
            )

        similaridades = cosine_similarity(
            embedding_consulta,
            self.embeddings,
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

            resultados.append(
                {
                    "id": noticia["id"],
                    "titulo": noticia["titulo"],
                    "fonte": noticia["fonte"],
                    "data": noticia["data"],
                    "similaridade": round(
                        float(similaridades[indice]), 4
                    ),
                    "trecho": trecho,
                }
            )

        return resultados
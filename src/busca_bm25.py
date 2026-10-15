
import json
import re
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenizar(texto):
    """Normaliza o texto e separa em termos para a busca lexical."""
    texto = unicodedata.normalize("NFKC", texto).casefold()
    return re.findall(r"\w+", texto, flags=re.UNICODE)


class MotorBuscaBM25:
    def __init__(self, caminho_dados="dados/noticias_limpas.json"):
        caminho_dados = Path(caminho_dados)

        if not caminho_dados.is_file():
            raise FileNotFoundError(
                f"Arquivo de notícias não encontrado: {caminho_dados}"
            )

        with caminho_dados.open("r", encoding="utf-8") as arquivo:
            self.noticias = json.load(arquivo)

        if not isinstance(self.noticias, list) or not self.noticias:
            raise ValueError(
                "O arquivo deve conter uma lista não vazia de notícias."
            )

        self.documentos = []

        for indice, noticia in enumerate(self.noticias):
            if not isinstance(noticia, dict):
                raise ValueError(f"Notícia inválida na posição {indice}.")

            titulo = noticia.get("titulo", "")
            texto = noticia.get("texto_limpo", "")

            if not isinstance(titulo, str) or not isinstance(texto, str):
                raise ValueError(
                    f"Título ou texto inválido na notícia da posição {indice}."
                )

            conteudo = f"{titulo} {titulo} {texto}".strip()
            tokens = tokenizar(conteudo)

            if not tokens:
                raise ValueError(
                    f"A notícia na posição {indice} não possui termos pesquisáveis."
                )

            self.documentos.append(tokens)

        self.modelo = BM25Okapi(self.documentos)

    def buscar(self, consulta, top_k=5):
        if not isinstance(consulta, str) or not consulta.strip():
            raise ValueError("A consulta deve ser um texto não vazio.")

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k deve ser um inteiro maior que zero.")

        tokens_consulta = tokenizar(consulta)

        if not tokens_consulta:
            raise ValueError("A consulta não contém termos pesquisáveis.")

        pontuacoes = self.modelo.get_scores(tokens_consulta)
        quantidade = min(top_k, len(self.noticias))

        indices = sorted(
            range(len(pontuacoes)),
            key=lambda indice: (-pontuacoes[indice], indice),
        )[:quantidade]

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
                "fonte": noticia.get("fonte", ""),
                "data": noticia.get("data", ""),
                "pontuacao": round(float(pontuacoes[indice]), 4),
                "trecho": trecho,
            })

        return resultados


import json
from pathlib import Path

from src.busca import MotorBuscaSemantica


def precision_at_k(ids_retorno, ids_relevantes, k=5):
    if k <= 0:
        raise ValueError("k deve ser maior que zero.")

    resultados = ids_retorno[:k]
    if not resultados:
        return 0.0

    relevantes = set(ids_relevantes)
    acertos = sum(1 for item_id in resultados if item_id in relevantes)

    return acertos / len(resultados)


def recall_at_k(ids_retorno, ids_relevantes, k=5):
    if k <= 0:
        raise ValueError("k deve ser maior que zero.")

    relevantes = set(ids_relevantes)
    if not relevantes:
        return 0.0

    resultados = ids_retorno[:k]
    acertos = sum(1 for item_id in resultados if item_id in relevantes)

    return acertos / len(relevantes)


def reciprocal_rank(ids_retorno, ids_relevantes):
    relevantes = set(ids_relevantes)

    for posicao, item_id in enumerate(ids_retorno, start=1):
        if item_id in relevantes:
            return 1.0 / posicao

    return 0.0


def avaliar_motor(
    caminho_consultas="dados/consultas_avaliacao.json",
    top_k=5,
):
    caminho = Path(caminho_consultas)

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Arquivo de avaliação não encontrado: {caminho}"
        )

    with caminho.open("r", encoding="utf-8") as arquivo:
        consultas = json.load(arquivo)

    if not isinstance(consultas, list) or not consultas:
        raise ValueError(
            "O arquivo de avaliação deve conter uma lista não vazia."
        )

    motor = MotorBuscaSemantica()
    resultados_avaliacao = []

    for item in consultas:
        consulta = item["consulta"]
        relevantes = item["relevantes"]

        resultados = motor.buscar(consulta, top_k=top_k)
        ids_retornados = [
            resultado["id"] for resultado in resultados
        ]

        resultados_avaliacao.append({
            "consulta": consulta,
            "ids_relevantes": relevantes,
            "ids_retornados": ids_retornados,
            "precision_at_k": precision_at_k(
                ids_retornados, relevantes, top_k
            ),
            "recall_at_k": recall_at_k(
                ids_retornados, relevantes, top_k
            ),
            "mrr": reciprocal_rank(
                ids_retornados, relevantes
            ),
        })

    quantidade = len(resultados_avaliacao)

    medias = {
        metrica: sum(item[metrica] for item in resultados_avaliacao)
        / quantidade
        for metrica in (
            "precision_at_k",
            "recall_at_k",
            "mrr",
        )
    }

    return {
        "top_k": top_k,
        "quantidade_consultas": quantidade,
        "medias": medias,
        "detalhes": resultados_avaliacao,
    }


if __name__ == "__main__":
    avaliacao = avaliar_motor()

    print(f"\n=== Avaliação do motor (K={avaliacao['top_k']}) ===")
    print(f"Consultas avaliadas: {avaliacao['quantidade_consultas']}")

    for metrica, valor in avaliacao["medias"].items():
        print(f"{metrica}: {valor:.4f}")

    print("\n=== Resultados por consulta ===")

    for item in avaliacao["detalhes"]:
        print(f"\nConsulta: {item['consulta']}")
        print(f"IDs relevantes: {item['ids_relevantes']}")
        print(f"IDs retornados: {item['ids_retornados']}")
        print(f"Precision@K: {item['precision_at_k']:.4f}")
        print(f"Recall@K:    {item['recall_at_k']:.4f}")
        print(f"MRR:         {item['mrr']:.4f}")


import json

from src.avaliacao import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.busca import MotorBuscaSemantica
from src.busca_bm25 import MotorBuscaBM25
from src.busca_hibrida import MotorBuscaHibrida


def avaliar_metodo(motor, consultas, top_k=5):
    resultados = []

    for item in consultas:
        ids_relevantes = item["relevantes"]
        encontrados = motor.buscar(item["consulta"], top_k=top_k)
        ids_retornados = [r["id"] for r in encontrados]

        resultados.append({
            "consulta": item["consulta"],
            "precision": precision_at_k(
                ids_retornados, ids_relevantes, top_k
            ),
            "recall": recall_at_k(
                ids_retornados, ids_relevantes, top_k
            ),
            "mrr": reciprocal_rank(
                ids_retornados, ids_relevantes
            ),
            "ids": ids_retornados,
        })

    quantidade = len(resultados)

    medias = {
        metrica: sum(r[metrica] for r in resultados) / quantidade
        for metrica in ("precision", "recall", "mrr")
    }

    return medias, resultados


def main():
    with open(
        "dados/consultas_avaliacao.json",
        encoding="utf-8",
    ) as arquivo:
        consultas = json.load(arquivo)

    semantico = MotorBuscaSemantica()
    bm25 = MotorBuscaBM25()

    hibrido = MotorBuscaHibrida(
        motor_semantico=semantico,
        motor_bm25=bm25,
    )

    motores = {
        "Semântica": semantico,
        "BM25": bm25,
        "Híbrida RRF": hibrido,
    }

    avaliacoes = {
        nome: avaliar_metodo(motor, consultas)
        for nome, motor in motores.items()
    }

    print("\n=== COMPARAÇÃO DOS MÉTODOS: TOP 5 ===")
    print(f"{'Métrica':<15}", end="")

    for nome in motores:
        print(f"{nome:>16}", end="")

    print()

    for metrica in ("precision", "recall", "mrr"):
        print(f"{metrica:<15}", end="")

        for nome in motores:
            medias, _ = avaliacoes[nome]
            print(f"{medias[metrica]:>16.4f}", end="")

        print()

    print("\n=== RESULTADOS POR CONSULTA ===")

    for indice, consulta in enumerate(consultas):
        print(f"\nConsulta: {consulta['consulta']}")

        for nome in motores:
            _, detalhes = avaliacoes[nome]
            resultado = detalhes[indice]

            print(
                f"{nome:<13} IDs: {resultado['ids']} | "
                f"P@5={resultado['precision']:.2f} | "
                f"R@5={resultado['recall']:.2f} | "
                f"MRR={resultado['mrr']:.2f}"
            )


if __name__ == "__main__":
    main()

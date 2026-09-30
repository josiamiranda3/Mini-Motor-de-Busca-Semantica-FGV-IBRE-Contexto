from src.limpeza import gerar_dados_limpos
from src.embeddings import gerar_embeddings
from src.busca import MotorBuscaSemantica


def executar_pipeline():
    print("Etapa 1 - Limpando dados...")
    gerar_dados_limpos()

    print("\nEtapa 2 - Gerando embeddings...")
    gerar_embeddings()

    print("\nEtapa 3 - Testando busca semântica...")
    motor = MotorBuscaSemantica()

    consultas = [
        "mudanças na taxa de juros",
        "mercado de trabalho e desemprego",
        "inflação e preços ao consumidor"
    ]

    for consulta in consultas:
        print("\n" + "=" * 80)
        print(f"Consulta: {consulta}")
        print("=" * 80)

        resultados = motor.buscar(consulta, top_k=5)

        for resultado in resultados:
            print(f"\nID: {resultado['id']}")
            print(f"Título: {resultado['titulo']}")
            print(f"Fonte: {resultado['fonte']}")
            print(f"Data: {resultado['data']}")
            print(f"Similaridade: {resultado['similaridade']}")
            print(f"Trecho: {resultado['trecho']}")


if __name__ == "__main__":
    executar_pipeline()
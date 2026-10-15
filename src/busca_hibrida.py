
from src.busca import MotorBuscaSemantica
from src.busca_bm25 import MotorBuscaBM25


class MotorBuscaHibrida:
    """Combina rankings semântico e lexical usando Reciprocal Rank Fusion."""

    def __init__(
        self,
        constante_rrf=60,
        motor_semantico=None,
        motor_bm25=None,
    ):
        if (
            isinstance(constante_rrf, bool)
            or not isinstance(constante_rrf, int)
            or constante_rrf <= 0
        ):
            raise ValueError(
                "constante_rrf deve ser um inteiro positivo."
            )

        # Permite reutilizar motores já inicializados.
        # Isso evita carregar o modelo semântico novamente
        # durante a comparação entre métodos.
        self.semantico = (
            motor_semantico
            if motor_semantico is not None
            else MotorBuscaSemantica()
        )

        self.bm25 = (
            motor_bm25
            if motor_bm25 is not None
            else MotorBuscaBM25()
        )

        self.constante_rrf = constante_rrf

    def buscar(self, consulta, top_k=5):
        """Retorna os documentos mais bem classificados pela fusão RRF."""

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
                "top_k deve ser um inteiro positivo."
            )

        # Recupera os rankings completos para que a fusão considere
        # todos os documentos disponíveis nos dois motores.
        resultados_semanticos = self.semantico.buscar(
            consulta,
            top_k=len(self.semantico.noticias),
        )

        resultados_bm25 = self.bm25.buscar(
            consulta,
            top_k=len(self.bm25.noticias),
        )

        # Indexa os documentos por ID para preservar os metadados.
        por_id = {
            resultado["id"]: resultado
            for resultado in resultados_semanticos
        }

        # Adiciona documentos que eventualmente existam apenas
        # no ranking lexical.
        for resultado in resultados_bm25:
            por_id.setdefault(resultado["id"], resultado)

        pontuacoes = {}

        rankings = (
            resultados_semanticos,
            resultados_bm25,
        )

        for ranking in rankings:
            for posicao, resultado in enumerate(ranking, start=1):
                item_id = resultado["id"]

                pontuacoes[item_id] = (
                    pontuacoes.get(item_id, 0.0)
                    + 1.0 / (self.constante_rrf + posicao)
                )

        # Desempate determinístico para facilitar a reprodução dos testes.
        ids_ordenados = sorted(
            pontuacoes,
            key=lambda item_id: (
                -pontuacoes[item_id],
                str(item_id),
            ),
        )

        resultados = []

        for item_id in ids_ordenados[:top_k]:
            resultado = por_id[item_id].copy()
            resultado["pontuacao_rrf"] = round(
                pontuacoes[item_id],
                6,
            )
            resultados.append(resultado)

        return resultados

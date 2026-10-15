
import pytest

from src.busca_hibrida import MotorBuscaHibrida


class MotorFalso:
    """Motor simplificado para testar a fusão sem carregar modelos reais."""

    def __init__(self, resultados):
        self.resultados = resultados
        self.noticias = [
            {"id": resultado["id"]}
            for resultado in resultados
        ]
        self.ultimas_consultas = []

    def buscar(self, consulta, top_k=5):
        self.ultimas_consultas.append({
            "consulta": consulta,
            "top_k": top_k,
        })
        return self.resultados[:top_k]


def criar_resultado(item_id, titulo=None, **extras):
    resultado = {
        "id": item_id,
        "titulo": titulo or f"Notícia {item_id}",
        "fonte": "Fonte de teste",
        "data": "2026-01-01",
        "trecho": f"Trecho da notícia {item_id}",
    }
    resultado.update(extras)
    return resultado


@pytest.fixture
def motores():
    semantico = MotorFalso([
        criar_resultado(1, "Notícia semântica 1", similaridade=0.9),
        criar_resultado(2, "Notícia semântica 2", similaridade=0.8),
        criar_resultado(3, "Notícia semântica 3", similaridade=0.7),
    ])

    bm25 = MotorFalso([
        criar_resultado(1, "Notícia BM25 1", pontuacao=8.0),
        criar_resultado(3, "Notícia BM25 3", pontuacao=6.0),
        criar_resultado(2, "Notícia BM25 2", pontuacao=5.0),
        criar_resultado(4, "Notícia exclusiva BM25", pontuacao=4.0),
    ])

    return semantico, bm25


def test_fusao_prioriza_documento_bem_classificado_nos_dois_motores(
    motores,
):
    semantico, bm25 = motores
    motor = MotorBuscaHibrida(
        constante_rrf=60,
        motor_semantico=semantico,
        motor_bm25=bm25,
    )

    resultados = motor.buscar("consulta de teste", top_k=4)
    ids = [resultado["id"] for resultado in resultados]

    assert ids[0] == 1
    assert ids.index(1) < ids.index(2)
    assert ids.index(1) < ids.index(3)


def test_pontuacao_rrf_e_calculada_corretamente(motores):
    semantico, bm25 = motores
    motor = MotorBuscaHibrida(
        constante_rrf=60,
        motor_semantico=semantico,
        motor_bm25=bm25,
    )

    resultados = motor.buscar("consulta", top_k=4)
    por_id = {resultado["id"]: resultado for resultado in resultados}

    # O documento 1 ocupa a primeira posição nos dois rankings.
    assert por_id[1]["pontuacao_rrf"] == round(1 / 61 + 1 / 61, 6)

    # O documento 2 ocupa a segunda posição no ranking semântico
    # e a terceira no ranking BM25.
    assert por_id[2]["pontuacao_rrf"] == round(1 / 62 + 1 / 63, 6)


def test_resultados_respeitam_top_k(motores):
    semantico, bm25 = motores
    motor = MotorBuscaHibrida(
        motor_semantico=semantico,
        motor_bm25=bm25,
    )

    resultados = motor.buscar("consulta", top_k=2)

    assert len(resultados) == 2


def test_documento_exclusivo_do_bm25_e_preservado(motores):
    semantico, bm25 = motores
    motor = MotorBuscaHibrida(
        motor_semantico=semantico,
        motor_bm25=bm25,
    )

    resultados = motor.buscar("consulta", top_k=10)
    por_id = {resultado["id"]: resultado for resultado in resultados}

    assert 4 in por_id
    assert por_id[4]["titulo"] == "Notícia exclusiva BM25"
    assert "pontuacao_rrf" in por_id[4]


def test_metadados_semanticos_sao_preservados_quando_id_e_compartilhado(
    motores,
):
    semantico, bm25 = motores
    motor = MotorBuscaHibrida(
        motor_semantico=semantico,
        motor_bm25=bm25,
    )

    resultados = motor.buscar("consulta", top_k=4)
    por_id = {resultado["id"]: resultado for resultado in resultados}

    assert por_id[1]["titulo"] == "Notícia semântica 1"
    assert por_id[1]["similaridade"] == 0.9
    assert "pontuacao_rrf" in por_id[1]


def test_busca_recebe_a_consulta_e_recupera_rankings_completos(motores):
    semantico, bm25 = motores
    motor = MotorBuscaHibrida(
        motor_semantico=semantico,
        motor_bm25=bm25,
    )

    motor.buscar("inflação brasileira", top_k=3)

    assert semantico.ultimas_consultas[-1]["consulta"] == (
        "inflação brasileira"
    )
    assert bm25.ultimas_consultas[-1]["consulta"] == (
        "inflação brasileira"
    )

    assert semantico.ultimas_consultas[-1]["top_k"] == 3
    assert bm25.ultimas_consultas[-1]["top_k"] == 4


@pytest.mark.parametrize("constante", [0, -1, 1.5, True, "60", None])
def test_constante_rrf_invalida(constante):
    with pytest.raises(ValueError):
        MotorBuscaHibrida(constante_rrf=constante)


@pytest.mark.parametrize("consulta", ["", "   ", None, 123])
def test_consulta_invalida(motores, consulta):
    semantico, bm25 = motores
    motor = MotorBuscaHibrida(
        motor_semantico=semantico,
        motor_bm25=bm25,
    )

    with pytest.raises(ValueError):
        motor.buscar(consulta)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "5", None])
def test_top_k_invalido(motores, top_k):
    semantico, bm25 = motores
    motor = MotorBuscaHibrida(
        motor_semantico=semantico,
        motor_bm25=bm25,
    )

    with pytest.raises(ValueError):
        motor.buscar("consulta válida", top_k=top_k)


def test_empate_no_rrf_tem_desempate_deterministico():
    semantico = MotorFalso([
        criar_resultado(2),
        criar_resultado(1),
    ])
    bm25 = MotorFalso([
        criar_resultado(1),
        criar_resultado(2),
    ])

    motor = MotorBuscaHibrida(
        motor_semantico=semantico,
        motor_bm25=bm25,
    )

    resultados = motor.buscar("consulta", top_k=2)

    assert [resultado["id"] for resultado in resultados] == [1, 2]


def test_lista_vazia_de_resultados():
    semantico = MotorFalso([])
    bm25 = MotorFalso([])

    motor = MotorBuscaHibrida(
        motor_semantico=semantico,
        motor_bm25=bm25,
    )

    assert motor.buscar("consulta", top_k=5) == []

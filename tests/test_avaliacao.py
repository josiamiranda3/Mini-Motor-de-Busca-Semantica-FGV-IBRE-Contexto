
import pytest

from src.avaliacao import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    assert precision_at_k([1, 2, 3, 4], [1, 3], k=3) == pytest.approx(2 / 3)


def test_recall_at_k():
    assert recall_at_k([1, 2, 3, 4], [1, 3, 5], k=3) == pytest.approx(2 / 3)


def test_precision_sem_resultados():
    assert precision_at_k([], [1, 2], k=5) == 0.0


def test_recall_sem_relevantes():
    assert recall_at_k([1, 2], [], k=5) == 0.0


def test_mrr_primeiro_relevante_na_segunda_posicao():
    assert reciprocal_rank([4, 2, 7], [2]) == pytest.approx(0.5)


def test_mrr_sem_resultado_relevante():
    assert reciprocal_rank([4, 5, 7], [2]) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_precision_rejeita_k_invalido(k):
    with pytest.raises(ValueError):
        precision_at_k([1, 2], [1], k)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_rejeita_k_invalido(k):
    with pytest.raises(ValueError):
        recall_at_k([1, 2], [1], k)

"""Correr con: python -m tests.test_metricas

Cada caso está calculado a mano en el comentario que lo acompaña.
"""

import numpy as np

from src.metricas import METRICAS, en_presupuesto, evaluar, llamadas, puntajes, scoring


def _cerca(a, b, tol=1e-12):
    return abs(a - b) < tol


def test_presupuesto_calculado_a_mano():
    # 10 clientes, 2 «yes» (posiciones 0 y 1). Con q = 20 % se llama a 2: los de puntaje 0,9
    # (un «yes») y 0,8 (un «no»). Aciertos 1: recall 1/2, precisión 1/2, F1 1/2.
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    s = [0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.05]
    r = en_presupuesto(y, s, 0.2)
    assert r["llamadas"] == 2
    assert _cerca(r["recall_q"], 0.5) and _cerca(r["precision_q"], 0.5) and _cerca(r["f1_q"], 0.5)
    print("ok  presupuesto del 20 % sobre 10 clientes: recall, precisión y F1 de 0,5")


def test_empate_total_alcanza_exactamente_q():
    # Todos con el mismo puntaje: se reparte el corte en proporción. Con 1000 clientes, 113 «yes»
    # y 200 llamadas, se esperan 0,2 · 113 = 22,6 aciertos: recall 0,2 y precisión 0,113.
    y = np.zeros(1000, int)
    y[:113] = 1
    r = en_presupuesto(y, np.ones(1000), 0.2)
    assert _cerca(r["recall_q"], 0.2) and _cerca(r["precision_q"], 0.113)
    print("ok  con todos empatados, el recall es exactamente q y no depende del orden de las filas")


def test_empate_parcial_en_el_corte():
    # 5 clientes, «yes» en 0 y 2; puntajes 0,9 / 0,5 / 0,5 / 0,5 / 0,1; q = 40 % → 2 llamadas.
    # Arriba del corte: el 0 (un acierto). Falta 1 llamada entre 3 empatados, con 1 «yes» entre
    # ellos: 1/3 de acierto esperado. Aciertos 4/3: recall (4/3)/2 = 2/3, precisión (4/3)/2 = 2/3.
    r = en_presupuesto([1, 0, 1, 0, 0], [0.9, 0.5, 0.5, 0.5, 0.1], 0.4)
    assert _cerca(r["recall_q"], 2 / 3) and _cerca(r["precision_q"], 2 / 3)
    print("ok  empate parcial en el corte: recall y precisión de 2/3")


def test_orden_perfecto():
    # 2 «yes» con los dos puntajes más altos y 2 llamadas: recall 1, precisión 1.
    r = en_presupuesto([0, 1, 0, 1, 0, 0, 0, 0, 0, 0], [0, 9, 1, 8, 2, 3, 4, 5, 6, 7], 0.2)
    assert _cerca(r["recall_q"], 1.0) and _cerca(r["precision_q"], 1.0)
    print("ok  con el orden perfecto el presupuesto captura todos los «yes»")


def test_auc_en_los_extremos():
    y = [0, 0, 1, 1]
    assert _cerca(evaluar(y, [1, 2, 3, 4])["auc"], 1.0)
    assert _cerca(evaluar(y, [4, 3, 2, 1])["auc"], 0.0)
    assert _cerca(evaluar(y, [1, 1, 1, 1])["auc"], 0.5)
    print("ok  AUC: 1 con el orden perfecto, 0 con el invertido y 0,5 con todos empatados")


def test_cantidad_de_llamadas():
    # Un fold de validación tiene 6588 filas: 0,2 · 6588 = 1317,6 → 1318. El test tiene 8236:
    # 1647,2 → 1647.
    assert llamadas(6588) == 1318 and llamadas(8236) == 1647 and llamadas(3) == 1
    print("ok  llamadas: 1318 por fold de validación, 1647 en test, al menos 1 siempre")


class _NaiveBayesFalsoNB:
    """Un NB de juguete: la probabilidad satura en 1,0 para dos filas, el log-odds no."""

    def predict_proba(self, X):
        return np.array([[0.0, 1.0], [0.0, 1.0], [0.6, 0.4]])

    def predict_log_proba(self, X):
        return np.array([[-50.0, 0.0], [-20.0, 0.0], [np.log(0.6), np.log(0.4)]])


class _ConDecision:
    def decision_function(self, X):
        return np.array([2.0, -1.0])

    def predict_proba(self, X):
        raise AssertionError("no debería usar predict_proba si hay decision_function")


class _ConProbabilidad:
    def predict_proba(self, X):
        return np.array([[0.3, 0.7], [0.9, 0.1]])


def test_puntajes_elige_la_salida_correcta():
    assert list(puntajes(_ConDecision(), None)) == [2.0, -1.0]
    assert list(puntajes(_ConProbabilidad(), None)) == [0.7, 0.1]
    print("ok  puntajes usa decision_function si existe y, si no, la probabilidad de «yes»")


def test_naive_bayes_puntua_por_log_odds():
    s = puntajes(_NaiveBayesFalsoNB(), None)
    assert s[0] > s[1] > s[2] and np.isfinite(s).all()
    print("ok  en Naive Bayes el puntaje es el log-odds, que desempata donde la probabilidad satura")


def test_sin_positivos_el_recall_es_nan():
    r = en_presupuesto([0, 0, 0, 0, 0], [5, 4, 3, 2, 1], 0.4)
    assert np.isnan(r["recall_q"]) and np.isnan(r["f1_q"]) and r["precision_q"] == 0.0
    print("ok  sin «yes» en el conjunto, recall y F1 son NaN y no 0")


def test_scoring_devuelve_las_cinco_metricas():
    r = scoring(0.5)(_ConProbabilidad(), None, np.array([1, 0]))
    assert tuple(r) == METRICAS
    assert _cerca(r["auc"], 1.0) and _cerca(r["recall_q"], 1.0)
    print("ok  scoring devuelve las cinco métricas con una sola llamada a los puntajes")


def main():
    test_presupuesto_calculado_a_mano()
    test_empate_total_alcanza_exactamente_q()
    test_empate_parcial_en_el_corte()
    test_orden_perfecto()
    test_auc_en_los_extremos()
    test_cantidad_de_llamadas()
    test_puntajes_elige_la_salida_correcta()
    test_naive_bayes_puntua_por_log_odds()
    test_sin_positivos_el_recall_es_nan()
    test_scoring_devuelve_las_cinco_metricas()
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()

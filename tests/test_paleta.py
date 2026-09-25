"""Correr con: python -m tests.test_paleta

Portado del TP1 (tp1/tests/test_paleta.py): contraste WCAG, croma y distancia de color en OKLab,
en visión normal y bajo protanopía, deuteranopía y tritanopía.
"""

import numpy as np

from src.estilo import (
    COLOR_CLASE,
    COLOR_CONJUNTO,
    COLOR_LINEA_BASE,
    COLOR_MODELO,
    SUPERFICIE,
)

PISO_DELTA_E = 8.0
PISO_CONTRASTE = 3.0
PISO_CROMA = 5.0

MATRICES_DALTONISMO = {
    "protanopía": np.array([[0.1121, 0.8853, -0.0005],
                            [0.1127, 0.8897, -0.0001],
                            [0.0045, 0.0000, 1.0019]]),
    "deuteranopía": np.array([[0.2920, 0.7054, -0.0003],
                              [0.2934, 0.7089, 0.0000],
                              [-0.0202, 0.0270, 0.9915]]),
    "tritanopía": np.array([[1.0170, 0.1097, -0.1269],
                            [0.0000, 0.9578, 0.0423],
                            [0.0000, 0.3355, 0.6645]]),
}
VISIONES = [None, *MATRICES_DALTONISMO]

_A_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
_A_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])


def _a_lineal(color_hex):
    h = color_hex.lstrip("#")
    canales = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255
    return np.where(canales <= 0.04045, canales / 12.92, ((canales + 0.055) / 1.055) ** 2.4)


def _a_oklab(lineal):
    return _A_LAB @ np.cbrt(np.clip(_A_LMS @ lineal, 0, None))


def luminancia(color_hex):
    return float(np.dot([0.2126, 0.7152, 0.0722], _a_lineal(color_hex)))


def contraste(color_a, color_b):
    la, lb = luminancia(color_a), luminancia(color_b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def croma(color_hex):
    _, a, b = _a_oklab(_a_lineal(color_hex))
    return float(np.hypot(a, b) * 100)


def delta_e(color_a, color_b, daltonismo=None):
    lin_a, lin_b = _a_lineal(color_a), _a_lineal(color_b)
    if daltonismo is not None:
        m = MATRICES_DALTONISMO[daltonismo]
        lin_a, lin_b = m @ lin_a, m @ lin_b
    return float(np.linalg.norm(_a_oklab(lin_a) - _a_oklab(lin_b)) * 100)


def _separados(colores, nombre_familia):
    nombres = sorted(colores)
    for i, uno in enumerate(nombres):
        for otro in nombres[i + 1:]:
            for vision in VISIONES:
                d = delta_e(colores[uno], colores[otro], vision)
                assert d >= PISO_DELTA_E, (
                    f"{nombre_familia}: {uno} contra {otro} en {vision or 'visión normal'}: "
                    f"ΔE {d:.1f} < {PISO_DELTA_E}"
                )


def test_negro_y_blanco_dan_el_contraste_maximo():
    assert abs(contraste("#000000", "#ffffff") - 21.0) < 1e-9
    print("ok  contraste de negro contra blanco da 21:1")


def test_un_gris_no_tiene_croma():
    assert croma("#808080") < 0.1
    print("ok  un gris puro da croma ≈ 0")


def test_un_color_no_se_separa_de_si_mismo():
    for vision in VISIONES:
        assert delta_e(COLOR_CLASE["no"], COLOR_CLASE["no"], vision) < 1e-9
    print("ok  ΔE de un color contra sí mismo es 0 en las cuatro visiones")


def test_el_verde_intuitivo_falla_contra_el_naranja():
    assert delta_e("#2e8b57", COLOR_CLASE["yes"]) > PISO_DELTA_E
    assert delta_e("#2e8b57", COLOR_CLASE["yes"], "protanopía") < PISO_DELTA_E
    print("ok  el validador rechaza el verde 'obvio': colapsa con el naranja en protanopía")


def test_cada_familia_se_separa_en_las_cuatro_visiones():
    for nombre, familia in [("clase", COLOR_CLASE), ("conjunto", COLOR_CONJUNTO),
                            ("modelo", COLOR_MODELO)]:
        _separados(familia, nombre)
    print("ok  clase, conjunto y los cuatro modelos superan el piso de ΔE en las 4 visiones")


def test_la_linea_de_base_se_separa_de_todos_los_colores():
    todos = {**COLOR_CLASE, **COLOR_CONJUNTO, **COLOR_MODELO}
    for nombre, color in todos.items():
        for vision in VISIONES:
            d = delta_e(COLOR_LINEA_BASE, color, vision)
            assert d >= PISO_DELTA_E, f"línea de base contra {nombre} en {vision}: ΔE {d:.1f}"
    print("ok  el gris de la línea de base se separa de todos los colores en las 4 visiones")


def test_todos_los_colores_contrastan_contra_la_superficie():
    todos = {**COLOR_CLASE, **COLOR_CONJUNTO, **COLOR_MODELO, "linea_base": COLOR_LINEA_BASE}
    for nombre, color in todos.items():
        k = contraste(color, SUPERFICIE)
        assert k >= PISO_CONTRASTE, f"{nombre} ({color}): contraste {k:.2f} < {PISO_CONTRASTE}"
    print("ok  todos los colores superan 3:1 contra la superficie")


def test_los_colores_de_datos_tienen_croma_suficiente():
    for nombre, color in {**COLOR_CLASE, **COLOR_CONJUNTO, **COLOR_MODELO}.items():
        c = croma(color)
        assert c >= PISO_CROMA, f"{nombre} ({color}): croma {c:.1f} < {PISO_CROMA}"
    print("ok  los colores de datos superan el piso de croma")


def main():
    test_negro_y_blanco_dan_el_contraste_maximo()
    test_un_gris_no_tiene_croma()
    test_un_color_no_se_separa_de_si_mismo()
    test_el_verde_intuitivo_falla_contra_el_naranja()

    test_cada_familia_se_separa_en_las_cuatro_visiones()
    test_la_linea_de_base_se_separa_de_todos_los_colores()
    test_todos_los_colores_contrastan_contra_la_superficie()
    test_los_colores_de_datos_tienen_croma_suficiente()

    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()

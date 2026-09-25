"""Correr con: python -m tests.test_aislamiento

El test se abre una sola vez, en el punto 4 (D-04). Este test lo hace cumplir por construcción:
fuera de una lista blanca, ningún archivo .py del repo puede nombrar `cargar_test`, `cargar` (que
lee el CSV completo, test incluido), `RUTA_TEST`, `RUTA_CSV`, `DIR_PARTICION` ni el literal
`test.csv`. Se buscan los nombres, no las llamadas: un alias (`import cargar_test as f`) también
cuenta. Queda como riesgo residual armar el nombre con cadenas (`getattr(datos, "RUTA_" + "TEST")`).
"""

import re

from src.datos import RAIZ

LISTA_BLANCA = {
    "src/datos.py",                # define las tres
    "src/evidencia_particion.py",  # mide la partición misma, no elige nada del modelo (D-02, D-03)
    "src/evaluar_test.py",         # la evaluación única del punto 4 (ola 5)
    "tests/test_datos.py",         # verifica que train y test sean disjuntos y reconstruyan el dataset
    "tests/test_aislamiento.py",   # nombra los patrones para buscarlos
}

PROHIBIDOS = {
    "cargar_test": re.compile(r"\bcargar_test\b"),
    "cargar": re.compile(r"\bcargar\b"),
    "RUTA_TEST": re.compile(r"\bRUTA_TEST\b"),
    "RUTA_CSV": re.compile(r"\bRUTA_CSV\b"),
    "DIR_PARTICION": re.compile(r"\bDIR_PARTICION\b"),
    "test.csv": re.compile(r"test\.csv"),
}
IGNORAR = {".git", "__pycache__", "_build", "resumen-tp2"}


def infracciones(texto):
    """Los patrones prohibidos que aparecen en un texto, con su número de línea."""
    return [(numero, nombre)
            for numero, linea in enumerate(texto.splitlines(), start=1)
            for nombre, patron in PROHIBIDOS.items() if patron.search(linea)]


def test_el_detector_encuentra_cada_patron():
    assert infracciones("df = cargar_test()") == [(1, "cargar_test")]
    assert infracciones("from src.datos import cargar_test as f") == [(1, "cargar_test")]
    assert infracciones("from src import datos\ndf = datos.cargar()") == [(2, "cargar")]
    assert infracciones("pd.read_csv(RUTA_TEST)") == [(1, "RUTA_TEST")]
    assert infracciones("pd.read_csv(RUTA_CSV, sep=';')") == [(1, "RUTA_CSV")]
    assert infracciones('pd.read_csv("data/particion/test.csv")') == [(1, "test.csv")]
    assert infracciones("ruta = DIR_PARTICION / nombre") == [(1, "DIR_PARTICION")]
    assert infracciones("df = cargar_train()") == []
    print("ok  el detector encuentra los seis nombres prohibidos, con alias, y deja pasar cargar_train()")


def test_nadie_fuera_de_la_lista_blanca_toca_el_test():
    encontradas = []
    for ruta in sorted(RAIZ.rglob("*.py")):
        relativa = ruta.relative_to(RAIZ).as_posix()
        if relativa in LISTA_BLANCA or set(ruta.relative_to(RAIZ).parts[:-1]) & IGNORAR:
            continue
        encontradas += [f"{relativa}:{n}: {nombre}"
                        for n, nombre in infracciones(ruta.read_text(encoding="utf-8"))]
    assert not encontradas, "acceso al test fuera de la lista blanca:\n" + "\n".join(encontradas)
    print("ok  ningún archivo .py fuera de la lista blanca nombra el test ni el CSV completo")


def main():
    test_el_detector_encuentra_cada_patron()
    test_nadie_fuera_de_la_lista_blanca_toca_el_test()
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()

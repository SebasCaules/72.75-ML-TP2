"""Correr con: python -m tests.test_entregar

Arma los entregables sobre un repo de juguete en un directorio temporal, con archivos en cada
carpeta, incluidos los que nunca deben entrar al zip, y corre sobre él la prueba de clon limpio.
"""

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from src.datos import RAIZ
from src.entregar import (
    NOMBRE_CODIGO,
    NOMBRE_PRESENTACION,
    NOMBRE_ZIP,
    FaltanArchivos,
    armar,
    excluido,
    probar,
)

CARPETA = NOMBRE_CODIGO + "/"

SUITE_QUE_PASA = f'''"""Suite de juguete: corre en el clon y lee los datos que viajan en el zip."""

from src.modulo import RAIZ


def main():
    assert RAIZ.name == {NOMBRE_CODIGO!r}, RAIZ
    assert (RAIZ / "data" / "particion" / "train.csv").read_text().startswith("edad,y")
    print("ok  la suite corre en el clon y lee sus datos")
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
'''

SUITE_QUE_FALLA = '''def main():
    assert 1 + 1 == 3, "falla a propósito"
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
'''

SUITE_SIN_CIERRE = 'print("ok  el caso pasa, pero la suite no dice que terminó")\n'

README_REPO = "# README del repo\n"
README_ENTREGA = "# README de entrega\n"

# Lo que entra al zip, con la ruta que tiene dentro de la carpeta raíz. El README.md va aparte.
ENTRAN = {
    "requirements.txt": "numpy>=2.0\n",
    "data/raw/datos.csv": "edad;y\n30;no\n",
    "data/raw/descripcion.txt": "diccionario de variables\n",
    "data/particion/train.csv": "edad,y\n30,no\n",
    "data/particion/validacion.csv": "edad,y\n41,yes\n",
    "src/__init__.py": "",
    "src/modulo.py": "from pathlib import Path\n\nRAIZ = Path(__file__).resolve().parent.parent\n",
    "tests/__init__.py": "",
    "tests/test_x.py": SUITE_QUE_PASA,
    "resultados/tabla.csv": "modelo,valor\nrf,0.8\n",
    "resultados/eda/eda.html": "<p>EDA</p>\n",
    "figuras/curva.png": "png de juguete\n",
    "figuras/presentacion/portada.png": "png de juguete\n",
}

NO_ENTRAN = [
    "plan/PLAN.md", "resumen-tp2/README.md", "informe/guion.md", "informe/presentacion.tex",
    "entregables/anterior.zip", ".git/config", ".git/objects/ab/cdef",
    "src/__pycache__/modulo.cpython-312.pyc", "tests/__pycache__/test_x.cpython-312.pyc",
    "resultados/__pycache__/x.cpython-312.pyc", "src/suelto.pyc", "figuras/_build/deck.aux",
    ".DS_Store", "data/raw/.DS_Store", "resultados/eda/.DS_Store", ".gitignore", ".gitattributes",
    # Fuera de las carpetas que entran: la raíz sólo aporta README.md y requirements.txt, y
    # data/ sólo raw/ y particion/.
    "enunciado.pdf", "data/notas.txt",
    # Entra sólo con --con-bitacora (S-01).
    "DECISIONES.md",
]


def _repo(base, readme_entrega=True, presentacion=True):
    raiz = Path(base) / "repo"
    archivos = {**ENTRAN, **{r: "no entra\n" for r in NO_ENTRAN}, "README.md": README_REPO}
    if readme_entrega:
        archivos["informe/README-entrega.md"] = README_ENTREGA
    if presentacion:
        archivos["informe/presentacion.pdf"] = "%PDF-1.5 de juguete\n"
    for relativa, texto in archivos.items():
        ruta = raiz / relativa
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(texto, encoding="utf-8")
    return raiz


def _entradas(ruta_zip):
    with zipfile.ZipFile(ruta_zip) as zf:
        return zf.infolist()


def _archivos(ruta_zip):
    """Los archivos del zip, con la ruta relativa a la carpeta raíz."""
    return {i.filename.removeprefix(CARPETA) for i in _entradas(ruta_zip) if not i.is_dir()}


def _leer(ruta_zip, relativa):
    with zipfile.ZipFile(ruta_zip) as zf:
        return zf.read(CARPETA + relativa).decode("utf-8")


def _cli(*argumentos):
    return subprocess.run([sys.executable, "-m", "src.entregar", *argumentos], cwd=RAIZ,
                          capture_output=True, encoding="utf-8", timeout=300)


def test_nombres_del_grupo_7(base):
    assert NOMBRE_CODIGO == "TP2-grupo7-codigo"
    assert NOMBRE_ZIP == "TP2-grupo7-codigo.zip"
    assert NOMBRE_PRESENTACION == "TP2-grupo7-presentacion.pdf"
    print("ok  los entregables son TP2-grupo7-codigo.zip y TP2-grupo7-presentacion.pdf (N0-2)")


def test_excluido_clasifica_cada_caso(base):
    for ruta in ("plan/PLAN.md", "informe/guion.md", ".git/config", "src/__pycache__/m.pyc",
                 "resultados/__pycache__/m.py", "figuras/_build/x.aux", "resumen-tp2/a.py",
                 "entregables/x.zip", "resultados/eda/.DS_Store", ".gitignore",
                 ".gitattributes", "src/suelto.pyc", "resultados/entrega/" + NOMBRE_ZIP):
        assert excluido(ruta), ruta
    for ruta in ("src/datos.py", "resultados/eda/eda.html", "data/raw/datos.csv",
                 "figuras/presentacion/portada.png", "README.md"):
        assert not excluido(ruta), ruta
    print("ok  excluido() descarta carpetas y archivos prohibidos a cualquier profundidad")


def test_zip_con_carpeta_raiz_deflate_y_barras_normales(base):
    _, ruta_zip, _ = armar(_repo(base), base / "salida")
    assert ruta_zip.name == NOMBRE_ZIP and ruta_zip.parent == (base / "salida").resolve()
    entradas = _entradas(ruta_zip)
    assert entradas[0].filename == CARPETA and entradas[0].is_dir()
    assert all(i.filename.startswith(CARPETA) for i in entradas)
    assert not any("\\" in i.filename for i in entradas)
    assert all(i.compress_type == zipfile.ZIP_DEFLATED for i in entradas if not i.is_dir())
    with zipfile.ZipFile(ruta_zip) as zf:
        assert zf.testzip() is None
    print(f"ok  todo cuelga de {CARPETA}, con DEFLATE, barras normales y CRC correctos")


def test_entra_exactamente_lo_que_debe(base):
    _, ruta_zip, _ = armar(_repo(base), base / "salida")
    esperados = set(ENTRAN) | {"README.md"}
    archivos = _archivos(ruta_zip)
    assert archivos == esperados, f"sobran {archivos - esperados}; faltan {esperados - archivos}"
    for relativa, texto in ENTRAN.items():
        assert _leer(ruta_zip, relativa) == texto, relativa
    print(f"ok  entran los {len(esperados)} archivos esperados, con su contenido, y nada más")


def test_no_entra_nada_de_lo_excluido(base):
    _, ruta_zip, _ = armar(_repo(base), base / "salida")
    nombres = [i.filename for i in _entradas(ruta_zip)]
    for prohibido in ("__pycache__", "/plan/", "/informe/", "/.git/", "resumen-tp2",
                      "entregables", "_build", ".DS_Store", ".gitignore", ".gitattributes",
                      ".pyc", "DECISIONES.md", "enunciado.pdf", "notas.txt"):
        assert not any(prohibido in n for n in nombres), prohibido
    print("ok  quedan fuera __pycache__, plan/, informe/, .git/, .DS_Store, *.pyc y todo lo demás")


def test_readme_de_entrega_reemplaza_al_del_repo(base):
    _, con, _ = armar(_repo(base / "con"), base / "salida-con")
    _, sin, _ = armar(_repo(base / "sin", readme_entrega=False), base / "salida-sin")
    assert _leer(con, "README.md") == README_ENTREGA
    assert _leer(sin, "README.md") == README_REPO
    assert [n for n in _archivos(con) if n.endswith(".md")] == ["README.md"]
    print("ok  con informe/README-entrega.md, ése es el README.md del zip; sin él, el del repo")


def test_presentacion_se_copia_si_existe_y_si_no_se_sigue(base):
    raiz = _repo(base / "con")
    _, _, pdf = armar(raiz, base / "salida-con")
    assert pdf == (base / "salida-con").resolve() / NOMBRE_PRESENTACION
    assert pdf.read_bytes() == (raiz / "informe" / "presentacion.pdf").read_bytes()
    _, ruta_zip, pdf = armar(_repo(base / "sin", presentacion=False), base / "salida-sin")
    assert pdf is None and ruta_zip.is_file()
    assert not (base / "salida-sin" / NOMBRE_PRESENTACION).exists()
    print("ok  el PDF se copia como TP2-grupo7-presentacion.pdf; si no existe, el zip sale igual")


def test_sin_lo_obligatorio_no_se_escribe_nada(base):
    for falta in ("requirements.txt", "data/particion", "README.md"):
        raiz = _repo(base / falta.replace("/", "-"), readme_entrega=False)
        ruta = raiz / falta
        if ruta.is_dir():
            shutil.rmtree(ruta)
        else:
            ruta.unlink()
        salida = base / ("salida-" + falta.replace("/", "-"))
        try:
            armar(raiz, salida)
        except FaltanArchivos as e:
            assert falta in str(e), str(e)
        else:
            raise AssertionError(f"sin {falta} debió fallar")
        assert not salida.exists()
    print("ok  sin README, requirements.txt o data/particion/ falla antes de escribir nada")


def test_rearmar_no_mete_los_entregables_en_el_zip(base):
    raiz = _repo(base)
    salida = raiz / "resultados" / "entrega"  # a propósito, dentro de una carpeta que entra
    armar(raiz, salida)
    _, ruta_zip, _ = armar(raiz, salida)
    assert _archivos(ruta_zip) == set(ENTRAN) | {"README.md"}
    assert not list(salida.glob("*.parcial"))
    print("ok  rearmar con la salida dentro de resultados/ no mete el zip ni el PDF en el zip")


def test_bitacora_solo_con_la_opcion(base):
    raiz = _repo(base)
    _, sin, _ = armar(raiz, base / "salida-sin")
    seleccion, con, _ = armar(raiz, base / "salida-con", con_bitacora=True)
    assert "DECISIONES.md" not in _archivos(sin) and "DECISIONES.md" in _archivos(con)
    assert "GLOSARIO.md" in seleccion.ausentes
    print("ok  DECISIONES.md entra sólo con --con-bitacora; el GLOSARIO.md ausente se informa")


def test_el_clon_no_necesita_el_repo_original(base):
    raiz = _repo(base)
    _, ruta_zip, _ = armar(raiz, base / "salida")
    shutil.rmtree(raiz)
    with contextlib.redirect_stdout(io.StringIO()):
        suites = probar(ruta_zip)
    assert [(s.modulo, s.paso) for s in suites] == [("tests.test_x", True)], suites
    print("ok  con el repo original borrado, la suite del clon pasa: los datos viajan en el zip")


def test_ayuda_de_la_linea_de_comandos(base):
    r = _cli("--help")
    assert r.returncode == 0, r.stderr
    assert all(opcion in r.stdout for opcion in ("--raiz", "--salida", "--probar"))
    print("ok  python -m src.entregar --help muestra --raiz, --salida y --probar")


def test_probar_devuelve_0_si_todas_las_suites_cierran(base):
    r = _cli("--raiz", str(_repo(base)), "--salida", str(base / "salida"), "--probar")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tests.test_x" in r.stdout and "FALLA" not in r.stdout
    print("ok  --probar devuelve 0 si todas las suites del clon cierran con TODOS LOS TESTS OK")


def test_probar_devuelve_1_si_una_suite_falla(base):
    for caso, suite in (("falla", SUITE_QUE_FALLA), ("sin-cierre", SUITE_SIN_CIERRE)):
        raiz = _repo(base / caso)
        (raiz / "tests" / "test_y.py").write_text(suite, encoding="utf-8")
        r = _cli("--raiz", str(raiz), "--salida", str(base / caso / "salida"), "--probar")
        assert r.returncode == 1, (caso, r.stdout + r.stderr)
        assert "FALLA  tests.test_y" in r.stdout, r.stdout
        assert "falla 1 de 2 suites (tests.test_y)." in r.stdout, r.stdout
    print("ok  --probar devuelve 1 si una suite sale con error o no cierra con TODOS LOS TESTS OK")


def main():
    for test in (
        test_nombres_del_grupo_7,
        test_excluido_clasifica_cada_caso,
        test_zip_con_carpeta_raiz_deflate_y_barras_normales,
        test_entra_exactamente_lo_que_debe,
        test_no_entra_nada_de_lo_excluido,
        test_readme_de_entrega_reemplaza_al_del_repo,
        test_presentacion_se_copia_si_existe_y_si_no_se_sigue,
        test_sin_lo_obligatorio_no_se_escribe_nada,
        test_rearmar_no_mete_los_entregables_en_el_zip,
        test_bitacora_solo_con_la_opcion,
        test_el_clon_no_necesita_el_repo_original,
        test_ayuda_de_la_linea_de_comandos,
        test_probar_devuelve_0_si_todas_las_suites_cierran,
        test_probar_devuelve_1_si_una_suite_falla,
    ):
        with tempfile.TemporaryDirectory(prefix="test-entregar-") as tmp:
            test(Path(tmp))
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()

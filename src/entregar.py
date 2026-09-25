"""Correr con: python -m src.entregar   (opcional: --probar, --con-bitacora, --raiz, --salida)

Paso 8.2 del plan: arma en entregables/ los dos archivos que se entregan (plan, §2; grupo 7, N0-2).

- TP2-grupo7-codigo.zip, con la carpeta raíz TP2-grupo7-codigo/. Lleva el README de entrega
  (informe/README-entrega.md si existe; si no, el README.md del repo), requirements.txt,
  data/raw/, data/particion/, src/*.py, tests/*.py y, si existen, resultados/ y figuras/ con
  todas sus subcarpetas. Con --con-bitacora suma DECISIONES.md y GLOSARIO.md (S-01).
- TP2-grupo7-presentacion.pdf, la copia de informe/presentacion.pdf, si ya está compilada.

Nunca entran, a ninguna profundidad: plan/, resumen-tp2/, informe/, entregables/, .git/,
__pycache__/, _build/, .DS_Store, .gitignore, .gitattributes ni *.pyc.

Con --probar hace además la prueba de clon limpio: descomprime el zip en un directorio temporal
y corre ahí cada tests/test_*.py como `python -m tests.test_X`, con el directorio de trabajo en
la carpeta del clon y sin PYTHONPATH. Los datos viajan dentro del zip (data/), así que el clon no
necesita nada del repo original; lo único externo son los paquetes de requirements.txt. Termina
con código 1 si alguna suite no cierra con «TODOS LOS TESTS OK».
"""

import argparse
import fnmatch
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path, PurePosixPath

from src.datos import RAIZ

GRUPO = 7  # N0-2
NOMBRE_CODIGO = f"TP2-grupo{GRUPO}-codigo"
NOMBRE_ZIP = f"{NOMBRE_CODIGO}.zip"
NOMBRE_PRESENTACION = f"TP2-grupo{GRUPO}-presentacion.pdf"
PARCIAL = ".parcial"

README_ENTREGA = "informe/README-entrega.md"
README_REPO = "README.md"
PRESENTACION = "informe/presentacion.pdf"

# Rutas relativas a la raíz: «*» son los archivos de esa carpeta y «**» los de todo su árbol.
OBLIGATORIOS = ("requirements.txt", "data/raw/*", "data/particion/*", "src/*.py", "tests/*.py")
OPCIONALES = ("resultados/**", "figuras/**")
BITACORA = ("DECISIONES.md", "GLOSARIO.md")

CARPETAS_EXCLUIDAS = {"plan", "resumen-tp2", "informe", "entregables", ".git", "__pycache__",
                      "_build"}
# Los entregables mismos también: si --salida cae dentro de una carpeta que entra al zip, el zip
# no se mete a sí mismo al rearmarlo.
ARCHIVOS_EXCLUIDOS = (".DS_Store", ".gitignore", ".gitattributes", "*.pyc",
                      NOMBRE_ZIP, NOMBRE_ZIP + PARCIAL, NOMBRE_PRESENTACION)

CIERRE = "TODOS LOS TESTS OK"
TIEMPO_MAXIMO = 900  # segundos por suite en --probar: una suite colgada cuenta como falla
LINEAS_DE_COLA = 15


class FaltanArchivos(Exception):
    """Falta una pieza obligatoria del zip. Se lanza antes de escribir nada."""


@dataclass
class Seleccion:
    entradas: list  # (ruta dentro de la carpeta raíz del zip, archivo de origen), en orden
    readme: Path  # el archivo que entra como README.md
    ausentes: list = field(default_factory=list)  # opcionales que no existen


@dataclass
class Suite:
    modulo: str
    paso: bool
    segundos: float
    motivo: str = ""
    cola: str = ""


def excluido(relativa):
    """Si una ruta relativa a la raíz, con barras normales, queda siempre fuera del zip."""
    partes = PurePosixPath(relativa).parts
    return (any(parte in CARPETAS_EXCLUIDAS for parte in partes[:-1])
            or any(fnmatch.fnmatchcase(partes[-1], patron) for patron in ARCHIVOS_EXCLUIDOS))


def _expandir(raiz, patron):
    carpeta, _, final = patron.rpartition("/")
    base = raiz / carpeta if carpeta else raiz
    candidatos = base.rglob("*") if final == "**" else base.glob(final)
    return sorted(c for c in candidatos if c.is_file())


def seleccionar(raiz, con_bitacora=False):
    """Qué entra al zip y desde dónde. Lanza FaltanArchivos si falta algo obligatorio."""
    raiz = Path(raiz)
    readme = next((raiz / r for r in (README_ENTREGA, README_REPO) if (raiz / r).is_file()), None)
    entradas, faltan, ausentes = {}, [], []
    if readme is None:
        faltan.append(f"{README_ENTREGA} o {README_REPO}")
    else:
        entradas["README.md"] = readme
    grupos = [(OBLIGATORIOS, faltan), (OPCIONALES, ausentes)]
    if con_bitacora:
        grupos.append((BITACORA, ausentes))
    for patrones, vacios in grupos:
        for patron in patrones:
            relativas = [(a.relative_to(raiz).as_posix(), a) for a in _expandir(raiz, patron)]
            relativas = [(r, a) for r, a in relativas if not excluido(r)]
            if not relativas:
                vacios.append(patron)
            for relativa, archivo in relativas:
                entradas.setdefault(relativa, archivo)
    if faltan:
        raise FaltanArchivos("faltan piezas obligatorias: " + ", ".join(faltan))
    # Primero los archivos sueltos de la raíz (README.md, requirements.txt), después las carpetas.
    orden = sorted(entradas.items(), key=lambda e: ("/" in e[0], e[0]))
    return Seleccion(orden, readme, ausentes)


def _entrada_de_carpeta(nombre):
    info = zipfile.ZipInfo(nombre + "/", date_time=time.localtime()[:6])
    info.external_attr = (0o40755 << 16) | 0x10  # drwxr-xr-x y el atributo de carpeta de MS-DOS
    return info


def armar_zip(entradas, destino):
    """Escribe el zip con la carpeta raíz NOMBRE_CODIGO/, lo verifica y devuelve su ruta.

    Se escribe primero a un archivo .parcial y se renombra al final: un corte a mitad de camino
    no deja un zip roto con el nombre del entregable.
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + PARCIAL)
    carpetas = set()
    with zipfile.ZipFile(parcial, "w", compression=zipfile.ZIP_DEFLATED,
                         strict_timestamps=False) as zf:
        for relativa, origen in entradas:
            nombre = f"{NOMBRE_CODIGO}/{relativa}"
            for carpeta in reversed(list(PurePosixPath(nombre).parents)[:-1]):
                if carpeta not in carpetas:
                    carpetas.add(carpeta)
                    zf.writestr(_entrada_de_carpeta(carpeta.as_posix()), b"")
            zf.write(origen, nombre)
    with zipfile.ZipFile(parcial) as zf:
        roto = zf.testzip()
    if roto is not None:
        parcial.unlink()
        raise zipfile.BadZipFile(f"el CRC de {roto} no coincide al releer el zip")
    parcial.replace(destino)
    return destino


def copiar_presentacion(raiz, salida):
    """Copia informe/presentacion.pdf a la salida; None si todavía no está compilada."""
    origen = Path(raiz) / PRESENTACION
    if not origen.is_file():
        return None
    destino = Path(salida) / NOMBRE_PRESENTACION
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origen, destino)
    return destino


def fuentes_posteriores(raiz):
    """Los .tex de informe/ posteriores al PDF: si hay alguno, el PDF puede estar desactualizado."""
    pdf = Path(raiz) / PRESENTACION
    compilado = pdf.stat().st_mtime
    return sorted(t.name for t in pdf.parent.glob("*.tex") if t.stat().st_mtime > compilado)


def armar(raiz=RAIZ, salida=None, con_bitacora=False):
    """Arma los dos entregables. Devuelve (selección, ruta del zip, ruta del PDF o None)."""
    raiz = Path(raiz).resolve()
    salida = Path(salida).resolve() if salida is not None else raiz / "entregables"
    seleccion = seleccionar(raiz, con_bitacora)
    ruta_zip = armar_zip(seleccion.entradas, salida / NOMBRE_ZIP)
    return seleccion, ruta_zip, copiar_presentacion(raiz, salida)


def listado(ruta_zip):
    """Los archivos del zip, sin las entradas de carpeta: (nombre, bytes, bytes comprimidos)."""
    with zipfile.ZipFile(ruta_zip) as zf:
        return [(i.filename, i.file_size, i.compress_size) for i in zf.infolist()
                if not i.is_dir()]


def paquetes(requirements):
    """Cada requisito de requirements.txt con la versión instalada en este intérprete, o None."""
    filas = []
    for linea in Path(requirements).read_text(encoding="utf-8").splitlines():
        linea = linea.split("#")[0].strip()
        if not linea or linea.startswith("-"):
            continue
        nombre = re.split(r"[\s<>=!~;\[]", linea, maxsplit=1)[0]
        try:
            filas.append((linea, metadata.version(nombre)))
        except metadata.PackageNotFoundError:
            filas.append((linea, None))
    return filas


def correr_suite(clon, modulo, tiempo_maximo=TIEMPO_MAXIMO):
    """Corre `python -m <modulo>` en el clon. Pasa si sale con 0 y su última línea es CIERRE."""
    entorno = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    entorno["PYTHONIOENCODING"] = "utf-8"
    inicio = time.perf_counter()
    try:
        r = subprocess.run([sys.executable or "python3", "-m", modulo], cwd=clon, env=entorno,
                           capture_output=True, encoding="utf-8", errors="replace",
                           timeout=tiempo_maximo)
    except subprocess.TimeoutExpired:
        return Suite(modulo, False, time.perf_counter() - inicio,
                     f"no terminó en {tiempo_maximo} s")
    segundos = time.perf_counter() - inicio
    ultima = next((linea.strip() for linea in reversed(r.stdout.splitlines()) if linea.strip()),
                  "")
    if r.returncode == 0 and ultima == CIERRE:
        return Suite(modulo, True, segundos)
    motivo = (f"código de salida {r.returncode}" if r.returncode != 0
              else f"termina en {ultima!r} y no en «{CIERRE}»")
    cola = (r.stderr if r.stderr.strip() else r.stdout).strip().splitlines()[-LINEAS_DE_COLA:]
    return Suite(modulo, False, segundos, motivo, "\n".join(cola))


def probar(ruta_zip, tiempo_maximo=TIEMPO_MAXIMO):
    """La prueba de clon limpio: descomprime el zip en un directorio temporal y corre ahí cada
    tests/test_*.py, en orden. Devuelve la lista de Suite; muestra cada una al terminar."""
    with tempfile.TemporaryDirectory(prefix="clon-tp2-", ignore_cleanup_errors=True) as tmp:
        with zipfile.ZipFile(ruta_zip) as zf:
            zf.extractall(tmp)
        clon = Path(tmp) / NOMBRE_CODIGO
        _presentar_clon(clon)
        suites = []
        for archivo in sorted((clon / "tests").glob("test_*.py")):
            suites.append(correr_suite(clon, f"tests.{archivo.stem}", tiempo_maximo))
            _mostrar_suite(suites[-1])
    return suites


def _legible(n):
    """Bytes en texto: 950 -> '950 B', 73385 -> '71,7 KB', 5834924 -> '5,6 MB'."""
    if n < 1024:
        return f"{n} B"
    for unidad in ("KB", "MB", "GB"):
        n /= 1024
        if n < 1024 or unidad == "GB":
            return f"{n:.1f} {unidad}".replace(".", ",")


def _entero(n):
    return f"{n:,}".replace(",", ".")


def _cuenta(n, sustantivo):
    return f"{n} {sustantivo}{'' if n == 1 else 's'}"


def _mostrar(ruta, raiz):
    try:
        return Path(ruta).relative_to(raiz).as_posix()
    except ValueError:
        return str(ruta)


def mostrar_zip(ruta_zip, raiz):
    archivos = listado(ruta_zip)
    print(f"Zip: {_mostrar(ruta_zip, raiz)}  (carpeta raíz {NOMBRE_CODIGO}/, compresión DEFLATE)")
    print(f"  {'tamaño':>10}  {'comprimido':>10}  archivo")
    for nombre, peso, comprimido in archivos:
        print(f"  {_legible(peso):>10}  {_legible(comprimido):>10}  {nombre}")
    total = sum(peso for _, peso, _ in archivos)
    bytes_zip = ruta_zip.stat().st_size
    print(f"Total: {_cuenta(len(archivos), 'archivo')}, {_legible(total)} sin comprimir; "
          f"el zip ocupa {_legible(bytes_zip)} ({_entero(bytes_zip)} bytes).")


def _presentar_clon(clon):
    raw, particion = (_cuenta(sum(a.is_file() for a in (clon / "data" / c).rglob("*")), "archivo")
                      for c in ("raw", "particion"))
    interprete = Path(sys.executable).name
    print(f"Prueba de clon limpio en {clon}")
    print(f"  Los datos viajan dentro del zip: data/raw/ con {raw} y data/particion/ con "
          f"{particion}.")
    print(f"  Cada suite corre como «{interprete} -m tests.test_X», con el directorio de trabajo "
          "en el clon\n  y sin PYTHONPATH: no lee nada del repo original.")
    print("  Lo único externo son los paquetes de requirements.txt; se usan los de este intérprete"
          f"\n  ({interprete} {platform.python_version()}):")
    for requisito, version in paquetes(clon / "requirements.txt"):
        print(f"    {requisito:<24} {version or 'NO INSTALADO'}")
    print(flush=True)


def _mostrar_suite(suite):
    estado = "ok   " if suite.paso else "FALLA"
    segundos = f"{suite.segundos:.1f} s".replace(".", ",")
    print(f"  {estado}  {suite.modulo:<30} {segundos:>8}"
          + ("" if suite.paso else f"  ({suite.motivo})"), flush=True)
    for linea in suite.cola.splitlines():
        print(f"         | {linea}")


def _parser():
    parser = argparse.ArgumentParser(prog="python -m src.entregar", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raiz", type=Path, default=RAIZ,
                        help="raíz del repo que se empaqueta (por defecto, este repo)")
    parser.add_argument("--salida", type=Path,
                        help="carpeta de los entregables (por defecto, <raiz>/entregables)")
    parser.add_argument("--probar", action="store_true",
                        help="prueba de clon limpio: descomprime el zip en un directorio temporal "
                             "y corre ahí cada tests/test_*.py")
    parser.add_argument("--con-bitacora", action="store_true",
                        help="suma DECISIONES.md y GLOSARIO.md al zip (S-01, se decide en "
                             "la ola 8)")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    raiz = args.raiz.resolve()
    if not raiz.is_dir():
        print(f"No existe la carpeta {raiz}.")
        return 1
    try:
        seleccion, ruta_zip, pdf = armar(raiz, args.salida, args.con_bitacora)
    except FaltanArchivos as e:
        print(f"No se arma el zip: {e}.")
        return 1

    readme = _mostrar(seleccion.readme, raiz)
    print(f"README.md del zip: {readme}" + ("" if readme == README_ENTREGA
                                            else f" (todavía no existe {README_ENTREGA})"))
    for patron in seleccion.ausentes:
        print(f"No existe {patron.removesuffix('**')}: no entra al zip.")
    mostrar_zip(ruta_zip, raiz)
    print()
    if pdf is None:
        print(f"Presentación: todavía no existe {PRESENTACION}; se sigue sin ella.")
    else:
        print(f"Presentación: {PRESENTACION} -> {_mostrar(pdf, raiz)} "
              f"({_legible(pdf.stat().st_size)})")
        posteriores = fuentes_posteriores(raiz)
        if posteriores:
            print(f"  Aviso: {', '.join(posteriores)} cambiaron después de compilar el PDF; "
                  "conviene recompilarlo antes de entregar.")
    if not args.probar:
        return 0

    print()
    suites = probar(ruta_zip)
    fallan = [s.modulo for s in suites if not s.paso]
    if not suites:
        print("Clon limpio: el zip no trae ninguna suite tests/test_*.py; no hay nada que probar.")
        return 1
    if fallan:
        print(f"Clon limpio: {'falla' if len(fallan) == 1 else 'fallan'} {len(fallan)} de "
              f"{_cuenta(len(suites), 'suite')} ({', '.join(fallan)}).")
        return 1
    todas = "la suite cierra" if len(suites) == 1 else f"las {len(suites)} suites cierran"
    print(f"Clon limpio: {todas} con «{CIERRE}».")
    return 0


if __name__ == "__main__":
    sys.exit(main())

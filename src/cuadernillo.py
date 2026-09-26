r"""Correr con: python3 -m src.cuadernillo   (opcional: --salida, --dpi, --conservar DIR)

Paso 7.5 del plan: el cuadernillo de ensayo, informe/slides-y-guion.pdf, generado desde el fuente
con un solo comando. En el TP1 se armó a mano el día de la defensa (plan, §1).

Lee, sin modificarlos:
- informe/presentacion.pdf, el deck compilado. De cada frame se toma su última página, que es el
  estado final con todos los overlays revelados, y se la renderiza con pdftoppm (200 dpi: 1260 ×
  709 px, como en el cuadernillo del TP1).
- informe/guion.md. De cada encabezado «### N · título — inicio → fin (~P palabras)» salen la
  ventana de reloj y las palabras; debajo, quién habla («**Habla: …**»), las indicaciones de
  ensayo (los párrafos sueltos), el texto que se dice (las líneas que empiezan con «>», con sus
  [→]) y el bloque «#### A aclarar». También «Los 10 minutos de preguntas» y el «Reloj de ensayo».
- informe/resultados-test.tex, sólo para saber si el test ya se evaluó (\testpendientetrue): en
  las slides con dos variantes habladas va primero la que corresponde a la imagen, con el párrafo
  del guion que la presenta completo (dice hasta cuándo vale y qué muestra la imagen), y la otra
  debajo, en gris.

CÓMO SE ELIGE LA PÁGINA FINAL DE CADA FRAME. Cada página del deck lleva al pie su número de slide:
«n/N» en la charla y «Rn/M» en el respaldo (\indicediapo en presentacion.tex, también en la
portada). pdftotext -layout extrae el texto de todas las páginas en una sola llamada; el pie es la
última línea de cada una, y las páginas consecutivas con el mismo rótulo son los overlays de un
mismo frame: la última es la final. Se lee del PDF mismo, así que nunca queda desfasado de lo que
se renderiza. Si existe informe/_build/presentacion.nav (latexmk lo deja ahí hasta que se borra
_build/), sus \beamer@framepages sirven de control cruzado, sólo si ese .nav es de un PDF con la
misma cantidad de páginas. Contar frames y overlays en el .tex no se usa: exigiría reimplementar
las especificaciones de overlay de beamer (<2->, \only, \visible, `visible on`). Límite conocido:
un frame con [noframenumbering] repetiría el número del anterior y se fundiría con él; el deck no
usa esa opción.

El PDF se compone con LuaLaTeX, con la tipografía y la paleta del deck (Fira Sans; src/estilo.py),
en un directorio temporal: el .log se lee, y sus avisos se muestran, antes de borrarlo. LuaLaTeX
corre hasta que el .log deja de pedir otra pasada («Rerun to get…», «Label(s) may have changed»):
dos veces, si todo converge, y tres como máximo. Con --conservar DIR, el .tex, las imágenes y los
auxiliares quedan en DIR.

Contenido: una portada con el grupo, la fecha de la defensa y el reloj total; el reloj de ensayo
del guion (tabla, puntos de control y orden de recorte); una página por slide, con la imagen en su
estado final, la ventana de reloj, quién habla, las indicaciones en gris cursiva, el texto hablado
con sus [→] y «A aclarar» (sin justificar); el respaldo; y el banco de preguntas al final, a dos
columnas, con cada pregunta en una pieza que no se corta entre columnas ni entre páginas.

Cada slide ocupa exactamente una página. La imagen va a todo el ancho de la caja de texto mientras
el texto le deje lugar, aunque para eso haya que cerrar hasta 8 pt las separaciones; si no, se
achica exactamente lo que falta (el alto de la página se mide, no se estima) y, por debajo del
62 % de su alto, se achica también la letra. La salida dice qué slides quedaron con la imagen más
angosta y a qué porcentaje del ancho; una página que ni así entra sale en el .log como «Overfull
\vbox». Si el guion y el deck no coinciden (una slide sin texto o sin imagen, [→] que no son los
pasos del PDF, el mapa de páginas del guion desactualizado), se avisa en la salida y en la página
afectada.
"""

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from src.datos import RAIZ
from src.entregar import GRUPO
from src.estilo import AZUL, AZUL_OSCURO, GRIS_REFERENCIA, NARANJA, REJILLA, TINTA, TINTA_SECUNDARIA

INFORME = RAIZ / "informe"
PRESENTACION = INFORME / "presentacion.pdf"
GUION = INFORME / "guion.md"
RESULTADOS_TEST = INFORME / "resultados-test.tex"
NAV = INFORME / "_build" / "presentacion.nav"
SALIDA = INFORME / "slides-y-guion.pdf"

DPI = 200
MOTOR = "lualatex"
CORRIDAS_MAXIMAS = 3
HERRAMIENTAS = ("pdftotext", "pdftoppm", "pdfinfo")
RITMO = 2.7        # palabras por segundo, si el guion no declara el suyo («Convenciones»)
TOPE = "9:45"      # tope hablado del plan (§5, ola 7; gate G6)
MATERIA = "72.75 Aprendizaje Automático — ITBA — 2026 Q2"
POR_PAGINA_RESPALDO = 2   # a todo el ancho: dos slides de 16:9 entran en una página A4


class ErrorCuadernillo(Exception):
    """Algo impide generar el cuadernillo: falta un archivo o una herramienta, o LaTeX falló."""


# =================================================================================================
# El deck: qué página es el estado final de cada frame
# =================================================================================================

@dataclass(frozen=True)
class Rotulo:
    """El número de slide que el deck imprime al pie: «7/20» en la charla, «R2/4» en el respaldo."""
    respaldo: bool
    numero: int
    total: int

    def __str__(self):
        return f"{'R' if self.respaldo else ''}{self.numero}/{self.total}"


@dataclass
class Frame:
    rotulo: Rotulo | None       # None: la página no tiene número al pie
    paginas: list[int]          # páginas del PDF, desde 1; una por overlay

    @property
    def final(self):
        return self.paginas[-1]

    @property
    def pasos(self):
        """Las pulsaciones del frame: una por overlay después del primero."""
        return len(self.paginas) - 1

    @property
    def nombre(self):
        """«7» en la charla y «R2» en el respaldo, como el mapa de páginas del guion."""
        if self.rotulo is None:
            return f"p{self.paginas[0]}"
        return f"R{self.rotulo.numero}" if self.rotulo.respaldo else str(self.rotulo.numero)


# El pie cierra la última línea de la página: «EDA          7/20». Anclado al final de la línea,
# un «80/20» del cuerpo de la slide no se confunde con el número.
PIE = re.compile(r"(?:^|\s)(R?)(\d+)\s*/\s*(\d+)\s*$")


def rotulo_de_pie(texto):
    """El rótulo «n/N» o «Rn/M» de la última línea con texto de una página, o None."""
    lineas = [linea for linea in texto.splitlines() if linea.strip()]
    if not lineas:
        return None
    m = PIE.search(lineas[-1])
    if not m:
        return None
    return Rotulo(bool(m.group(1)), int(m.group(2)), int(m.group(3)))


def frames_de_rotulos(rotulos):
    """Agrupa las páginas consecutivas con el mismo rótulo: cada grupo es un frame con sus overlays.

    Una página sin rótulo queda como un frame propio, y revisar_frames lo avisa.
    """
    frames = []
    for pagina, rotulo in enumerate(rotulos, start=1):
        if frames and rotulo is not None and frames[-1].rotulo == rotulo:
            frames[-1].paginas.append(pagina)
        else:
            frames.append(Frame(rotulo, [pagina]))
    return frames


def revisar_frames(frames):
    """Avisos si la numeración del pie no es 1..N en la charla y R1..RM en el respaldo."""
    avisos = []
    for parte, es_respaldo, prefijo in (("la charla", False, ""), ("el respaldo", True, "R")):
        propios = [f for f in frames if f.rotulo is not None and f.rotulo.respaldo == es_respaldo]
        numeros = [f.rotulo.numero for f in propios]
        if numeros != list(range(1, len(numeros) + 1)):
            avisos.append(f"En {parte}, los números al pie no van de {prefijo}1 en adelante sin "
                          f"saltos ni repeticiones: {', '.join(prefijo + str(n) for n in numeros)}.")
        totales = sorted({f.rotulo.total for f in propios})
        if totales and totales != [len(numeros)]:
            avisos.append(f"En {parte} hay {len(numeros)} frames, pero el pie dice «de "
                          f"{' o de '.join(map(str, totales))}».")
    sueltas = [f.paginas[0] for f in frames if f.rotulo is None]
    if sueltas:
        avisos.append("Páginas del PDF sin número de slide al pie, que quedan fuera del "
                      "cuadernillo: " + ", ".join(map(str, sueltas)) + ".")
    return avisos


FRAMEPAGES = re.compile(r"\\beamer@framepages\s*\{(\d+)\}\s*\{(\d+)\}")


def rangos_del_nav(texto):
    """Las páginas (primera, última) de cada frame según el .nav de beamer, en orden."""
    return [(int(a), int(b)) for a, b in FRAMEPAGES.findall(texto)]


def comparar_con_nav(frames, rangos, paginas_pdf):
    """Control cruzado con el .nav: avisos, o [] si coincide o si el .nav es de otro PDF."""
    if not rangos or rangos[-1][1] != paginas_pdf:
        return []   # quedó de otra compilación: no dice nada de este PDF
    propios = [(f.paginas[0], f.final) for f in frames]
    if propios == rangos:
        return []
    distintos = [f"frame {i}: {a}–{b} en el .nav y {c}–{d} por el pie"
                 for i, ((a, b), (c, d)) in enumerate(zip(rangos, propios), start=1)
                 if (a, b) != (c, d)]
    if len(rangos) != len(propios):
        distintos.append(f"{len(rangos)} frames en el .nav y {len(propios)} por el pie")
    return ["El .nav de informe/_build no coincide con los números al pie: "
            + "; ".join(distintos[:4]) + ("; …" if len(distintos) > 4 else "") + "."]


def _correr(argumentos, **kwargs):
    try:
        return subprocess.run(argumentos, capture_output=True, text=True, check=True,
                              encoding="utf-8", errors="replace", **kwargs)
    except FileNotFoundError as e:
        raise ErrorCuadernillo(f"No se encuentra {argumentos[0]}: hace falta instalarlo.") from e
    except subprocess.CalledProcessError as e:
        salida = (e.stderr or e.stdout or "").strip()
        raise ErrorCuadernillo(f"{argumentos[0]} terminó con código {e.returncode}: "
                               f"{salida[-400:]}") from e


def metadatos_pdf(pdf):
    """Los campos de pdfinfo (Title, Author, Pages, Page size…) como diccionario de texto."""
    datos = {}
    for linea in _correr(["pdfinfo", "-enc", "UTF-8", str(pdf)]).stdout.splitlines():
        clave, separador, valor = linea.partition(":")
        if separador:
            datos[clave.strip()] = valor.strip()
    return datos


def textos_por_pagina(pdf, paginas):
    """El texto de cada página con pdftotext -layout: una sola llamada, cortada en los saltos \\f."""
    salida = _correr(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"]).stdout
    textos = salida.split("\f")[:paginas]
    return textos + [""] * (paginas - len(textos))


def frames_del_pdf(pdf, paginas=None):
    """(frames, páginas del PDF, avisos) de un deck compilado con beamer."""
    if paginas is None:
        paginas = int(metadatos_pdf(pdf).get("Pages", 0))
    if paginas == 0:
        raise ErrorCuadernillo(f"{pdf} no tiene páginas.")
    frames = frames_de_rotulos([rotulo_de_pie(t) for t in textos_por_pagina(pdf, paginas)])
    return frames, paginas, revisar_frames(frames)


def renderizar(pdf, pagina, destino, dpi=DPI):
    """Una página del PDF como PNG, con pdftoppm. `destino` es la ruta sin la extensión."""
    destino = Path(destino)
    _correr(["pdftoppm", "-f", str(pagina), "-l", str(pagina), "-r", str(dpi), "-png",
             "-singlefile", str(pdf), str(destino)])
    return destino.with_name(destino.name + ".png")


def renderizar_varias(pdf, pedidos, dpi=DPI):
    """Renderiza {destino: página} en paralelo: cada pdftoppm es un proceso aparte."""
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as ejecutor:
        futuros = [ejecutor.submit(renderizar, pdf, pagina, destino, dpi)
                   for destino, pagina in pedidos.items()]
        return [f.result() for f in futuros]


def tamano_png(ruta):
    """(ancho, alto) en píxeles, leídos de la cabecera IHDR del PNG."""
    cabecera = Path(ruta).read_bytes()[:24]
    if cabecera[:8] != b"\x89PNG\r\n\x1a\n":
        raise ErrorCuadernillo(f"{ruta} no es un PNG.")
    return int.from_bytes(cabecera[16:20], "big"), int.from_bytes(cabecera[20:24], "big")


def estado_del_test(ruta=RESULTADOS_TEST):
    """True si el test sigue pendiente, False si ya se evaluó, None si no se sabe."""
    ruta = Path(ruta)
    if not ruta.exists():
        return None
    texto = ruta.read_text(encoding="utf-8")
    if re.search(r"\\testpendientetrue\b", texto):
        return True
    if re.search(r"\\testpendientefalse\b", texto):
        return False
    return None


# =================================================================================================
# El guion: markdown por bloques
# =================================================================================================

def secciones(texto, nivel):
    """Parte un markdown en sus secciones de un nivel: [(título, cuerpo)], con (None, preámbulo)."""
    patron = re.compile(rf"^{'#' * nivel}(?!#)\s+(.+?)\s*$")
    partes, titulo, lineas = [], None, []
    for linea in texto.splitlines():
        m = patron.match(linea)
        if m:
            partes.append((titulo, "\n".join(lineas)))
            titulo, lineas = m.group(1), []
        else:
            lineas.append(linea)
    partes.append((titulo, "\n".join(lineas)))
    return [(t, c) for t, c in partes if t is not None or c.strip()]


@dataclass
class Bloque:
    tipo: str      # parrafo, cita, lista, enumeracion, tabla o titulo
    lineas: list   # parrafo, cita y tabla: sus líneas; lista: un texto por ítem;
    #                enumeracion: (número, texto) por ítem; titulo: [nivel, texto]

    @property
    def texto(self):
        return " ".join(linea.strip() for linea in self.lineas if linea.strip())


ENCABEZADO = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
REGLA = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
ITEM = re.compile(r"^[-*+]\s+(.*)$")
ITEM_NUMERADO = re.compile(r"^(\d+)[.)]\s+(.*)$")


def bloques_md(texto):
    """Los bloques de un markdown sencillo: párrafos, citas «>», listas, enumeraciones y tablas.

    Una línea en blanco cierra el bloque. Una línea sin marca después de un ítem lo continúa (el
    guion sangra las continuaciones), y una sin «>» después de una cita también, como en
    CommonMark. Las reglas «---» se descartan.
    """
    bloques, actual = [], None

    def cerrar():
        nonlocal actual
        if actual is not None:
            bloques.append(actual)
        actual = None

    def abrir(tipo):
        nonlocal actual
        if actual is None or actual.tipo != tipo:
            cerrar()
            actual = Bloque(tipo, [])
        return actual

    for crudo in texto.splitlines():
        linea = crudo.rstrip()
        if not linea.strip():
            cerrar()
        elif (m := ENCABEZADO.match(linea)):
            cerrar()
            bloques.append(Bloque("titulo", [len(m.group(1)), m.group(2)]))
        elif REGLA.match(linea):
            cerrar()
        elif linea.startswith(">"):
            abrir("cita").lineas.append(re.sub(r"^>\s?", "", linea))
        elif linea.startswith("|"):
            abrir("tabla").lineas.append(linea)
        elif (m := ITEM.match(linea)):
            abrir("lista").lineas.append(m.group(1))
        elif (m := ITEM_NUMERADO.match(linea)):
            abrir("enumeracion").lineas.append((m.group(1), m.group(2)))
        elif actual is not None and actual.tipo == "lista":
            actual.lineas[-1] += " " + linea.strip()
        elif actual is not None and actual.tipo == "enumeracion":
            numero, previo = actual.lineas[-1]
            actual.lineas[-1] = (numero, previo + " " + linea.strip())
        elif actual is not None and actual.tipo in ("cita", "parrafo"):
            actual.lineas.append(linea)
        else:
            abrir("parrafo").lineas.append(linea)
    cerrar()
    return bloques


def _normalizar(texto):
    descompuesto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in descompuesto if not unicodedata.combining(c)).lower().strip()


# =================================================================================================
# El guion: slides, preguntas y reloj
# =================================================================================================

@dataclass
class Dicho:
    """Un texto hablado: los tramos entre [→]; el primero se dice antes de la primera pulsación."""
    segmentos: list[str]
    etiqueta: str | None = None   # el párrafo que presenta una variante («Mientras el test…»)

    @property
    def avances(self):
        return len(self.segmentos) - 1


@dataclass
class SlideGuion:
    numero: int
    titulo: str
    inicio: str | None = None
    fin: str | None = None
    palabras: int | None = None
    habla: str | None = None
    bloques: list = field(default_factory=list)   # ("nota", md), ("dicho", Dicho), ("bloque", Bloque)
    titulo_aclarar: str | None = None
    aclarar: str = ""                            # el markdown debajo de «#### A aclarar»

    @property
    def dichos(self):
        return [b for tipo, b in self.bloques if tipo == "dicho"]


@dataclass
class Pregunta:
    numero: int
    pregunta: str
    etiqueta: str | None      # «slide 14 · Sebastián»: a qué slide volver y quién responde primero
    respuesta: str


@dataclass
class GrupoPreguntas:
    titulo: str | None
    elementos: list           # Pregunta o Bloque, en el orden del guion


@dataclass
class Guion:
    titulo: str = ""
    slides: list[SlideGuion] = field(default_factory=list)
    preguntas_titulo: str | None = None
    preguntas_intro: list[Bloque] = field(default_factory=list)
    grupos: list[GrupoPreguntas] = field(default_factory=list)
    reloj_titulo: str | None = None
    reloj: list[Bloque] = field(default_factory=list)
    ritmo: float = RITMO
    controles: dict = field(default_factory=dict)        # slide → reloj al terminarla
    mapa_paginas: dict = field(default_factory=dict)     # «7» o «R2» → página, según el guion
    avisos: list[str] = field(default_factory=list)

    @property
    def palabras(self):
        return sum(s.palabras or 0 for s in self.slides)

    @property
    def duracion(self):
        """El reloj al final de la última ventana, en segundos, o None si no hay ventanas."""
        fines = [a_segundos(s.fin) for s in self.slides if s.fin]
        return max(fines) if fines else None

    @property
    def preguntas(self):
        return [e for g in self.grupos for e in g.elementos if isinstance(e, Pregunta)]


ENCABEZADO_SLIDE = re.compile(
    r"^(?P<numero>\d+)\s*·\s*(?P<titulo>.+?)"
    r"(?:\s+—\s+(?P<inicio>\d+:\d{2})\s*→\s*(?P<fin>\d+:\d{2}))?"
    r"(?:\s*\(\s*~?\s*(?P<palabras>\d[\d\s.]*?)\s+palabras\s*\))?\s*$")
HABLA = re.compile(r"^\*\*Habla:\s*(?P<quien>[^*]+?)\.?\s*\*\*\s*(?P<resto>.*)$", re.S)
MARCA_AVANCE = re.compile(r"\s*(?:\*\*)?\[→\](?:\*\*)?\s*")
PREGUNTA = re.compile(r"^\*\*(?P<numero>\d+)\.\s*(?P<pregunta>.+?)\*\*\s*"
                      r"(?:\[(?P<etiqueta>[^\]]*)\])?\s*(?:—\s*)?(?P<respuesta>.*)$", re.S)
RITMO_GUION = re.compile(r"(\d+(?:,\d+)?)\s*palabras por segundo")
CONTROL = re.compile(r"Al terminar la \*\*slide (\d+)\*\* tienen que haber pasado "
                     r"\*\*(\d+:\d{2})\*\*")
PAR_DE_PAGINAS = re.compile(r"\b(R?\d+)\s*→\s*(\d+)\b")
RESPALDO_CITADO = re.compile(r"\bR(\d+)\b")


def a_segundos(reloj):
    minutos, segundos = reloj.split(":")
    return int(minutos) * 60 + int(segundos)


def a_reloj(segundos):
    s = int(round(segundos))
    return f"{s // 60}:{s % 60:02d}"


def parsear_slide(titulo, cuerpo):
    """Una sección «### N · …» del guion, o None si el título no es el de una slide."""
    m = ENCABEZADO_SLIDE.match(titulo.strip())
    if not m:
        return None
    palabras = m["palabras"]
    slide = SlideGuion(int(m["numero"]), m["titulo"].strip(), m["inicio"], m["fin"],
                       int(re.sub(r"\D", "", palabras)) if palabras else None)
    partes = secciones(cuerpo, 4)
    previo = "\n".join(c for t, c in partes if t is None)
    con_titulo = [(t, c) for t, c in partes if t is not None]
    if con_titulo:
        slide.titulo_aclarar, slide.aclarar = con_titulo[0]
        for t, c in con_titulo[1:]:          # un segundo «####» queda dentro del mismo bloque
            slide.aclarar += f"\n\n#### {t}\n{c}"
    for bloque in bloques_md(previo):
        if bloque.tipo == "cita":
            segmentos = [s.strip() for s in MARCA_AVANCE.split(bloque.texto)]
            etiqueta = None
            if slide.dichos and slide.bloques and slide.bloques[-1][0] == "nota":
                etiqueta = slide.bloques.pop()[1]   # el párrafo que presenta la variante
            slide.bloques.append(("dicho", Dicho(segmentos, etiqueta)))
        elif bloque.tipo == "parrafo":
            texto = bloque.texto
            if slide.habla is None and (h := HABLA.match(texto)):
                slide.habla = h["quien"].strip().rstrip(".")
                texto = h["resto"].strip()
                if not texto:
                    continue
            slide.bloques.append(("nota", texto))
        else:
            slide.bloques.append(("bloque", bloque))
    return slide


def parsear_preguntas(cuerpo):
    """(intro, grupos) de «Los 10 minutos de preguntas»: cada «### …» es un grupo."""
    intro, grupos = [], []
    for titulo, texto in secciones(cuerpo, 3):
        bloques = bloques_md(texto)
        if titulo is None:
            intro = bloques
            continue
        elementos = []
        for b in bloques:
            m = PREGUNTA.match(b.texto) if b.tipo == "parrafo" else None
            if m:
                elementos.append(Pregunta(int(m["numero"]), m["pregunta"].strip(),
                                          m["etiqueta"].strip() if m["etiqueta"] else None,
                                          m["respuesta"].strip()))
            else:
                elementos.append(b)
        grupos.append(GrupoPreguntas(titulo, elementos))
    return intro, grupos


def parsear_guion(texto):
    """El guion completo. Lo que no encuentra queda vacío y se anota en `avisos`."""
    guion = Guion()
    guion.titulo = next((t for t, _ in secciones(texto, 1) if t), "")
    if (m := RITMO_GUION.search(texto)):
        guion.ritmo = float(m.group(1).replace(",", "."))
    for titulo, cuerpo in secciones(texto, 2):
        if titulo is None:
            continue
        clave = _normalizar(titulo)
        if clave.startswith("el guion"):
            for t, c in secciones(cuerpo, 3):
                if t is None:
                    continue
                slide = parsear_slide(t, c)
                if slide is None:
                    guion.avisos.append(f"Sección del guion que no es una slide: «### {t}».")
                else:
                    guion.slides.append(slide)
        elif "preguntas" in clave and guion.preguntas_titulo is None:
            guion.preguntas_titulo = titulo
            guion.preguntas_intro, guion.grupos = parsear_preguntas(cuerpo)
        elif clave.startswith("reloj"):
            guion.reloj_titulo = titulo
            guion.reloj = bloques_md(cuerpo)
            guion.controles = {int(n): r for n, r in CONTROL.findall(cuerpo)}
    for b in guion.preguntas_intro:
        if "Para volver a una slide" in b.texto:
            guion.mapa_paginas = {k: int(v) for k, v in PAR_DE_PAGINAS.findall(b.texto)}
    if not guion.slides:
        guion.avisos.append("El guion no tiene la sección «## El guion» con encabezados "
                            "«### N · …».")
    numeros = [s.numero for s in guion.slides]
    if len(set(numeros)) != len(numeros):
        guion.avisos.append("El guion repite números de slide: " + ", ".join(map(str, numeros)) + ".")
    if guion.preguntas_titulo is None:
        guion.avisos.append("El guion no tiene la sección de preguntas.")
    return guion


def variantes(dichos, test_pendiente):
    """(principal, otras): la variante hablada que corresponde a la imagen va primero.

    Con el test pendiente, la imagen es la del test cerrado, así que la principal es la variante
    cuya etiqueta lo dice («Mientras el test siga cerrado…»). Si no, la primera, como en el guion.
    """
    if not dichos:
        return None, []
    indice = 0
    if test_pendiente:
        indice = next((i for i, d in enumerate(dichos)
                       if d.etiqueta and "cerrado" in _normalizar(d.etiqueta)), 0)
    return dichos[indice], [d for i, d in enumerate(dichos) if i != indice]


# =================================================================================================
# Markdown → LaTeX
# =================================================================================================

_ESCAPES = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "%": r"\%", "&": r"\&", "#": r"\#",
    "$": r"\$", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "<": r"\textless{}", ">": r"\textgreater{}", "|": r"\textbar{}", '"': r"\textquotedbl{}",
}
_ESPECIALES = re.compile(r'[\\{}%&#$_~^<>|"]')
_CODIGO = re.compile(r"`([^`]+)`")
_MATEMATICA = re.compile(r"(?<![\\\w])\$([^$\n]+?)\$(?!\w)")
_AVANCE = re.compile(r"(?:\*\*)?\[→\](?:\*\*)?")
_NEGRITA = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_CURSIVA = re.compile(r"(?<![*\w])\*(?=\S)(.+?)(?<=\S)\*(?![*\w])")
# «41 188» → 41\,188 y «11,3 %» → 11,3\,%: espacio fino, sin corte de línea en el medio.
_MILES = re.compile(r"(?<![\d,.])(\d{1,3})((?: \d{3})+)(?!\d)")
_PORCENTAJE = re.compile(r"(\d) \\%")
_RESERVA = 0xF0000   # caracteres de uso privado: guardan lo ya convertido mientras se escapa el resto


def escapar(texto):
    """Texto plano a LaTeX: los caracteres especiales, escapados."""
    return _ESPECIALES.sub(lambda m: _ESCAPES[m.group()], texto)


def _codigo(texto):
    # \mbox{-}: ni la ligadura «--» → «–» ni un corte de línea que parezca una división en sílabas.
    tex = escapar(texto).replace("-", r"\mbox{-}")
    tex = tex.replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
    return r"\texttt{" + tex + "}"


def en_linea(md):
    """Markdown en línea a LaTeX: **negrita**, *cursiva*, `código`, $matemática$ y [→].

    Además, el espacio de miles y el de «11,3 %» pasan a espacio fino sin corte.
    """
    guardados = []

    def guardar(tex):
        guardados.append(tex)
        return chr(_RESERVA + len(guardados) - 1)

    s = _CODIGO.sub(lambda m: guardar(_codigo(m.group(1))), md)
    s = _MATEMATICA.sub(lambda m: guardar("$" + m.group(1) + "$"), s)
    s = _AVANCE.sub(lambda m: guardar(r"\avanza{}"), s)
    s = escapar(s)
    s = _NEGRITA.sub(r"\\textbf{\1}", s)
    s = _CURSIVA.sub(r"\\emph{\1}", s)
    s = _MILES.sub(lambda m: m.group(1) + m.group(2).replace(" ", r"\,"), s)
    s = _PORCENTAJE.sub(r"\1\\,\\%", s)
    return re.sub("[\U000F0000-\U000FFFFD]", lambda m: guardados[ord(m.group()) - _RESERVA], s)


def texto_plano(md):
    """Markdown en línea sin marcas, para los marcadores del PDF y la salida por pantalla."""
    s = _AVANCE.sub("[→]", md)
    s = _CODIGO.sub(lambda m: m.group(1), s)
    s = _MATEMATICA.sub(lambda m: m.group(1), s)
    s = _NEGRITA.sub(r"\1", s)
    return _CURSIVA.sub(r"\1", s)


def _celdas(fila):
    """Las celdas de una fila de tabla markdown; un «|» dentro de `código` no corta."""
    fila = fila.strip()
    fila = fila[1:] if fila.startswith("|") else fila
    fila = fila[:-1] if fila.endswith("|") and not fila.endswith("\\|") else fila
    celdas, actual, en_codigo = [], "", False
    for c in fila:
        if c == "`":
            en_codigo = not en_codigo
        if c == "|" and not en_codigo and not actual.endswith("\\"):
            celdas.append(actual.strip())
            actual = ""
        else:
            actual += c
    celdas.append(actual.strip())
    return [c.replace("\\|", "|") for c in celdas]


def latex_de_tabla(lineas):
    filas = [_celdas(linea) for linea in lineas]
    alineaciones = []
    if len(filas) > 1 and all(re.fullmatch(r":?-{3,}:?", c) for c in filas[1] if c):
        alineaciones = ["c" if c.startswith(":") and c.endswith(":") else "r" if c.endswith(":")
                        else "l" for c in filas[1]]
        filas = [filas[0]] + filas[2:]
    columnas = max(len(f) for f in filas)
    alineaciones = (alineaciones + ["l"] * columnas)[:columnas]
    tex = [r"\begin{tabular}{@{}" + "".join(alineaciones) + r"@{}}", r"\toprule"]
    for i, fila in enumerate(filas):
        celdas = [en_linea(c) for c in fila + [""] * (columnas - len(fila))]
        if i == 0:
            celdas = [r"\textbf{" + c + "}" if c else "" for c in celdas]
        tex.append(" & ".join(celdas) + r" \\")
        if i == 0:
            tex.append(r"\midrule")
    return "\n".join(tex + [r"\bottomrule", r"\end{tabular}"])


def latex_de_bloques(bloques):
    """Bloques de markdown a LaTeX, en la letra y el color del grupo que los rodea."""
    partes = []
    for b in bloques:
        if b.tipo == "parrafo":
            partes.append(en_linea(b.texto) + r"\par\vspace{3pt}")
        elif b.tipo == "cita":
            partes.append(r"{\itshape " + en_linea(b.texto) + r"\par}\vspace{3pt}")
        elif b.tipo == "lista":
            items = "\n".join(r"\item " + en_linea(i) for i in b.lineas)
            partes.append("\\begin{itemize}\n" + items + "\n\\end{itemize}\\vspace{2pt}")
        elif b.tipo == "enumeracion":
            items = "\n".join(rf"\item[{n}.] " + en_linea(t) for n, t in b.lineas)
            partes.append("\\begin{enumerate}\n" + items + "\n\\end{enumerate}\\vspace{2pt}")
        elif b.tipo == "tabla":
            partes.append("{\\centering\\footnotesize\n" + latex_de_tabla(b.lineas)
                          + "\\par}\\vspace{8pt}")
        elif b.tipo == "titulo":
            partes.append(r"{\bfseries " + en_linea(b.lineas[1]) + r"\par}\vspace{2pt}")
    return "\n".join(partes)


# =================================================================================================
# El documento
# =================================================================================================

PREAMBULO = r"""\documentclass[11pt,a4paper]{article}
\usepackage[a4paper,left=1.8cm,right=1.8cm,top=1.5cm,bottom=2.0cm,footskip=0.9cm]{geometry}
\usepackage[spanish,es-noquoting]{babel}
% babel en español agrega su propio espacio antes de %; el texto ya trae el suyo (11,3\,\%).
\spanishplainpercent
\usepackage[sfdefault,lining,tabular]{FiraSans}
\usepackage[scaled=0.86]{FiraMono}
\usepackage{unicode-math}
\setmathfont{Fira Math}
\usepackage[tracking=true]{microtype}
\usepackage[table]{xcolor}   % table: \arrayrulecolor para las reglas de booktabs
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{multicol}
\usepackage{array}
\usepackage{booktabs}
\usepackage{fancyhdr}
\usepackage{hyperref}
\tracinglostchars=2

% La paleta del deck y de las figuras (src/estilo.py).
\definecolor{azul}{HTML}{<<AZUL>>}
\definecolor{naranja}{HTML}{<<NARANJA>>}
\definecolor{azuloscuro}{HTML}{<<AZUL_OSCURO>>}
\definecolor{tinta}{HTML}{<<TINTA>>}
\definecolor{tintasec}{HTML}{<<TINTA_SECUNDARIA>>}
\definecolor{apagado}{HTML}{<<GRIS>>}
\definecolor{rejilla}{HTML}{<<REJILLA>>}

\hypersetup{
  pdftitle={<<PDFTITULO>>},
  pdfauthor={<<PDFAUTOR>>},
  pdfsubject={<<PDFTEMA>>},
  pdfcreator={python3 -m src.cuadernillo},
  hidelinks, bookmarksopen=true, bookmarksopenlevel=0, pdfstartview=FitH,
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}
\raggedbottom
\setlength{\columnsep}{1.4em}
\arrayrulecolor{rejilla}
\setlength{\fboxsep}{0pt}
\setlength{\fboxrule}{0.5pt}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\newcommand{\textopie}{}
\fancyfoot[C]{\footnotesize\color{apagado}\textopie}

% La marca de avance, como \avanza en las notas del deck: una pulsación, un overlay.
\newcommand{\avanza}{\textcolor{naranja}{\textbf{[→]}}}
\newcommand{\etiqueta}[1]{\textls[90]{\MakeUppercase{#1}}}
\newcommand{\cuadrado}{\raisebox{0.2em}{\textcolor{apagado}{\rule{0.36em}{0.36em}}}}
\setlist[itemize]{label=\cuadrado, leftmargin=1.1em, labelsep=0.5em, itemsep=1.4pt, parsep=0pt,
  topsep=2pt, partopsep=0pt}
\setlist[enumerate]{leftmargin=1.6em, labelsep=0.5em, itemsep=1.4pt, parsep=0pt, topsep=2pt,
  partopsep=0pt}

% LA PÁGINA DE UNA SLIDE. El encabezado (\cabeza) y el texto (\cuerpo) se componen primero en
% cajas; con ellas, la página entera se compone una vez con la imagen (\laimagen) a todo el ancho de
% la caja de texto, y se mide su alto natural. Si no entra en \textheight, primero se cierran las
% dos separaciones (hasta \encogible, 8 pt entre las dos) y, si aun así falta, la imagen se achica
% exactamente lo que falta: nunca más ancha que la caja de texto, ni más angosta de lo necesario.
% Si le quedaría menos del 62 % del alto que tiene a todo el ancho, el texto se vuelve a componer un
% punto más pequeño (\tamano = 1); por debajo del 45 %, la imagen se queda en el 45 % y la página se
% desborda. La caja es de alto fijo y su único relleno (\vfill) estira pero no encoge, así que un
% desborde de más de \vfuzz sale en el .log como «Overfull \vbox». Una slide nunca ocupa dos
% páginas, y la línea «cuadernillo: slide …» del .log dice cómo quedó cada imagen. Entre la imagen
% y el texto no va interlineado (\nointerlineskip), sólo \sepimagen: cuando la caja del texto
% empieza con un \color (una indicación en gris), su alto es 0, y TeX agregaba ahí hasta 13 pt de
% interlineado que ninguna cuenta veía.
\newcount\tamano
\newcommand{\fuentedicho}{\ifcase\tamano\normalsize\else\small\fi}
\newcommand{\fuentenota}{\ifcase\tamano\small\else\footnotesize\fi}
\newcommand{\fuenteaclarar}{\ifcase\tamano\footnotesize\else\scriptsize\fi}
\newcommand{\cabeza}{}
\newcommand{\cuerpo}{}
\newcommand{\laimagen}{}
\newsavebox{\cajacabeza}
\newsavebox{\cajacuerpo}
\newsavebox{\cajapagina}
\newlength{\anchoimagen}
\newlength{\altomaximo}
\newlength{\altoimagen}
\newlength{\sepcabeza}\setlength{\sepcabeza}{7pt minus 3pt}
\newlength{\sepimagen}\setlength{\sepimagen}{12pt minus 5pt}
\newlength{\encogible}
\setlength{\encogible}{\dimexpr\glueshrink\sepcabeza+\glueshrink\sepimagen\relax}
\newcommand{\componerpagina}{%
  \usebox{\cajacabeza}\par\vspace{\sepcabeza}%
  {\centering\laimagen\par}%
  \vspace{\sepimagen}\nointerlineskip
  \usebox{\cajacuerpo}\par}
% El espacio que cierra el último bloque (\vspace, el \topsep de una lista) quedaría al pie de la
% página sin verse; se recorta para que lo aproveche la imagen.
\newcommand{\recortarfinal}{\par\unskip\unpenalty\unskip\unpenalty\unskip\unpenalty\unskip}
\newcommand{\medirpagina}{%
  \sbox{\cajacuerpo}{\begin{minipage}[t]{\textwidth}\cuerpo\recortarfinal\end{minipage}}%
  \setlength{\altoimagen}{\altomaximo}%
  \setbox\cajapagina=\vbox{\componerpagina}%
  \setlength{\altoimagen}{\dimexpr\textheight+\encogible+\altomaximo-\ht\cajapagina
    -\dp\cajapagina\relax}%
  \ifdim\altoimagen>\altomaximo\setlength{\altoimagen}{\altomaximo}\fi}
\newcommand{\paginaslide}[1]{%
  \setlength{\anchoimagen}{\dimexpr\textwidth-2\fboxrule\relax}%
  \setlength{\altomaximo}{\dimexpr\anchoimagen*<<ALTO_PX>>/<<ANCHO_PX>>\relax}%
  \sbox{\cajacabeza}{\begin{minipage}[t]{\textwidth}\cabeza\end{minipage}}%
  \tamano=0 \medirpagina
  \ifdim\altoimagen<0.62\altomaximo\tamano=1 \medirpagina\fi
  \ifdim\altoimagen<0.45\altomaximo\setlength{\altoimagen}{0.45\altomaximo}\fi
  \typeout{cuadernillo: slide #1, imagen \the\altoimagen\space de \the\altomaximo,
    letra \the\tamano}%
  \vbox to\textheight{\componerpagina\vfill}}
\newcommand{\imagenslide}[1]{\fcolorbox{rejilla}{white}{%
  \includegraphics[width=\anchoimagen,height=\altoimagen,keepaspectratio]{#1}}}
\newcommand{\sinimagen}[1]{\fcolorbox{rejilla}{rejilla!25}{%
  \parbox[c][\altoimagen][c]{\anchoimagen}{\centering\color{apagado}#1}}}

\newcommand{\meta}[1]{{\small\color{tintasec}#1\par}}
\newcommand{\aviso}[1]{{\small\color{naranja}\textbf{Aviso:} #1\par}}
\newcommand{\nota}[1]{{\fuentenota\itshape\color{tintasec}#1\par}\vspace{4pt}}
\newcommand{\dicho}[1]{{\fuentedicho #1\par}\vspace{3pt}}
\newcommand{\rotulovariante}[1]{{\fuentenota\color{naranja}#1\par}\vspace{1pt}}
\newcommand{\variante}[2]{\vspace{1pt}{\fuentenota\color{apagado}\leftskip=1em
  \textit{#1}\par\vspace{1pt}#2\par}\vspace{4pt}}
\newcommand{\aclarar}[1]{\vspace{4pt}{\color{rejilla}\rule{\linewidth}{0.6pt}}\par\vspace{3pt}%
  {\fuenteaclarar\bfseries\color{azuloscuro}\etiqueta{#1}\par}\vspace{1pt}}

% EL BANCO DE PREGUNTAS, a dos columnas, como en el cuadernillo del TP1. Cada pregunta es una pieza
% (\begin{pieza}…\end{pieza}, una minipage del ancho de la columna) que multicols no puede cortar:
% el título nunca queda al pie de una columna con la respuesta en la siguiente, ni la respuesta
% partida en dos. El título de un grupo entra en la pieza de su primer elemento. Un \nopagebreak
% entre párrafos no alcanzaba: el \color en modo vertical deja un nodo que habilita el corte.
\newenvironment{pieza}[1]{\par\vspace{#1}\noindent\begin{minipage}[b]{\linewidth}}
  {\end{minipage}\par}
\newcommand{\grupopreguntas}[1]{{\footnotesize\bfseries\color{azul}\etiqueta{#1}\par}\vspace{6pt}}
\newcommand{\pregunta}[3]{{\small\bfseries\color{azuloscuro}#1\par}
  {\footnotesize\color{apagado}#2\par}\vspace{1pt}{\small #3\par}}

% LAS PORTADAS DE SECCIÓN (reloj, respaldo, preguntas).
\newcommand{\titulo}[1]{{\LARGE\bfseries\color{azuloscuro}#1\par}\vspace{6pt}}

% LA PÁGINA AJUSTADA (el reloj de ensayo): su contenido (\ajustado), título incluido, se compone
% en una caja y se prueba en tres tamaños de letra hasta que entra en una página. Si no entra ni en
% el más pequeño, fluye en varias páginas en ese tamaño.
\newcount\tamanoajuste
\newcommand{\fuenteajustada}{\ifcase\tamanoajuste\small\or\footnotesize\else\scriptsize\fi}
\newcommand{\ajustado}{}
\newsavebox{\cajaajustada}
\newcommand{\medirajustado}{%
  \sbox{\cajaajustada}{\begin{minipage}[t]{\textwidth}\fuenteajustada\ajustado\end{minipage}}}
\newcommand{\altoajustado}{\dimexpr\ht\cajaajustada+\dp\cajaajustada\relax}
\newcommand{\paginaajustada}{%
  \tamanoajuste=0 \medirajustado
  \ifdim\altoajustado>\dimexpr\textheight-6pt\relax\tamanoajuste=1 \medirajustado\fi
  \ifdim\altoajustado>\dimexpr\textheight-6pt\relax\tamanoajuste=2 \medirajustado\fi
  \ifdim\altoajustado>\dimexpr\textheight-6pt\relax
    {\fuenteajustada\ajustado\par}%
  \else
    \noindent\usebox{\cajaajustada}\par
  \fi}

\begin{document}
\color{tinta}
"""


def _hex(color):
    return color.lstrip("#").upper()


def _entero(n, tex=True):
    return f"{n:,}".replace(",", r"\," if tex else " ")


def _decimal(x, decimales=1):
    return f"{x:.{decimales}f}".replace(".", ",")


def _enumerar(partes):
    """«a», «a y b», «a, b y c»."""
    partes = list(partes)
    if len(partes) <= 1:
        return "".join(partes)
    return ", ".join(partes[:-1]) + " y " + partes[-1]


def _rangos(numeros):
    """[1, 2, 3, 8, 9, 12] → «1–3, 8–9 y 12»."""
    tramos = []
    for n in sorted(numeros):
        if tramos and n == tramos[-1][1] + 1:
            tramos[-1][1] = n
        else:
            tramos.append([n, n])
    return _enumerar(f"{a}–{b}" if a != b else str(a) for a, b in tramos)


def _pie(texto):
    return r"\renewcommand{\textopie}{" + escapar(f"TP2 · Grupo {GRUPO} · {texto}") + "}"


@dataclass
class Pagina:
    """Una página de slide del cuadernillo: el texto del guion, el frame del PDF, o los dos."""
    numero: int
    slide: SlideGuion | None
    frame: Frame | None
    avisos: list[str] = field(default_factory=list)

    @property
    def imagen(self):
        return f"slide-{self.numero:02d}.png" if self.frame else None


def armar_paginas(guion, frames, test_pendiente=None):
    """Empareja cada slide del guion con su frame de la charla y avisa lo que no coincide."""
    de_la_charla = {f.rotulo.numero: f for f in frames if f.rotulo and not f.rotulo.respaldo}
    del_guion = {s.numero: s for s in guion.slides}
    paginas = []
    for n in sorted(set(de_la_charla) | set(del_guion)):
        slide, frame = del_guion.get(n), de_la_charla.get(n)
        pagina = Pagina(n, slide, frame)
        if frame is None:
            pagina.avisos.append(f"El PDF no tiene la slide {n}: el deck tiene "
                                 f"{len(de_la_charla)} en la charla.")
        if slide is None:
            pagina.avisos.append("El guion no tiene texto para esta slide.")
        if slide is not None and frame is not None and slide.dichos:
            principal, _ = variantes(slide.dichos, test_pendiente)
            if principal.avances != frame.pasos:
                pagina.avisos.append(
                    f"El guion marca {principal.avances} [→] y en el PDF esta slide tiene "
                    f"{frame.pasos} {'paso' if frame.pasos == 1 else 'pasos'} después del primero.")
        paginas.append(pagina)
    return paginas


def comparar_mapa(guion, frames):
    """El mapa «slide → página» que escribe el guion contra el del PDF: [(slide, guion, PDF)]."""
    reales = {f.nombre: f.final for f in frames if f.rotulo is not None}
    return [(clave, pagina, reales.get(clave)) for clave, pagina in guion.mapa_paginas.items()
            if reales.get(clave) != pagina]


def _cabeza(pagina, guion, test_pendiente):
    s, f = pagina.slide, pagina.frame
    titulo = en_linea(s.titulo) if s else "(sin texto en el guion)"
    datos = []
    if s and s.inicio and s.fin:
        datos.append(r"\textbf{\textcolor{azul}{" + f"{s.inicio} → {s.fin}" + "}}")
    if pagina.numero in guion.controles:
        datos.append(r"\textcolor{naranja}{\textbf{punto de control}}")
    if s and s.palabras is not None:
        datos.append(r"\textasciitilde{}" + _entero(s.palabras) + " palabras")
    if s and s.dichos:
        principal, _ = variantes(s.dichos, test_pendiente)
        datos.append(("sin" if principal.avances == 0 else str(principal.avances)) + r" \avanza{}")
    if f is not None:
        datos.append(f"página {f.final} del PDF")
    derecha = (r"Habla: \textbf{" + escapar(s.habla) + "}") if s and s.habla else ""
    lineas = [r"{\Large\bfseries\color{azuloscuro}\raggedright " + f"{pagina.numero} · {titulo}"
              + r"\par}",
              r"\vspace{3pt}",
              r"\meta{" + r"\enspace·\enspace ".join(datos) + r"\hfill " + derecha + "}"]
    lineas += [r"\aviso{" + escapar(a).replace("[→]", r"\avanza{}") + "}" for a in pagina.avisos]
    return "\n".join(lineas)


def _segmentos(d, separador):
    return separador.join((r"\avanza{} " if i > 0 else "") + en_linea(segmento)
                          for i, segmento in enumerate(d.segmentos) if i > 0 or segmento)


# En el guion, la variante del test cerrado va después de la de la defensa y su párrafo dice «…, en
# lugar de lo anterior:». Si el cuadernillo la adelanta, lo que reemplaza queda debajo, en gris.
_LO_ANTERIOR = re.compile(r"\ben lugar de lo anterior\b")


def rotulo_de_variante(etiqueta, adelantada=False):
    """El párrafo del guion que presenta una variante, completo, en LaTeX.

    Su negrita inicial, si la tiene, va como rótulo en mayúsculas espaciadas; el resto del párrafo
    sigue en la misma línea, tal cual, porque suele decir hasta cuándo vale la variante y qué
    muestra la imagen. `adelantada`: la variante va antes que la que en el guion la precede, así que
    su «en lugar de lo anterior» pasa a apuntar a la de abajo.
    """
    if adelantada:
        etiqueta = _LO_ANTERIOR.sub("en lugar de la variante de abajo, en gris", etiqueta)
    m = _NEGRITA.match(etiqueta)
    if not m:
        return en_linea(etiqueta)
    return r"\etiqueta{" + en_linea(texto_plano(m.group(1))) + "}" + en_linea(etiqueta[m.end():])


def _cuerpo(pagina, test_pendiente):
    s = pagina.slide
    if s is None:
        return r"\nota{El guion no tiene texto para esta slide.}"
    principal, otras = variantes(s.dichos, test_pendiente)
    partes, dicho_puesto = [], False
    for tipo, contenido in s.bloques:
        if tipo == "nota":
            partes.append(r"\nota{" + en_linea(contenido) + "}")
        elif tipo == "bloque":
            partes.append(r"{\fuentenota\itshape\color{tintasec}" + latex_de_bloques([contenido])
                          + r"\par}\vspace{3pt}")
        elif tipo == "dicho" and not dicho_puesto:
            # Todas las variantes van donde estaba la primera: la principal adelante y las otras,
            # en gris, debajo, cada una con el párrafo que la presenta, completo.
            dicho_puesto = True
            if otras and principal.etiqueta:
                adelantada = next(i for i, d in enumerate(s.dichos) if d is principal) > 0
                partes.append(r"\rotulovariante{"
                              + rotulo_de_variante(principal.etiqueta, adelantada) + "}")
            partes.append(r"\dicho{" + _segmentos(principal, r"}" + "\n" + r"\dicho{") + "}")
            for otra in otras:
                etiqueta = (en_linea(otra.etiqueta) if otra.etiqueta
                            else "Con el test evaluado, en la defensa, en lugar de lo anterior:")
                partes.append(r"\variante{" + etiqueta + "}{" + _segmentos(otra, r"\par ") + "}")
    if s.aclarar.strip():
        titulo = s.titulo_aclarar or "A aclarar"
        if _normalizar(titulo) == "a aclarar":
            titulo = "A aclarar · preguntas probables"
        partes.append(r"\aclarar{" + en_linea(titulo) + "}")
        # Sin justificar: en letra pequeña, con `código` y cifras largas, la justificación estiraba
        # algunas líneas hasta el «Underfull \hbox (badness 10000)».
        partes.append(r"{\fuenteaclarar\raggedright " + latex_de_bloques(bloques_md(s.aclarar))
                      + r"\par}")
    return "\n".join(partes)


def latex_slide(pagina, total, guion, test_pendiente, primera=False):
    s = pagina.slide
    marcador = escapar(f"{pagina.numero} · " + (texto_plano(s.titulo) if s else "sin guion"))
    imagen = (r"\imagenslide{" + pagina.imagen + "}" if pagina.imagen
              else r"\sinimagen{Esta slide no está en el PDF.}")
    # El marcador «Slides» va dentro de la página de la primera: suelto antes del \clearpage,
    # dejaría una página en blanco.
    return "\n".join(linea for linea in [
        r"\clearpage",
        r"\pdfbookmark[0]{Slides}{slides}" if primera else "",
        _pie(f"slide {pagina.numero} de {total}"),
        rf"\pdfbookmark[1]{{{marcador}}}{{slide-{pagina.numero}}}\label{{slide-{pagina.numero}}}",
        r"\renewcommand{\cabeza}{" + _cabeza(pagina, guion, test_pendiente) + "}",
        r"\renewcommand{\cuerpo}{" + _cuerpo(pagina, test_pendiente) + "}",
        r"\renewcommand{\laimagen}{" + imagen + "}",
        r"\paginaslide{" + str(pagina.numero) + "}",
    ] if linea)


def _quien_habla(guion):
    """Por orador, en el orden en que entra: (nombre, palabras, slides)."""
    oradores = {}
    for s in guion.slides:
        if s.habla:
            nombre = s.habla
            palabras, slides = oradores.setdefault(nombre, [0, []])
            oradores[nombre][0] = palabras + (s.palabras or 0)
            slides.append(s.numero)
    return [(nombre, palabras, slides) for nombre, (palabras, slides) in oradores.items()]


def _fecha_de_defensa(titulo):
    fecha = re.search(r"\d{1,2}/\d{1,2}/\d{4}(?:\s*\([^)]*\))?", titulo)
    duracion = re.search(r"\d+\s*\+\s*\d+\s*minutos", titulo)
    return (fecha.group(0) if fecha else None), (duracion.group(0) if duracion else None)


def latex_portada(guion, meta, frames, paginas, test_pendiente, ahora):
    titulo_deck = meta.get("Title", "").replace(" - ", " — ")
    autores = re.sub(r"^Grupo\s+\d+\s*[—–-]+\s*", "", meta.get("Author", "")).replace("), ", ") · ")
    fecha, duracion = _fecha_de_defensa(guion.titulo)
    charla = [f for f in frames if f.rotulo and not f.rotulo.respaldo]
    respaldo = [f for f in frames if f.rotulo and f.rotulo.respaldo]
    avances = sum(variantes(p.slide.dichos, test_pendiente)[0].avances
                  for p in paginas if p.slide and p.slide.dichos)
    partes = [r"\thispagestyle{empty}",
              r"\pdfbookmark[0]{Portada}{portada}\label{portada}",
              r"\vspace*{1.6cm}",
              r"{\small\color{azul}\etiqueta{Cuadernillo de ensayo}\par}\vspace{10pt}",
              r"{\fontsize{30}{34}\selectfont\bfseries\color{azuloscuro}Slides y guion\par}"
              r"\vspace{10pt}",
              r"{\Large Defensa del TP2 --- Grupo " + str(GRUPO) + r"\par}\vspace{5pt}"]
    if titulo_deck:
        partes.append(r"{\large\color{tintasec}" + escapar(titulo_deck) + r"\par}")
    partes.append(r"\vspace{16pt}{\normalsize " + escapar(MATERIA) + r"\par}\vspace{2pt}")
    if autores:
        partes.append(r"{\normalsize " + escapar(autores) + r"\par}\vspace{2pt}")
    defensa = (r"Defensa: \textbf{" + escapar(fecha) + "}") if fecha else "Defensa: a confirmar"
    if duracion:
        defensa += r"\enspace·\enspace " + escapar(duracion)
    partes.append(r"{\normalsize " + defensa + r"\par}")

    # El reloj total: el final de la última ventana del guion y sus palabras al ritmo del guion.
    partes.append(r"\vspace{28pt}{\footnotesize\color{tintasec}\etiqueta{Reloj total}\par}"
                  r"\vspace{3pt}")
    if guion.duracion is not None:
        partes.append(r"{\fontsize{40}{44}\selectfont\bfseries\color{azul}"
                      + a_reloj(guion.duracion) + r"\par}\vspace{6pt}")
    lineas = [f"{_entero(guion.palabras)} palabras a {_decimal(guion.ritmo)} por segundo dan "
              f"{a_reloj(guion.palabras / guion.ritmo)}"
              + (f"; las ventanas del guion terminan en {a_reloj(guion.duracion)}"
                 if guion.duracion is not None else "")
              + f". El tope hablado es {TOPE}.",
              f"{len(guion.slides)} slides en el guion, con {avances} " + r"\avanza{}"
              + f"; el PDF de proyección tiene {sum(len(f.paginas) for f in charla)} páginas en la "
              "charla" + (f" y {len(respaldo)} de respaldo" if respaldo else "") + "."]
    partes.append(r"{\small\color{tintasec}" + r"\par\vspace{2pt}".join(lineas) + r"\par}")

    oradores = _quien_habla(guion)
    if oradores:
        filas = [r"\textbf{" + escapar(nombre) + "} & " + f"{_entero(palabras)} palabras & "
                 f"{a_reloj(palabras / guion.ritmo)} & slides {_rangos(slides)}" + r" \\"
                 for nombre, palabras, slides in oradores]
        partes.append(r"\vspace{18pt}{\footnotesize\color{tintasec}\etiqueta{Quién habla}\par}"
                      r"\vspace{4pt}{\small\begin{tabular}{@{}l@{\hspace{1.2em}}r@{\hspace{1.2em}}"
                      r"r@{\hspace{1.2em}}l@{}}" + "\n".join(filas) + r"\end{tabular}\par}")

    como = [
        r"Cada página es una slide del deck en su \textbf{estado final}, con todos los overlays "
        r"revelados, junto a su texto hablado. Arriba van la ventana de reloj, las palabras, las "
        r"pulsaciones y la página del PDF, para volver a ella en las preguntas.",
        r"Lo que va en letra normal se dice, y la marca \avanza{} indica dónde se avanza un overlay. "
        r"Lo que va en gris cursiva son indicaciones de ensayo, y el bloque «A aclarar» es material "
        r"para las preguntas: nada de eso se dice en voz alta.",
    ]
    con_variantes = [p.numero for p in paginas if p.slide and len(p.slide.dichos) > 1]
    if con_variantes and test_pendiente is not None:
        cuales = ("en la slide " if len(con_variantes) == 1 else "en las slides ") + _enumerar(
            map(str, con_variantes))
        if test_pendiente:
            como.append(r"\textbf{El test todavía no se evaluó} ("
                        + _codigo("informe/resultados-test.tex") + f"): {cuales} va primero la "
                        "variante con el test cerrado, que es la que muestra la imagen, y debajo, "
                        "en gris, la de la defensa.")
        else:
            como.append(r"\textbf{El test ya se evaluó}: " + cuales + " va primero la variante de "
                        "la defensa, y debajo, en gris, la que se ensayaba con el test cerrado.")
    partes.append(r"\vspace{22pt}{\footnotesize\color{tintasec}\etiqueta{Cómo leerlo}\par}"
                  r"\vspace{4pt}{\small " + r"\par\vspace{4pt}".join(como) + r"\par}")
    generado = (f"Generado el {ahora:%d/%m/%Y} a las {ahora:%H:%M} con "
                + _codigo("python3 -m src.cuadernillo") + ", desde "
                + _codigo("informe/presentacion.pdf") + " e " + _codigo("informe/guion.md") + ".")
    partes.append(r"\vfill{\footnotesize\color{apagado}" + generado + r"\par}")
    return "\n".join(partes)


def latex_reloj(guion):
    if not guion.reloj:
        return ""
    titulo = guion.reloj_titulo or "Reloj de ensayo"
    return "\n".join([r"\clearpage", _pie(texto_plano(titulo).lower()),
                      r"\pdfbookmark[0]{" + escapar(texto_plano(titulo)) + r"}{reloj}\label{reloj}",
                      r"\renewcommand{\ajustado}{\titulo{" + en_linea(titulo) + "}",
                      latex_de_bloques(guion.reloj) + "}",
                      r"\paginaajustada"])


def latex_respaldo(guion, frames):
    respaldo = [f for f in frames if f.rotulo and f.rotulo.respaldo]
    if not respaldo:
        return ""
    citas = {}
    for p in guion.preguntas:
        if p.etiqueta and "respaldo" in _normalizar(p.etiqueta):
            for n in RESPALDO_CITADO.findall(p.etiqueta):
                citas.setdefault(int(n), []).append(p.numero)
    charla = [f for f in frames if f.rotulo and not f.rotulo.respaldo]
    ultima = f" de la {charla[-1].rotulo.numero}" if charla else ""
    intro = (f"Las {len(respaldo)} slides de respaldo van después{ultima}, fuera del recorrido "
             f"(páginas {respaldo[0].paginas[0]} a {respaldo[-1].final} del PDF), y se muestran "
             "sólo si una pregunta las pide.")
    partes = []
    for i, f in enumerate(respaldo):
        if i % POR_PAGINA_RESPALDO == 0:
            partes += [r"\clearpage", _pie("respaldo")]
            if i == 0:
                partes += [r"\pdfbookmark[0]{Respaldo}{respaldo}\label{respaldo}",
                           r"\titulo{Respaldo}",
                           r"{\small\color{tintasec}" + escapar(intro) + r"\par}\vspace{10pt}"]
        else:
            partes.append(r"\vspace{16pt}")
        pie = f"R{f.rotulo.numero} · página {f.final} del PDF"
        if f.rotulo.numero in citas:
            numeros = citas[f.rotulo.numero]
            pie += (" · pregunta " if len(numeros) == 1 else " · preguntas ") + _enumerar(
                map(str, numeros))
        partes += [r"{\centering\fcolorbox{rejilla}{white}{\includegraphics[width=\dimexpr"
                   r"\textwidth-2\fboxrule\relax]{" + f"respaldo-{f.rotulo.numero}.png" + r"}}\par}",
                   r"\vspace{4pt}{\small\color{tintasec}" + escapar(pie) + r"\par}"]
    return "\n".join(partes)


def latex_preguntas(guion, diferencias_mapa):
    titulo = guion.preguntas_titulo or "Preguntas probables"
    partes = [r"\clearpage", _pie("preguntas"),
              r"\pdfbookmark[0]{" + escapar(texto_plano(titulo)) + r"}{preguntas}\label{preguntas}",
              r"\titulo{" + en_linea(titulo) + "}"]
    if guion.preguntas_intro:
        partes.append(r"{\small\color{tintasec}" + latex_de_bloques(guion.preguntas_intro) + r"\par}")
    if diferencias_mapa:
        detalle = "; ".join(f"{clave} → {g} en el guion y {'no está' if r is None else r} en el PDF"
                            for clave, g, r in diferencias_mapa)
        partes.append(r"\aviso{El mapa de páginas del guion no coincide con el PDF actual: "
                      + escapar(detalle) + ".}")
    partes.append(r"\vspace{4pt}\raggedcolumns\begin{multicols}{2}")
    for i, grupo in enumerate(guion.grupos, start=1):
        # El marcador y el título del grupo van dentro de la pieza de su primer elemento: así no
        # quedan solos al pie de una columna.
        cabecera = ""
        if grupo.titulo:
            cabecera = (rf"\pdfbookmark[1]{{{escapar(texto_plano(grupo.titulo))}}}{{grupo-{i}}}"
                        + r"\grupopreguntas{" + en_linea(grupo.titulo) + "}\n")
        for e in grupo.elementos or [None]:
            if isinstance(e, Pregunta):
                contenido = (r"\pregunta{" + f"{e.numero}. " + en_linea(e.pregunta) + "}{"
                             + (en_linea(f"[{e.etiqueta}]") if e.etiqueta else "") + "}{"
                             + en_linea(e.respuesta) + "}")
            elif e is not None:
                contenido = r"{\small " + latex_de_bloques([e]) + r"\par}"
            else:
                contenido = ""
            partes.append(r"\begin{pieza}{" + ("9pt" if cabecera else "6pt") + "}\n" + cabecera
                          + contenido + "\n" + r"\end{pieza}")
            cabecera = ""
    partes += [r"\end{multicols}", r"\label{fin}"]
    return "\n".join(partes)


def documento(guion, meta, frames, paginas, test_pendiente, diferencias_mapa, tamano_imagen,
              ahora):
    """El .tex completo del cuadernillo."""
    autor = meta.get("Author", f"Grupo {GRUPO}")
    reemplazos = {
        "<<AZUL>>": _hex(AZUL), "<<NARANJA>>": _hex(NARANJA), "<<AZUL_OSCURO>>": _hex(AZUL_OSCURO),
        "<<TINTA>>": _hex(TINTA), "<<TINTA_SECUNDARIA>>": _hex(TINTA_SECUNDARIA),
        "<<GRIS>>": _hex(GRIS_REFERENCIA), "<<REJILLA>>": _hex(REJILLA),
        "<<PDFTITULO>>": escapar(f"TP2 Grupo {GRUPO} — Slides y guion"),
        "<<PDFAUTOR>>": escapar(autor),
        "<<PDFTEMA>>": "Defensa del TP2: cada slide en su estado final, con el guion hablado",
        "<<ANCHO_PX>>": str(tamano_imagen[0]), "<<ALTO_PX>>": str(tamano_imagen[1]),
    }
    tex = PREAMBULO
    for clave, valor in reemplazos.items():
        tex = tex.replace(clave, valor)
    total = max([len(paginas)] + [f.rotulo.total for f in frames
                                  if f.rotulo and not f.rotulo.respaldo])
    partes = [tex, latex_portada(guion, meta, frames, paginas, test_pendiente, ahora),
              latex_reloj(guion)]
    partes += [latex_slide(p, total, guion, test_pendiente, primera=(i == 0))
               for i, p in enumerate(paginas)]
    partes += [latex_respaldo(guion, frames), latex_preguntas(guion, diferencias_mapa),
               r"\end{document}", ""]
    return "\n".join(p for p in partes if p)


# =================================================================================================
# Compilar y revisar
# =================================================================================================

# Los avisos con los que LaTeX pide otra pasada: el del núcleo («Label(s) may have changed. Rerun to
# get cross-references right»), el de rerunfilecheck para los marcadores («Rerun to get outlines
# right»), el de hyperref («Rerun to get /PageLabels entry») y el de longtable («Rerun LaTeX»). No
# se busca el nombre del paquete: «rerunfilecheck» está en todo .log de hyperref, en la línea que
# lo carga y en «File `….out' has not changed», y con él la compilación nunca paraba antes de la
# tercera pasada. \s+ entre palabras: el .log corta las líneas largas.
RELEER = re.compile(r"Rerun\s+to\s+get|Label\(s\)\s+may\s+have\s+changed|Rerun\s+LaTeX")
# Los underfull \hbox se muestran desde badness 10000, el máximo que TeX informa: una línea tan
# estirada que se ve. Los de menos quedan fuera: la justificación a dos columnas los produce sin
# que se noten.
AVISOS_LOG = re.compile(r"^(Overfull \\[hv]box.*|Underfull \\vbox.*"
                        r"|Underfull \\hbox \(badness [1-9]\d{4,}\).*|Missing character: .*"
                        r"|LaTeX(?: Font)? Warning: .*|Package \S+ Warning: .*"
                        r"|pdfTeX warning.*|luaotfload.*warning.*)$", re.M | re.I)
ETIQUETA_AUX = re.compile(r"\\newlabel\{([^}]*)\}\{\{[^}]*\}\{(\d+)\}")
IMAGEN_LOG = re.compile(r"^cuadernillo: slide (\d+), imagen ([\d.]+)pt de ([\d.]+)pt, "
                        r"letra (\d+)\s*$", re.M)
ANCHO_COMPLETO = 0.9995   # desde aquí, la imagen va a todo el ancho (el resto es redondeo)


def pide_otra_pasada(log):
    """True si el .log de una pasada de LaTeX pide otra: cambió una etiqueta o los marcadores."""
    return RELEER.search(log) is not None


def imagenes_del_log(log):
    """{slide: (fracción del ancho, letra)} de las líneas «cuadernillo: slide …» del .log.

    La imagen conserva su proporción, así que la fracción del alto es la del ancho. `letra` es 0
    con la letra normal y 1 si el texto se compuso un punto más pequeño.
    """
    return {int(n): (float(alto) / float(maximo), int(letra))
            for n, alto, maximo, letra in IMAGEN_LOG.findall(log)}


def compilar(tex, directorio, motor=MOTOR):
    """Compila cuadernillo.tex en `directorio` hasta que no pida otra pasada.

    Devuelve (pdf, .log de la última pasada, pasadas): dos, si todo converge; CORRIDAS_MAXIMAS
    como mucho.
    """
    fuente = Path(directorio) / "cuadernillo.tex"
    fuente.write_text(tex, encoding="utf-8")
    log = ""
    for corrida in range(1, CORRIDAS_MAXIMAS + 1):
        try:
            proceso = subprocess.run([motor, "-interaction=nonstopmode", "-halt-on-error",
                                      "-file-line-error", fuente.name], cwd=directorio,
                                     capture_output=True, text=True, encoding="utf-8",
                                     errors="replace")
        except FileNotFoundError as e:
            raise ErrorCuadernillo(f"No se encuentra {motor}: hace falta una instalación "
                                   "de TeX.") from e
        ruta_log = fuente.with_suffix(".log")
        log = (ruta_log.read_text(encoding="utf-8", errors="replace") if ruta_log.exists()
               else proceso.stdout)
        if proceso.returncode != 0:
            errores = [linea for linea in log.splitlines()
                       if linea.startswith("!") or re.match(r"^\S+\.tex:\d+:", linea)]
            raise ErrorCuadernillo(f"{motor} falló en la pasada {corrida}:\n"
                                   + "\n".join((errores or log.splitlines()[-25:])[:15]))
        if corrida >= 2 and not pide_otra_pasada(log):
            break
    return fuente.with_suffix(".pdf"), log, corrida


def revisar_log(log):
    """Los avisos que importan del .log, sin repetir: cajas desbordadas, glifos faltantes, etc."""
    vistos, avisos = set(), []
    for m in AVISOS_LOG.finditer(log):
        linea = m.group(1).strip()
        if linea not in vistos:
            vistos.add(linea)
            avisos.append(linea)
    return avisos


def paginas_de_etiquetas(aux):
    """{etiqueta: página} de los \\label del .aux."""
    return {k: int(v) for k, v in ETIQUETA_AUX.findall(aux)}


def revisar_estructura(etiquetas, paginas_pdf, paginas):
    """Cada slide en su página, consecutivas, y el total del PDF igual a la última etiqueta."""
    avisos = []
    esperadas = [etiquetas.get(f"slide-{p.numero}") for p in paginas]
    if None in esperadas:
        avisos.append("Falta la etiqueta de alguna slide en el .aux.")
    elif esperadas and esperadas != list(range(esperadas[0], esperadas[0] + len(esperadas))):
        avisos.append("Alguna slide ocupa más de una página: " + ", ".join(
            f"{p.numero} en la {e}" for p, e in zip(paginas, esperadas)) + ".")
    if etiquetas.get("fin") != paginas_pdf:
        avisos.append(f"El PDF tiene {paginas_pdf} páginas y la última etiqueta cae en la "
                      f"{etiquetas.get('fin')}.")
    return avisos


@dataclass
class Resultado:
    salida: Path
    paginas: int                 # del cuadernillo
    guion: Guion
    frames: list[Frame]
    paginas_deck: int
    slides: list[Pagina]
    etiquetas: dict              # etiqueta → página del cuadernillo
    test_pendiente: bool | None
    avisos: list[str]            # del guion contra el deck
    avisos_log: list[str]        # del .log de LuaLaTeX y de la estructura del resultado
    corridas: int = 0            # pasadas de LuaLaTeX hasta que el .log dejó de pedir otra
    imagenes: dict = field(default_factory=dict)   # slide → (fracción del ancho, letra)


def resumen_de_imagenes(imagenes):
    """Una línea: qué slides quedaron con la imagen más angosta que la caja de texto, y cuánto."""
    if not imagenes:
        return "Imágenes: el .log no dice cómo quedaron."
    achicadas = [(n, fraccion, letra) for n, (fraccion, letra) in sorted(imagenes.items())
                 if fraccion < ANCHO_COMPLETO or letra]
    if not achicadas:
        return f"Imágenes: las {len(imagenes)} a todo el ancho de la caja de texto."
    detalle = _enumerar(f"la de la slide {n} va al {_decimal(100 * fraccion)} % del ancho"
                        + (", con la letra un punto más pequeña" if letra else "")
                        for n, fraccion, letra in achicadas)
    return (f"Imágenes: {len(imagenes) - len(achicadas)} de {len(imagenes)} a todo el ancho de la "
            f"caja de texto; por el largo del texto, {detalle}.")


def generar(presentacion=PRESENTACION, guion=GUION, salida=SALIDA, resultados_test=RESULTADOS_TEST,
            nav=NAV, dpi=DPI, conservar=None, ahora=None):
    """Genera el cuadernillo y devuelve un Resultado. Lanza ErrorCuadernillo si no puede."""
    presentacion, guion, salida = Path(presentacion), Path(guion), Path(salida)
    faltan = [str(p) for p in (presentacion, guion) if not p.exists()]
    if faltan:
        raise ErrorCuadernillo("Faltan los archivos de entrada: " + ", ".join(faltan) + ".")
    faltan = [h for h in HERRAMIENTAS + (MOTOR,) if shutil.which(h) is None]
    if faltan:
        raise ErrorCuadernillo("Faltan herramientas: " + ", ".join(faltan)
                               + " (poppler y una instalación de TeX con LuaLaTeX).")
    ahora = ahora or dt.datetime.now()

    g = parsear_guion(guion.read_text(encoding="utf-8"))
    meta = metadatos_pdf(presentacion)
    frames, paginas_deck, avisos = frames_del_pdf(presentacion, int(meta.get("Pages", 0)))
    if nav and Path(nav).exists():
        avisos += comparar_con_nav(frames, rangos_del_nav(Path(nav).read_text(encoding="utf-8",
                                                                             errors="replace")),
                                   paginas_deck)
    avisos += g.avisos
    test_pendiente = estado_del_test(resultados_test)
    paginas = armar_paginas(g, frames, test_pendiente)
    avisos += [f"Slide {p.numero}: {a}" for p in paginas for a in p.avisos]
    diferencias = comparar_mapa(g, frames)
    if diferencias:
        avisos.append("El mapa «slide → página» del guion no coincide con el PDF: " + "; ".join(
            f"{c} → {a} en el guion y {b} en el PDF" for c, a, b in diferencias) + ".")

    with tempfile.TemporaryDirectory(prefix="cuadernillo-") as temporal:
        directorio = Path(conservar) if conservar else Path(temporal)
        directorio.mkdir(parents=True, exist_ok=True)
        pedidos = {directorio / f"slide-{p.numero:02d}": p.frame.final for p in paginas if p.frame}
        pedidos |= {directorio / f"respaldo-{f.rotulo.numero}": f.final
                    for f in frames if f.rotulo and f.rotulo.respaldo}
        imagenes = renderizar_varias(presentacion, pedidos, dpi)
        tamano = tamano_png(imagenes[0]) if imagenes else (16, 9)
        tex = documento(g, meta, frames, paginas, test_pendiente, diferencias, tamano, ahora)
        pdf, log, corridas = compilar(tex, directorio)
        avisos_log = revisar_log(log)
        if pide_otra_pasada(log):
            avisos_log.append(f"{MOTOR} sigue pidiendo otra pasada después de {corridas}: los "
                              "marcadores o las etiquetas pueden haber quedado desfasados.")
        imagenes_slides = imagenes_del_log(log)
        aux = pdf.with_suffix(".aux")
        etiquetas = paginas_de_etiquetas(aux.read_text(encoding="utf-8", errors="replace")
                                         if aux.exists() else "")
        paginas_salida = int(metadatos_pdf(pdf).get("Pages", 0))
        avisos_log += revisar_estructura(etiquetas, paginas_salida, paginas)
        salida.parent.mkdir(parents=True, exist_ok=True)
        parcial = salida.with_name(salida.name + ".parcial")
        shutil.copyfile(pdf, parcial)
        os.replace(parcial, salida)
    return Resultado(salida, paginas_salida, g, frames, paginas_deck, paginas, etiquetas,
                     test_pendiente, avisos, avisos_log, corridas, imagenes_slides)


# =================================================================================================
# Línea de comandos
# =================================================================================================

def _relativa(ruta):
    ruta = Path(ruta).resolve()
    try:
        return ruta.relative_to(RAIZ).as_posix()
    except ValueError:
        return str(ruta)


def _secciones_del_resultado(r):
    e = r.etiquetas
    partes = []
    if "portada" in e:
        partes.append(f"portada {e['portada']}")
    if "reloj" in e:
        partes.append(f"reloj {e['reloj']}")
    paginas_slides = [e[f"slide-{p.numero}"] for p in r.slides if f"slide-{p.numero}" in e]
    if paginas_slides:
        partes.append(f"slides {min(paginas_slides)}–{max(paginas_slides)}")
    if "respaldo" in e:
        partes.append(f"respaldo {e['respaldo']}")
    if "preguntas" in e:
        fin = e.get("fin", e["preguntas"])
        partes.append(f"preguntas {e['preguntas']}" + (f"–{fin}" if fin != e["preguntas"] else ""))
    return ", ".join(partes)


def _mostrar(r, presentacion, guion):
    charla = [f for f in r.frames if f.rotulo and not f.rotulo.respaldo]
    respaldo = [f for f in r.frames if f.rotulo and f.rotulo.respaldo]
    print(f"Deck: {_relativa(presentacion)}, {r.paginas_deck} páginas: {len(charla)} slides en la "
          f"charla ({sum(f.pasos for f in charla)} [→]) y {len(respaldo)} de respaldo.")
    print("Página final de cada slide: " + " · ".join(f"{f.nombre}→{f.final}" for f in r.frames))
    g = r.guion
    print(f"Guion: {_relativa(guion)}, {len(g.slides)} slides, {_entero(g.palabras, tex=False)} "
          f"palabras ({a_reloj(g.palabras / g.ritmo)} a {_decimal(g.ritmo)} por segundo), "
          f"reloj {a_reloj(g.duracion) if g.duracion is not None else '?'}; "
          f"{len(g.preguntas)} preguntas.")
    estado = {True: "pendiente", False: "evaluado", None: "desconocido"}[r.test_pendiente]
    print(f"Test: {estado} ({_relativa(RESULTADOS_TEST)}).")
    print("Avisos del guion contra el deck: " + ("ninguno." if not r.avisos else ""))
    for a in r.avisos:
        print(f"  - {a}")
    print(resumen_de_imagenes(r.imagenes))
    print(f".log de {MOTOR} ({r.corridas} {'pasada' if r.corridas == 1 else 'pasadas'}) y "
          "estructura: " + ("sin avisos." if not r.avisos_log else ""))
    for a in r.avisos_log:
        print(f"  - {a}")
    print(f"Escrito: {_relativa(r.salida)}, {r.paginas} páginas ({_secciones_del_resultado(r)}).")


def _parser():
    p = argparse.ArgumentParser(description="Genera el cuadernillo de ensayo (slides y guion).")
    p.add_argument("--presentacion", type=Path, default=PRESENTACION)
    p.add_argument("--guion", type=Path, default=GUION)
    p.add_argument("--salida", type=Path, default=SALIDA)
    p.add_argument("--dpi", type=int, default=DPI,
                   help=f"resolución de las slides (por defecto {DPI})")
    p.add_argument("--conservar", type=Path, default=None, metavar="DIR",
                   help="deja el .tex, las imágenes y los auxiliares en DIR en lugar de borrarlos")
    return p


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        r = generar(args.presentacion, args.guion, args.salida, dpi=args.dpi,
                    conservar=args.conservar)
    except ErrorCuadernillo as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    _mostrar(r, args.presentacion, args.guion)
    return 0


if __name__ == "__main__":
    sys.exit(main())

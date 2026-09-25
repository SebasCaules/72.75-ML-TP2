"""Estilo de los gráficos de un dossier: la misma letra, el mismo tamaño y la misma
paleta que el PDF.

    from estilo_graficos import plt, P, SERIES, figura, guardar, num, pct, barras_h, rotular

Cada figura se dibuja a su tamaño final (el ancho del texto de dossier.sty es 174 mm) y
LaTeX la incluye sin escalar, así la letra del gráfico mide lo mismo en el papel que en
el código. La paleta se lee de dossier.sty y de los \\definecolor del .tex de esta carpeta:
si el documento adopta los colores de un proyecto, los gráficos los siguen solos.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

AQUI = Path(__file__).resolve().parent
FIG = AQUI / "fig"
ANCHO_TEXTO_MM = 174
MM = 1 / 25.4

_DEFINECOLOR = re.compile(r"\\definecolor\{(\w+)\}\{HTML\}\{([0-9A-Fa-f]{6})\}")


def leer_paleta(*archivos: Path) -> dict[str, str]:
    """Colores `\\definecolor{nombre}{HTML}{RRGGBB}` de los archivos dados; el último gana.
    Ignora las líneas comentadas."""
    paleta: dict[str, str] = {}
    for archivo in archivos:
        if not archivo.exists():
            continue
        for linea in archivo.read_text(encoding="utf-8").splitlines():
            codigo = re.split(r"(?<!\\)%", linea, maxsplit=1)[0]
            for nombre, valor in _DEFINECOLOR.findall(codigo):
                paleta[nombre] = "#" + valor.lower()
    return paleta


P = leer_paleta(AQUI / "dossier.sty", *sorted(AQUI.glob("*.tex")))

# Colores para distinguir categorías o series: la paleta de referencia de la skill dataviz,
# que pasa su validador sobre fondo blanco en este orden (ver dataviz/references/palette.md).
# Los colores del documento (P) sirven para destacar una serie contra gris, no para
# distinguir categorías: el acento es demasiado oscuro y apagado para ese uso. Si el
# proyecto trae su propia paleta categórica, reemplazar esta lista y validarla.
# Con puntos o mapas, donde cualquier par de colores queda junto, usar solo los tres primeros.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def _fuente() -> str:
    """La familia del documento si el sistema la tiene; si no, una sin serifa cualquiera.

    Avenir Next viene en una colección .ttc y matplotlib solo lee su primera cara, que es
    la Bold: todo el gráfico saldría en negrita. Por eso se extraen la Regular, la Demi
    Bold (la negrita del documento) y la Italic a archivos sueltos, una sola vez.
    """
    ttc = Path("/System/Library/Fonts/Avenir Next.ttc")
    if not ttc.exists():
        return "DejaVu Sans"
    try:
        import logging

        from fontTools.ttLib import TTCollection

        logging.getLogger("fontTools").setLevel(logging.ERROR)
        cache = Path(matplotlib.get_cachedir()) / "dossier-fuentes"
        cache.mkdir(parents=True, exist_ok=True)
        coleccion = None
        for cara in ("Regular", "Demi Bold", "Italic"):
            destino = cache / f"AvenirNext-{cara.replace(' ', '')}.ttf"
            if not destino.exists():
                coleccion = coleccion or TTCollection(str(ttc))
                for fuente in coleccion.fonts:
                    if fuente["name"].getDebugName(4) == f"Avenir Next {cara}":
                        fuente.save(str(destino))
                        break
            fm.fontManager.addfont(str(destino))
        return "Avenir Next"
    except Exception:
        return "DejaVu Sans"


plt.rcParams.update({
    "font.family": [_fuente(), "DejaVu Sans"],
    "font.size": 8.5,
    "text.color": P.get("tinta", "#16233a"),
    "axes.edgecolor": P.get("linea", "#b9c0cc"),
    "axes.labelcolor": P.get("rotulo", "#3d4d66"),
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 9,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "xtick.color": P.get("apagado", "#5f6779"),
    "ytick.color": P.get("apagado", "#5f6779"),
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def figura(alto_mm: float = 60, ancho: float = 1.0, **kw):
    """`plt.subplots` al tamaño final. `ancho` es la fracción del ancho del texto."""
    return plt.subplots(figsize=(ANCHO_TEXTO_MM * ancho * MM, alto_mm * MM), **kw)


def guardar(fig, nombre: str) -> Path:
    """Guarda en fig/ como PDF vectorial (texto seleccionable) y cierra la figura."""
    FIG.mkdir(exist_ok=True)
    ruta = FIG / nombre
    fig.savefig(ruta, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return ruta


def num(x: float, dec: int = 0) -> str:
    """12345.6 -> '12.345,6' (miles con punto, decimales con coma)."""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x: float, dec: int = 1) -> str:
    return f"{num(x, dec)} %"


def _luminancia(hexa: str) -> float:
    canales = [int(hexa.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lineales = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in canales]
    return 0.2126 * lineales[0] + 0.7152 * lineales[1] + 0.0722 * lineales[2]


def texto_sobre(fondo: str) -> str:
    """Blanco o tinta, el que más contraste tenga sobre `fondo`: para rótulos dentro de una
    barra. Sobre el naranja de SERIES, el blanco no llega a 3,5:1 y la tinta sí."""
    tinta = P.get("tinta", "#16233a")
    lf = _luminancia(fondo)
    con_blanco = 1.05 / (lf + 0.05)
    con_tinta = (max(lf, _luminancia(tinta)) + 0.05) / (min(lf, _luminancia(tinta)) + 0.05)
    return "white" if con_blanco >= con_tinta else tinta


def rotular(ax, barras, fmt=num, color=None):
    """Escribe el valor al final de cada barra (columnas o barras horizontales).

    `barras` es lo que devuelven `ax.bar` o `ax.barh`. Cuando el valor está escrito, no
    hace falta eje: conviene quitarlo con `ax.set_yticks([])` (o `set_xticks`).
    """
    color = color or P.get("tinta", "#16233a")
    horizontal = barras.orientation == "horizontal"
    tope = 0.0
    for barra in barras:
        x, y, ancho, alto = barra.get_x(), barra.get_y(), barra.get_width(), barra.get_height()
        if horizontal:
            tope = max(tope, x + ancho)
            ax.annotate(fmt(ancho), (x + ancho, y + alto / 2), xytext=(3, 0),
                        textcoords="offset points", va="center", ha="left", color=color)
        else:
            tope = max(tope, y + alto)
            ax.annotate(fmt(alto), (x + ancho / 2, y + alto), xytext=(0, 2),
                        textcoords="offset points", va="bottom", ha="center", color=color)
    # aire para que el rótulo más largo no choque con el título ni con el borde
    if horizontal:
        ax.set_xlim(right=tope * 1.14)
    else:
        ax.set_ylim(top=tope * 1.16)
    return ax


def barras_h(ax, etiquetas, valores, destacar=None, fmt=num, color=None, color_destacado=None):
    """Barras horizontales desde cero, de mayor a menor, con el valor al final de cada una.

    Es la forma que conviene cuando los valores están cerca (49,6 y 50,4): las etiquetas
    encima de puntos se pisan y una barra que arranca en cero no exagera la diferencia.
    `destacar` es la etiqueta (o lista de etiquetas) que va en el color de acento.
    """
    pares = sorted(zip(etiquetas, valores), key=lambda par: par[1])
    destacadas = {destacar} if isinstance(destacar, str) else set(destacar or [])
    base = color or P.get("linea", "#b9c0cc")
    fuerte = color_destacado or P.get("acento", "#22456f")
    colores = [fuerte if e in destacadas else base for e, _ in pares]
    ys = range(len(pares))
    ax.barh(list(ys), [v for _, v in pares], color=colores, height=0.62)
    ax.set_yticks(list(ys), [e for e, _ in pares])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    tope = max(v for _, v in pares) if pares else 1
    for y, (e, v) in zip(ys, pares):
        ax.text(v + tope * 0.012, y, fmt(v), va="center", ha="left",
                color=P.get("tinta", "#16233a"),
                fontweight="semibold" if e in destacadas else "normal")
    ax.set_xlim(0, tope * 1.14)
    return ax

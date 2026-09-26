r"""Correr con: python -m tests.test_cuadernillo

Pruebas de src/cuadernillo.py (paso 7.5). Las seis primeras no necesitan nada instalado: el parser
del guion sobre un guion sintético, la conversión de markdown a LaTeX, el párrafo que presenta una
variante hablada, la elección de la página final de cada frame a partir de los números al pie, el
emparejamiento del guion con el deck, con sus avisos, y la lectura del .log (cuándo pide otra
pasada, qué avisos se muestran y cómo quedó cada imagen). Las tres últimas generan el cuadernillo de
verdad y se saltan, con un aviso, si falta LuaLaTeX o poppler: dos sobre un deck beamer sintético
que se compila aquí (con overlays \pause, un «80/20» en el cuerpo y un frame de respaldo), la
segunda con páginas cuyo texto no deja lugar a la imagen y un banco de 40 preguntas; y una sobre
informe/presentacion.pdf e informe/guion.md, si existen, que exige que el guion y el deck coincidan
y que el .log salga limpio. Ninguna escribe en informe/: la salida va a un directorio temporal.
"""

import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.cuadernillo import (
    ANCHO_COMPLETO,
    GUION,
    HERRAMIENTAS,
    MOTOR,
    POR_PAGINA_RESPALDO,
    PREAMBULO,
    PRESENTACION,
    RESULTADOS_TEST,
    Pregunta,
    Rotulo,
    a_reloj,
    armar_paginas,
    bloques_md,
    comparar_con_nav,
    comparar_mapa,
    en_linea,
    escapar,
    frames_de_rotulos,
    generar,
    imagenes_del_log,
    latex_de_tabla,
    latex_slide,
    metadatos_pdf,
    parsear_guion,
    pide_otra_pasada,
    rangos_del_nav,
    resumen_de_imagenes,
    revisar_frames,
    revisar_log,
    rotulo_de_pie,
    rotulo_de_variante,
    texto_plano,
    variantes,
    _rangos,
)

GUION_SINTETICO = """\
# Guion sintético — TP2 · Grupo 7 · 07/10/2026 (o 14/10) · 10 + 10 minutos

Texto de presentación que no es ninguna sección.

## Convenciones

- Ritmo de referencia: **3,0 palabras por segundo**.

## El guion

### 1 · Portada — 0:00 → 0:05 (~6 palabras)

**Habla: Ana** (bloque 1, slides 1–2).

> Hola. Somos el grupo siete.

#### A aclarar
- **Una aclaración** con `codigo_con_guion-bajo` y 41 188 filas
  que sigue en la línea de abajo.
- Otra, con 11,3 % y $F_1$.

### 2 · Tres pasos — 0:05 → 0:20 (~30 palabras)

**Habla: Beto.** Una indicación de ensayo, que no se dice.

> Primero esto.
> **[→]** Después esto otro.
> **[→]** Y al final, lo último.

#### A aclarar
- **¿Una pregunta?** Una respuesta.

### 3 · Con variantes — 0:20 → 0:30 (~12 palabras)

**Habla: Beto.**

> Da ? de AUC en test.

**Mientras el test siga cerrado** (ensayos hasta el 01/10), en lugar de lo anterior:

> Todavía no lo abrimos.

El encabezado cuenta la variante más larga.

#### A aclarar
- Nada más.

---

## Los 10 minutos de preguntas

Respuestas cortas.

**Para volver a una slide.** Página de la última variante de cada slide: 1 → 1 · 2 → 4 ·
3 → 5 · R1 → 6.

### Sobre los datos

**1. ¿Por qué así?** [slide 2 · Ana] — Porque sí: 0,795 de AUC de validación.

**2. ¿Y el respaldo?** [slide 3, respaldo R1 · Beto]
— Está en el respaldo.

Un párrafo que no es una pregunta.

## Reloj de ensayo

| Slide | Título | Empieza |
| --- | :---: | ---: |
| 1 | Portada | 0:00 |
| 2 | Tres pasos | 0:05 |

**Puntos de control al ensayar.**

- Al terminar la **slide 2** tienen que haber pasado **0:20**: el control.

**Orden de recorte.**

1. **Primero**, lo último.
   Sigue el ítem.
2. Después, lo otro.
"""

VARIANTE_CERRADA = ("**Mientras el test siga cerrado** (ensayos hasta el 01/10), en lugar de lo "
                    "anterior:")

DECK_SINTETICO = r"""\documentclass[aspectratio=169]{beamer}
\setbeamertemplate{navigation symbols}{}
\setbeamertemplate{footline}{\hbox to\paperwidth{\hfill\tiny
  \ifnum\insertframenumberinappendix>0 R\insertframenumberinappendix/\insertappendixframenumber
  \else\insertframenumber/\insertmainframenumber\fi\hspace*{1em}}\vspace*{3pt}}
\begin{document}
\begin{frame}{Portada}Uno\end{frame}
\begin{frame}{Tres pasos}Partimos 80/20 estratificado.\pause{} Paso dos.\pause{}
  Paso tres.\end{frame}
\begin{frame}{Con variantes}Final\end{frame}
\appendix
\begin{frame}{Respaldo}Una figura de reserva.\end{frame}
\end{document}
"""

TEST_PENDIENTE = "\\newif\\iftestpendiente\\testpendientetrue\n"

# Un .log de LuaLaTeX que ya convergió: hyperref carga rerunfilecheck, así que su nombre está en
# todo .log, pida otra pasada o no.
LOG_CONVERGIDO = (
    "(/usr/local/texlive/2025/texmf-dist/tex/latex/rerunfilecheck/rerunfilecheck.sty\n"
    "Package: rerunfilecheck 2022-07-10 v1.10 Rerun checks for auxiliary files (HO)\n"
    "Package uniquecounter Info: New unique counter `rerunfilecheck' on input line 285.\n"
    "Package rerunfilecheck Info: File `cuadernillo.out' has not changed.\n"
    "(rerunfilecheck)             Checksum: 3B24060B4747C4663937AD403A4BC2BC;8315.\n")

# Palabras para rellenar las respuestas del banco largo, con largos distintos.
RELLENO = ("la respuesta sigue con palabras de relleno para que cada pieza tenga un largo distinto "
           "y los cortes de columna caigan en lugares variados del banco").split()


def _r(texto):
    """Un rótulo a partir de «2/3» o «R1/1»."""
    return rotulo_de_pie("EDA   " + texto)


def _frames_sinteticos():
    return frames_de_rotulos([_r("1/3"), _r("2/3"), _r("2/3"), _r("2/3"), _r("3/3"), _r("R1/1")])


def _faltan_herramientas():
    return [h for h in HERRAMIENTAS + (MOTOR,) if shutil.which(h) is None]


def _texto_de_pagina(pdf, pagina):
    return subprocess.run(["pdftotext", "-f", str(pagina), "-l", str(pagina), "-layout",
                           "-enc", "UTF-8", str(pdf), "-"], capture_output=True, text=True,
                          check=True, encoding="utf-8").stdout


def _compilar_deck(directorio):
    """Compila DECK_SINTETICO en `directorio`, con el test pendiente, y devuelve la ruta del PDF."""
    (directorio / "deck.tex").write_text(DECK_SINTETICO, encoding="utf-8")
    for _ in range(2):   # la segunda pasada fija \insertmainframenumber
        subprocess.run([MOTOR, "-interaction=nonstopmode", "-halt-on-error", "deck.tex"],
                       cwd=directorio, capture_output=True, check=True)
    (directorio / "resultados-test.tex").write_text(TEST_PENDIENTE, encoding="utf-8")
    return directorio / "deck.pdf"


def _palabras(pdf, desde=1):
    """Cada palabra del PDF, desde la página `desde`: (texto, página, columna, arriba, abajo).

    Sale de pdftotext -tsv, en puntos desde el borde superior (-bbox falla en este poppler con el
    campo Keywords vacío). La columna es «izq» o «der»: de qué lado de la mitad de la página
    empieza la palabra.
    """
    tsv = subprocess.run(["pdftotext", "-f", str(desde), "-tsv", "-enc", "UTF-8", str(pdf), "-"],
                         capture_output=True, text=True, check=True, encoding="utf-8").stdout
    anchos, palabras = {}, []
    for fila in tsv.splitlines()[1:]:
        c = fila.split("\t")
        if c[0] == "1":          # la página: su ancho
            anchos[int(c[1])] = float(c[8])
        elif c[0] == "5":        # una palabra
            pagina, izquierda, arriba = int(c[1]), float(c[6]), float(c[7])
            columna = "izq" if izquierda < anchos[pagina] / 2 else "der"
            palabras.append((c[11], pagina, columna, arriba, arriba + float(c[9])))
    return palabras


def _fondo_del_area(pdf):
    """Dónde termina el área de texto, en puntos desde arriba.

    Es el alto de la página, según pdfinfo, menos el margen inferior del PREAMBULO.
    """
    alto = float(re.search(r"x ([\d.]+) pts", metadatos_pdf(pdf)["Page size"]).group(1))
    margen = float(re.search(r"bottom=([\d.]+)cm", PREAMBULO).group(1))
    return alto - margen / 2.54 * 72


def _con_paginas_llenas(texto, lineas):
    """El guion con un «A aclarar» de lineas[i] ítems de una línea en la slide i + 1."""
    partes = re.split(r"#### A aclarar\n(?:- .*\n(?:  .*\n)?)+", texto)
    assert len(partes) == len(lineas) + 1, len(partes)
    aclarar = ["#### A aclarar\n" + "".join(f"- **Punto {i}.** Una línea de relleno para medir "
                                            "el alto.\n" for i in range(1, n + 1))
               for n in lineas]
    return "".join(parte + extra for parte, extra in zip(partes, aclarar + [""]))


def _con_banco_largo(texto, grupos=(("Alfa", 12), ("Beta", 15), ("Gama", 13))):
    """El guion con un banco de preguntas de largos variados en lugar del suyo.

    La pregunta k se llama «¿Qué pasa con Tk?» y su respuesta termina en «Fk.»: dos palabras que
    sólo aparecen ahí, para ubicarlas en el PDF.
    """
    banco, k = [], 0
    for nombre, cuantas in grupos:
        banco.append(f"### Grupo {nombre}\n")
        for _ in range(cuantas):
            k += 1
            relleno = " ".join(RELLENO[(7 * k + i) % len(RELLENO)] for i in range(8 + 13 * k % 50))
            banco.append(f"**{k}. ¿Qué pasa con T{k}?** [slide 2 · Ana] — {relleno} F{k}.\n")
    inicio, fin = texto.index("### Sobre los datos"), texto.index("## Reloj de ensayo")
    return texto[:inicio] + "\n".join(banco) + "\n" + texto[fin:]


# =================================================================================================

def test_parser_del_guion_sintetico():
    g = parsear_guion(GUION_SINTETICO)
    assert g.titulo.startswith("Guion sintético") and g.avisos == [], g.avisos
    assert g.ritmo == 3.0
    assert [s.numero for s in g.slides] == [1, 2, 3]
    assert [s.titulo for s in g.slides] == ["Portada", "Tres pasos", "Con variantes"]
    assert [(s.inicio, s.fin, s.palabras) for s in g.slides] == [
        ("0:00", "0:05", 6), ("0:05", "0:20", 30), ("0:20", "0:30", 12)]
    assert [s.habla for s in g.slides] == ["Ana", "Beto", "Beto"]
    assert g.palabras == 48 and g.duracion == 30 and a_reloj(g.duracion) == "0:30"

    uno, dos, tres = g.slides
    # «Habla» sale del párrafo y lo que sigue queda como indicación de ensayo.
    assert uno.bloques[0] == ("nota", "(bloque 1, slides 1–2).")
    assert uno.dichos[0].segmentos == ["Hola. Somos el grupo siete."] and uno.dichos[0].avances == 0
    items = [b for b in bloques_md(uno.aclarar) if b.tipo == "lista"][0].lineas
    assert items == ["**Una aclaración** con `codigo_con_guion-bajo` y 41 188 filas que sigue en la "
                     "línea de abajo.", "Otra, con 11,3 % y $F_1$."], items
    assert uno.titulo_aclarar == "A aclarar"

    assert dos.bloques[0] == ("nota", "Una indicación de ensayo, que no se dice.")
    assert dos.dichos[0].segmentos == ["Primero esto.", "Después esto otro.",
                                       "Y al final, lo último."]
    assert dos.dichos[0].avances == 2

    # Dos variantes: el párrafo que presenta la segunda es su etiqueta, entero, no una indicación.
    assert [t for t, _ in tres.bloques] == ["dicho", "dicho", "nota"]
    assert tres.dichos[1].etiqueta == VARIANTE_CERRADA
    assert variantes(tres.dichos, True)[0].segmentos == ["Todavía no lo abrimos."]
    assert variantes(tres.dichos, False)[0].segmentos == ["Da ? de AUC en test."]
    assert variantes(tres.dichos, None)[0].segmentos == ["Da ? de AUC en test."]
    assert len(variantes(tres.dichos, True)[1]) == 1

    # Preguntas: número, pregunta, etiqueta y respuesta, aunque la etiqueta corte la línea.
    assert [p.numero for p in g.preguntas] == [1, 2]
    p1, p2 = g.preguntas
    assert (p1.pregunta, p1.etiqueta) == ("¿Por qué así?", "slide 2 · Ana")
    assert p1.respuesta == "Porque sí: 0,795 de AUC de validación."
    assert (p2.etiqueta, p2.respuesta) == ("slide 3, respaldo R1 · Beto", "Está en el respaldo.")
    assert g.grupos[0].titulo == "Sobre los datos"
    assert not isinstance(g.grupos[0].elementos[-1], Pregunta)
    assert g.mapa_paginas == {"1": 1, "2": 4, "3": 5, "R1": 6}

    # Reloj: la tabla y las listas del guion, y los puntos de control.
    assert [b.tipo for b in g.reloj] == ["tabla", "parrafo", "lista", "parrafo", "enumeracion"]
    assert g.reloj[-1].lineas == [("1", "**Primero**, lo último. Sigue el ítem."),
                                  ("2", "Después, lo otro.")]
    assert g.controles == {2: "0:20"}

    # Un guion sin la sección de slides no rompe: avisa.
    vacio = parsear_guion("# Nada\n\nSólo texto.\n")
    assert vacio.slides == [] and len(vacio.avisos) == 2
    print("ok  el parser lee encabezados, quién habla, indicaciones, [→], variantes, A aclarar, "
          "preguntas y reloj")


def test_markdown_a_latex():
    assert en_linea("**Hola** y *Bank Marketing*") == r"\textbf{Hola} y \emph{Bank Marketing}"
    assert en_linea("`day_of_week`") == r"\texttt{day\_\allowbreak{}of\_\allowbreak{}week}"
    assert en_linea("`src/a-b.py`") == r"\texttt{src/\allowbreak{}a\mbox{-}b.py}"
    # Negrita que contiene código, y la marca de avance.
    assert en_linea("**¿`campaign` existe?**") == r"\textbf{¿\texttt{campaign} existe?}"
    assert en_linea("**[→]** Sigue.") == r"\avanza{} Sigue."
    # Miles y porcentajes con espacio fino, sin tocar lo que no son miles.
    assert en_linea("41 188 filas, 3 711 «yes» y 11,3 %") == r"41\,188 filas, 3\,711 «yes» y 11,3\,\%"
    assert en_linea("1 492 palabras / 2,7 = 552,6 s") == r"1\,492 palabras / 2,7 = 552,6 s"
    for igual in ("entre 3 y 5", "0,782 ± 0,007", "1 → 1 · 2 → 4", "slides 88–89", "k = 801"):
        assert en_linea(igual) == igual, igual
    # Matemática tal cual; los especiales de LaTeX, escapados.
    assert en_linea("¿Y $F_1$?") == "¿Y $F_1$?"
    assert en_linea(r"50 % & #1 ~ <b> {x} \note") == (
        r"50\,\% \& \#1 \textasciitilde{} \textless{}b\textgreater{} \{x\} \textbackslash{}note")
    assert escapar('a_b "c"') == r"a\_b \textquotedbl{}c\textquotedbl{}"
    assert texto_plano("**¿`pdays` < 999?** *sí* **[→]**") == "¿pdays < 999? sí [→]"
    tabla = latex_de_tabla(["| a | b | c |", "| --- | :---: | ---: |", "| **1** | 2 | 3 |"])
    assert r"\begin{tabular}{@{}lcr@{}}" in tabla and r"\textbf{1} & 2 & 3 \\" in tabla
    assert _rangos([1, 2, 3, 8, 9, 12]) == "1–3, 8–9 y 12" and _rangos([4]) == "4"
    print("ok  markdown a LaTeX: negrita, cursiva, código, [→], miles, %, matemática, escapes y tablas")


def test_parrafo_de_la_variante_completo():
    # El párrafo entero: la negrita, como rótulo, y el resto tal cual. Si la variante se adelanta a
    # la que reemplaza, su «en lugar de lo anterior» apunta a la de abajo.
    assert rotulo_de_variante(VARIANTE_CERRADA, adelantada=True) == (
        r"\etiqueta{Mientras el test siga cerrado} (ensayos hasta el 01/10), en lugar de la "
        r"variante de abajo, en gris:")
    assert rotulo_de_variante(VARIANTE_CERRADA) == (
        r"\etiqueta{Mientras el test siga cerrado} (ensayos hasta el 01/10), en lugar de lo "
        r"anterior:")
    assert rotulo_de_variante("**Mientras el test siga cerrado**, la matriz de la slide es la de "
                              "validación, y se dice:", adelantada=True) == (
        r"\etiqueta{Mientras el test siga cerrado}, la matriz de la slide es la de validación, "
        r"y se dice:")
    assert rotulo_de_variante("Con el test *cerrado*, se dice:") == (
        r"Con el test \emph{cerrado}, se dice:")

    # En la página: con el test pendiente, el párrafo completo encabeza la variante principal; con
    # el test evaluado, va completo sobre la variante en gris, donde «lo anterior» sí está arriba.
    g = parsear_guion(GUION_SINTETICO)
    pendiente = armar_paginas(g, _frames_sinteticos(), test_pendiente=True)[2]
    tex = latex_slide(pendiente, 3, g, True)
    assert (r"\rotulovariante{\etiqueta{Mientras el test siga cerrado} (ensayos hasta el 01/10), "
            r"en lugar de la variante de abajo, en gris:}") in tex, tex
    evaluado = armar_paginas(g, _frames_sinteticos(), test_pendiente=False)[2]
    tex = latex_slide(evaluado, 3, g, False)
    assert r"\rotulovariante" not in tex, tex
    assert (r"\variante{\textbf{Mientras el test siga cerrado} (ensayos hasta el 01/10), en lugar "
            r"de lo anterior:}") in tex, tex
    print("ok  el párrafo que presenta una variante va completo, con su «lo anterior» bien "
          "apuntado")


def test_pagina_final_de_cada_frame_sobre_un_caso_sintetico():
    # El pie es la última línea: un «80/20» del cuerpo no se toma por el número de slide.
    pagina = "Train decide\n 80/20 estratificado por y\n\n\nDATOS                   3/20\n"
    assert rotulo_de_pie(pagina) == Rotulo(False, 3, 20)
    assert rotulo_de_pie("título\nRESPALDO      R2/4  \n\n") == Rotulo(True, 2, 4)
    assert rotulo_de_pie("sin número al pie") is None and rotulo_de_pie("") is None
    assert rotulo_de_pie("Partimos 80/20\n") == Rotulo(False, 80, 20)   # por eso se revisa

    frames = _frames_sinteticos()
    assert [f.nombre for f in frames] == ["1", "2", "3", "R1"]
    assert [f.paginas for f in frames] == [[1], [2, 3, 4], [5], [6]]
    assert [f.final for f in frames] == [1, 4, 5, 6] and [f.pasos for f in frames] == [0, 2, 0, 0]
    assert revisar_frames(frames) == []

    # Numeración rota, total que no coincide y páginas sin pie: se avisa.
    salteado = frames_de_rotulos([_r("1/3"), _r("3/3"), None])
    avisos = revisar_frames(salteado)
    assert len(avisos) == 3, avisos
    assert "1, 3" in avisos[0] and "2 frames" in avisos[1] and "3" in avisos[2]
    assert [f.paginas for f in salteado] == [[1], [2], [3]]

    # El .nav de beamer como control cruzado.
    nav = ("\\headcommand {\\beamer@framepages {1}{1}}\n\\headcommand {\\beamer@framepages {2}{4}}\n"
           "\\headcommand {\\beamer@framepages {5}{5}}\n\\headcommand {\\beamer@framepages {6}{6}}\n")
    rangos = rangos_del_nav(nav)
    assert rangos == [(1, 1), (2, 4), (5, 5), (6, 6)]
    assert comparar_con_nav(frames, rangos, 6) == []
    otro = [(1, 1), (2, 3), (4, 5), (6, 6)]
    assert len(comparar_con_nav(frames, otro, 6)) == 1
    assert comparar_con_nav(frames, [(1, 1), (2, 7)], 6) == []   # un .nav de otra compilación
    print("ok  la página final de cada frame sale de los números al pie (con overlays, un «80/20» "
          "en el cuerpo y el respaldo), con el .nav como control")


def test_emparejamiento_del_guion_con_el_deck():
    g = parsear_guion(GUION_SINTETICO)
    frames = _frames_sinteticos()
    paginas = armar_paginas(g, frames, test_pendiente=True)
    assert [(p.numero, p.imagen, p.frame.final) for p in paginas] == [
        (1, "slide-01.png", 1), (2, "slide-02.png", 4), (3, "slide-03.png", 5)]
    assert all(p.avisos == [] for p in paginas)
    assert comparar_mapa(g, frames) == []

    # Un overlay de menos en el PDF, una slide del guion sin frame y un frame sin guion.
    otros = frames_de_rotulos([_r("1/4"), _r("2/4"), _r("2/4"), _r("4/4")])
    paginas = {p.numero: p for p in armar_paginas(g, otros, test_pendiente=True)}
    assert sorted(paginas) == [1, 2, 3, 4]
    assert "marca 2 [→]" in paginas[2].avisos[0] and "1 paso" in paginas[2].avisos[0]
    assert paginas[3].imagen is None and "no tiene la slide 3" in paginas[3].avisos[0]
    assert paginas[4].slide is None and "no tiene texto" in paginas[4].avisos[0]
    assert comparar_mapa(g, otros) == [("2", 4, 3), ("3", 5, None), ("R1", 6, None)]
    print("ok  el guion se empareja con el deck por número y avisa [→] distintos, slides sin "
          "imagen o sin texto y un mapa de páginas viejo")


def test_lectura_del_log():
    # Otra pasada sólo si un aviso la pide: el nombre de rerunfilecheck, solo, no la pide.
    assert not pide_otra_pasada(LOG_CONVERGIDO)
    for pedido in ("Package rerunfilecheck Warning: File `cuadernillo.out' has changed.\n"
                   "(rerunfilecheck)                Rerun to get outlines right\n"
                   "(rerunfilecheck)                or use package `bookmark'.\n",
                   "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references "
                   "right.\n",
                   "LaTeX Warning: Label(s) may have changed. Rerun to get\ncross-references "
                   "right.\n",
                   "Package hyperref Warning: Rerun to get /PageLabels entry.\n",
                   "Package longtable Warning: Table widths have changed. Rerun LaTeX.\n"):
        assert pide_otra_pasada(LOG_CONVERGIDO + pedido), pedido

    # Los underfull \hbox, desde badness 10000; los desbordes, siempre; sin repetir.
    log = ("Underfull \\hbox (badness 10000) in paragraph at lines 250--250\n"
           "[]\\TU/FiraSans(0)/b/n/9 ¿\n"
           "Underfull \\hbox (badness 1102) in paragraph at lines 619--619\n"
           "Overfull \\hbox (1.2pt too wide) in paragraph at lines 3--4\n"
           "Overfull \\vbox (392.87346pt too high) detected at line 398\n"
           "Underfull \\hbox (badness 10000) in paragraph at lines 250--250\n")
    assert revisar_log(log) == ["Underfull \\hbox (badness 10000) in paragraph at lines 250--250",
                                "Overfull \\hbox (1.2pt too wide) in paragraph at lines 3--4",
                                "Overfull \\vbox (392.87346pt too high) detected at line 398"]

    # Cómo quedó cada imagen, según las líneas «cuadernillo: slide …» del .log.
    imagenes = imagenes_del_log("cuadernillo: slide 1, imagen 278.01678pt de 278.01678pt, letra 0\n"
                                "cuadernillo: slide 7, imagen 271.04382pt de 278.01678pt, letra 0\n"
                                "cuadernillo: slide 9, imagen 150.0pt de 278.01678pt, letra 1\n")
    assert imagenes[1] == (1.0, 0) and imagenes[9][1] == 1
    assert math.isclose(imagenes[7][0], 271.04382 / 278.01678)
    assert resumen_de_imagenes(imagenes) == (
        "Imágenes: 1 de 3 a todo el ancho de la caja de texto; por el largo del texto, la de la "
        "slide 7 va al 97,5 % del ancho y la de la slide 9 va al 54,0 % del ancho, con la letra un "
        "punto más pequeña.")
    assert resumen_de_imagenes({1: (1.0, 0), 2: (1.0, 0)}) == (
        "Imágenes: las 2 a todo el ancho de la caja de texto.")
    print("ok  el .log: otra pasada sólo si un aviso la pide, underfull desde badness 10000 y el "
          "ancho de cada imagen")


def test_generacion_sobre_un_deck_beamer_sintetico():
    faltan = _faltan_herramientas()
    if faltan:
        print(f"--  saltado: faltan {', '.join(faltan)} para compilar un deck y el cuadernillo")
        return
    with tempfile.TemporaryDirectory() as temporal:
        d = Path(temporal)
        deck = _compilar_deck(d)
        (d / "guion.md").write_text(GUION_SINTETICO, encoding="utf-8")
        r = generar(presentacion=deck, guion=d / "guion.md", salida=d / "salida.pdf",
                    resultados_test=d / "resultados-test.tex", nav=None)

        assert [f.final for f in r.frames] == [1, 4, 5, 6], [f.paginas for f in r.frames]
        assert r.avisos == [] and r.avisos_log == [], (r.avisos, r.avisos_log)
        assert r.test_pendiente is True
        # Converge en dos pasadas: la segunda ya no pide otra.
        assert r.corridas == 2, r.corridas
        # Con textos cortos, las tres imágenes van a todo el ancho y en la letra normal.
        assert sorted(r.imagenes) == [1, 2, 3], r.imagenes
        assert all(f >= ANCHO_COMPLETO and letra == 0 for f, letra in r.imagenes.values())
        paginas = int(metadatos_pdf(d / "salida.pdf")["Pages"])
        # portada, reloj, las 3 slides, 1 página de respaldo y las preguntas
        assert paginas == r.paginas == r.etiquetas["fin"] == 7, (paginas, r.etiquetas)
        assert [r.etiquetas[f"slide-{n}"] for n in (1, 2, 3)] == [3, 4, 5]

        dos = _texto_de_pagina(d / "salida.pdf", 4)
        assert "slide 2 de 3" in dos and "Tres pasos" in dos and "Habla: Beto" in dos
        assert "Después esto otro." in dos and dos.count("[→]") >= 2
        tres = _texto_de_pagina(d / "salida.pdf", 5)
        # Con el test pendiente, primero la variante del test cerrado, con su párrafo completo.
        assert tres.index("Todavía no lo abrimos.") < tres.index("Da ? de AUC en test.")
        assert ("MIENTRAS EL TEST SIGA CERRADO (ensayos hasta el 01/10), en lugar de la variante "
                "de abajo, en gris:") in " ".join(tres.split()), tres
        preguntas = _texto_de_pagina(d / "salida.pdf", 7)
        assert "¿Por qué así?" in preguntas and "[slide 2 · Ana]" in preguntas

        # Un guion desfasado del deck también compila: un [→] de menos y una slide sin imagen,
        # avisados en la salida y en su página.
        roto = GUION_SINTETICO.replace("> **[→]** Y al final, lo último.\n", "").replace(
            "---\n\n## Los 10 minutos", "### 4 · Sin imagen — 0:30 → 0:35 (~3 palabras)\n\n"
            "> Una slide de más.\n\n---\n\n## Los 10 minutos")
        (d / "roto.md").write_text(roto, encoding="utf-8")
        r = generar(presentacion=deck, guion=d / "roto.md", salida=d / "roto.pdf",
                    resultados_test=d / "resultados-test.tex", nav=None)
        assert len(r.avisos) == 2 and r.avisos_log == [], (r.avisos, r.avisos_log)
        assert r.paginas == int(metadatos_pdf(d / "roto.pdf")["Pages"]) == 8
        dos = _texto_de_pagina(d / "roto.pdf", r.etiquetas["slide-2"])
        assert "Aviso: El guion marca 1 [→]" in dos
        cuatro = _texto_de_pagina(d / "roto.pdf", r.etiquetas["slide-4"])
        assert "Esta slide no está en el PDF." in cuatro and "slide 4 de 4" in cuatro
    print("ok  sobre un deck beamer sintético: la página final de cada frame, una página por "
          "slide, dos pasadas, el orden de las variantes, 7 páginas según pdfinfo y, con el guion "
          "desfasado, los avisos")


def test_paginas_llenas_y_banco_sin_cortes():
    faltan = _faltan_herramientas()
    if faltan:
        print(f"--  saltado: faltan {', '.join(faltan)} para compilar un deck y el cuadernillo")
        return
    with tempfile.TemporaryDirectory() as temporal:
        d = Path(temporal)
        deck = _compilar_deck(d)
        guion = _con_banco_largo(_con_paginas_llenas(GUION_SINTETICO, (34, 36, 80)))
        (d / "largo.md").write_text(guion, encoding="utf-8")
        r = generar(presentacion=deck, guion=d / "largo.md", salida=d / "largo.pdf",
                    resultados_test=d / "resultados-test.tex", nav=None)
        assert r.avisos == [] and r.corridas == 2, (r.avisos, r.corridas)
        assert len(r.guion.preguntas) == 40

        # Las tres maneras en que el texto no deja lugar a la imagen: se achica la imagen, con la
        # letra normal (34 líneas); se achica también la letra (36); y ni así entra (80): la imagen
        # se queda en el 45 % y el .log lo avisa. Cada slide sigue en su página.
        (f1, letra1), (f2, letra2), (f3, letra3) = (r.imagenes[n] for n in (1, 2, 3))
        assert 0.62 <= f1 < ANCHO_COMPLETO and letra1 == 0, r.imagenes
        assert 0.45 < f2 and letra2 == 1, r.imagenes
        assert math.isclose(f3, 0.45, abs_tol=1e-3) and letra3 == 1, r.imagenes
        assert len(r.avisos_log) == 1, r.avisos_log
        assert r.avisos_log[0].startswith("Overfull \\vbox"), r.avisos_log
        assert [r.etiquetas[f"slide-{n}"] for n in (1, 2, 3)] == [3, 4, 5], r.etiquetas
        assert "la de la slide 1 va al" in resumen_de_imagenes(r.imagenes)

        # La imagen se achica exactamente lo que falta: la página llega al fondo del área de texto
        # sin pasarse. pdftotext cuenta el descendente entero de la letra y TeX sólo el de la «p»
        # de la última línea: 0,5 pt de diferencia. Con el margen fijo de antes, la página se
        # pasaba sin aviso: la última línea terminaba 2,5 pt más abajo.
        fondo = _fondo_del_area(d / "largo.pdf")
        palabras = _palabras(d / "largo.pdf")
        for pagina in (3, 4):
            abajo = max(p[4] for p in palabras if p[1] == pagina and p[3] < fondo + 8)
            assert abs(abajo - fondo) < 1.5, (pagina, abajo, fondo)

        # El banco: ninguna pregunta partida entre columnas o páginas, y el título de cada grupo en
        # la columna de su primera pregunta. El banco ocupa varias columnas: los cortes existen.
        lugar = {}
        for texto, pagina, columna, _, _ in _palabras(d / "largo.pdf", r.etiquetas["preguntas"]):
            lugar.setdefault(texto, (pagina, columna))
        cortadas = [k for k in range(1, 41) if lugar[f"T{k}?"] != lugar[f"F{k}."]]
        assert not cortadas, [(k, lugar[f"T{k}?"], lugar[f"F{k}."]) for k in cortadas]
        for grupo, primera in (("ALFA", 1), ("BETA", 13), ("GAMA", 28)):
            assert lugar[grupo] == lugar[f"T{primera}?"], (grupo, lugar[grupo])
        assert len({lugar[f"T{k}?"] for k in range(1, 41)}) >= 4, lugar
    print("ok  con textos largos, la imagen se achica lo justo, después la letra, y un desborde se "
          "avisa; en el banco de 40 preguntas, ninguna queda partida")


def test_generacion_sobre_los_archivos_reales():
    if not (PRESENTACION.exists() and GUION.exists()):
        print("--  saltado: no están informe/presentacion.pdf e informe/guion.md (clon de entrega)")
        return
    faltan = _faltan_herramientas()
    if faltan:
        print(f"--  saltado: faltan {', '.join(faltan)}")
        return
    with tempfile.TemporaryDirectory() as temporal:
        salida = Path(temporal) / "slides-y-guion.pdf"
        r = generar(salida=salida, resultados_test=RESULTADOS_TEST)
        paginas = int(metadatos_pdf(salida)["Pages"])
        e = r.etiquetas
        slides = [e[f"slide-{p.numero}"] for p in r.slides]
        respaldo = [f for f in r.frames if f.rotulo and f.rotulo.respaldo]

        # El guion y el deck tienen que coincidir: un [→] de más o de menos, una slide sin texto o
        # sin imagen, un número al pie salteado o el mapa de páginas del guion desactualizado hacen
        # fallar esta prueba (la sintética muestra que cada caso produce su aviso). Hoy no hay
        # ningún aviso legítimo que tolerar, y el .log tiene que salir tan limpio como ellos.
        assert not r.avisos, "El guion y el deck no coinciden:\n" + "\n".join(r.avisos)
        assert not r.avisos_log, "Avisos del .log de LuaLaTeX:\n" + "\n".join(r.avisos_log)
        assert r.corridas == 2, r.corridas

        assert paginas == r.paginas == e["fin"], (paginas, r.paginas, e.get("fin"))
        assert slides == list(range(slides[0], slides[0] + len(slides))), slides
        assert e["portada"] == 1 and slides[0] > e.get("reloj", e["portada"])
        despues = slides[-1] + 1
        if respaldo:
            assert e["respaldo"] == despues
            despues += math.ceil(len(respaldo) / POR_PAGINA_RESPALDO)
        assert e["preguntas"] == despues
        total = len(r.slides)
        assert sorted(r.imagenes) == [p.numero for p in r.slides], r.imagenes
        for p in r.slides:
            texto = _texto_de_pagina(salida, e[f"slide-{p.numero}"])
            assert f"slide {p.numero} de {total}" in texto, p.numero
    print(f"    {resumen_de_imagenes(r.imagenes)}")
    print(f"ok  sobre los archivos reales: {paginas} páginas según pdfinfo ({total} slides desde la "
          f"{slides[0]}, preguntas desde la {e['preguntas']}), cada slide en su página, sin avisos "
          "del guion contra el deck ni del .log")


def main():
    test_parser_del_guion_sintetico()
    test_markdown_a_latex()
    test_parrafo_de_la_variante_completo()
    test_pagina_final_de_cada_frame_sobre_un_caso_sintetico()
    test_emparejamiento_del_guion_con_el_deck()
    test_lectura_del_log()
    test_generacion_sobre_un_deck_beamer_sintetico()
    test_paginas_llenas_y_banco_sin_cortes()
    test_generacion_sobre_los_archivos_reales()
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()

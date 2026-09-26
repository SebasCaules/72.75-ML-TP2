r"""Correr con: python -m tests.test_cuadernillo

Pruebas de src/cuadernillo.py (paso 7.5). Las cuatro primeras no necesitan nada instalado: el
parser del guion sobre un guion sintético, la conversión de markdown a LaTeX, la elección de la
página final de cada frame a partir de los números al pie, y el emparejamiento del guion con el
deck, con sus avisos. Las dos últimas generan el cuadernillo de verdad y se saltan, con un aviso,
si falta LuaLaTeX o poppler: una sobre un deck beamer sintético que se compila aquí (con overlays
\pause, un «80/20» en el cuerpo y un frame de respaldo), y otra sobre informe/presentacion.pdf e
informe/guion.md, si existen. Ninguna escribe en informe/: la salida va a un directorio temporal.
"""

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.cuadernillo import (
    GUION,
    HERRAMIENTAS,
    MOTOR,
    POR_PAGINA_RESPALDO,
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
    latex_de_tabla,
    metadatos_pdf,
    parsear_guion,
    rangos_del_nav,
    revisar_frames,
    rotulo_de_pie,
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

**Mientras el test siga cerrado**, se dice:

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

| Slide | Título | Arranca |
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

    # Dos variantes: el párrafo que presenta la segunda es su etiqueta, no una indicación suelta.
    assert [t for t, _ in tres.bloques] == ["dicho", "dicho", "nota"]
    assert tres.dichos[1].etiqueta == "**Mientras el test siga cerrado**, se dice:"
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


def test_generacion_sobre_un_deck_beamer_sintetico():
    faltan = _faltan_herramientas()
    if faltan:
        print(f"--  saltado: faltan {', '.join(faltan)} para compilar un deck y el cuadernillo")
        return
    with tempfile.TemporaryDirectory() as temporal:
        d = Path(temporal)
        (d / "deck.tex").write_text(DECK_SINTETICO, encoding="utf-8")
        for _ in range(2):   # la segunda pasada fija \insertmainframenumber
            subprocess.run([MOTOR, "-interaction=nonstopmode", "-halt-on-error", "deck.tex"],
                           cwd=d, capture_output=True, check=True)
        (d / "guion.md").write_text(GUION_SINTETICO, encoding="utf-8")
        (d / "resultados-test.tex").write_text(TEST_PENDIENTE, encoding="utf-8")
        r = generar(presentacion=d / "deck.pdf", guion=d / "guion.md", salida=d / "salida.pdf",
                    resultados_test=d / "resultados-test.tex", nav=None)

        assert [f.final for f in r.frames] == [1, 4, 5, 6], [f.paginas for f in r.frames]
        assert r.avisos == [] and r.avisos_log == [], (r.avisos, r.avisos_log)
        assert r.test_pendiente is True
        paginas = int(metadatos_pdf(d / "salida.pdf")["Pages"])
        # portada, reloj, las 3 slides, 1 página de respaldo y las preguntas
        assert paginas == r.paginas == r.etiquetas["fin"] == 7, (paginas, r.etiquetas)
        assert [r.etiquetas[f"slide-{n}"] for n in (1, 2, 3)] == [3, 4, 5]

        dos = _texto_de_pagina(d / "salida.pdf", 4)
        assert "slide 2 de 3" in dos and "Tres pasos" in dos and "Habla: Beto" in dos
        assert "Después esto otro." in dos and dos.count("[→]") >= 2
        tres = _texto_de_pagina(d / "salida.pdf", 5)
        # Con el test pendiente, primero la variante del test cerrado.
        assert tres.index("Todavía no lo abrimos.") < tres.index("Da ? de AUC en test.")
        preguntas = _texto_de_pagina(d / "salida.pdf", 7)
        assert "¿Por qué así?" in preguntas and "[slide 2 · Ana]" in preguntas

        # Un guion desfasado del deck también compila: un [→] de menos y una slide sin imagen,
        # avisados en la salida y en su página.
        roto = GUION_SINTETICO.replace("> **[→]** Y al final, lo último.\n", "").replace(
            "---\n\n## Los 10 minutos", "### 4 · Sin imagen — 0:30 → 0:35 (~3 palabras)\n\n"
            "> Una slide de más.\n\n---\n\n## Los 10 minutos")
        (d / "roto.md").write_text(roto, encoding="utf-8")
        r = generar(presentacion=d / "deck.pdf", guion=d / "roto.md", salida=d / "roto.pdf",
                    resultados_test=d / "resultados-test.tex", nav=None)
        assert len(r.avisos) == 2 and r.avisos_log == [], (r.avisos, r.avisos_log)
        assert r.paginas == int(metadatos_pdf(d / "roto.pdf")["Pages"]) == 8
        dos = _texto_de_pagina(d / "roto.pdf", r.etiquetas["slide-2"])
        assert "Aviso: El guion marca 1 [→]" in dos
        cuatro = _texto_de_pagina(d / "roto.pdf", r.etiquetas["slide-4"])
        assert "Esta slide no está en el PDF." in cuatro and "slide 4 de 4" in cuatro
    print("ok  sobre un deck beamer sintético: la página final de cada frame, una página por "
          "slide, el orden de las variantes, 7 páginas según pdfinfo y, con el guion desfasado, "
          "los avisos")


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

        assert paginas == r.paginas == e["fin"], (paginas, r.paginas, e.get("fin"))
        assert slides == list(range(slides[0], slides[0] + len(slides))), slides
        assert e["portada"] == 1 and slides[0] > e.get("reloj", e["portada"])
        despues = slides[-1] + 1
        if respaldo:
            assert e["respaldo"] == despues
            despues += math.ceil(len(respaldo) / POR_PAGINA_RESPALDO)
        assert e["preguntas"] == despues
        total = len(r.slides)
        for p in r.slides:
            texto = _texto_de_pagina(salida, e[f"slide-{p.numero}"])
            assert f"slide {p.numero} de {total}" in texto, p.numero
        graves = [a for a in r.avisos_log if a.startswith(("Overfull \\vbox", "Missing character"))]
        assert not graves, graves
    for aviso in r.avisos + r.avisos_log:
        print(f"    aviso: {aviso}")
    print(f"ok  sobre los archivos reales: {paginas} páginas según pdfinfo ({total} slides desde la "
          f"{slides[0]}, preguntas desde la {e['preguntas']}), cada slide en su página")


def main():
    test_parser_del_guion_sintetico()
    test_markdown_a_latex()
    test_pagina_final_de_cada_frame_sobre_un_caso_sintetico()
    test_emparejamiento_del_guion_con_el_deck()
    test_generacion_sobre_un_deck_beamer_sintetico()
    test_generacion_sobre_los_archivos_reales()
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()

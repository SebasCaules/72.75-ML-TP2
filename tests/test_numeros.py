"""Correr con: python3 -m tests.test_numeros

src/numeros.py (paso 7.2, gate G3): el formato español de las cifras contra casos calculados a
mano, que los macros se lean igual en texto y dentro de $…$ (compilando con pdflatex si está), los
nombres de los macros, el verificador de números escritos a mano (lo que marca y lo que no), la
línea de comandos sobre un directorio de resultados sintético (dos corridas idénticas, «?» cuando
falta un archivo) y la generación sobre los resultados reales: sólo falta lo que tiene que faltar, y
una cifra de cada familia de macros se recalcula aquí con pandas, sin las funciones del módulo.
"""

import functools
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.ablaciones import VARIANTES
from src.datos import RAIZ, cargar_train
from src.eda_html import analizar, leer_names
from src.evaluar_test import MACROS as MACROS_TEST
from src.evaluar_test import N_BOOTSTRAP, N_TEST, renderizar_tex
from src.evidencia_particion import escribir as escribir_evidencia
from src.evidencia_particion import evidencia
from src.metricas import METRICAS, llamadas
from src.numeros import (
    DEFINICION_MENOS,
    DESCONOCIDO,
    EVIDENCIA_PARTICION,
    MENOS,
    PERMITIDOS,
    POR_ANIO,
    POR_MES,
    RUTA_RESULTADOS_TEST,
    Fuentes,
    _auc_dentro_de_cada_mes,
    _columnas_de,
    _llamando_dentro,
    _pesos_en_el_corte,
    _repetidos,
    a_texto,
    catalogo,
    formatear_decimal,
    formatear_delta,
    formatear_miles,
    formatear_parametro,
    formatear_porcentaje,
    generar,
    nombre_de_macro,
    numero_en_palabras,
    numeros_a_mano,
)
from src.numeros import main as numeros_main
from src.resultados import DIR

# Los que nombra el esqueleto del deck (plan, paso 7.2): tienen que existir con este nombre.
ESPERADOS = {
    "filasTrain", "pctYesTrain", "aucRfVal", "aucRfValDesvio", "aucKnnVal", "aucNbVal",
    "aucSvmVal", "aucSinModelo", "recallRfVal", "recallSinModelo", "brechaRf", "profundidadRf",
    "arbolesRf", "vecinosKnn", "cSvm", "aucRfAdelante", "aucRfAdelanteDesvio",
    "aucRfAdelanteFoldUno", "aucRfAdelanteFoldDos", "aucRfAdelanteFoldTres",
    "aucRfAdelanteFoldCuatro", "aucRfAdelanteFoldCinco", "pctYesBloqueUno", "pctYesBloqueDos",
    "pctYesBloqueTres", "pctYesBloqueCuatro", "pctYesBloqueCinco", "aucRfSinMacroAdelante",
    "deltaAucDurationRf", "aucRfConDuration", "deltaAucSinMacroRf", "deltaAucSinMacroNiMonthRf",
    "gananciaRfVeinte", "recallRfDiez", "recallRfTreinta", "precisionRfVeinte",
    "aucNbGaussianoVal", "deltaAucNbCategorico", "columnasReferencia", "columnasFinales",
    "filasAtipicasIqr", "pctAtipicasIqr", "aucRfProfundidadSinLimite", "aucRfProfundidadDos",
    "aucKnnUno", "aucSvmRbfReferencia", "aucSvmBalanceada", "filasTest", "llamadasTest",
    "yesTrain", "noTrain", "exactitudSiempreNo",
    # Cifras del esqueleto y de DECISIONES.md que la primera versión no tenía: el 999 del título
    # de la slide 6, el factor del IQR, los extremos de las grillas, la regla de 1 ES de los
    # árboles (N0-11), los parámetros del NB categórico (D-18), las 19 predictoras (D-07) y la
    # evidencia de la partición (D-02, D-03).
    "pdaysCentinela", "factorIqr", "profundidadRfMinima", "vecinosKnnMinimo", "vecinosKnnMeseta",
    "arbolesRfRegla", "umbralUnoEsRfArboles", "errorEstandarRfProfundidadMejor",
    "errorEstandarKnnMejor", "cortesNb", "alfaNb", "predictorasDisponibles",
    "desvioPctYesTestSinEstratificar", "pctYesTestSinEstratificarMinimo",
    "pctYesTestSinEstratificarMaximo", "pctYesTrainTemporal", "pctYesTestTemporal",
    # Los que agregó el deck (paso 7.1): el nivel y los remuestreos del intervalo de test (D-24),
    # las llamadas por «yes» de la matriz fuera de fold, y las tres lecturas del hallazgo
    # (resultados/conclusiones.md, §2): pares de años distintos, el AUC barajado en las filas de
    # cada bloque hacia adelante, cuánto pierde hacia adelante y con cuántos «yes» entrena.
    "nivelIntervalo", "remuestreosBootstrap", "llamadasPorYesRf", "llamadasPorYesSinModelo",
    "pctParesEntreAnios", "aucOofEntreAnios", "aucOofDentroAnio", "aucOofDosMilOcho",
    "aucRfBarajadoBloqueUno", "aucRfBarajadoBloqueDos", "aucRfBarajadoBloqueTres",
    "aucRfBarajadoBloqueCuatro", "aucRfBarajadoBloqueCinco", "caidaAucRfBloqueCuatro",
    "caidaAucRfBloqueCinco", "yesEntrenamientoAdelanteUno",
} | {
    # Los que agregó el cierre de la ola 8 para el guion: RF eligiendo a quién llamar dentro de
    # cada año y de cada mes (entonces la pregunta 27), la profundidad 6 con cuatro decimales contra el umbral
    # de 1 ES (slide 10), la calibración de Naive Bayes (slide 16), las filas con las mismas
    # predictoras (pregunta 7) y las diferencias pareadas de los pesos de clase (D-21, N0-12). Con
    # N0-15 el guion ya no cita varios de ellos, porque no se ven en las slides; siguen en la tabla.
    "recallRfDentroAnio", "llamadasPorYesRfDentroAnio", "recallRfDentroAnioDosMilOcho",
    "recallRfDentroAnioDosMilNueve", "recallRfDentroAnioDosMilDiez", "llamadasDentroAnioDosMilDiez",
    "yesDosMilDiez", "recallTopeDosMilDiez", "recallRfDentroMes", "aucOofDentroMes",
    "aucRfProfundidadSeisConCuatroDecimales", "aucRfProfundidadOchoConCuatroDecimales",
    "probabilidadAltaNb", "pctNbProbabilidadAlta", "pctYesNbProbabilidadAlta",
    "probabilidadMediaNbDecilSuperior", "tasaYesNbDecilSuperior", "vectoresRepetidos",
    "filasVectoresRepetidos", "pctFilasVectoresRepetidos", "vectoresRepetidosClasesDistintas",
    "filasVectoresRepetidosClasesDistintas", "vectoresRepetidosConDuration",
    "deltaAucPesosRfDesvio", "foldsPositivosPesosRf", "deltaAucPesosSvmDesvio",
    "foldsPositivosPesosSvm",
}
MACROS_DE_TEST = ("yesTest", "pctYesTest", "diferenciaAucTestValidacion",
                  "diferenciaRecallTestValidacion")
MACROS_DE_PARTICION = ("semillasSinEstratificar", "desvioPctYesTestSinEstratificar",
                       "pctYesTestSinEstratificarMinimo", "pctYesTestSinEstratificarMaximo",
                       "pctYesTrainTemporal", "pctYesTestTemporal")


def _macros(texto_tex):
    return dict(re.findall(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}\s*%", texto_tex, re.M))


@functools.lru_cache(maxsize=1)
def _generado_real():
    return generar(Fuentes())


# --- Formato ------------------------------------------------------------------------------------

def test_formato_espanol_con_casos_a_mano():
    # En LaTeX la coma decimal va entre llaves, para que dentro de $…$ no deje un espacio detrás;
    # en el guion (a_texto) es una coma.
    assert formatear_decimal(0.7952) == "0{,}795" and a_texto("0{,}795") == "0,795"
    assert formatear_decimal(0.7952, 4) == "0{,}7952"
    assert formatear_decimal(0.5, 1) == "0{,}5"
    assert formatear_decimal(1234.5, 1) == r"1\,234{,}5"
    assert formatear_decimal(-0.0276) == r"\signoMenos 0{,}028"
    assert formatear_porcentaje(11.2663) == r"11{,}3\,\%"
    assert formatear_porcentaje(0.0) == r"0{,}0\,\%"
    assert formatear_porcentaje(20, 0) == r"20\,\%"
    assert formatear_miles(32940) == r"32\,940"
    assert formatear_miles(928) == "928"
    assert formatear_miles(1647) == r"1\,647"
    assert formatear_miles(1234567) == r"1\,234\,567"
    assert formatear_miles(32940.0) == r"32\,940"
    assert formatear_miles(-12) == r"\signoMenos 12"
    # Los Δ con la regla de src/graficos.py: tres decimales desde 0,01 y cuatro por debajo.
    assert formatear_delta(0.1697) == "+0{,}170"
    assert formatear_delta(-0.0276) == r"\signoMenos 0{,}028"
    assert formatear_delta(-0.0012327) == r"\signoMenos 0{,}0012"
    assert formatear_delta(0.00029) == "+0{,}0003"
    assert formatear_delta(0.0) == "+0{,}0000"
    assert formatear_parametro(0.001) == "0{,}001"
    assert formatear_parametro(0.01) == "0{,}01"
    assert formatear_parametro(801) == "801"
    assert formatear_parametro(1.0) == "1"
    assert formatear_parametro("balanced") == "balanced"
    for malo in (lambda: formatear_miles(3.5), lambda: formatear_decimal(float("nan"))):
        try:
            malo()
        except ValueError:
            continue
        raise AssertionError("un valor inválido se escribió como cifra")
    # En el guion: coma, espacio común y el menos tipográfico, como en el TP1.
    assert a_texto(r"32\,940") == "32 940"
    assert a_texto(r"11{,}3\,\%") == "11,3 %"
    assert a_texto(r"\signoMenos 0{,}028") == "\N{MINUS SIGN}0,028"
    assert a_texto(r"1\,234{,}5") == "1 234,5"
    # Los intervalos de informe/resultados-test.tex llevan la raya de LaTeX.
    assert a_texto("0{,}812--0{,}834") == "0,812\N{EN DASH}0,834"
    print("ok  formato español: 0{,}795, 11{,}3\\,\\%, 32\\,940, +0{,}170 y \\signoMenos 0{,}0012; "
          "en el guion, 0,795, 11,3 %, 32 940, −0,028 y el intervalo 0,812–0,834")


def test_los_macros_se_leen_igual_en_texto_y_en_modo_matematico():
    """El hallazgo del verificador: con «0,795» y \\textminus, $\\aucRfVal$ se veía «0, 795» y
    $\\deltaAucSinMacroRf$ perdía el signo. Se comprueba en el .tex y, si hay pdflatex, compilando
    todos los macros de los resultados reales en texto y en $…$."""
    generado = _generado_real()
    assert DEFINICION_MENOS in generado.tex
    assert "\\ifmmode-\\else\\textminus\\fi" in DEFINICION_MENOS
    assert "DeclareRobustCommand" in DEFINICION_MENOS, "robusto: sirve en títulos y en tablas"
    assert generado.tex.index(DEFINICION_MENOS) < generado.tex.index("\\newcommand")
    for nombre, valor in generado.valores.items():
        sin_llaves = valor.replace("{,}", "")
        assert "\\textminus" not in valor, (nombre, valor)
        if re.search(r"\d", valor) and nombre != "rotuloLineaBase":
            assert not re.search(r"\d,\d", sin_llaves), f"coma decimal suelta en \\{nombre}: {valor}"
        if valor.startswith("\\"):
            assert valor.startswith(MENOS), (nombre, valor)
    assert generado.valores["deltaAucSinMacroRf"].startswith(MENOS)

    if not shutil.which("pdflatex"):
        print("ok  los macros usan {,} y \\signoMenos (sin pdflatex: no se compiló)")
        return
    nombres = list(generado.valores)
    positivos = [n for n, v in generado.valores.items()
                 if re.search(r"\d(?:\{,\}|,)\d", v) and not v.startswith("\\")]
    anchos = "\n".join(f"\\setbox0\\hbox{{\\{n}}}\\setbox2\\hbox{{$\\{n}$}}"
                       f"\\typeout{{ANCHO {n} \\the\\wd0 \\space y \\the\\wd2}}" for n in positivos)
    cuerpo = "\n".join(f"\\noindent\\{n}{{}} y ${{\\{n}}}$\\par" for n in nombres)
    documento = ("\\documentclass{article}\n\\usepackage[T1]{fontenc}\n"
                 "\\usepackage[spanish]{babel}\n\\spanishplainpercent\n\\input{numeros.tex}\n"
                 "\\begin{document}\n" + anchos + "\n" + cuerpo + "\n"
                 "\\noindent PRUEBA $\\Delta = \\deltaAucSinMacroRf$ y $\\aucRfVal \\pm "
                 "\\aucRfValDesvio$ FIN\n\\end{document}\n")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "numeros.tex").write_text(generado.tex, encoding="utf-8")
        (tmp / "prueba.tex").write_text(documento, encoding="utf-8")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "prueba.tex"],
                           cwd=tmp, capture_output=True, text=True, timeout=300)
        log = (tmp / "prueba.log").read_text(encoding="utf-8", errors="replace")
        if r.returncode != 0 and re.search(r"not found|! Font .* not loadable", log):
            # Una instalación mínima de LaTeX (sin babel-spanish o sin las fuentes T1) no dice
            # nada sobre los macros: la prueba se saltea en lugar de fallar por el entorno.
            print("ok  los macros usan {,} y \\signoMenos (a esta instalación de LaTeX le falta un "
                  "paquete o una fuente: no se compiló)")
            return
        assert r.returncode == 0, log[-3000:]
        problemas = [l for l in log.splitlines()
                     if "invalid in math mode" in l or "Missing character" in l or l.startswith("!")]
        assert not problemas, problemas
        medidas = re.findall(r"^ANCHO (\w+) ([\d.]+)pt y ([\d.]+)pt$", log, re.M)
        assert len(medidas) == len(positivos), (len(medidas), len(positivos))
        distintos = [(n, a, b) for n, a, b in medidas if abs(float(a) - float(b)) > 0.1]
        assert not distintos, f"se ven distinto en $…$ que en texto: {distintos[:5]}"
        if shutil.which("pdftotext"):
            texto = subprocess.run(["pdftotext", str(tmp / "prueba.pdf"), "-"], capture_output=True,
                                   text=True).stdout
            prueba = " ".join(texto[texto.index("PRUEBA"):texto.index("FIN")].split())
            esperado = (a_texto(generado.valores["deltaAucSinMacroRf"]) + " y "
                        + a_texto(generado.valores["aucRfVal"]) + " ± "
                        + a_texto(generado.valores["aucRfValDesvio"]))
            assert esperado in prueba, (esperado, prueba)
    print(f"ok  los {len(nombres)} macros compilan en texto y en $…$ sin avisos; los "
          f"{len(positivos)} con decimales miden lo mismo en los dos modos, y un negativo "
          "conserva su signo dentro de $…$")


def test_numeros_en_palabras():
    casos = {0: "cero", 1: "uno", 16: "dieciséis", 21: "veintiuno", 31: "treinta y uno",
             100: "cien", 101: "ciento uno", 151: "ciento cincuenta y uno", 800: "ochocientos",
             1000: "mil", 2008: "dos mil ocho", 2010: "dos mil diez", 32940: "treinta y dos mil "
             "novecientos cuarenta"}
    for n, palabras in casos.items():
        assert numero_en_palabras(n) == palabras, (n, numero_en_palabras(n))
    try:
        numero_en_palabras(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("un negativo no tiene nombre de macro")
    print("ok  los números en palabras: 21 veintiuno, 151 ciento cincuenta y uno, 2008 dos mil ocho")


def test_nombres_de_macro_solo_letras():
    casos = {("auc", "rf", "val"): "aucRfVal", ("auc", "RF", "val"): "aucRfVal",
             ("pct yes bloque", 1): "pctYesBloqueUno",
             ("auc rf adelante fold", 5): "aucRfAdelanteFoldCinco",
             ("ganancia rf", 20): "gananciaRfVeinte", ("recall rf", 30): "recallRfTreinta",
             ("auc rf profundidad", "sin límite"): "aucRfProfundidadSinLimite",
             ("auc rf profundidad", 2): "aucRfProfundidadDos",
             ("atípicos", "cons.conf.idx"): "atipicosConsConfIdx",
             ("pct yes", 2008): "pctYesDosMilOcho", ("auc knn", 801): "aucKnnOchocientosUno",
             ("delta auc", "SinMacroNiMonth", "Rf"): "deltaAucSinMacroNiMonthRf",
             ("x", "16"): "xDieciseis"}
    for partes, nombre in casos.items():
        assert nombre_de_macro(*partes) == nombre, (partes, nombre_de_macro(*partes))
        assert re.fullmatch(r"[A-Za-z]+", nombre)
    try:
        nombre_de_macro("", "  ")
    except ValueError:
        pass
    else:
        raise AssertionError("un nombre vacío tiene que fallar")

    nombres = [numero.nombre for _, numeros in catalogo() for numero in numeros]
    assert all(re.fullmatch(r"[A-Za-z]+", n) for n in nombres)
    assert len(nombres) == len(set(nombres)), "hay macros repetidos"
    assert not set(nombres) & set(MACROS_TEST), "choca con un macro de resultados-test.tex"
    assert "signoMenos" not in nombres
    faltan = ESPERADOS - set(nombres)
    assert not faltan, f"faltan macros del esqueleto: {sorted(faltan)}"
    print(f"ok  {len(nombres)} macros, todos de letras, sin repetir ni chocar con los de test; "
          f"están los {len(ESPERADOS)} que nombran el esqueleto y DECISIONES.md")


def test_los_titulos_del_esqueleto_se_escriben_con_macros():
    """El plan (§5, ola 7) proyecta el título «999 no significa "nunca contactado"»: con el macro
    pasa el verificador, y tipeado no."""
    frame = "\\begin{document}\n\\begin{frame}{%s no significa «nunca contactado»}\n\\end{frame}\n"
    assert numeros_a_mano(frame % "999") == [(2, "999")]
    assert numeros_a_mano(frame % "\\pdaysCentinela{}") == []
    assert _generado_real().valores["pdaysCentinela"] == "999"
    print("ok  el título de la slide 6 pasa el verificador con \\pdaysCentinela, y con 999 no")


# --- Verificador --------------------------------------------------------------------------------

DECK_LIMPIO = r"""\documentclass[aspectratio=169,11pt]{beamer}
\usepackage[scaled=0.96]{FiraSans}
\definecolor{azul}{HTML}{2A78D6}
\setbeamerfont{title}{size=\fontsize{20}{24}}
\newcommand{\figgrandealto}{0.90}
\newcommand{\figgrande}[2][0.9]{\includegraphics[width=#1\textwidth]{#2}}
\input{numeros}
\begin{document}
\section[Métricas]{Dos métricas que no premian decir siempre «no»}
\begin{frame}<2->[t]{El modelo aprende la época}
  \setbeamerfont{frametitle}{size=\fontsize{18}{22}}
  \includegraphics[width=0.8\textwidth]{03-curva-rf-max-depth.png}
  \figgrande[0.99]{01-ablaciones.png}
  \figgrande{figuras/presentacion/03-curva-rf}
  \renewcommand{\figgrandealto}{0.86}
  \begin{itemize}[<+->]
    \item<2-> Validación con 5 folds, 80/20 y $k = 5$, en 10 minutos.
    \item<3->[a)] \only<3>{Se llama al 20\,\% de la lista con un presupuesto del 20\,\%.}
    \item[1.] Recall al 20\,\%: el 20\,\% de las llamadas. \item [(2)] Otro.
  \end{itemize}
  De 2008 a 2010 (Clase~7, slide~50; Alpaydin §8.3, p.~192; D-22, N0-11).
  Datos de Moro et al., 2014; KNN en Mitchell 1997, cap.~8, pp.~230--236.
  Clase 4, slides 45–51 y slides 26 y 27; atributos 1 a 7; punto 2.3; pasos 6.1 a 6.3.
  El fold 3 y el bloque 5; entrega el 06/10/2026 y consulta el 30/09.
  \vspace{0.6em}\hspace*{-1.5em}{\color{azul!20}texto} \textcolor{naranja!80}{\aucRfVal}
  \kern2pt\phantom{0}\hphantom{00}\vphantom{1}\transduration{2}
  % un comentario con 0,79 no cuenta
  \begin{tikzpicture}[x=1cm]
    \foreach \x in {1,...,5} { \draw (\x,0) circle (0.1); }
    \node[above=1.6mm,visible on=<2->] (a) at (6.025,1.05) {\filasTrain{} filas};
    \draw[azul!20] (0,0) -- node[pos=0.5] {\pctYesTrain} (3,4);
    \matrix (m) [matrix of nodes, row sep=2mm] {
      |[fill=azul!10]| \vpRfVal & \fpRfVal \\ \fnRfVal & \vnRfVal \\ };
  \end{tikzpicture}
  \tikz[baseline] \node[inner sep=2pt] at (0.5,0.5) {\aucRfVal};
  \begin{tabular}{lrr} \rowcolors{2}{azul!10}{white} \multicolumn{2}{c}{\aucKnnVal} \\ \end{tabular}
  \begin{columns}[T]\begin{column}{0.48\textwidth} euribor3m, A7, H1 y TP2 \end{column}
  \end{columns}
  \\[0.4em] \pause[3] $F_1$, $F_{1}$ y $x^2$; \verb|x = 0,79|
  \note{Se dice \aucRfVal{} en voz alta.}
\end{frame}
\end{document}
"""


def _con_linea(deck, linea):
    """El deck limpio con una línea más antes de \\end{frame}; devuelve el texto y su número."""
    lineas = deck.split("\n")
    donde = lineas.index(r"\end{frame}")
    lineas.insert(donde, linea)
    return "\n".join(lineas), donde + 1


def test_el_verificador_no_marca_sintaxis():
    assert numeros_a_mano(DECK_LIMPIO) == [], numeros_a_mano(DECK_LIMPIO)
    print("ok  el verificador no marca preámbulo, \\newcommand con un factor de medida, "
          "\\includegraphics ni \\figgrande sin extensión, \\definecolor, \\setbeamerfont, "
          "\\phantom, \\rowcolors, overlays, opciones, la sintaxis de TikZ ni los nodos con macros, "
          "comentarios, años, fechas, referencias, rótulos de \\item ni el 20 % del presupuesto")


def test_el_verificador_detecta_numeros_tipeados():
    deck, n = _con_linea(DECK_LIMPIO, "  El AUC de RF es 0,79 en validación.")
    assert numeros_a_mano(deck) == [(n, "0,79")]

    deck, n = _con_linea(DECK_LIMPIO, r"  Train: 32\,940 filas y 11,3\,\% de «yes».")
    assert numeros_a_mano(deck) == [(n, r"32\,940"), (n, "11,3")]

    deck, n = _con_linea(DECK_LIMPIO, r"  \note{Sin modelo acierta el 88,7\,\%.}")
    assert numeros_a_mano(deck) == [(n, "88,7")], "el guion dentro de \\note también cuenta"

    deck, n = _con_linea(DECK_LIMPIO, r"\begin{frame}{Decir siempre «no» acierta el 88,7\,\%}")
    assert numeros_a_mano(deck) == [(n, "88,7")]

    deck, n = _con_linea(DECK_LIMPIO, "  Con k = 801 vecinos y datos de 2011.")
    assert numeros_a_mano(deck) == [(n, "801"), (n, "2011")]
    assert numeros_a_mano(deck, list(PERMITIDOS) + [r"\b2011\b"]) == [(n, "801")]

    casos = {
        # Un número después de una referencia con un conector (hallazgo del verificador, a).
        "Según la Clase 7, slide 61 y 200 árboles bastan.": ["200"],
        "Como dice la clase 7, 0,79 es el AUC.": ["0,79"],
        "En el punto 4, 0,795 de AUC.": ["0,795"],
        "En la ola 4, 801 vecinos.": ["801"],
        "En los slides 45, 0,79 de AUC.": ["0,79"],
        # Un número pegado a una letra que no es una unidad de TeX (b).
        "Tardó 600s.": ["600"],
        "La tasa se multiplica por x27.": ["27"],
        "Son 33k filas.": ["33"],
        "Sube 0,31pp.": ["0,31"],
        # El opcional de \item y el título corto de una sección son texto (c).
        r"\item[0,79] AUC": ["0,79"],
        r"\section[Hallazgo 2]{El modelo aprende la época}": ["2"],
        # Una fracción no es una fecha (d).
        "Con 1/3 de los «yes».": ["1", "3"],
        "Entre 4/5 y 1/5.": ["4", "5", "1", "5"],
        # El texto de un nodo de TikZ se proyecta.
        r"\begin{tikzpicture}\node[font=\small] at (4.75,0.525) {1070 filas};\end{tikzpicture}":
            ["1070"],
        r"\tikz \node {32};": ["32"],
        r"\begin{tikzpicture}\matrix [matrix of nodes] { |[fill=azul!10]| 0,79 & \x \\ };"
        r"\end{tikzpicture}": ["0,79"],
        # Una definición cuyo cuerpo es una cifra (hallazgo bajo).
        r"\newcommand{\miauc}{0,79}": ["0,79"],
        r"\def\miauc{0.795}": ["0.795"],
        # El 20 % que es un resultado, no la definición del presupuesto.
        r"Sin modelo se alcanza el 20\,\% de los «yes».": ["20"],
    }
    for linea, esperados in casos.items():
        deck, n = _con_linea(DECK_LIMPIO, linea)
        hallados = numeros_a_mano(deck)
        assert hallados == [(n, e) for e in esperados], (linea, hallados)

    # En el preámbulo, una definición con una cifra también cuenta; en los macros generados, no.
    deck = DECK_LIMPIO.replace("\\input{numeros}", "\\input{numeros}\n\\newcommand{\\otro}{0,79}")
    assert numeros_a_mano(deck) == [(8, "0,79")]
    assert numeros_a_mano("\\newcommand{\\x}{0{,}79}", generado=True) == []
    print(f"ok  el verificador marca un 0,79 tipeado en un frame, un 32\\,940, un número en un "
          f"título, en \\note o en un nodo de TikZ, un año fuera de 2008–2010 y {len(casos)} casos "
          "que la primera versión dejaba pasar; --permitir lo amplía")


def _resultados_test_a_verificar(carpeta):
    """Los resultados-test.tex que se verifican: los marcadores que escribe
    src/evaluar_test.py --marcadores, generados en `carpeta` con el nombre de siempre (--verificar
    reconoce por el nombre los macros generados), y además informe/resultados-test.tex si existe.
    El clon de entrega no trae informe/ (src/entregar.py): ahí se verifican sólo los marcadores."""
    marcadores = Path(carpeta) / RUTA_RESULTADOS_TEST.name
    marcadores.write_text(renderizar_tex(None), encoding="utf-8")
    return [marcadores] + ([RUTA_RESULTADOS_TEST] if RUTA_RESULTADOS_TEST.exists() else [])


def test_el_verificador_sin_preambulo_y_en_los_macros_generados():
    # Un fragmento sin \begin{document} se lee entero, salvo comentarios.
    fragmento = "\\newcommand{\\x}{0,79}\n% 0,79\n\\x y 0,79"
    assert numeros_a_mano(fragmento) == [(1, "0,79"), (3, "0,79")]
    assert numeros_a_mano(fragmento, generado=True) == [(3, "0,79")]
    # Un tikzpicture sin cerrar oculta su sintaxis hasta el final, sin fallar; su nodo se lee.
    assert numeros_a_mano("\\begin{tikzpicture}\n\\draw (0,0) -- (3,4);\n\\node at (1,2) {32};") \
        == [(3, "32")]
    with tempfile.TemporaryDirectory() as tmp:
        rutas = _resultados_test_a_verificar(tmp)
        for ruta in rutas:
            assert numeros_a_mano(ruta.read_text(encoding="utf-8"), generado=True) == [], ruta
    print("ok  un fragmento sin preámbulo se lee entero; los marcadores de resultados-test.tex"
          + (" e informe/resultados-test.tex no tienen" if len(rutas) > 1 else " no tienen (sin "
             "informe/, como en el clon de entrega)") + " números a mano")


def _verificar_cli(*argumentos):
    salida = io.StringIO()
    with redirect_stdout(salida):
        codigo = numeros_main(["--verificar", *map(str, argumentos)])
    return codigo, salida.getvalue()


def test_cli_verificar_da_archivo_linea_y_codigo():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "frames.tex").write_text("\\begin{frame}{Título}\n  AUC de 0,79.\n\\end{frame}\n",
                                        encoding="utf-8")
        (tmp / "numeros.tex").write_text("\\newcommand{\\x}{0,79}\n", encoding="utf-8")
        (tmp / "deck.tex").write_text("\\documentclass{beamer}\n\\input{numeros}\n"
                                      "\\begin{document}\n\\input{frames}\n\\end{document}\n",
                                      encoding="utf-8")
        codigo, salida = _verificar_cli(tmp / "deck.tex")
        assert codigo == 1
        assert re.search(r"frames\.tex:2: 0,79", salida), salida
        assert _verificar_cli(tmp / "deck.tex", "--permitir", r"\b0,79\b")[0] == 0
        # Los macros generados se verifican solos: sus definiciones no cuentan.
        assert _verificar_cli(tmp / "numeros.tex")[0] == 0
        for ruta in _resultados_test_a_verificar(tmp):
            assert _verificar_cli(ruta)[0] == 0, ruta

        # Dos niveles de \input, cada uno relativo a la carpeta del .tex raíz, como en LaTeX.
        secciones = tmp / "v" / "secciones"
        secciones.mkdir(parents=True)
        (secciones / "eda.tex").write_text("\\begin{frame}{EDA}Sin números.\\end{frame}\n"
                                           "\\input{secciones/fuga}\n", encoding="utf-8")
        (secciones / "fuga.tex").write_text("\\begin{frame}{Fuga}\nEl AUC es 0,94.\\end{frame}\n",
                                            encoding="utf-8")
        (tmp / "v" / "deck.tex").write_text("\\documentclass{beamer}\n\\begin{document}\n"
                                            "\\input{secciones/eda}\n\\end{document}\n",
                                            encoding="utf-8")
        codigo, salida = _verificar_cli(tmp / "v" / "deck.tex")
        assert codigo == 1 and re.search(r"fuga\.tex:2: 0,94", salida), salida
        # Un \input que no se encuentra no se saltea en silencio.
        (tmp / "v" / "deck.tex").write_text("\\documentclass{beamer}\n\\begin{document}\n"
                                            "\\input{secciones/no-existe}\n\\end{document}\n",
                                            encoding="utf-8")
        codigo, salida = _verificar_cli(tmp / "v" / "deck.tex")
        assert codigo == 1 and re.search(r"deck\.tex:3: \\input\{secciones/no-existe\} no se "
                                         r"encuentra", salida), salida
        # Babel en español sin \spanishplainpercent: aviso, sin fallar.
        (tmp / "babel.tex").write_text("\\documentclass{beamer}\n\\usepackage[spanish,"
                                       "es-noquoting]{babel}\n\\begin{document}\nHola.\n"
                                       "\\end{document}\n", encoding="utf-8")
        codigo, salida = _verificar_cli(tmp / "babel.tex")
        assert codigo == 0 and "AVISO" in salida and "spanishplainpercent" in salida, salida
        texto = (tmp / "babel.tex").read_text(encoding="utf-8")
        (tmp / "babel.tex").write_text(texto.replace("{babel}\n", "{babel}\n\\spanishplainpercent\n"),
                                       encoding="utf-8")
        assert "AVISO" not in _verificar_cli(tmp / "babel.tex")[1]
    print("ok  --verificar sigue los \\input del cuerpo a cualquier profundidad, relativos a la "
          "carpeta del .tex raíz; informa archivo:línea y termina con código 1 si hay números o un "
          "\\input que no se encuentra; avisa si falta \\spanishplainpercent")


# --- Línea de comandos sobre resultados sintéticos -----------------------------------------------

def _resultados_sinteticos(carpeta):
    """El mínimo: cv_final_resumen.csv y modelo_elegido.json. Todo lo demás falta."""
    medias = {"rf": 0.8123, "knn": 0.8012, "nb_categorico": 0.7901, "svm": 0.7812,
              "sin_modelo": 0.5}
    configuracion = {"rf": {"max_depth": 6, "n_estimators": 100}, "knn": {"n_neighbors": 51},
                     "nb_categorico": {"alpha": 1.0, "n_cortes": 10},
                     "svm": {"C": 0.01, "kernel": "linear"}, "sin_modelo": {"estrategia": "prior"}}
    filas = []
    for modelo, auc in medias.items():
        for conjunto, desplazamiento in (("train", 0.02), ("validacion", 0.0)):
            for metrica in METRICAS:
                filas.append({"modelo": modelo, "conjunto": conjunto, "metrica": metrica,
                              "media": (auc if metrica == "auc" else auc / 2) + desplazamiento})
        filas.append({"modelo": modelo, "conjunto": "brecha", "metrica": "auc", "media": 0.02})
    tabla = pd.DataFrame(filas).assign(
        etiqueta="final", configuracion=lambda t: t["modelo"].map(lambda m: json.dumps(configuracion[m])),
        desvio=0.006, n=5, error_estandar=0.0027)
    tabla[["etiqueta", "modelo", "configuracion", "conjunto", "metrica", "media", "desvio", "n",
           "error_estandar"]].to_csv(carpeta / "cv_final_resumen.csv", index=False)
    ranking = [{"modelo": m, "auc": medias[m], "auc_error_estandar": 0.0027,
                "recall_q": medias[m] / 2} for m in ("rf", "knn", "nb_categorico", "svm")]
    elegido = {"modelo": "rf", "k_folds": 5, "semilla": 42, "ranking": ranking,
               "empate_dentro_1es": [], "linea_base": {"auc": 0.5, "recall_q": 0.2},
               "metricas": {"validacion": {"auc": {"media": 0.8123, "error_estandar": 0.0027},
                                           "recall_q": {"media": 0.40615}}}}
    (carpeta / "modelo_elegido.json").write_text(json.dumps(elegido), encoding="utf-8")


def _correr(*argumentos):
    return subprocess.run([sys.executable, "-m", "src.numeros", *map(str, argumentos)], cwd=RAIZ,
                          capture_output=True, text=True, timeout=300)


def test_cli_sintetico_es_determinista_y_escribe_interrogacion_si_falta_un_archivo():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        resultados = tmp / "resultados"
        resultados.mkdir()
        _resultados_sinteticos(resultados)
        salidas = []
        for corrida in (1, 2):
            tex, md = tmp / f"numeros{corrida}.tex", tmp / f"numeros{corrida}.md"
            r = _correr("--resultados", resultados, "--salida-tex", tex, "--salida-md", md,
                        "--resultados-test", tmp / "no-existe.tex")
            assert r.returncode == 0, r.stderr
            salidas.append((tex.read_bytes(), md.read_bytes()))
        assert salidas[0] == salidas[1], "dos corridas seguidas no dan los mismos bytes"

        texto_tex, texto_md = (s.decode("utf-8") for s in salidas[0])
        macros = _macros(texto_tex)
        assert macros["aucRfVal"] == "0{,}812" and macros["aucSinModelo"] == "0{,}500"
        assert macros["profundidadRf"] == "6" and macros["cSvm"] == "0{,}01"
        assert macros["margenAucFinal"] == "0{,}011" and macros["modeloFinal"] == "Random Forest"
        assert macros["cortesNb"] == "10" and macros["alfaNb"] == "1"
        assert macros["filasTrain"] == r"32\,940", "el EDA sale de train aunque falten resultados"
        assert macros["aucRfAdelante"] == DESCONOCIDO, "robustez_temporal_resumen.csv falta"
        assert macros["gananciaRfVeinte"] == DESCONOCIDO and macros["yesTest"] == DESCONOCIDO
        assert all(macros[n] == DESCONOCIDO for n in MACROS_DE_PARTICION)
        assert "resultados/robustez_temporal_resumen.csv" in texto_md
        assert "`resultados/evaluacion_test.json`: `\\yesTest`" in texto_md
        assert "`resultados/evidencia_particion.json`: `\\semillasSinEstratificar`" in texto_md
        assert "Falta `" in texto_md and "no-existe.tex" in texto_md
        assert all(f"`\\{n}` | ? |" in texto_md for n in MACROS_TEST)
        assert numeros_a_mano(texto_tex, generado=True) == [], "numeros.tex no pasa su verificador"

        # Con resultados de otro lado no se pisan los macros del deck.
        r = _correr("--resultados", resultados)
        assert r.returncode == 1 and "--salida-tex" in r.stderr
    print("ok  sobre resultados sintéticos, dos corridas dan bytes idénticos; lo que falta vale «?» "
          "y se anota en el .md; otro --resultados sin --salida-* se niega")


def test_con_la_evaluacion_de_test_y_la_evidencia_de_la_particion_se_derivan_sus_macros():
    """Con un evaluacion_test.json sintético, con las claves que escribe src/evaluar_test.py, y un
    evidencia_particion.json con el contrato que documenta src/numeros.py."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _resultados_sinteticos(tmp)
        registro = {"n_test": 8236, "n_yes_test": 928,
                    "metricas": {"auc": {"valor": 0.8234, "ic95": [0.8121, 0.8339]},
                                 "recall_q": {"valor": 0.5462, "ic95": [0.5102, 0.5768]}}}
        (tmp / "evaluacion_test.json").write_text(json.dumps(registro), encoding="utf-8")
        evidencia = {"n_semillas": 400,
                     "sin_estratificar": {"desvio_pp": 0.3123, "minimo_pct": 10.4199,
                                          "maximo_pct": 12.1297},
                     "temporal": {"pct_yes_train": 6.3812, "pct_yes_test": 30.8287}}
        (tmp / "evidencia_particion.json").write_text(json.dumps(evidencia), encoding="utf-8")
        muestra = cargar_train().iloc[::8].reset_index(drop=True)
        generado = generar(Fuentes(tmp, train=muestra, resultados_test=tmp / "no.tex"))
    v = generado.valores
    assert v["yesTest"] == "928" and v["pctYesTest"] == r"11{,}3\,\%"
    assert v["diferenciaAucTestValidacion"] == "+0{,}011"       # 0,8234 − 0,8123
    assert v["diferenciaRecallTestValidacion"] == "+0{,}140"    # 0,5462 − 0,40615
    assert v["semillasSinEstratificar"] == "400"
    assert v["desvioPctYesTestSinEstratificar"] == "0{,}31"
    assert v["pctYesTestSinEstratificarMinimo"] == r"10{,}4\,\%"
    assert v["pctYesTestSinEstratificarMaximo"] == r"12{,}1\,\%"
    assert v["pctYesTrainTemporal"] == r"6{,}4\,\%" and v["pctYesTestTemporal"] == r"30{,}8\,\%"
    assert not {"resultados/evaluacion_test.json", "resultados/evidencia_particion.json"} \
        & set(generado.faltan)
    print("ok  con evaluacion_test.json y evidencia_particion.json, sus macros se calculan: 928, "
          "11,3 %, +0,011 y +0,140; 400 semillas, 0,31 pp, 10,4 % a 12,1 %, 6,4 % y 30,8 %")


def test_evidencia_particion_escribe_el_archivo_que_se_lee_aqui():
    """src/evidencia_particion.py escribe el JSON con el contrato que lee generar(). Un dataset
    sintético de 1 000 filas con 100 «yes»: 60 en las 800 primeras y 40 en las 200 últimas, así
    que la partición temporal da 7,5 % en train (60 / 800) y 20,0 % en test (40 / 200). Las
    cifras sin estratificar dependen del sorteo: se comparan con las que calculó el módulo."""
    df = pd.DataFrame({"y": ["yes"] * 60 + ["no"] * 740 + ["yes"] * 40 + ["no"] * 160})
    datos = evidencia(df, n_semillas=20)
    assert abs(datos["temporal"]["pct_yes_train"] - 7.5) < 1e-9
    assert abs(datos["temporal"]["pct_yes_test"] - 20.0) < 1e-9
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        escribir_evidencia(tmp / EVIDENCIA_PARTICION, datos)
        muestra = cargar_train().iloc[::8].reset_index(drop=True)
        generado = generar(Fuentes(tmp, train=muestra, resultados_test=tmp / "no.tex"))
    v, s = generado.valores, datos["sin_estratificar"]
    assert v["semillasSinEstratificar"] == "20"
    assert v["desvioPctYesTestSinEstratificar"] == formatear_decimal(s["desvio_pp"], 2)
    assert v["pctYesTestSinEstratificarMinimo"] == formatear_porcentaje(s["minimo_pct"])
    assert v["pctYesTestSinEstratificarMaximo"] == formatear_porcentaje(s["maximo_pct"])
    assert v["pctYesTrainTemporal"] == r"7{,}5\,\%" and v["pctYesTestTemporal"] == r"20{,}0\,\%"
    assert f"resultados/{EVIDENCIA_PARTICION}" not in generado.faltan
    print("ok  src/evidencia_particion.py escribe el JSON que se lee aquí: con 60 «yes» en las "
          "800 primeras filas y 40 en las 200 últimas, la partición temporal da 7,5 % y 20,0 %, y "
          "los seis macros de D-02 y D-03 dejan de valer «?»")


def test_sin_ningun_resultado_no_falla():
    with tempfile.TemporaryDirectory() as tmp:
        muestra = cargar_train().iloc[::8].reset_index(drop=True)
        generado = generar(Fuentes(Path(tmp), train=muestra, resultados_test=Path(tmp) / "no.tex"))
    valores = generado.valores
    assert valores["filasTrain"] == formatear_miles(len(muestra))
    assert valores["aucRfVal"] == DESCONOCIDO and valores["columnasFinales"] == "58"
    assert valores["pdaysCentinela"] == "999" and valores["factorIqr"] == "1{,}5"
    assert valores["vecinosKnnMeseta"] == DESCONOCIDO, "sin hiperparametros.json no se valida"
    assert "resultados/cv_final_resumen.csv" in generado.faltan
    assert "## Pendientes" in generado.md
    print(f"ok  sin ningún archivo de resultados, {sum(v == DESCONOCIDO for v in valores.values())} "
          f"macros valen «?» y los {sum(v != DESCONOCIDO for v in valores.values())} de train y de "
          "código se calculan")


def test_la_meseta_de_knn_vale_interrogacion_si_deja_de_estar_dentro_de_un_error_estandar():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        curva = {"puntos_dentro_1es": ["401", "601", "801"], "mejor": "601", "valor": 801}
        (tmp / "hiperparametros.json").write_text(
            json.dumps({"curvas": {"knn_n_neighbors_uniform": curva}}), encoding="utf-8")
        muestra = cargar_train().iloc[::8].reset_index(drop=True)
        generado = generar(Fuentes(tmp, train=muestra, resultados_test=tmp / "no.tex"))
    assert generado.valores["vecinosKnnMeseta"] == DESCONOCIDO
    assert any("D-22" in que and "vecinosKnnMeseta" in nombres
               for que, nombres in generado.faltan.items()), generado.faltan
    print("ok  \\vecinosKnnMeseta, copiado de D-22, vale «?» si el punto sale de 1 error estándar")


def test_knn_por_distancia_sale_de_la_curva_y_no_de_hiperparametros_json():
    """La grilla de KNN con weights = distance se extendió hasta 801 después de escribir
    hiperparametros.json, que quedó con su entrada vieja (201). Los cuatro macros de ese punto
    aplican D-22 a la curva misma. Aquí el mejor es 101, pero 801 queda dentro de un error estándar
    y es el de más vecinos: la regla elige 801, y el JSON viejo dice 201."""
    validacion = {1: [0.60, 0.61, 0.62, 0.60, 0.61],
                  101: [0.770, 0.772, 0.768, 0.771, 0.770],        # media 0,7702, ES 0,00066
                  801: [0.7698, 0.7700, 0.7699, 0.7701, 0.7702]}   # media 0,7700
    train = {1: 0.99, 101: 1.0, 801: 1.0}
    filas = []
    for k, valores in validacion.items():
        configuracion = json.dumps({"n_neighbors": k, "weights": "distance"}, sort_keys=True)
        for fold, valor in enumerate(valores, start=1):
            for conjunto, v in (("train", train[k]), ("validacion", valor)):
                filas.append({"parametro": "n_neighbors", "punto": str(k), "modelo": "knn",
                              "configuracion": configuracion, "fold": fold, "conjunto": conjunto,
                              "metrica": "auc", "valor": v})
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "curvas").mkdir()
        pd.DataFrame(filas).to_csv(tmp / "curvas" / "knn_n_neighbors_distance.csv", index=False)
        vieja = {"valor": 201, "auc_validacion_media": 0.7685, "auc_train_media": 0.9999,
                 "brecha": 0.2314, "mejor": "201", "puntos_dentro_1es": ["151", "201"]}
        (tmp / "hiperparametros.json").write_text(
            json.dumps({"curvas": {"knn_n_neighbors_distance": vieja}}), encoding="utf-8")
        muestra = cargar_train().iloc[::8].reset_index(drop=True)
        v = generar(Fuentes(tmp, train=muestra, resultados_test=tmp / "no.tex")).valores
    assert v["vecinosKnnDistancia"] == "801", v["vecinosKnnDistancia"]
    assert v["aucKnnDistancia"] == "0{,}770" and v["aucTrainKnnDistancia"] == "1{,}000"
    assert v["brechaKnnDistancia"] == "0{,}230", v["brechaKnnDistancia"]
    print("ok  KNN por distancia: D-22 sobre la curva elige 801 (0,770, brecha 0,230), no el 201 de "
          "una entrada vieja de hiperparametros.json")


# --- Sobre los resultados reales ----------------------------------------------------------------

def _leer(nombre, **kwargs):
    return pd.read_csv(DIR / nombre, **kwargs)


def _curva(nombre):
    """Media por punto de una curva, sin src/curvas.py: `punto` como texto («None» incluido)."""
    tabla = _leer(f"curvas/{nombre}.csv", dtype={"punto": str}, keep_default_na=False)
    tabla["valor"] = tabla["valor"].astype(float)
    return tabla[tabla["metrica"] == "auc"].groupby(["punto", "conjunto"])["valor"].mean()


def _recalculados():
    """{macro: valor esperado}: una cifra de cada familia de macros, calculada aquí con pandas desde
    su archivo, sin las funciones de src/numeros.py (sólo las de formato, que tienen sus casos a
    mano). Cubre las columnas que la primera versión de la suite no distinguía."""
    final = _leer("cv_final_resumen.csv").set_index(["modelo", "conjunto", "metrica"])
    referencia = _leer("cv_referencia_resumen.csv").set_index(["modelo", "conjunto", "metrica"])
    ablaciones = _leer("ablaciones_resumen.csv").set_index(["modelo", "variante"])
    robustez = _leer("robustez_temporal_resumen.csv").set_index(
        ["modelo", "esquema", "variante", "metrica"])
    por_fold = _leer("robustez_temporal.csv")
    bloques = _leer("robustez_folds.csv").set_index(["esquema", "fold"])
    sensibilidad = _leer("sensibilidad_q_final.csv")
    ganancia = _leer("ganancia_final.csv").set_index(["modelo", "p"])["pct_yes_alcanzado"]
    hiper = json.loads((DIR / "hiperparametros.json").read_text(encoding="utf-8"))["curvas"]
    pareado = _leer("cv_referencia.csv")
    pareado = pareado[(pareado["conjunto"] == "validacion") & (pareado["metrica"] == "auc")]
    pareado = pareado.pivot_table(index="fold", columns="modelo", values="valor")
    profundidad, pesos = _curva("rf_max_depth"), _curva("rf_pesos_clase_depth8")

    train = cargar_train()
    y = (train["y"] == "yes").to_numpy()
    campaign = train["campaign"]
    q1, q3 = campaign.quantile(0.25), campaign.quantile(0.75)
    fuera = (campaign < q1 - 1.5 * (q3 - q1)) | (campaign > q3 + 1.5 * (q3 - q1))
    barajado = StratifiedKFold(5, shuffle=True, random_state=42)
    euribor = [train["euribor3m"].iloc[v].mean() for _, v in barajado.split(train, y)]
    nb = json.loads(final.loc[("nb_categorico", "validacion", "auc"), "configuracion"])
    adelante = por_fold[(por_fold["modelo"] == "rf") & (por_fold["esquema"] == "hacia_adelante")
                        & (por_fold["variante"] == "todas") & (por_fold["fold"] == 3)
                        & (por_fold["conjunto"] == "validacion") & (por_fold["metrica"] == "auc")]
    rf_diez = sensibilidad[(sensibilidad["modelo"] == "rf") & np.isclose(sensibilidad["q"], 0.1)
                           & (sensibilidad["metrica"] == "recall_q")]["valor"]

    def desvio_delta(x):
        return formatear_decimal(x, 3 if abs(x) >= 0.01 else 4)

    # El hallazgo (conclusiones.md, §2), por otro camino que src/numeros.py: el año se infiere aquí
    # a mano (el CSV va por fecha y el año cambia cada vez que el mes retrocede) y el AUC de los
    # pares de años distintos se mide par de años por par de años, con roc_auc_score sobre los «yes»
    # de un año y los «no» del otro, en lugar de por diferencia con el AUC de todo junto.
    meses = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    por_fila = train.sort_values("fila")
    vuelta = np.diff(por_fila["month"].map(meses.index).to_numpy()) < 0
    anio_de = pd.Series(leer_names().anio_ini + np.concatenate([[0], np.cumsum(vuelta)]),
                        index=por_fila["fila"].to_numpy())
    oof = _leer("oof_final_rf.csv", float_precision="round_trip")
    oof["y"] = oof["fila"].map(pd.Series(y.astype(int), index=train["fila"].to_numpy()))
    oof["anio"] = oof["fila"].map(anio_de)
    sumas = {"dentro": [0.0, 0], "entre": [0.0, 0]}
    auc_2008 = None
    for a in sorted(oof["anio"].unique()):
        for b in sorted(oof["anio"].unique()):
            si = oof.loc[(oof["anio"] == a) & (oof["y"] == 1), "puntaje"].to_numpy()
            no = oof.loc[(oof["anio"] == b) & (oof["y"] == 0), "puntaje"].to_numpy()
            if not len(si) or not len(no):
                continue
            auc_ab = roc_auc_score(np.r_[np.ones(len(si)), np.zeros(len(no))], np.r_[si, no])
            clave = "dentro" if a == b else "entre"
            sumas[clave][0] += auc_ab * len(si) * len(no)
            sumas[clave][1] += len(si) * len(no)
            if a == b == 2008:
                auc_2008 = auc_ab
    pares = sumas["dentro"][1] + sumas["entre"][1]
    assert pares == int(y.sum()) * int((~y).sum())

    def barajado_en_bloque(fold):
        b = bloques.loc[("hacia_adelante", fold)]
        filas = oof[oof["fila"].between(b["fila_min_validacion"], b["fila_max_validacion"])]
        return roc_auc_score(filas["y"], filas["puntaje"])

    adelante_cinco = por_fold[(por_fold["modelo"] == "rf") & (por_fold["esquema"] == "hacia_adelante")
                              & (por_fold["variante"] == "todas") & (por_fold["fold"] == 5)
                              & (por_fold["conjunto"] == "validacion")
                              & (por_fold["metrica"] == "auc")]["valor"].item()
    primer_corte = bloques.loc[("hacia_adelante", 1), "fila_max_train"]
    # La matriz fuera de fold: los llamados son las llamadas(n) filas de mayor puntaje.
    k = llamadas(len(oof))
    vp = int(oof.sort_values("puntaje", ascending=False, kind="stable")["y"].iloc[:k].sum())

    # KNN con weights = distance: D-22 sobre la curva tal como está, no la entrada de
    # hiperparametros.json. El mejor punto de validación y, dentro de un error estándar de él, el de
    # más vecinos; el error estándar es el desvío (ddof = 1) sobre la raíz de los folds.
    distancia = _leer("curvas/knn_n_neighbors_distance.csv", dtype={"punto": str},
                      keep_default_na=False)
    distancia = distancia[distancia["metrica"] == "auc"].astype({"valor": float})
    por_k = distancia.pivot_table(index="punto", columns="conjunto", values="valor",
                                  aggfunc=["mean", "std", "count"])
    medias_k = por_k[("mean", "validacion")]
    mejor_k = medias_k.idxmax()
    umbral_k = medias_k[mejor_k] - (por_k.loc[mejor_k, ("std", "validacion")]
                                    / np.sqrt(por_k.loc[mejor_k, ("count", "validacion")]))
    k_distancia = max(medias_k.index[medias_k >= umbral_k], key=int)

    return {
        "vecinosKnnDistancia": formatear_miles(int(k_distancia)),
        "aucKnnDistancia": formatear_decimal(medias_k[k_distancia]),
        "aucTrainKnnDistancia": formatear_decimal(por_k.loc[k_distancia, ("mean", "train")]),
        "brechaKnnDistancia": formatear_decimal(por_k.loc[k_distancia, ("mean", "train")]
                                                - medias_k[k_distancia]),
        "pctParesEntreAnios": formatear_porcentaje(100 * sumas["entre"][1] / pares),
        "aucOofEntreAnios": formatear_decimal(sumas["entre"][0] / sumas["entre"][1]),
        "aucOofDentroAnio": formatear_decimal(sumas["dentro"][0] / sumas["dentro"][1]),
        "aucOofDosMilOcho": formatear_decimal(auc_2008),
        "aucRfBarajadoBloqueCuatro": formatear_decimal(barajado_en_bloque(4)),
        "caidaAucRfBloqueCinco": formatear_decimal(barajado_en_bloque(5) - adelante_cinco),
        "yesEntrenamientoAdelanteUno": formatear_miles(int(y[train["fila"] <= primer_corte].sum())),
        "llamadasPorYesRf": formatear_decimal(k / vp, 1),
        "llamadasPorYesSinModelo": formatear_decimal(
            1 / final.loc[("sin_modelo", "validacion", "precision_q"), "media"], 1),
        "nivelIntervalo": r"95\,\%",
        "remuestreosBootstrap": formatear_miles(N_BOOTSTRAP),
        "aucKnnVal": formatear_decimal(final.loc[("knn", "validacion", "auc"), "media"]),
        "recallRfVal": formatear_decimal(final.loc[("rf", "validacion", "recall_q"), "media"]),
        "recallRfValDesvio": formatear_decimal(final.loc[("rf", "validacion", "recall_q"), "desvio"]),
        "brechaRf": formatear_decimal(final.loc[("rf", "brecha", "auc"), "media"]),
        "aucTrainSvm": formatear_decimal(final.loc[("svm", "train", "auc"), "media"]),
        "aucRfReferencia": formatear_decimal(referencia.loc[("rf", "validacion", "auc"), "media"]),
        "aucKnnReferenciaDesvio": formatear_decimal(
            referencia.loc[("knn", "validacion", "auc"), "desvio"]),
        "deltaAucSinMacroRf": formatear_delta(ablaciones.loc[("rf", "A7"), "delta_media"]),
        "deltaAucSinMacroRfDesvio": desvio_delta(ablaciones.loc[("rf", "A7"), "delta_desvio"]),
        "aucRfConDuration": formatear_decimal(ablaciones.loc[("rf", "A1"), "auc_variante"]),
        "columnasSinMacro": formatear_miles(ablaciones.loc[("rf", "A7"), "columnas"]),
        "aucRfAdelante": formatear_decimal(
            robustez.loc[("rf", "hacia_adelante", "todas", "auc"), "media"]),
        "aucRfAdelanteDesvio": formatear_decimal(
            robustez.loc[("rf", "hacia_adelante", "todas", "auc"), "desvio"]),
        "aucKnnSinMacroNiMonthAdelante": formatear_decimal(
            robustez.loc[("knn", "hacia_adelante", "sin_macro_ni_month", "auc"), "media"]),
        "aucRfAdelanteFoldTres": formatear_decimal(adelante["valor"].item()),
        "pctYesBloqueCinco": formatear_porcentaje(
            bloques.loc[("hacia_adelante", 5), "pct_yes_validacion"]),
        "recallRfDiez": formatear_decimal(rf_diez.mean()),
        "gananciaRfVeinte": formatear_porcentaje(ganancia[("rf", 20)]),
        "gananciaSinModeloVeinte": formatear_porcentaje(ganancia[("sin_modelo", 20)]),
        "deltaAucNbCategorico": formatear_delta(
            (pareado["nb_categorico"] - pareado["nb_gaussiano"]).mean()),
        "aucRfProfundidadDiez": formatear_decimal(profundidad[("10", "validacion")]),
        "aucRfProfundidadSinLimite": formatear_decimal(profundidad[("None", "validacion")]),
        "deltaAucPesosRf": formatear_delta(pesos[("balanced", "validacion")]
                                           - pesos[("None", "validacion")]),
        "aucRfProfundidadElegida": formatear_decimal(hiper["rf_max_depth"]["auc_validacion_media"]),
        "errorEstandarRfProfundidadMejor": formatear_decimal(
            hiper["rf_max_depth"]["error_estandar_mejor"], 4),
        "arbolesRfRegla": formatear_miles(hiper["rf_n_estimators_depth8"]["valor"]),
        "umbralUnoEsRfArboles": formatear_decimal(hiper["rf_n_estimators_depth8"]["umbral_1es"], 4),
        "errorEstandarKnnMejor": formatear_decimal(
            hiper["knn_n_neighbors_uniform"]["error_estandar_mejor"], 4),
        "cortesNb": formatear_miles(nb["n_cortes"]),
        "alfaNb": formatear_parametro(nb["alpha"]),
        "euriborFoldBarajadoMinimo": formatear_decimal(min(euribor), 2),
        "atipicosCampaign": formatear_miles(int(fuera.sum())),
        "desvioPdays": formatear_decimal(train["pdays"].std(), 1),
        "pdaysCentinela": "999",
        "profundidadRfMinima": "2",
        "vecinosKnnMinimo": "1",
        "predictorasDisponibles": "19",
    }


def test_resultados_reales_coinciden_con_las_otras_fuentes():
    generado = _generado_real()
    v = generado.valores
    assert generar(Fuentes()).tex == generado.tex, "dos generaciones no dan el mismo .tex"

    # Sólo falta lo que tiene que faltar: la evaluación del test (N0-1) y la evidencia de la
    # partición, mientras no existan. Si un filtro deja de encontrar su fila, aparece aquí.
    esperados = {f"resultados/{n}" for n in ("evaluacion_test.json", "evidencia_particion.json")
                 if not (DIR / n).exists()}
    assert set(generado.faltan) == esperados, generado.faltan
    assert sum(len(n) for n in generado.faltan.values()) == sum(x == DESCONOCIDO for x in v.values())

    # La partición (D-01, D-02) y el presupuesto (D-20).
    assert v["filasTrain"] == r"32\,940" and v["yesTrain"] == r"3\,711"
    assert v["filasTest"] == formatear_miles(N_TEST) == r"8\,236"
    assert v["llamadasTest"] == r"1\,647" and v["duplicados"] == "12"

    # Las columnas: la cuenta pura de src/preproceso.py contra la de las ablaciones.
    resumen = pd.read_csv(DIR / "ablaciones_resumen.csv")
    for variante, var in VARIANTES.items():
        if variante == "A0":
            continue
        medidas = set(resumen.loc[resumen["variante"] == variante, "columnas"])
        assert medidas == {_columnas_de(var.opciones)}, (variante, medidas)
    assert v["columnasReferencia"] == "62" and v["columnasFinales"] == "58"

    # El modelo final: cv_final_resumen.csv y modelo_elegido.json dicen lo mismo.
    elegido = json.loads((DIR / "modelo_elegido.json").read_text(encoding="utf-8"))
    assert v["aucRfVal"] == formatear_decimal(elegido["metricas"]["validacion"]["auc"]["media"])
    assert v["profundidadRf"] == str(elegido["hiperparametros"]["max_depth"])
    assert v["arbolesRf"] == str(elegido["hiperparametros"]["n_estimators"])

    # Las cifras del EDA, contra src/eda_html.py (analizar), que publica eda.html.
    r = analizar(cargar_train(), leer_names())
    assert v["exactitudSiempreNo"] == formatear_porcentaje(r.acc_base)
    assert v["pctYesTrain"] == formatear_porcentaje(r.tasa)
    assert v["filasCentinelaConPrevio"] == formatear_miles(r.n_contra)
    assert v["pctPdaysCentinela"] == formatear_porcentaje(r.pct999)
    assert v["pctYesPrimerBloqueEda"] == formatear_porcentaje(r.bloques["tasa"].iloc[0])
    assert v["pctYesUltimoBloqueEda"] == formatear_porcentaje(r.bloques["tasa"].iloc[-1])
    assert v["pctYesDosMilOcho"] == formatear_porcentaje(r.tasa_anio[2008])
    assert v["pctYesDosMilDiez"] == formatear_porcentaje(r.tasa_anio[2010])
    assert v["pctYesDurationDecilLargo"] == formatear_porcentaje(r.dur_dec["tasa"].iloc[-1])
    assert v["aucDurationSola"] == formatear_decimal(r.auc_dur)
    assert v["razonDesviosNumericas"] == formatear_decimal(r.sd_ratio, 0)
    assert v["correlacionMacroMinima"] == formatear_decimal(min(r.r_trio), 2)
    assert v["correlacionMacroMaxima"] == formatear_decimal(max(r.r_trio), 2)
    # El factor del IQR es el de resultados/eda/reporte.txt, §5.
    assert "1.5 IQR" in (DIR / "eda" / "reporte.txt").read_text(encoding="utf-8")
    assert v["factorIqr"] == "1{,}5"

    # Una cifra de cada familia, recalculada aquí con pandas.
    distintos = {n: (v[n], esperado) for n, esperado in _recalculados().items() if v[n] != esperado}
    assert not distintos, distintos

    # La matriz fuera de fold cierra: VP + FN son los «yes» y VP + FP, las llamadas; FP > FN
    # porque se llama a más clientes (6 588) de los que contratan (3 711).
    vp, fp, fn = (int(v[c].replace("\\,", "")) for c in ("vpRfVal", "fpRfVal", "fnRfVal"))
    assert vp + fn == 3711 and vp + fp == int(v["llamadasTrain"].replace("\\,", ""))

    # La meseta de KNN que copia D-22 sigue dentro de 1 error estándar.
    assert v["vecinosKnnMeseta"] == "301"

    # El test no se evaluó (N0-1): sus macros valen «?» mientras falte evaluacion_test.json.
    evaluado = (DIR / "evaluacion_test.json").exists()
    assert all((v[n] == DESCONOCIDO) != evaluado for n in MACROS_DE_TEST)
    assert numeros_a_mano(generado.tex, generado=True) == []
    print(f"ok  sobre resultados/: {len(v)} macros; sólo faltan {sorted(esperados)}; partición, "
          f"columnas, modelo final y EDA coinciden con las otras fuentes; {len(_recalculados())} "
          "cifras recalculadas con pandas coinciden; la matriz fuera de fold cierra")


def test_el_corte_reparte_los_empates_en_proporcion():
    """El decil superior de Naive Bayes se corta como el presupuesto: con puntajes 3, 2, 2, 2, 1 y
    q = 0,4 se llama a 2 filas; la de 3 entra entera y la segunda llamada se reparte entre las tres
    de 2, un tercio cada una."""
    pesos = _pesos_en_el_corte([3, 2, 2, 2, 1], 0.4)
    assert np.allclose(pesos, [1, 1 / 3, 1 / 3, 1 / 3, 0]), pesos
    assert np.isclose(pesos.sum(), 2)
    assert np.allclose(_pesos_en_el_corte([5, 4, 3, 2, 1], 0.4), [1, 1, 0, 0, 0])
    print("ok  el corte del decil superior reparte los empates en proporción: 1, 1/3, 1/3, 1/3, 0")


def _fuentes_sinteticas(carpeta, train, oof):
    """Fuentes con un train y un OOF de RF sintéticos (y el names.txt real, para el año)."""
    oof.to_csv(Path(carpeta) / "oof_final_rf.csv", index=False)
    return Fuentes(Path(carpeta), train=train, resultados_test=Path(carpeta) / "no.tex")


def test_llamar_dentro_de_cada_anio_y_de_cada_mes_sobre_un_caso_a_mano():
    """20 filas: las 10 primeras de mayo (2008) y las 10 siguientes de marzo y abril, que por el mes
    que retrocede son de 2009. Llamando al 20 % de cada año (2 llamadas en cada uno): en 2008, la
    fila de 0,95 («yes») y la de 0,90 («no»), 1 de sus 2 «yes»; en 2009, la de 0,85 («no») y la de
    0,80 («yes»), 1 de 5, con tope 2 / 5. En total 2 de 7 «yes», y 4 llamadas / 2 alcanzados = 2
    por «yes». Llamando al 20 % de cada mes (2, 1 y 1 llamadas): mayo, 1; marzo, la de 0,85
    («no»), 0; abril, la de 0,75 («no»), 0: 1 de 7. El AUC de los pares del mismo mes: mayo 8 de
    16 pares bien ordenados, marzo 4 de 6, abril 3 de 6, es decir 15 / 28."""
    meses = ["may"] * 10 + ["mar"] * 5 + ["apr"] * 5
    yes = {0, 1, 10, 11, 15, 16, 17}
    puntaje = [0.95, 0.10, 0.90, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26,
               0.80, 0.30, 0.85, 0.05, 0.06,
               0.70, 0.60, 0.02, 0.75, 0.01]
    train = pd.DataFrame({"fila": range(20), "month": meses,
                          "y": ["yes" if i in yes else "no" for i in range(20)]})
    oof = pd.DataFrame({"fila": range(20), "fold": 1, "puntaje": puntaje})
    with tempfile.TemporaryDirectory() as tmp:
        f = _fuentes_sinteticas(tmp, train, oof)
        assert np.isclose(_llamando_dentro(POR_ANIO, "recall", (2008,))(f), 1 / 2)
        assert np.isclose(_llamando_dentro(POR_ANIO, "recall", (2009,))(f), 1 / 5)
        assert _llamando_dentro(POR_ANIO, "llamadas", (2009,))(f) == 2
        assert np.isclose(_llamando_dentro(POR_ANIO, "tope", (2009,))(f), 2 / 5)
        assert np.isclose(_llamando_dentro(POR_ANIO, "recall")(f), 2 / 7)
        assert np.isclose(_llamando_dentro(POR_ANIO, "llamadas_por_yes")(f), 4 / 2)
        assert np.isclose(_llamando_dentro(POR_MES, "recall")(f), 1 / 7)
        assert _llamando_dentro(POR_MES, "llamadas")(f) == 4
        assert np.isclose(_auc_dentro_de_cada_mes(f), 15 / 28)
    print("ok  llamando dentro de cada año: 1/2 en 2008, 1/5 en 2009 (tope 2/5), 2/7 en total y 2 "
          "llamadas por «yes»; dentro de cada mes, 1/7; el AUC de los pares del mismo mes, 15/28")


def test_vectores_repetidos_sobre_un_caso_a_mano():
    """Seis filas: el vector (1, 1) aparece tres veces, con «yes» y «no»; el (2, 2), dos veces, las
    dos «no»; el (3, 3), una. Sin duration hay 2 vectores repetidos en 5 filas, y 1 de ellos, con 3
    filas, tiene las dos clases. Con duration, que es distinta en cada fila, ninguno."""
    train = pd.DataFrame({"fila": range(6), "a": [1, 1, 1, 2, 2, 3], "b": [1, 1, 1, 2, 2, 3],
                          "duration": [10, 20, 30, 40, 50, 60],
                          "y": ["no", "yes", "no", "no", "no", "yes"]})
    f = Fuentes(Path(tempfile.gettempdir()) / "no-existe", train=train)
    assert _repetidos("vectores")(f) == 2 and _repetidos("filas")(f) == 5
    assert np.isclose(_repetidos("pct_filas")(f), 100 * 5 / 6)
    assert _repetidos("vectores_distintas")(f) == 1 and _repetidos("filas_distintas")(f) == 3
    assert _repetidos("vectores", con_duration=True)(f) == 0
    print("ok  vectores repetidos: 2 en 5 filas, 1 con las dos clases en 3; con duration, ninguno")


def test_macros_del_cierre_de_la_ola_ocho_por_otro_camino():
    """Los macros que agregó el cierre de la ola 8, recalculados aquí sin las funciones de
    src/numeros.py: el año, a mano desde el mes que retrocede; el corte del 20 % con nlargest; el
    AUC de cada mes con la U de Mann–Whitney sobre rangos promedio, no con roc_auc_score; el decil
    superior de Naive Bayes con rank(method="first"), y P(yes) con la logística directa; los
    vectores repetidos con duplicated(); y las diferencias pareadas de los pesos desde el CSV de
    cada curva."""
    v = _generado_real().valores
    train = cargar_train()
    y = (train["y"] == "yes").astype(int).to_numpy()
    meses = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    por_fila = train.sort_values("fila")
    vuelta = np.diff(por_fila["month"].map(meses.index).to_numpy()) < 0
    fechas = pd.DataFrame({"fila": por_fila["fila"].to_numpy(),
                           "anio": leer_names().anio_ini + np.concatenate([[0], np.cumsum(vuelta)]),
                           "mes": por_fila["month"].to_numpy()})
    objetivo = pd.DataFrame({"fila": train["fila"].to_numpy(), "y": y})
    rf = (_leer("oof_final_rf.csv", float_precision="round_trip")
          .merge(objetivo, on="fila").merge(fechas, on="fila"))
    assert len(rf) == len(train)

    def llamando(claves):
        """(alcanzados, llamadas, «yes») de cada grupo, con el valor esperado de los empates."""
        cifras = {}
        for clave, g in rf.groupby(claves):
            k = max(1, int(round(0.2 * len(g))))
            corte = g["puntaje"].nlargest(k).min()
            arriba, empate = g["puntaje"] > corte, g["puntaje"] == corte
            alcanzados = g.loc[arriba, "y"].sum() + (k - arriba.sum()) * g.loc[empate, "y"].mean()
            cifras[clave] = (alcanzados, k, g["y"].sum())
        return cifras

    anual, mensual = llamando(["anio"]), llamando(["anio", "mes"])
    alcanzados = sum(a for a, _, _ in anual.values())
    esperado = {
        "recallRfDentroAnio": formatear_decimal(alcanzados / y.sum()),
        "llamadasPorYesRfDentroAnio": formatear_decimal(
            sum(k for _, k, _ in anual.values()) / alcanzados, 1),
        "recallRfDentroMes": formatear_decimal(sum(a for a, _, _ in mensual.values()) / y.sum()),
        "recallTopeDosMilDiez": formatear_decimal(anual[(2010,)][1] / anual[(2010,)][2]),
    }
    for anio, palabra in ((2008, "Ocho"), (2009, "Nueve"), (2010, "Diez")):
        a, k, p = anual[(anio,)]
        esperado[f"recallRfDentroAnioDosMil{palabra}"] = formatear_decimal(a / p)
        esperado[f"llamadasDentroAnioDosMil{palabra}"] = formatear_miles(k)
        esperado[f"yesDosMil{palabra}"] = formatear_miles(int(rf.loc[rf["anio"] == anio, "y"].sum()))

    # El AUC de los pares del mismo año y mes, con la U de Mann–Whitney.
    suma = pares = 0
    for _, g in rf.groupby(["anio", "mes"]):
        p, n = int(g["y"].sum()), int((1 - g["y"]).sum())
        if p and n:
            rangos = g["puntaje"].rank(method="average")
            suma += rangos[g["y"] == 1].sum() - p * (p + 1) / 2
            pares += p * n
    esperado["aucOofDentroMes"] = formatear_decimal(suma / pares)

    # Naive Bayes categórico: su puntaje es el log-odds.
    nb = _leer("oof_final_nb_categorico.csv", float_precision="round_trip").merge(objetivo,
                                                                                    on="fila")
    probabilidad = 1 / (1 + np.exp(-nb["puntaje"]))
    alta = probabilidad >= 0.99
    decil = nb["puntaje"].rank(method="first", ascending=False) <= round(0.1 * len(nb))
    esperado.update({
        "probabilidadAltaNb": "0{,}99",
        "filasNbProbabilidadAlta": formatear_miles(int(alta.sum())),
        "pctNbProbabilidadAlta": formatear_porcentaje(100 * alta.mean()),
        "pctYesNbProbabilidadAlta": formatear_porcentaje(100 * nb.loc[alta, "y"].mean()),
        "probabilidadMediaNbDecilSuperior": formatear_decimal(probabilidad[decil].mean()),
        "tasaYesNbDecilSuperior": formatear_decimal(nb.loc[decil, "y"].mean()),
    })

    # Las filas con las mismas predictoras disponibles (sin fila, y ni duration).
    columnas = [c for c in train.columns if c not in ("fila", "y", "duration")]
    en_repetido = train.duplicated(subset=columnas, keep=False)
    clases = train.groupby(columnas, dropna=False)["y"].transform("nunique")
    distintas = en_repetido & (clases > 1)
    esperado.update({
        "vectoresRepetidos": formatear_miles(len(train.loc[en_repetido, columnas].drop_duplicates())),
        "filasVectoresRepetidos": formatear_miles(int(en_repetido.sum())),
        "pctFilasVectoresRepetidos": formatear_porcentaje(100 * en_repetido.mean()),
        "vectoresRepetidosClasesDistintas": formatear_miles(
            len(train.loc[distintas, columnas].drop_duplicates())),
        "filasVectoresRepetidosClasesDistintas": formatear_miles(int(distintas.sum())),
        "vectoresRepetidosConDuration": formatear_miles(len(
            train.loc[train.duplicated(subset=columnas + ["duration"], keep=False)]
            .drop_duplicates(subset=columnas + ["duration"]))),
    })

    # La curva de profundidad de RF, con cuatro decimales.
    profundidad = _curva("rf_max_depth")
    palabras = {"2": "Dos", "4": "Cuatro", "6": "Seis", "8": "Ocho", "10": "Diez", "12": "Doce",
                "15": "Quince", "20": "Veinte", "25": "Veinticinco", "None": "SinLimite"}
    for punto, palabra in palabras.items():
        esperado[f"aucRfProfundidad{palabra}ConCuatroDecimales"] = formatear_decimal(
            profundidad[(punto, "validacion")], 4)

    # Los pesos de clase, fold a fold (D-21, N0-12).
    for curva, modelo in (("rf_pesos_clase_depth8", "Rf"), ("svm_pesos_clase", "Svm")):
        tabla = _leer(f"curvas/{curva}.csv", dtype={"punto": str}, keep_default_na=False)
        tabla = tabla[(tabla["conjunto"] == "validacion") & (tabla["metrica"] == "auc")]
        ancho = tabla.pivot(index="fold", columns="punto", values="valor").astype(float)
        diferencia = ancho["balanced"] - ancho["None"]
        desvio = diferencia.std()
        esperado[f"deltaAucPesos{modelo}Desvio"] = formatear_decimal(
            desvio, 3 if abs(desvio) >= 0.01 else 4)
        esperado[f"foldsPositivosPesos{modelo}"] = formatear_miles(int((diferencia > 0).sum()))

    distintos = {n: (v[n], e) for n, e in esperado.items() if v[n] != e}
    assert not distintos, distintos
    # Lo que DECISIONES.md, N0-12 y D-21, dice en palabras: positiva en los cinco folds.
    assert v["foldsPositivosPesosRf"] == v["foldsPositivosPesosSvm"] == v["kFolds"]
    print(f"ok  los {len(esperado)} macros del cierre de la ola 8, recalculados por otro camino, "
          f"coinciden: recall dentro del año {a_texto(v['recallRfDentroAnio'])} y dentro del mes "
          f"{a_texto(v['recallRfDentroMes'])}, AUC dentro del mes {a_texto(v['aucOofDentroMes'])}, "
          f"profundidad 6 {a_texto(v['aucRfProfundidadSeisConCuatroDecimales'])}, "
          f"{a_texto(v['vectoresRepetidos'])} vectores repetidos")


def main():
    test_formato_espanol_con_casos_a_mano()
    test_los_macros_se_leen_igual_en_texto_y_en_modo_matematico()
    test_numeros_en_palabras()
    test_nombres_de_macro_solo_letras()
    test_los_titulos_del_esqueleto_se_escriben_con_macros()
    test_el_verificador_no_marca_sintaxis()
    test_el_verificador_detecta_numeros_tipeados()
    test_el_verificador_sin_preambulo_y_en_los_macros_generados()
    test_cli_verificar_da_archivo_linea_y_codigo()
    test_cli_sintetico_es_determinista_y_escribe_interrogacion_si_falta_un_archivo()
    test_con_la_evaluacion_de_test_y_la_evidencia_de_la_particion_se_derivan_sus_macros()
    test_evidencia_particion_escribe_el_archivo_que_se_lee_aqui()
    test_sin_ningun_resultado_no_falla()
    test_la_meseta_de_knn_vale_interrogacion_si_deja_de_estar_dentro_de_un_error_estandar()
    test_knn_por_distancia_sale_de_la_curva_y_no_de_hiperparametros_json()
    test_resultados_reales_coinciden_con_las_otras_fuentes()
    test_el_corte_reparte_los_empates_en_proporcion()
    test_llamar_dentro_de_cada_anio_y_de_cada_mes_sobre_un_caso_a_mano()
    test_vectores_repetidos_sobre_un_caso_a_mano()
    test_macros_del_cierre_de_la_ola_ocho_por_otro_camino()
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()

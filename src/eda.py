"""Correr con: python -m src.eda   (antes: python -m src.datos)

Análisis exploratorio del dataset Bank Marketing, **sólo sobre train**. Las decisiones que
salgan de aquí (qué variables excluir, qué categorías agrupar, qué escalar) no pueden haber
mirado el test: si lo miran, la estimación del punto 4 sale optimista. Ver DECISIONES.md, D-04.

Imprime un reporte por consola (también lo guarda en resultados/eda/reporte.txt) y guarda las
figuras en resultados/eda/.

Cubre lo que pide el enunciado del TP2 (sección 1): balance de clases,
distribuciones, relación predictor-objetivo, faltantes ("unknown"), categorías
poco frecuentes, diferencias entre clases, correlaciones y posibles fuentes de
data leakage (duration).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datos import FILA, OBJETIVO, RAIZ, RUTA_TRAIN, cargar_train

SALIDA = RAIZ / "resultados" / "eda"
SALIDA.mkdir(parents=True, exist_ok=True)

UMBRAL_RARA = 0.01  # categoría "poco frecuente": menos del 1 % de las filas
# Centinela de pdays: no hay días registrados desde un contacto previo. NO equivale a «nunca
# contactado», aunque bank-additional-names.txt lo describa así: hay filas con pdays == 999 y
# previous >= 1. La sección 2 del reporte las cuenta.
PDAYS_CENTINELA = 999

ORDEN_MESES = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
ORDEN_DIAS = ["mon", "tue", "wed", "thu", "fri"]

_lineas = []


def log(texto=""):
    print(texto)
    _lineas.append(str(texto))


def titulo(texto):
    log()
    log("=" * 78)
    log(texto)
    log("=" * 78)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(SALIDA / nombre, dpi=130)
    plt.close(fig)


def cramers_v(x, y):
    """V de Cramér (con corrección de sesgo) entre dos variables categóricas."""
    tabla = pd.crosstab(x, y).to_numpy()
    n = tabla.sum()
    esperado = tabla.sum(1, keepdims=True) @ tabla.sum(0, keepdims=True) / n
    chi2 = ((tabla - esperado) ** 2 / esperado).sum()
    r, k = tabla.shape
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    r_c = r - (r - 1) ** 2 / (n - 1)
    k_c = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2 / max(1e-12, min(k_c - 1, r_c - 1)))


# --------------------------------------------------------------------------
# 1. Carga y estructura
# --------------------------------------------------------------------------
df = cargar_train()
df["y_bin"] = (df[OBJETIVO] == "yes").astype(int)

numericas = [c for c in df.select_dtypes(include="number").columns if c not in (FILA, "y_bin")]
categoricas = [c for c in df.select_dtypes(exclude="number").columns if c != OBJETIVO]

titulo("1. ESTRUCTURA")
log(f"Archivo: {RUTA_TRAIN.relative_to(RAIZ)}  (train: 80 % del dataset sin duplicados)")
log("El test (data/particion/test.csv) no se abre hasta el punto 4.")
log(f"Filas: {len(df):,}   Columnas: {len(numericas) + len(categoricas)} predictoras + y")
log(f"Numéricas ({len(numericas)}): {', '.join(numericas)}")
log(f"Categóricas ({len(categoricas)}): {', '.join(categoricas)}")

duplicadas = df.drop(columns=FILA).duplicated().sum()
log(f"Filas duplicadas exactas: {duplicadas} (los 12 duplicados del CSV se quitaron antes de partir)")

# --------------------------------------------------------------------------
# 2. Faltantes: NaN explícitos y "unknown" implícitos
# --------------------------------------------------------------------------
titulo("2. VALORES FALTANTES")
nan = df.isna().sum()
log(f"NaN explícitos en todo el dataset: {int(nan.sum())}")

unknown = pd.DataFrame({
    "n_unknown": [(df[c] == "unknown").sum() for c in categoricas],
}, index=categoricas)
unknown["pct"] = 100 * unknown["n_unknown"] / len(df)
unknown["tasa_yes_si_unknown"] = [
    100 * df.loc[df[c] == "unknown", "y_bin"].mean() if (df[c] == "unknown").any() else np.nan
    for c in categoricas
]
unknown = unknown[unknown["n_unknown"] > 0].sort_values("pct", ascending=False)
log('Categóricas con "unknown" (faltante codificado como categoría):')
log(unknown.round(2).to_string())

centinela = df["pdays"] == PDAYS_CENTINELA
log()
log(f"pdays == {PDAYS_CENTINELA} (centinela): {100 * centinela.mean():.2f} % de las filas")
log("  -> es un centinela, no una distancia: tratarlo como numérico directo distorsiona")
log("     escalado y distancias (KNN/SVM).")
contactados = centinela & (df["previous"] >= 1)
log(f"  Pero NO significa 'nunca contactado': {contactados.sum():,} filas con pdays == "
    f"{PDAYS_CENTINELA} tienen previous >= 1")
log(f"  (poutcome de esas filas: {df.loc[contactados, 'poutcome'].value_counts().to_dict()}).")
log("  -> para marcar 'contactado antes' conviene usar previous > 0, no pdays != 999.")
log(f"poutcome == 'nonexistent': {100 * (df['poutcome'] == 'nonexistent').mean():.2f} % "
    "(coherente con previous == 0)")
log(f"  Filas con previous == 0 y poutcome != 'nonexistent': "
    f"{((df['previous'] == 0) & (df['poutcome'] != 'nonexistent')).sum()}")

# --------------------------------------------------------------------------
# 3. Balance de clases
# --------------------------------------------------------------------------
titulo("3. BALANCE DE CLASES")
conteo = df[OBJETIVO].value_counts()
for clase, n in conteo.items():
    log(f"  {clase:>3}: {n:6,}  ({100 * n / len(df):.2f} %)")
log(f"  Ratio no/yes: {conteo['no'] / conteo['yes']:.1f} : 1")
log("  -> desbalanceado: accuracy engaña (predecir siempre 'no' da "
    f"{100 * conteo['no'] / len(df):.1f} %). Usar métricas como recall/precision/F1/AUC-PR")
log("     y StratifiedKFold barajado: folds() de src/validacion.py (D-06), nunca cv=5.")

fig, ax = plt.subplots(figsize=(4, 3.5))
ax.bar(conteo.index, conteo.values, color=["#8a9bb0", "#d9822b"])
for i, v in enumerate(conteo.values):
    ax.text(i, v, f"{100 * v / len(df):.1f} %", ha="center", va="bottom")
ax.set_title("Balance de clases (y)")
ax.set_ylabel("Clientes")
guardar(fig, "01-balance-clases.png")

# --------------------------------------------------------------------------
# 4. Numéricas: resumen, por clase, asimetría
# --------------------------------------------------------------------------
titulo("4. VARIABLES NUMÉRICAS")
resumen = df[numericas].describe().T
resumen["asimetria"] = df[numericas].skew()
resumen["n_unicos"] = df[numericas].nunique()
log(resumen.round(3).to_string())

log()
log("Medias y medianas por clase:")
por_clase = df.groupby(OBJETIVO)[numericas].agg(["mean", "median"]).T.unstack()
log(por_clase.round(3).to_string())

log()
log("Correlación punto-biserial con y (Pearson contra y_bin), ordenada por |r|:")
corr_y = df[numericas + ["y_bin"]].corr()["y_bin"].drop("y_bin")
corr_y = corr_y.reindex(corr_y.abs().sort_values(ascending=False).index)
log(corr_y.round(3).to_string())

log()
log("Escalas: rango (max - min) de cada numérica:")
log((df[numericas].max() - df[numericas].min()).round(2).to_string())
log("  -> escalas muy distintas: KNN y SVM necesitan escalado; RF y NB no.")

# Histogramas por clase (densidad, para comparar pese al desbalance)
cols = 4
filas = int(np.ceil(len(numericas) / cols))
fig, axes = plt.subplots(filas, cols, figsize=(4 * cols, 3 * filas))
for ax, c in zip(axes.flat, numericas):
    datos = df[c]
    if c == "pdays":
        datos = datos[datos != PDAYS_CENTINELA]
    bins = min(40, datos.nunique())
    for clase, color in [("no", "#8a9bb0"), ("yes", "#d9822b")]:
        ax.hist(datos[df.loc[datos.index, OBJETIVO] == clase], bins=bins,
                density=True, alpha=0.55, color=color, label=clase)
    ax.set_title(c + (" (sin 999)" if c == "pdays" else ""))
for ax in axes.flat[len(numericas):]:
    ax.axis("off")
axes.flat[0].legend()
fig.suptitle("Distribución de numéricas por clase (densidad)")
guardar(fig, "02-numericas-por-clase.png")

# Boxplots por clase
fig, axes = plt.subplots(filas, cols, figsize=(4 * cols, 3 * filas))
for ax, c in zip(axes.flat, numericas):
    ax.boxplot([df.loc[df[OBJETIVO] == k, c] for k in ["no", "yes"]],
               tick_labels=["no", "yes"], showfliers=False)
    ax.set_title(c)
for ax in axes.flat[len(numericas):]:
    ax.axis("off")
fig.suptitle("Boxplots por clase (sin outliers)")
guardar(fig, "03-boxplots-por-clase.png")

# --------------------------------------------------------------------------
# 5. Outliers (regla de Tukey, 1.5 IQR)
# --------------------------------------------------------------------------
titulo("5. OUTLIERS (1.5 IQR)")
q1, q3 = df[numericas].quantile(0.25), df[numericas].quantile(0.75)
iqr = q3 - q1
fuera = ((df[numericas] < q1 - 1.5 * iqr) | (df[numericas] > q3 + 1.5 * iqr)).mean() * 100
log(fuera.round(2).rename("% filas fuera").to_string())

# --------------------------------------------------------------------------
# 6. Correlación entre numéricas (multicolinealidad)
# --------------------------------------------------------------------------
titulo("6. CORRELACIÓN ENTRE NUMÉRICAS")
corr = df[numericas].corr()
pares = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
         .stack().rename("r").reset_index())
pares = pares.reindex(pares["r"].abs().sort_values(ascending=False).index)
log("Pares con |r| > 0.5:")
log(pares[pares["r"].abs() > 0.5].round(3).to_string(index=False))
log("  -> los indicadores macro (emp.var.rate, euribor3m, nr.employed) están casi")
log("     colineales: viola la independencia condicional de Naive Bayes y es redundante.")

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(numericas)), numericas, rotation=45, ha="right")
ax.set_yticks(range(len(numericas)), numericas)
for i in range(len(numericas)):
    for j in range(len(numericas)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(corr.iloc[i, j]) > 0.6 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Correlación de Pearson entre numéricas")
guardar(fig, "04-correlacion-numericas.png")

# --------------------------------------------------------------------------
# 7. Categóricas: cardinalidad, categorías raras, tasa de 'yes'
# --------------------------------------------------------------------------
titulo("7. VARIABLES CATEGÓRICAS")
tasa_global = df["y_bin"].mean()
log(f"Tasa global de 'yes': {100 * tasa_global:.2f} %")

asociacion = {}
for c in categoricas:
    tabla = (df.groupby(c)["y_bin"].agg(n="size", tasa_yes="mean")
             .assign(pct=lambda t: 100 * t["n"] / len(df), tasa_yes=lambda t: 100 * t["tasa_yes"])
             .sort_values("n", ascending=False))
    raras = tabla[tabla["pct"] < 100 * UMBRAL_RARA]
    asociacion[c] = cramers_v(df[c], df[OBJETIVO])
    log()
    log(f"--- {c}  ({len(tabla)} categorías, V de Cramér con y = {asociacion[c]:.3f})")
    log(tabla[["n", "pct", "tasa_yes"]].round(2).to_string())
    if len(raras):
        log(f"  Poco frecuentes (< {100 * UMBRAL_RARA:.0f} %): {', '.join(map(str, raras.index))}")

log()
log("Asociación categórica-objetivo (V de Cramér), ordenada:")
log(pd.Series(asociacion).sort_values(ascending=False).round(3).to_string())

# Tasa de 'yes' por categoría
cols = 3
filas = int(np.ceil(len(categoricas) / cols))
fig, axes = plt.subplots(filas, cols, figsize=(5.5 * cols, 3.5 * filas))
for ax, c in zip(axes.flat, categoricas):
    orden = {"month": ORDEN_MESES, "day_of_week": ORDEN_DIAS}.get(c)
    g = df.groupby(c)["y_bin"].agg(["mean", "size"])
    g = g.reindex([o for o in orden if o in g.index]) if orden else g.sort_values("mean")
    colores = ["#c44e52" if n / len(df) < UMBRAL_RARA else "#4c72b0" for n in g["size"]]
    ax.barh(g.index.astype(str), 100 * g["mean"], color=colores)
    ax.axvline(100 * tasa_global, color="gray", ls="--", lw=1)
    if orden:
        ax.invert_yaxis()  # meses y días de arriba hacia abajo en orden cronológico
    ax.set_title(f"{c}  (V={asociacion[c]:.2f})")
    ax.set_xlabel("% yes")
    ax.tick_params(axis="y", labelsize=8)
for ax in axes.flat[len(categoricas):]:
    ax.axis("off")
fig.suptitle("Tasa de 'yes' por categoría (rojo: < 1 % de filas; línea: tasa global)")
guardar(fig, "05-tasa-yes-por-categoria.png")

# --------------------------------------------------------------------------
# 8. Data leakage: duration
# --------------------------------------------------------------------------
titulo("8. DATA LEAKAGE: duration")
log("Según bank-additional-names.txt, duration (segundos de la última llamada) no se")
log("conoce antes de llamar y si duration == 0 entonces y == 'no'. Debe excluirse")
log("del modelo realista (o usarse solo como benchmark).")
log(f"  Filas con duration == 0: {(df['duration'] == 0).sum()} "
    f"-> y: {df.loc[df['duration'] == 0, OBJETIVO].value_counts().to_dict()}")
bins_dur = pd.qcut(df["duration"], 10, duplicates="drop")
tasa_dur = df.groupby(bins_dur, observed=True)["y_bin"].mean() * 100
log("  Tasa de 'yes' por decil de duration:")
log(tasa_dur.round(2).to_string())
log(f"  Correlación duration-y: {corr_y['duration']:.3f} (la mayor de las numéricas)")

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(range(len(tasa_dur)), tasa_dur.values, color="#d9822b")
ax.set_xticks(range(len(tasa_dur)), [f"{max(0, iv.left):.0f}-{iv.right:.0f}" for iv in tasa_dur.index],
              rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Decil de duration (s)")
ax.set_ylabel("% yes")
ax.set_title("duration vs. tasa de 'yes' (leakage)")
guardar(fig, "06-duration-leakage.png")

# --------------------------------------------------------------------------
# 9. Estructura temporal: el archivo está ordenado por fecha (may-2008 a nov-2010)
# --------------------------------------------------------------------------
titulo("9. ORDEN TEMPORAL")
log("El archivo está ordenado cronológicamente. La tasa de 'yes' cambia a lo largo")
log("del archivo, así que un split sin mezclar (shuffle=False) no es representativo.")
log(f"Los bloques se arman por la columna '{FILA}' (posición en el CSV original).")
bloques = df["y_bin"].groupby(df[FILA] // 2000).mean() * 100
log("Tasa de 'yes' por bloque de 2000 filas del CSV original (sólo las de train):")
log(bloques.round(1).to_string())

fig, ax1 = plt.subplots(figsize=(9, 3.5))
ax1.plot(bloques.index * 2000, bloques.values, marker="o", color="#d9822b", label="% yes")
ax1.set_xlabel("Fila del CSV original (orden temporal)")
ax1.set_ylabel("% yes", color="#d9822b")
ax2 = ax1.twinx()
ax2.plot(df[FILA], df["euribor3m"], color="#4c72b0", lw=0.8, label="euribor3m")
ax2.set_ylabel("euribor3m", color="#4c72b0")
ax1.set_title("Tasa de 'yes' y euribor3m a lo largo del archivo (train)")
guardar(fig, "07-orden-temporal.png")

# --------------------------------------------------------------------------
# 10. Resumen de decisiones sugeridas para el pipeline
# --------------------------------------------------------------------------
titulo("10. CONSECUENCIAS PARA EL PIPELINE (a discutir)")
decisiones = [
    "HECHO (D-01): duplicados exactos eliminados antes del split.",
    "HECHO (D-02, D-03): split 80/20 estratificado y con shuffle antes de cualquier transformación.",
    "HECHO (D-05): duration fuera del modelo (leakage); a lo sumo, un modelo con duration como techo.",
    "HECHO (D-06): la CV usa folds() = StratifiedKFold barajado; nunca cv=5 (no baraja).",
    "Métricas acordes al desbalance (no accuracy sola).",
    "pdays: el 999 es un centinela; 'contactado antes' sale de previous > 0 (y/o descartar pdays).",
    "'unknown': mantener como categoría (su tasa de 'yes' difiere) o imputar dentro del pipeline.",
    "default: casi no tiene 'yes' conocidos; evaluar descartarla (ver tabla de la sección 7).",
    "Agrupar categorías < 1 % (p. ej. education 'illiterate', default 'yes') en 'otros'.",
    "One-hot para categóricas nominales; month y day_of_week también (no son lineales).",
    "Escalado (StandardScaler) para KNN y SVM, dentro del Pipeline.",
    "Macro colineales: considerar quedarse con una o dos (euribor3m, nr.employed), sobre todo para NB.",
]
for d in decisiones:
    log(f"  - {d}")

(SALIDA / "reporte.txt").write_text("\n".join(_lineas) + "\n", encoding="utf-8")
print(f"\nReporte y figuras guardados en: {SALIDA}")

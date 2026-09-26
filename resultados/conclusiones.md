# Conclusiones (punto 5)

Sin números de test (N0-1); archivos en `resultados/`. «OOF»: puntajes fuera de fold de RF con
`cargar_train()`; «año»: inferido como en el EDA.

## 1. Por qué cada modelo rindió lo que rindió

| Modelo | AUC de validación (final) | Qué supone o necesita | Qué muestran los datos | Consecuencia observada |
|---|---|---|---|---|
| Naive Bayes categórico | 0,782 ± 0,007 | Independencia condicional dada la clase (D-14); el gaussiano, además, una normal por clase (D-18) | **Variables muy correlacionadas:** tres variables macro, con r de 0,91 a 0,97 (`eda/reporte.txt`, §6). **Distribuciones no gaussianas:** asimetría de `campaign` 4,89, `pdays` = 999 en el 96,3 %, edad en U (`eda/eda.html`) | Tercero. Discretizar no supone forma: +0,0132 ± 0,0024 sobre el gaussiano (D-18). El bloque macro ayuda aunque viola la independencia: sacarlo cuesta 0,033 (`robustez_temporal.csv`). La redundancia cuenta tres veces la época (inferencia), y reducirla empeora el NB gaussiano (A6, −0,004; D-14) |
| SVM lineal | 0,774 ± 0,009 | Escalas comparables (Clase 8, slide 54; Alpaydin §8.3, p. 192); un puntaje lineal es monótono en cada variable (inferencia) | **Diferencias de escala:** desvíos de 0,50 a 186,6 (`eda/eda.html`); se escala (D-12). **Relaciones no lineales:** la U de la edad (fila RF) | Última, y terminó lineal: con C = 0,001, lineal 0,7741 y RBF 0,7681 (el RBF en su mejor C, 0,01, da 0,7721; D-22). El RBF puede captar la U (D-16), pero con γ sin ajustar no la aprovechó (inferencia) |
| KNN | 0,784 ± 0,008 | Distancias: sin escalar, las decide la variable de mayor varianza (Clase 8, slide 54) | **Diferencias de escala:** sin escalar, `pdays` y `nr.employed` son el 86,6 % y el 13,0 % de la varianza (D-12) | Segundo, con k = 801 (D-22). Sin escalar sube +0,0084 por la razón equivocada: mide historial y época (`nr.employed` es un reloj; D-12) |
| Random Forest | 0,795 ± 0,006 | Ni escala ni forma (D-12); sus cortes captan la U (D-16) | **Relaciones no lineales:** 24,2 % de «yes» antes de los 25 años, 8,0 % en 45–49 y 39,3 % desde los 60; `age` sola da AUC 0,507 (`eda/eda.html`) | Primero; brecha train − validación, 0,030 (D-22). Pero la U se mezcla con la época: de los 742 clientes de más de 60 años, 741 son de 2009–2010, y RF es el que más pierde sin macro ni `month`: −0,059 con la configuración final (R4, casi como NB, −0,058) y −0,093 con la de referencia (A8; D-15, D-25) |

AUC: media ± desvío en 5 folds (`cv_final_resumen.csv`). Los cuatro quedan a menos de 0,03 entre
sí y a 0,27–0,30 de «sin modelo» (0,500).

## 2. El hallazgo

**Se elige H1: el modelo aprende la época y, hacia adelante en el tiempo, en promedio casi no
supera el azar.** Es el de más evidencia. Hacia adelante (D-25; media de 5 bloques), RF pasa de
0,795 a 0,558 ± 0,123; KNN, de 0,784 a 0,539; NB, de 0,782 a 0,598; SVM, de 0,774 a 0,541.
Barajado, sin el bloque macro los cuatro pierden 0,020–0,033, y sin él ni `month`, 0,048–0,059
(`robustez_temporal_resumen.csv`).

**Slides 18 y 19** (figuras desde `robustez_temporal*.csv` y la OOF):
1. *Slide 18.* Texto: en promedio, mezclando épocas los cuatro modelos superan 0,77 de AUC;
   entrenados con el pasado y evaluados en el bloque siguiente, ninguno pasa de 0,60. Parte del
   0,795 de RF es distinguir años. Figura: puntos pareados por modelo (eje x: AUC, 0,40–0,85);
   lleno, barajado; hueco, hacia adelante con ±1 desvío; valores anotados; vertical en 0,5, «sin
   modelo».
2. *Slide 19.* Texto: en los tres bloques de 2008 RF da 0,489, 0,500 y 0,426, y ni barajado ordena
   bien esas filas (0,536, 0,519 y 0,584); en los de 2009–2010, 0,664 y 0,711, entre 0,05 y 0,08
   menos que barajado. Sin las macro, casi igual. Figura: por bloque (may–jul, jul–ago y ago–nov
   2008; nov 2008–may 2009; may 2009–nov 2010), AUC de RF hacia adelante con todas (llena), sin
   macro (discontinua) y barajado en las mismas filas (hueca); horizontales en 0,5 y 0,795; barras,
   tasa de «yes» (`robustez_folds.csv`).

**Lectura honesta.** La caída junta tres cosas. (a) Parte del 0,795 es ordenar épocas: el 66,3 %
de los pares «yes»–«no» de la OOF cruza años, y ahí el AUC es 0,864; dentro del año, 0,658. (b) En
2008 el modelo no sirve: ni barajado ordena bien ese año (0,598), y el primer bloque entrena con 159
«yes». (c) En 2009–2010 pierde 0,05 a 0,08 contra el barajado en las mismas filas: real, pero sin
caer al azar. Sin las macro, RF hacia adelante da 0,556, y sin ellas ni `month`, 0,547 (D-25): no
es sólo que fechen las filas.

**H2 y H3 quedan de apoyo.** H2 (A1: `duration` suma +0,17 a RF, hasta 0,94) es la fuga que
advierte names.txt (D-05). H3 (62,8 % de los «yes» llamando al 20 %; `ganancia_final.csv`)
repite el recall ya mostrado, y H1 lo matiza: hacia adelante, ese recall es 0,243.

## 3. Qué errores comete el modelo elegido

**Sobre validación, no sobre test** (N0-1). OOF de RF sobre la lista completa de validación (los 5
folds juntos), llamando al 20 %: 6 588 llamadas. Es el corte de la curva de ganancia y del test, y
la matriz de la slide 15 (`src/numeros.py`, `\vpRfVal` a `\vnRfVal`):

| | Llamado | No llamado |
|---|---|---|
| «yes» | VP = 2 329 | FN = 1 382 |
| «no» | FP = 4 259 | VN = 24 970 |

**Lectura de negocio.** Un FN es un cliente que habría contratado y no se llama: 1 382, el 37,2 %
de los «yes». Un FP es una llamada de más: 4 259, el 64,6 % de las 6 588 llamadas. Cada «yes»
cuesta 2,8 llamadas, contra 8,9 sin modelo. El error caro es el FN (D-19).

**Perfiles (OOF).**
- **Época:** el 76,0 % de los FN es de 2008; RF llama al 0,4 % de las filas de 2008 y a todas las
  de 2010. Mayo aporta el 37,0 % de los FN: RF deja fuera 512 de sus 704 «yes».
- **Sin historial:** el 92,2 % de los FN no tiene campaña previa; con `poutcome` = success, RF llama
  a los 1 094 y no pierde ninguno de sus 722 «yes».

## 4. Limitaciones

Por impacto:
1. **Época (D-03, D-25):** hacia adelante, RF da 0,558 en promedio. El test, aleatorio de la misma
   mezcla, medirá clientes nuevos de esas campañas, no una campaña futura.
2. **Presupuesto del 20 % (D-20):** con q de 10 a 30 %, el recall de RF va de 0,446 a 0,705, y RF
   siempre queda primero (`sensibilidad_q_final.csv`).
3. **Hiperparámetros elegidos en la misma CV que los reporta:** el máximo sale optimista. Cuánto, no
   se midió; la regla de 1 ES lo atenúa (profundidad 8 y no 10: 0,7954 contra 0,7968). Lo mediría la
   CV anidada de la Clase 3 (slides 98–100: una CV interna para elegir y otra externa para
   evaluar), que no se usó (D-22); la defensa es el test intacto (D-24).
4. **Desbalance sin tratar (D-21):** pesos balanceados en RF, +0,0006: una diferencia positiva en
   los cinco folds, pero despreciable por su tamaño (N0-12).
5. **`unknown` (D-08):** imputar es neutro (RF +0,0011); los clientes con `default` «unknown»
   (5,3 % de «yes») casi no se llaman y aportan el 21,6 % de los FN (OOF).
6. **`duration` fuera (D-05):** con ella, 0,94 (A1), pero no existe antes de llamar.
7. **`campaign` puede depender del resultado (inferencia, D-07):** media 2,053 en «yes» y 2,627 en
   «no», compatible pero no concluyente; efecto no medido.

## 5. Mejoras

Por impacto:
1. **Validación temporal como esquema principal (D-25):** mide una campaña futura.
2. **Más historial del cliente:** el 86,3 % no tiene campaña previa (`eda/reporte.txt`, §7) y
   concentra el 92,2 % de los FN.
3. **Costo explícito por tipo de error:** el umbral «depende del costo de cada error» (Clase 4,
   slides 41–44), no de un 20 % fijo (D-20).
4. **Calibración de probabilidades:** un costo por error necesita probabilidades (inferencia), y hoy
   sólo cuenta el orden (D-19).

## 6. Frases para el guion

- **§1.** «Los cuatro modelos quedan a dos centésimas entre sí y a casi tres décimas del azar».
- **§2.** «Mezclando épocas, Random Forest llega a 0,795, en parte por distinguir años; entrenado
  con el pasado, cae a 0,558 en promedio: en 2008 no sirve, y en 2009–2010 pierde cinco a ocho
  centésimas». «Sin variables económicas, pasa lo mismo: no es sólo que fechen a los clientes».
- **§3.** «El error caro es el cliente que habría contratado y no llamamos: tres de cada cuatro son
  de 2008».
- **§4.** «El número de test valdrá para clientes nuevos de estas campañas, no para una campaña
  futura».
- **§5.** «Lo primero sería validar hacia adelante en el tiempo, que es como se usaría el modelo».

## Cómo se calculó

**Copiadas.** AUC finales, brecha, error estándar y línea de base: `cv_final_resumen.csv`. Barajado
contra hacia adelante, sin macro (barajado y hacia adelante) y recall hacia adelante:
`robustez_temporal_resumen.csv`; fold a fold, `robustez_temporal.csv`; tasa de «yes» de cada
bloque, `robustez_folds.csv`. ΔAUC de A1, A6 (medida sobre el NB gaussiano de referencia), A8, A11
y A12: `ablaciones_resumen.csv`. Kernels con C = 0,001: `curvas/svm_kernel.csv`; el RBF en su
mejor C: `curvas/svm_C_balanced.csv` (D-22). Profundidad 8 contra 10: `curvas/rf_max_depth.csv`
(D-22). Pesos de RF: `curvas/rf_pesos_clase_depth8.csv`, 0,79608 − 0,79544 = +0,0006; fold a
fold, +0,0006 ± 0,0009, positiva en los cinco. Es despreciable por su tamaño, no por ruido: un
cuarto del error estándar del mejor punto, `balanced` (0,0024), que es la tolerancia de empate de
D-22; y D-21 pide una mejora que valga la pena (N0-12). q = 10 y 30 %:
`sensibilidad_q_final.csv`;
RF queda primero con los tres q, pero el orden de los otros tres cambia (con q = 10 %, SVM, KNN y
NB; con 20 %, KNN, NB y SVM; con 30 %, NB, SVM y KNN). El 62,8 %: `ganancia_final.csv`, p = 20.
Correlaciones, asimetría, `pdays`, medias de `campaign` y el 86,3 % sin campaña previa:
`eda/reporte.txt`, §2, §4, §6 y §7. La U de la edad, los desvíos y el 5,3 % de «yes» con `default`
«unknown» (D-08): `eda/eda.html`. Varianza de `pdays` y `nr.employed`: D-12. La asimetría de
`campaign` es 4,89 sobre train (`eda/reporte.txt`, §4: 4,886; `eda/eda.html`: 4,89).

**Calculadas.** Desde la raíz del repo; lee sólo train (`cargar_train()`) y no entrena nada:

```bash
python3 - <<'EOF'
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.datos import cargar_train
from src.eda_html import anio_inferido, leer_names
from src.metricas import en_presupuesto, llamadas

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
train = cargar_train()
t = train.assign(y01=(train["y"] == "yes").astype(int), anio=anio_inferido(train, leer_names()))
oof = pd.read_csv("resultados/oof_final_rf.csv", float_precision="round_trip")
d = oof.merge(t, on="fila", how="left", validate="one_to_one")

# §3. Llamar al 20 % de la lista completa, los 5 folds juntos: k = llamadas(n), los k de mayor
# puntaje; el corte de la curva de ganancia, de la slide 15 y del test
k = llamadas(len(d))
d["llamado"] = d["puntaje"].rank(method="first", ascending=False) <= k
r = en_presupuesto(d["y01"], d["puntaje"])  # sin empate partido en el corte: coincide
assert np.isclose(r["recall_q"] * d["y01"].sum(), (d["llamado"] & (d["y01"] == 1)).sum())
d["VP"] = d["llamado"] & (d["y01"] == 1)
d["FP"] = d["llamado"] & (d["y01"] == 0)
d["FN"] = ~d["llamado"] & (d["y01"] == 1)
d["VN"] = ~d["llamado"] & (d["y01"] == 0)
print("Matriz agregada:", d[["VP", "FP", "FN", "VN"]].sum().to_dict())


def perfil(col):
    g = d.groupby(col, observed=True)
    r = g[["VP", "FP", "FN", "llamado", "y01"]].sum().assign(n=g.size())
    r["pct_llamado"] = 100 * r["llamado"] / r["n"]
    r["recall"] = 100 * r["VP"] / r["y01"]
    r["pct_de_los_FN"] = 100 * r["FN"] / d["FN"].sum()
    return r.round(2)


for col in ["anio", "month", "poutcome", "default"]:
    print(perfil(col), "\n")

# §2. AUC fuera de fold dentro de cada año, contra todo junto. El AUC es la proporción de pares
# «yes»–«no» bien ordenados (empates, medio), así que el de los pares del mismo año es el de cada
# año ponderado por sus p·n pares, y el de los pares de años distintos sale por diferencia
auc = roc_auc_score(d["y01"], d["puntaje"])
por_anio = d.groupby("anio").apply(lambda g: pd.Series({
    "auc": roc_auc_score(g["y01"], g["puntaje"]),
    "pares": g["y01"].sum() * (1 - g["y01"]).sum()}), include_groups=False)
pares = d["y01"].sum() * (1 - d["y01"]).sum()
dentro = por_anio["pares"].sum()
auc_dentro = (por_anio["auc"] * por_anio["pares"]).sum() / dentro
auc_entre = (auc * pares - auc_dentro * dentro) / (pares - dentro)
print("AUC con todo junto:", round(auc, 4))
print(por_anio.round(4))
print(f"pares de años distintos: {100 * (pares - dentro) / pares:.1f} %; AUC dentro del año "
      f"{auc_dentro:.4f}, entre años {auc_entre:.4f}")

# §1. A quién se llamó según la edad: la mayor de cada año y los de más de 60
print("edad máxima:", d.groupby("anio")["age"].max().to_dict(),
      "| más de 60:", d[d["age"] > 60].groupby("anio").size().to_dict(),
      "| 60 o más en 2008:", int(((d["age"] >= 60) & (d["anio"] == 2008)).sum()),
      "con 60 justos:", int(((d["age"] == 60) & (d["anio"] == 2008)).sum()))

# §1 y §2. Diferencias pareadas por fold en el esquema barajado (configuración final), la curva
# hacia adelante de RF, el NB categórico contra el gaussiano (D-18) y los pesos de RF (N0-12)
rt = pd.read_csv("resultados/robustez_temporal.csv")
v = rt[(rt["conjunto"] == "validacion") & (rt["metrica"] == "auc")]
b = v[v["esquema"] == "barajado"].pivot_table(index=["modelo", "fold"], columns="variante", values="valor")
for var in ["sin_macro", "sin_macro_ni_month"]:
    print("todas -", var, (b["todas"] - b[var]).groupby("modelo").agg(["mean", "std"]).round(4).to_dict())
a = v[(v["esquema"] == "hacia_adelante") & (v["modelo"] == "rf")]
print(a.pivot_table(index="fold", columns="variante", values="valor").round(4))
cr = pd.read_csv("resultados/cv_referencia.csv")
c = cr[(cr["conjunto"] == "validacion") & (cr["metrica"] == "auc")].pivot_table(index="fold", columns="modelo", values="valor")
dif = c["nb_categorico"] - c["nb_gaussiano"]
print("NB categórico - gaussiano:", round(dif.mean(), 4), "±", round(dif.std(), 4))
cp = pd.read_csv("resultados/curvas/rf_pesos_clase_depth8.csv", keep_default_na=False)
cp = cp[(cp["conjunto"] == "validacion") & (cp["metrica"] == "auc")].pivot_table(index="fold", columns="punto", values="valor")
dp = cp["balanced"] - cp["None"]
print("RF balanced - sin pesos:", round(dp.mean(), 4), "±", round(dp.std(), 4), "; positiva en", int((dp > 0).sum()), "de 5")

# §2. Cada bloque de validación hacia adelante: período, «yes» con que entrena y AUC barajado (OOF)
# en esas mismas filas
rf = pd.read_csv("resultados/robustez_folds.csv").query("esquema == 'hacia_adelante'")
for _, r in rf.iterrows():
    blq = d[d["fila"].between(r["fila_min_validacion"], r["fila_max_validacion"])].sort_values("fila")
    print(int(r["fold"]), blq["month"].iloc[0], blq["anio"].iloc[0], "a", blq["month"].iloc[-1],
          blq["anio"].iloc[-1], (100 * blq["anio"].value_counts(normalize=True)).round(1).to_dict(),
          "| «yes» para entrenar:", int(t.loc[t["fila"] <= r["fila_max_train"], "y01"].sum()),
          "| AUC barajado en esas filas:", round(roc_auc_score(blq["y01"], blq["puntaje"]), 4))
EOF
```

De esa salida salen: la matriz de §3 y sus cocientes (1 382 / 3 711 = 37,2 %; 4 259 / 6 588 =
64,6 %; 6 588 / 2 329 = 2,8 llamadas por «yes»; sin modelo, 1 / 0,1127 = 8,9, con la precisión de
«sin modelo» de `cv_final_resumen.csv`); los perfiles por `anio`, `month`, `poutcome` y `default`
(76,0 %, 0,4 %, 37,0 %, 512 de 704, 92,2 %, 1 094 de 1 094; con `default` «unknown», 4,8 % de
filas llamadas y 299 de los 1 382 FN, 21,6 %); el AUC por año (0,598, 0,759 y 0,749; 0,794 junto)
y por pares (66,3 % de pares de años distintos; 0,658 dentro del año, 0,864 entre años); la edad
(máxima 61 en 2008; de más de 60, 1 en 2008, 483 en 2009 y 258 en 2010, 741 de 742 en 2009–2010; el
tramo de 60 o más de 2008 son 144, 143 de ellos con 60); el 0,033 del NB categórico sin macro, las
pérdidas barajadas sin macro (0,020–0,033) y sin macro ni `month` (0,048–0,059), el +0,0132 ± 0,0024
de D-18 y el +0,0006 ± 0,0009 de los pesos; la curva hacia adelante de RF con y sin macro de la
slide 19; y, por bloque, el período (los tres primeros, 100 % de 2008; el cuarto, 97,2 % de 2009; el
quinto, 69,7 % de 2009 y 30,3 % de 2010), los «yes» para entrenar (159 en el primero) y el AUC
barajado en las mismas filas (0,536, 0,519, 0,584, 0,748 y 0,762; hacia adelante pierde 0,084 y
0,051 en los dos últimos).

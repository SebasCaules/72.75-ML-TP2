# Decisiones metodológicas

El enunciado pide **justificar** las decisiones, no sólo tomarlas. Este documento las numera
(`D-01`, `D-02`, …) para poder citarlas desde el código y desde la presentación sin repetir el
argumento en cada lugar. Entra lo que tuvo una alternativa razonable; lo que no admitía
discusión no se documenta.

Las cifras salen de `python -m src.datos`, `python -m src.evidencia_particion`,
`python -m src.validacion` y `resultados/eda/reporte.txt`.

---

## 1. Partición de los datos (paso 1)

| # | Decisión | Por qué |
|---|---|---|
| **D-01** | Los **12 duplicados exactos** se eliminan **antes** de partir (41 188 → 41 176 filas). Se conserva la primera aparición | Si una copia cae en train y la otra en test, el test deja de ser independiente. Ignorando `y` también son 12: ningún par tiene etiquetas distintas, así que no hay conflicto que resolver |
| **D-02** | Split **80/20, estratificado por `y`, con barajado y `random_state=42`**. Train: 32 940 filas (3 711 «yes»). Test: 8 236 filas (928 «yes»). Las dos partes quedan con 11,27 % de «yes» | El punto 1 del enunciado pide partir antes de cualquier transformación aprendida de los datos. Sin estratificar, la proporción de «yes» del test cambiaría con la semilla: en 400 semillas, desvío de 0,31 pp y extremos de 10,4 % y 12,1 %. El efecto es pequeño, pero estratificar no cuesta nada y elimina esa fuente de variación. **La estratificación no se vio en clase**: hay que justificarla en una línea en la presentación |
| **D-03** | La partición es **aleatoria, no temporal** | El CSV está ordenado por fecha, de mayo de 2008 a noviembre de 2010 (`bank-additional-names.txt`, §4). Con el último 20 % como test, train tendría 6,38 % de «yes» y test 30,83 %: se entrenaría en una época y se evaluaría en otra. El enunciado plantea el problema como clasificación con *k-fold* y no menciona el tiempo, así que se sigue el esquema aleatorio. **El costo va al punto 5 como limitación**: la estimación supone que los clientes futuros se parecen a la mezcla de 2008–2010, y las variables macroeconómicas le permiten al modelo reconocer la época. Queda como consulta a la cátedra |
| **D-04** | **El test no se abre hasta el punto 4.** Todo lo anterior —EDA, validación cruzada, curvas de validación— lee sólo train (`cargar_train()`). El test se lee con `cargar_test()`, una sola vez | Si una decisión (qué variables excluir, qué categorías agrupar, qué hiperparámetro elegir) mira el test, el número del punto 4 deja de estimar el desempeño en datos nuevos y sale optimista. Por eso el EDA (`src/eda.py`) corre sobre train; la primera versión corría sobre el CSV completo (commit `0a9b311`) |

La partición está **versionada** en `data/particion/`. Así el test es un archivo concreto y no
depende de que otra versión de scikit-learn reproduzca el mismo sorteo; `tests/test_datos.py`
verifica que train + test reconstruyan el dataset sin duplicados, celda por celda.

## 2. Variables

| # | Decisión | Por qué |
|---|---|---|
| **D-05** | **`duration` queda fuera del modelo.** Se conserva en los archivos de la partición para poder reportar, como mucho, un modelo con `duration` como techo de referencia | Es la duración de la última llamada: no se conoce antes de llamar, y cuando termina la llamada `y` ya se sabe. El propio `bank-additional-names.txt` (atributo 11) dice que sólo sirve como referencia y que hay que descartarla para un modelo predictivo realista. En train, el decil más corto (≤ 59 s) tiene 0 % de «yes» y el más largo (> 551 s), 46 %. El enunciado pide analizar críticamente las variables con *data leakage* (p. 1) |

## 3. Validación cruzada

| # | Decisión | Por qué |
|---|---|---|
| **D-06** | La validación cruzada usa **`folds()`** (`src/validacion.py`): `StratifiedKFold(5, shuffle=True, random_state=42)`. **Nunca `cv=5` a secas** | `cv=5` en `cross_val_score`, `GridSearchCV` o `validation_curve` equivale a `StratifiedKFold(5)` **sin barajar**. Como train está ordenado por fecha, cada fold quedaría en una época: la misma proporción de «yes» en todos, pero el euribor medio iría de 4,87 en el primer fold a 1,10 en el último. Con `folds()`, los cinco quedan entre 3,58 y 3,64. `k = 5` es provisorio, igual que en el TP1: se revisa en el punto 2.2 |

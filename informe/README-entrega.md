# TP2 — Clasificación supervisada

**Grupo 7** — Andrés Cortese (64612) · Sebastián Caules (64331)

**72.75 Aprendizaje Automático (Machine Learning) — ITBA — 2026 Q2**

Defensa: 07/10/2026 según el enunciado (la otra fecha posible es el 14/10/2026). La presentación y
este código se envían 24 h antes.

Predecir, antes de llamar a un cliente, si va a contratar un depósito a plazo fijo (`y` = yes/no),
con el dataset **Bank Marketing** del punto 0 del enunciado y los cuatro clasificadores del punto 2.1
(Naive Bayes, SVM, KNN y Random Forest, con scikit-learn), evaluados con *k-fold cross-validation*
sobre train.

**Resultado en validación: Random Forest** (profundidad máxima 8, 200 árboles) **ordena a los
clientes con AUC 0,795 y, llamando al 20 % de la lista, alcanza un recall de 0,629**, contra 0,500 y
0,200 sin modelo. Son cifras de validación cruzada de 5 folds sobre las 32 940 filas de train.

**El test todavía no se evaluó.** Sus 8 236 filas quedaron aparte desde la partición: todo lo anterior
al punto 4, del análisis exploratorio a la elección del modelo, lee sólo train (D-04). Se abren una
sola vez, con `python -m src.evaluar_test`, después de la clase del 30/09, cuando la cátedra haya
respondido las consultas del grupo (N0-1): una respuesta puede cambiar variables o la partición, y
evaluar el test dos veces le resta independencia. Hasta entonces este README no trae ningún número
de test.

---

## Qué hay adentro

```text
TP2-grupo7-codigo/
├── README.md          este archivo
├── requirements.txt   numpy, pandas, matplotlib y scikit-learn
├── DECISIONES.md      la bitácora de decisiones (D-01, D-02, …), con su evidencia y la alternativa descartada
├── GLOSARIO.md        cada término y cada decisión en una línea, y el flujo de punta a punta
├── data/
│   ├── raw/           bank-additional-full.csv (41 188 filas × 21 columnas, separador «;») y
│   │                  bank-additional-names.txt, el diccionario de variables, tal como se descargaron
│   └── particion/     train.csv (32 940 filas) y test.csv (8 236), generados por src/datos.py
├── src/               el pipeline completo, un módulo por paso
├── tests/             las suites de tests, sin pytest
├── resultados/        cada CSV y JSON que produjo el código, más el EDA y las conclusiones
└── figuras/           las figuras de análisis y, en figuras/presentacion/, las de la presentación
```

**No hace falta correr nada para ver los números:** `resultados/` trae todo lo que produjo el código
y `figuras/`, todas las figuras. Los comandos de «Cómo correrlo» los reconstruyen desde cero.

### `data/`

`raw/` es el dataset que indica el enunciado (<https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing>):
el Bank Marketing de UCI con cinco indicadores macroeconómicos. No se edita. `particion/` es la salida
de `src/datos.py`: sin los 12 duplicados exactos (D-01), 80/20 estratificado por `y` con semilla 42
(D-02). Está versionada para que el test sea un archivo concreto y no dependa de que otra versión de
scikit-learn repita el mismo sorteo. Las dos partes llevan la columna `fila`, la posición en el CSV
original, que está ordenado por fecha: es el único rastro del orden temporal y **nunca es una
predictora**.

### `src/` — el pipeline

| Módulo | Qué hace | Escribe |
|---|---|---|
| `datos.py` | Carga el CSV, quita los duplicados y parte en train y test (D-01 a D-04). `cargar_train()` es la puerta de entrada de todo lo anterior al punto 4 | `data/particion/` |
| `validacion.py` | `folds()`: `StratifiedKFold(5, shuffle=True, random_state=42)`, la partición de toda validación cruzada (D-06) | — |
| `evidencia_particion.py` | Las cifras de D-02 (estratificar) y D-03 (partición aleatoria, no temporal). Mide la partición sobre el CSV completo; no elige nada del modelo | `resultados/evidencia_particion.json` |
| `eda.py` | Análisis exploratorio en texto, sólo sobre train (punto 1) | `resultados/eda/reporte.txt` y 7 figuras |
| `eda_html.py` | El mismo análisis en una página, con qué se ve y qué hacer en cada gráfico | `resultados/eda/eda.html` |
| `preproceso.py` | `Opciones` (las decisiones D-08 a D-16 y las ablaciones A1 a A12), `Derivadas` (reglas fijas, como sacar `duration`) y `codificador()` (one-hot, escalado o discretización, según el modelo) | — |
| `modelos.py` | `crear_modelo()`: un `Pipeline` por clasificador, Derivadas → codificación → modelo; `REFERENCIA` y `GRILLAS` | — |
| `configuracion.py` | La configuración vigente, que leen todos los módulos de cómputo: `OPCIONES_FINALES`, `HIPERPARAMETROS_FINALES` y `MODELOS_FINALES` | — |
| `metricas.py` | AUC, precisión promedio, y recall, precisión y $F_1$ llamando al 20 % de la lista con mayor puntaje (D-19, D-20) | — |
| `resultados.py` | El contrato de resultados, en formato largo (`modelo, configuracion, fold, conjunto, metrica, valor`), y `validacion_cruzada()`, el corredor común | — |
| `estilo.py` | La paleta de las figuras: azul para «no» y train, naranja para «yes» y validación; «sin modelo» es el único rótulo de la línea de base | — |
| `costos.py` | El presupuesto de cómputo: un ajuste por modelo en los extremos de cada grilla | `costos.csv`, `costos_grillas.csv` |
| `ablaciones.py` | La línea de base «sin modelo» y las ablaciones A1 a A12 contra la referencia A0, en los mismos folds | `linea_base.csv`, `ablaciones*.csv` |
| `experimentos.py` | La validación cruzada de los modelos (`--etiqueta referencia` o `final`), la sensibilidad al presupuesto y la curva de ganancia | `cv_*.csv`, `oof_*.csv`, `sensibilidad_q_*.csv`, `ganancia_*.csv` |
| `curvas.py` | Las curvas de validación, train y validación por valor del hiperparámetro, y la regla de 1 error estándar (D-22) | `curvas/*.csv`, `hiperparametros.json` |
| `seleccion.py` | Elige el modelo final con la regla de D-23; no ajusta nada ni abre el test | `modelo_elegido.json` |
| `robustez.py` | La validación hacia adelante en el tiempo, dentro de train, contra la barajada, con y sin las variables que fechan las filas (D-25) | `robustez_*.csv` |
| `evaluar_test.py` | La evaluación única del test (D-24): reentrena con todo train, puntúa el test una vez y calcula el AUC y el recall con su intervalo del 95 % por bootstrap, y la matriz de confusión del corte | `evaluacion_test.json`, `puntajes_test.csv`, `informe/resultados-test.tex` |
| `graficos.py` | Las figuras de análisis, 01 a 07 | `figuras/` |
| `graficos_presentacion.py` | Las figuras de la presentación: 16:9, sin título, la leyenda arriba y revelados en pasos | `figuras/presentacion/` |
| `numeros.py` | Cada número de la presentación como macro de LaTeX, con el archivo y la columna de donde sale; `--verificar` falla si un `.tex` tiene un número escrito a mano | `informe/numeros.tex`, `informe/numeros.md` |
| `entregar.py` | Arma este zip y el PDF de la presentación; `--con-bitacora` suma `DECISIONES.md` y `GLOSARIO.md`, que este zip lleva, y `--probar` descomprime el zip en un directorio temporal y corre ahí todas las suites | `entregables/` |
| `cuadernillo.py` | El cuadernillo de ensayo del grupo: cada slide de la presentación con su guion. Lee el deck compilado y el guion, `informe/presentacion.pdf` e `informe/guion.md`, que no vienen en este zip, y necesita LuaLaTeX y poppler | `informe/slides-y-guion.pdf` |

Las rutas de la columna «Escribe» sin carpeta están en `resultados/`. `informe/` no viene en el zip:
lo crean `src.numeros` y `src.evaluar_test` al correr, y la presentación se entrega aparte, en PDF.

**Sólo `evaluar_test.py` abre el test para evaluar.** Además lo tocan `datos.py`, que lo escribe;
`evidencia_particion.py`, que mide la partición sobre el CSV completo sin elegir nada; y
`tests/test_datos.py`, que comprueba que train y test reconstruyan el dataset.
`tests/test_aislamiento.py` falla si cualquier otro archivo `.py` nombra el test o el CSV completo.

### `tests/` — las suites

| Suite | Qué verifica |
|---|---|
| `test_datos` | Train y test reconstruyen el CSV sin duplicados, son disjuntos, están estratificados y conservan el orden original |
| `test_validacion` | `folds()` es estratificado, reproducible y mezcla todas las épocas |
| `test_preproceso` | Sin fuga: el escalado, los cortes de la discretización y la moda de la imputación salen sólo del fold de entrenamiento |
| `test_modelos` | Cada clasificador entrena y predice; RF es reproducible; las grillas arman modelos válidos |
| `test_metricas` | AUC y métricas en el presupuesto, contra casos calculados a mano, con empates en el corte |
| `test_resultados` | El contrato de resultados; los puntajes fuera de fold cubren cada fila una vez; 1 o 5 procesos dan lo mismo |
| `test_paleta` | Contraste y separación de los colores, también bajo tres formas de daltonismo |
| `test_aislamiento` | Ningún archivo fuera de la lista blanca nombra el test ni el CSV completo |
| `test_ablaciones` | Las columnas de cada variante y el criterio de efecto, calculados a mano; una corrida de punta a punta |
| `test_experimentos` | Resumen, sensibilidad al presupuesto y curva de ganancia contra cifras calculadas a mano; que se use la configuración vigente |
| `test_curvas` | La regla de 1 error estándar con curvas sintéticas; `--valores`, `--fijo` y los puntos omitidos |
| `test_seleccion` | La regla de D-23, con empate y sin él; que el modelo declarado sea el que se validó |
| `test_robustez` | Los folds hacia adelante, por sus índices; el resumen, contra una tabla calculada a mano |
| `test_evaluar_test` | Las funciones de la evaluación única con datos sintéticos, sin abrir el test |
| `test_graficos`, `test_graficos_2` | Cada figura de análisis dibuja lo que dicen los datos sintéticos, con los colores de la paleta y sin textos superpuestos |
| `test_graficos_presentacion` | Las figuras de la presentación, también con datos sintéticos |
| `test_numeros` | El formato de las cifras, los macros y el verificador de números escritos a mano; una cifra de cada familia, recalculada con pandas |
| `test_entregar` | El zip sobre un repositorio de juguete, y la prueba de clon limpio |
| `test_cuadernillo` | El guion leído y emparejado con el deck, con datos sintéticos; con LuaLaTeX y poppler, además, el cuadernillo generado sobre un deck sintético. La prueba sobre el deck y el guion reales se salta aquí, porque `informe/` no viene en el zip |

No usan pytest: cada suite se corre como módulo y termina imprimiendo `TODOS LOS TESTS OK`. Las que
corren algo de punta a punta lo hacen sobre una submuestra de train (`--rapido`) o con datos
sintéticos, en directorios temporales. La única que lee `resultados/` es `test_numeros`, que
recalcula desde ahí una cifra de cada familia de macros; `resultados/` viene en el zip, así que
corre. `informe/` no viene, y eso cambia dos cosas: `test_numeros` verifica sólo los marcadores de
los números de test, y `test_cuadernillo` salta su prueba sobre el deck y el guion reales.

### `resultados/` y `figuras/`

| Archivos | Qué son | Paso |
|---|---|---|
| `eda/reporte.txt`, `eda/eda.html` y 7 figuras | El análisis exploratorio sobre train | Punto 1 |
| `evidencia_particion.json` | Las cifras de D-02 y D-03: el % de «yes» del test sin estratificar, en 400 semillas, y el de train y test con una partición temporal | Punto 1 |
| `linea_base.csv` | «Sin modelo» en los mismos 5 folds: AUC 0,5 y recall 0,200 | Punto 2.3 |
| `ablaciones_<modelo>.csv`, `ablaciones.csv`, `ablaciones_resumen.csv`, `oof_ablacion_A0_<modelo>.csv` | Las ablaciones A1 a A12: ΔAUC pareado por fold contra A0 | Punto 1 |
| `costos.csv`, `costos_grillas.csv` | Cuánto tarda cada ajuste y cada grilla | — |
| `cv_referencia_<modelo>.csv`, `cv_referencia.csv`, `cv_referencia_resumen.csv`, `oof_referencia_<modelo>.csv` | Los cinco clasificadores sin ajustar, incluidas las dos variantes de Naive Bayes | Puntos 2.1 y 2.2 |
| `curvas/<modelo>_<parámetro>[_<sufijo>].csv`, `hiperparametros.json` | Las 12 curvas de validación y lo que elige la regla de 1 error estándar | Punto 3.1 |
| `cv_final_<modelo>.csv`, `cv_final.csv`, `cv_final_resumen.csv`, `oof_final_<modelo>.csv` | Los cuatro modelos con los hiperparámetros elegidos | Punto 4 |
| `sensibilidad_q_*.csv`, `ganancia_*.csv` | Recall y precisión llamando al 10, 20 y 30 %; la curva de ganancia | Puntos 2.3 y 5 |
| `modelo_elegido.json` | El modelo final, sus hiperparámetros y sus cinco métricas de validación | Punto 4 |
| `robustez_<modelo>.csv`, `robustez_folds.csv`, `robustez_temporal.csv`, `robustez_temporal_resumen.csv` | Validación barajada contra hacia adelante, por modelo, variante y fold | Punto 5 |
| `conclusiones.md` | Por qué rindió cada modelo, el hallazgo, los errores, las limitaciones y las mejoras | Punto 5 |
| `evaluacion_test.json`, `puntajes_test.csv` | La evaluación única del test; aparecen cuando se corre | Punto 4 |
| `figuras/01` a `07` | Ablaciones, modelos contra «sin modelo», las 16 curvas, robustez temporal, ganancia y sensibilidad al presupuesto | — |
| `figuras/presentacion/` | Las mismas lecturas, regeneradas para proyectar | — |

Los CSV de validación cruzada van en formato largo, una fila por modelo, configuración, fold,
conjunto (train o validación) y métrica, y guardan siempre las cinco métricas; los `oof_*.csv` son
los puntajes fuera de fold de cada fila de train.

---

## Cómo correrlo

Requiere **Python 3.10+**. En macOS el intérprete puede llamarse `python3`: en ese caso, cada
`python` de abajo es `python3`.

```bash
pip install -r requirements.txt
```

**Los tests.** Leen `data/` o datos sintéticos; sólo `test_numeros` lee además `resultados/`, que
viene en este zip (ver «`tests/` — las suites»). Tardan menos de dos minutos en total; la más lenta
es `test_cuadernillo`, que compila con LuaLaTeX si está instalado:

```bash
python -m tests.test_datos
python -m tests.test_validacion
python -m tests.test_preproceso
python -m tests.test_modelos
python -m tests.test_metricas
python -m tests.test_resultados
python -m tests.test_paleta
python -m tests.test_aislamiento
python -m tests.test_ablaciones
python -m tests.test_experimentos
python -m tests.test_curvas
python -m tests.test_seleccion
python -m tests.test_robustez
python -m tests.test_evaluar_test
python -m tests.test_graficos
python -m tests.test_graficos_2
python -m tests.test_graficos_presentacion
python -m tests.test_numeros
python -m tests.test_entregar
python -m tests.test_cuadernillo
```

**El pipeline**, en orden. Todo lo que está antes de `src.evaluar_test` lee sólo train:

```bash
# Punto 1: partición y análisis exploratorio
python -m src.datos                  # duplicados y partición -> data/particion/ (ya viene en el zip)
python -m src.evidencia_particion    # las cifras de D-02 y D-03
python -m src.validacion             # por qué la validación cruzada tiene que barajar (D-06)
python -m src.eda                    # EDA en texto -> resultados/eda/
python -m src.eda_html               # el mismo EDA en HTML -> resultados/eda/eda.html
python -m src.metricas               # un ejemplo de las cinco métricas
python -m src.costos                 # presupuesto de cómputo -> resultados/costos*.csv

# Punto 1: línea de base y ablaciones A1 a A12 (D-07 a D-17)
python -m src.ablaciones --linea-base
python -m src.ablaciones --modelo nb_gaussiano
python -m src.ablaciones --modelo svm
python -m src.ablaciones --modelo knn
python -m src.ablaciones --modelo rf
python -m src.ablaciones --resumen

# Puntos 2.1 a 2.3: los cinco clasificadores con su configuración de referencia (D-18)
python -m src.experimentos --etiqueta referencia --modelo nb_gaussiano nb_categorico svm knn rf
python -m src.experimentos --etiqueta referencia --resumen
python -m src.experimentos --etiqueta referencia --sensibilidad-q
python -m src.experimentos --etiqueta referencia --ganancia

# Punto 3.1: curvas de validación (D-21, D-22)
python -m src.curvas --modelo rf --parametro max_depth --fijo n_estimators=300
python -m src.curvas --modelo rf --parametro n_estimators --fijo max_depth=None
python -m src.curvas --modelo rf --parametro n_estimators --fijo max_depth=8 --sufijo depth8
python -m src.curvas --modelo rf --parametro pesos_clase --valores None,balanced --fijo n_estimators=300 max_depth=None
python -m src.curvas --modelo rf --parametro pesos_clase --valores None,balanced --fijo n_estimators=300 max_depth=8 --sufijo depth8
python -m src.curvas --modelo knn --parametro n_neighbors --fijo weights=uniform --sufijo uniform
python -m src.curvas --modelo knn --parametro n_neighbors --fijo weights=distance --sufijo distance
python -m src.curvas --modelo svm --parametro C --fijo kernel=rbf pesos_clase=None
python -m src.curvas --modelo svm --parametro pesos_clase --valores None,balanced --fijo kernel=rbf C=1
python -m src.curvas --modelo svm --parametro C --fijo kernel=rbf pesos_clase=balanced --sufijo balanced
python -m src.curvas --modelo svm --parametro kernel --valores linear,poly,rbf --fijo C=0.001 pesos_clase=balanced
python -m src.curvas --modelo svm --parametro C --fijo kernel=linear pesos_clase=balanced --sufijo linear_balanced
python -m src.curvas --elegir        # la regla de 1 error estándar -> resultados/hiperparametros.json

# Punto 4: los cuatro modelos con los hiperparámetros elegidos, la elección y la robustez temporal
python -m src.experimentos --etiqueta final
python -m src.experimentos --etiqueta final --resumen
python -m src.experimentos --etiqueta final --sensibilidad-q
python -m src.experimentos --etiqueta final --ganancia
python -m src.seleccion              # D-23 -> resultados/modelo_elegido.json
python -m src.robustez --modelo nb_categorico
python -m src.robustez --modelo svm
python -m src.robustez --modelo knn
python -m src.robustez --modelo rf
python -m src.robustez --resumen     # D-25 -> resultados/robustez_temporal*.csv

# Figuras y números de la presentación
python -m src.graficos               # figuras de análisis -> figuras/
python -m src.graficos_presentacion  # figuras de proyección -> figuras/presentacion/
python -m src.numeros                # los macros -> informe/numeros.tex e informe/numeros.md

# El cuadernillo de ensayo del grupo: lee el deck compilado y el guion de informe/, que no vienen
# en este zip, y necesita LuaLaTeX y poppler; tarda unos 8 s
python -m src.cuadernillo            # -> informe/slides-y-guion.pdf

# Punto 4: la evaluación única del test
python -m src.evaluar_test
```

Tres advertencias:

- **`python -m src.evaluar_test` abre el test, y se corre una sola vez.** Reentrena el modelo de
  `resultados/modelo_elegido.json` con todo train, puntúa el test y escribe
  `resultados/evaluacion_test.json`. Si ese archivo ya existe, pide escribir «si» antes de repetir,
  y la corrida nueva queda registrada como una evaluación más. Para ensayar sin abrir el test está
  `--rapido`, que parte 3 000 filas de train en 80/20 y escribe en un directorio temporal.
- **La etiqueta `referencia` no depende de la configuración vigente.** Con ella, `src.experimentos`
  toma los hiperparámetros de `REFERENCIA` (`src/modelos.py`) sin leer `src/configuracion.py`, así
  que los comandos de arriba reconstruyen `cv_referencia_*` con la misma configuración con que se
  midió (se comprobó contra la columna `configuracion` de los cinco CSV); `--etiqueta final` usa los
  hiperparámetros vigentes, los de `src/configuracion.py`. Las dos corren con las mismas opciones de
  preprocesamiento, `OPCIONES_FINALES`. `src.curvas`, en cambio, parte de la configuración vigente:
  por eso cada comando fija con `--fijo` la configuración con que se midió su CSV, y se comprobó,
  punto por punto, contra la columna `configuracion` de los 12 archivos de `resultados/curvas/`.
- Las ablaciones miden siempre contra la referencia A0 (`Opciones()` y `REFERENCIA`) y no dependen de
  `src/configuracion.py`.

**Tiempos.** Todo lo que no ajusta modelos corre en segundos, y lo lento es la SVM. Sobre un fold de
26 352 filas (`resultados/costos.csv`), un ajuste de Naive Bayes tarda menos de una décima de segundo
y uno de KNN, alrededor de un segundo con la predicción; RF con 800 árboles, 10 s; la SVM con RBF y
C = 1, 8 s en ajustar y 12 s en puntuar el fold de entrenamiento, que las curvas necesitan; con kernel
polinómico y C = 100, 57 s; y la SVM lineal de libsvm con C = 10 no terminó en 600 s, por eso la
lineal usa `LinearSVC`. La estimación en serie de cada grilla, hecha antes de correr las curvas
(`resultados/costos_grillas.csv`), va de 1 minuto (KNN) a 23 minutos (C de la SVM); con los cinco
folds en paralelo, el valor por defecto de `--n-jobs`, tarda menos. `src.costos` mide todo eso y puede
llevar hasta una hora. Las suites de tests tardan menos de dos minutos en total, y la más lenta es
`test_cuadernillo`, unos 25 s cuando LuaLaTeX está instalado.

---

## Dónde se resuelve cada punto del enunciado

Los códigos D-xx son las decisiones de `DECISIONES.md`; la columna «Slide» es la del esqueleto de la
presentación.

| Punto del enunciado | Dónde (módulo · función) | Decisiones | Slide |
|---|---|---|---|
| 0. Dataset Bank Marketing, `y` = yes/no | `data/raw/` · `datos.cargar_train()` | — | 2 |
| 1. Partir en train y test antes de cualquier transformación | `datos.limpiar()` (duplicados) · `datos.partir()` (80/20 estratificado) · `evidencia_particion.sin_estratificar()` y `temporal()` | D-01, D-02, D-03, D-04 | 3 |
| 1. Análisis exploratorio: balance de clases, distribuciones, relación con `y`, faltantes, categorías raras, diferencias entre clases | `eda.py` → `resultados/eda/reporte.txt` · `eda_html.analizar()` → `resultados/eda/eda.html` | D-04 | 4 a 6 |
| 1. Variables con fuga de datos: qué se conoce antes de llamar | `datos.EXCLUIDAS` (`duration`) · la ablación A1, con `duration` como techo | D-05, D-07 | 4 |
| 1. Que el análisis exploratorio decida el pipeline: excluir, codificar, escalar, agrupar categorías raras, con y sin variables problemáticas | `preproceso.Opciones` y `Derivadas` · `ablaciones.medir_variante()` (A1 a A12, ΔAUC pareado por fold) → `ablaciones_resumen.csv` | D-08 a D-17 | 7 |
| 1. Escalado, imputación y codificación ajustados sólo con train, dentro del pipeline | `modelos.crear_modelo()` (un `Pipeline` por modelo) · `tests/test_preproceso.py` | D-06, D-12 | 7 |
| 2.1 Naive Bayes, SVM, KNN y RF con scikit-learn | `modelos.crear_modelo()` (`CategoricalNB`, `GaussianNB`, `SVC`, `LinearSVC`, `KNeighborsClassifier`, `RandomForestClassifier`) · `experimentos.validar()` | D-18 | 9 |
| 2.2 k-fold sólo sobre train, sin fuga | `validacion.folds()` · `resultados.validacion_cruzada()` | D-06 | 3 y 7 |
| 2.3 Dos métricas de las vistas en clase, justificadas | `metricas.evaluar()` · `metricas.en_presupuesto()` · `ablaciones.linea_base()` («sin modelo») | D-19, D-20, D-21 | 8 |
| 3.1 Un hiperparámetro por modelo con curvas de validación: C y kernel, vecinos y ponderación, profundidad y árboles | `curvas.correr_curva()` · `curvas.elegir()` → `hiperparametros.json` | D-21, D-22, N0-11, N0-12 | 10 a 12 |
| 3.1 Sobreajuste y subajuste en una curva | La de profundidad de RF: `graficos.figura_curva()` y `graficos_presentacion.py` | D-22 | 10 |
| 4. Elegir el modelo final | `experimentos.py` (`--etiqueta final`) · `seleccion.regla_d23()` → `modelo_elegido.json` | D-23 | 13 |
| 4. Entrenarlo y estimar su desempeño en datos nuevos | `evaluar_test.evaluacion_unica()` · `evaluar_test.intervalo_bootstrap()` · `robustez.py`, para saber qué mide ese número | D-24, D-25 | 14 |
| 5. Qué modelos funcionaron mejor y por qué; si los datos cumplen sus supuestos (correlaciones, distribuciones no gaussianas, escalas, relaciones no lineales) | `resultados/conclusiones.md` §1 · ablaciones A6, A7 y A12 · Naive Bayes categórico contra gaussiano en `cv_referencia_resumen.csv` | D-12, D-14, D-16, D-18 | 16 |
| 5. Qué errores son más relevantes | `resultados/conclusiones.md` §3 (matriz fuera de fold de RF) · `evaluar_test.matriz_al_corte()` | D-19, D-20 | 15 |
| 5. Limitaciones y mejoras | `resultados/conclusiones.md` §4 y §5 · `experimentos.sensibilidad_q()` · `robustez.py` | D-03, D-20, D-21, D-25 | 17 |
| 5. El hallazgo: el modelo aprende la época | `robustez.medir()` · `graficos.figura_robustez()` y `figura_robustez_folds()` · `resultados/conclusiones.md` §2 | D-25 | 18 y 19 |
| Consejo del TP1: qué datos se usan en cada etapa | `datos.cargar_train()` en todo lo anterior al punto 4 · `tests/test_aislamiento.py` | D-04 | 3 |
| Consejo del TP1: gráficas con el valor anotado y los números en las slides | `graficos_presentacion.py` · `numeros.py`: cada número de la presentación es un macro generado | — | todas |

---

## El resultado

Validación cruzada de 5 folds (`folds()`) sobre las 32 940 filas de train, con los hiperparámetros
elegidos (`resultados/cv_final_resumen.csv`). Media ± desvío entre folds. El recall y la precisión son
llamando al 20 % de cada fold de validación con mayor puntaje, 1 318 llamadas por fold; la brecha es
el AUC de train menos el de validación.

| Modelo | Hiperparámetros | AUC | Recall al 20 % | Precisión al 20 % | Precisión promedio | Brecha |
|---|---|---:|---:|---:|---:|---:|
| **Random Forest** | profundidad 8, 200 árboles, sin pesos | **0,795 ± 0,006** | **0,629 ± 0,018** | 0,354 | 0,462 | 0,030 |
| KNN | k = 801, ponderación uniforme | 0,784 ± 0,008 | 0,617 ± 0,016 | 0,348 | 0,429 | 0,006 |
| Naive Bayes categórico | deciles, α = 1 | 0,782 ± 0,007 | 0,614 ± 0,017 | 0,346 | 0,422 | 0,001 |
| SVM lineal | C = 0,001, pesos balanceados | 0,774 ± 0,009 | 0,614 ± 0,018 | 0,346 | 0,405 | 0,002 |
| sin modelo | el mismo puntaje para todos | 0,500 | 0,200 | 0,113 | 0,113 | — |

**Random Forest gana por AUC** (D-23): el umbral de empate, un error estándar por debajo del mejor,
es 0,7927, y ningún otro modelo lo alcanza; en recall el orden es el mismo. Los cuatro quedan a menos
de 0,03 de AUC entre sí y a 0,27–0,30 de «sin modelo»: lo que separa a los modelos es menos que lo que
los separa de la línea de base. Ajustar los hiperparámetros subió el AUC de RF en 0,023, el de KNN en
0,029 y el de la SVM en 0,072; sin ajustar, el mejor era el Naive Bayes, con 0,782. Con profundidad 8,
la brecha de RF es 0,030: no sobreajusta; sin límite de profundidad llegaba a 0,228.

**El número de test**, cuando se evalúe, sale de reentrenar este RF con todo train y puntuar las
8 236 filas de test una sola vez: el AUC con su intervalo del 95 % y el recall llamando al 20 % de la
lista, 1 647 llamadas. Estima el desempeño sobre clientes nuevos **de las mismas campañas** de 2008 a
2010, no sobre una campaña futura: eso es lo que muestra el hallazgo.

## El hallazgo

**El modelo aprende la época.** El archivo está ordenado por fecha, de mayo de 2008 a noviembre de
2010, y la tasa de «yes» pasa de 4,9 % en 2008 a 51,9 % en 2010. Con los folds barajados, que mezclan
épocas, RF llega a 0,795; entrenado con el pasado y validado con el bloque siguiente (`TimeSeriesSplit`,
5 cortes dentro de train, D-25), cae a 0,558 ± 0,123, y los otros tres modelos quedan entre 0,539 y
0,598. La caída junta tres cosas. **(a)** Parte del 0,795 es ordenar épocas: en los puntajes fuera de
fold de RF, el 66,3 % de los pares «yes»–«no» cruza años, y ahí el AUC es 0,864, contra 0,658 entre
pares del mismo año. **(b)** En 2008 el modelo no sirve: en los tres bloques de ese año da 0,489, 0,500
y 0,426, y ni barajado ordena bien esas filas (0,536, 0,519 y 0,584); el primer bloque entrena con
159 «yes». **(c)** En 2009–2010 pierde entre 0,05 y 0,08 contra el barajado sobre las mismas filas
(0,748 → 0,664 y 0,762 → 0,711): una caída real, pero no «casi azar». Sacar las variables que fechan
las filas no lo arregla: sin el bloque macroeconómico, RF hacia adelante da 0,556, y sin él ni `month`,
0,547. El detalle, con el cálculo de cada cifra, está en `resultados/conclusiones.md`.

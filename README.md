# TP2 — Clasificación supervisada

**72.75 Aprendizaje Automático (Machine Learning) — ITBA — 2026 Q2**

**Grupo 7** — Andrés Cortese (64612) · Sebastián Caules (64331)

Defensa: 07/10/2026 según el enunciado (o el 14/10, la otra fecha posible); presentación y código se
envían 24 h antes. Enunciado: [`TP2-Clasificación supervisada.pdf`](TP2-Clasificación%20supervisada.pdf)

Predecir si un cliente contrata un depósito a plazo fijo (`y` = yes/no) a partir de datos
demográficos, de campañas previas y del contexto económico, con Naive Bayes, SVM, KNN y Random
Forest (scikit-learn) evaluados con *k-fold cross-validation*.

**Estado:** olas 0 a 7 hechas; falta la evaluación del test y la entrega. El test se evalúa una sola
vez después de la clase del 30/09, cuando la cátedra haya respondido las consultas (N0-1); después
viene la ola 8: auditoría, entregables, gates y ensayos. El avance, paso por paso, en
`plan/EXEC_STATE.md`.

**Resultado en validación:** Random Forest, con profundidad máxima 8 y 200 árboles, ordena a los
clientes con AUC 0,795 ± 0,006 y, llamando al 20 % de la lista, alcanza un recall de 0,629, contra
0,500 y 0,200 sin modelo. Ver [Resultados](#resultados).

---

## Datos

`data/raw/`: `bank-additional-full.csv` (41 188 filas × 21 columnas, separador `;`) y
`bank-additional-names.txt`, el diccionario de variables. Es el dataset Bank Marketing de UCI
enriquecido con cinco indicadores macroeconómicos, el que indica el enunciado:
<https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing>.

Puntos del dataset que condicionan todo lo demás (detalle en `resultados/eda/reporte.txt` y
`resultados/eda/eda.html`):

- **Desbalance:** 11,27 % de «yes». Predecir siempre «no» ya acierta el 88,73 %, así que la
  exactitud (*accuracy*) no sirve para comparar modelos (D-19).
- **`duration` es una fuga de información:** no se conoce antes de llamar. Queda fuera del
  modelo (D-05, D-07); con ella, RF llegaría a un AUC de 0,939.
- **Orden temporal:** el CSV está ordenado por fecha, de mayo de 2008 a noviembre de 2010, y la tasa
  de «yes» pasa de 4,9 % en 2008 a 51,9 % en 2010. De ahí D-03, D-06 y el hallazgo (D-25).
- **`pdays = 999` no significa «nunca contactado»:** en 3 302 filas de train con 999,
  `previous ≥ 1` (D-10).

## Partición (paso 1)

`python -m src.datos` elimina los 12 duplicados exactos y parte 80/20, estratificado por `y`,
con semilla 42:

| | Filas | «yes» | % «yes» | Archivo |
|---|---:|---:|---:|---|
| train | 32 940 | 3 711 | 11,27 % | `data/particion/train.csv` |
| test | 8 236 | 928 | 11,27 % | `data/particion/test.csv` |

Los dos archivos llevan la columna `fila`, la posición en el CSV original, que es el único rastro
del orden temporal. **Nunca es una predictora**: está en `EXCLUIDAS`, junto con `duration`.

Reglas de uso:

- Todo lo que viene antes del modelo final lee **`cargar_train()`**.
- `cargar_test()` se usa **una sola vez**, en el punto 4, desde `src/evaluar_test.py` (D-04);
  `tests/test_aislamiento.py` falla si otro archivo nombra el test.
- Toda validación cruzada usa **`folds()`**, nunca `cv=5` a secas (D-06).
- Las decisiones vigentes viven en un solo lugar, `src/configuracion.py` (N0-3); los módulos de
  cómputo leen de ahí.
- Todo número de la presentación sale de un macro generado: de `informe/numeros.tex`
  (`src/numeros.py`) o, los de test, de `informe/resultados-test.tex` (`src/evaluar_test.py`);
  ninguno se escribe a mano (gate G3).

## Instalación y uso

Requiere Python 3.10+. Los comandos van con `python -m`, como en el TP1; en macOS el intérprete
suele llamarse `python3`, y entonces cada `python` es `python3` (S-07).

```bash
pip install -r requirements.txt
```

**No hace falta correr nada para ver los números:** `resultados/` y `figuras/` están versionados.
Los tests, sin pytest, terminan en `TODOS LOS TESTS OK` y tardan alrededor de un minuto en total:

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
```

El pipeline completo, en orden. Todo lo que está antes de `src.evaluar_test` lee sólo train:

```bash
# Punto 1: partición y análisis exploratorio
python -m src.datos                  # duplicados y partición -> data/particion/ (versionada)
python -m src.evidencia_particion    # las cifras de D-02 y D-03
python -m src.validacion             # por qué la validación cruzada tiene que barajar (D-06)
python -m src.eda                    # EDA en texto -> resultados/eda/
python -m src.eda_html               # el mismo EDA en HTML -> resultados/eda/eda.html
python -m src.metricas               # un ejemplo de las cinco métricas
python -m src.costos                 # presupuesto de cómputo -> resultados/costos*.csv (hasta una hora)

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
python -m src.numeros --verificar informe/presentacion.tex   # gate G3: ningún número escrito a mano

# Punto 4: la evaluación única del test (N0-1: después del 30/09)
python -m src.evaluar_test

# Entrega: el zip del código y el PDF -> entregables/; --probar hace la prueba de clon limpio
python -m src.entregar --probar
```

- **`src.evaluar_test` abre el test, y se corre una sola vez.** Si `resultados/evaluacion_test.json`
  ya existe, pide escribir «si» y registra la corrida nueva como una evaluación más, que además hay
  que anotar en `DECISIONES.md`. `--marcadores` escribe `informe/resultados-test.tex` con «?» para
  que la presentación compile antes (así está hoy), y `--rapido` ensaya todo sin abrir el test.
- **`--etiqueta referencia` necesita la configuración de referencia.** `src.experimentos` usa lo que
  diga `src/configuracion.py` al correr, y hoy tiene los hiperparámetros elegidos: para rehacer
  `cv_referencia_*` hay que poner, durante esa corrida, `HIPERPARAMETROS_FINALES = REFERENCIA` (el
  diccionario de `src/modelos.py`), como estaba en el paso 3.3 (commit `2519d9b`). Las curvas no
  tienen ese problema: cada comando fija con `--fijo` la configuración con que se midió su CSV, y se
  comprobó contra la columna `configuracion` de los 12 archivos de `resultados/curvas/`. Las
  ablaciones miden siempre contra `Opciones()` y `REFERENCIA` (N0-9).
- **Tiempos** (`resultados/costos.csv`, sobre un fold de 26 352 filas): lo lento es la SVM. Con RBF y
  C = 1 tarda 8 s en ajustar y 12 s en puntuar el fold de entrenamiento; con kernel polinómico y
  C = 100, 57 s; la lineal de libsvm con C = 10 no terminó en 600 s, y por eso la lineal va con
  `LinearSVC`. La estimación en serie de cada grilla (`resultados/costos_grillas.csv`) va de 1 minuto
  (KNN) a 23 minutos (C de la SVM); con los cinco folds en paralelo, el valor por defecto de
  `--n-jobs`, tarda menos. Lo que no ajusta modelos corre en segundos.

## Estructura

```
data/raw/                    el dataset tal como se bajó (no se edita)
data/particion/              train.csv y test.csv, generados por src/datos.py
src/datos.py                 carga, duplicados, partición; cargar_train() y cargar_test()
src/validacion.py            folds(): StratifiedKFold(5) barajado, semilla 42 (D-06)
src/evidencia_particion.py   las cifras de D-02 y D-03: mide la partición, no elige nada del modelo
src/eda.py                   EDA sobre train, en texto (punto 1) -> resultados/eda/
src/eda_html.py              el mismo EDA en HTML -> resultados/eda/eda.html
src/preproceso.py            Opciones (D-08 a D-16, A1 a A12), Derivadas y la codificación por modelo
src/modelos.py               crear_modelo(): un Pipeline por clasificador; REFERENCIA y GRILLAS
src/configuracion.py         la configuración vigente: OPCIONES_FINALES, HIPERPARAMETROS_FINALES,
                             MODELOS_FINALES (N0-3)
src/metricas.py              AUC, precisión promedio, y recall, precisión y F1 en el presupuesto (D-19, D-20)
src/resultados.py            el contrato de resultados en formato largo y validacion_cruzada() (N0-4, N0-5)
src/estilo.py                paleta y estilo de las figuras; ROTULO_LINEA_BASE = "sin modelo"
src/costos.py                presupuesto de cómputo -> resultados/costos.csv y costos_grillas.csv
src/ablaciones.py            línea de base y ablaciones A1 a A12 -> linea_base.csv, ablaciones*.csv
src/experimentos.py          validación cruzada (referencia y final), sensibilidad a q y ganancia
src/curvas.py                curvas de validación y regla de 1 ES -> resultados/curvas/, hiperparametros.json
src/seleccion.py             el modelo final según D-23 -> modelo_elegido.json; no abre el test
src/robustez.py              validación hacia adelante dentro de train (D-25) -> robustez_*.csv
src/evaluar_test.py          la evaluación única del test (D-24): el único módulo de cómputo que lo abre
src/graficos.py              figuras de análisis -> figuras/
src/graficos_presentacion.py figuras de proyección -> figuras/presentacion/ (16:9, sin título, revelados)
src/numeros.py               los números de la presentación como macros -> informe/numeros.tex y .md
src/entregar.py              arma entregables/: el zip del código y el PDF; --probar, clon limpio
tests/                       las suites, sin pytest; cada una termina en TODOS LOS TESTS OK
  test_datos.py              la partición, contra el CSV original: disjunta, estratificada, en orden
  test_validacion.py         folds(): estratificado, reproducible y con todas las épocas
  test_preproceso.py         sin fuga: escalado, cortes e imputación salen del fold de entrenamiento
  test_modelos.py            cada clasificador entrena y predice; las grillas arman modelos válidos
  test_metricas.py           las métricas, contra casos calculados a mano
  test_resultados.py         el contrato de resultados y el corredor común de validación cruzada
  test_paleta.py             la paleta, con contraste y bajo simulación de daltonismo
  test_aislamiento.py        ningún archivo fuera de la lista blanca nombra el test
  test_ablaciones.py         columnas por variante y criterio de efecto; una corrida rápida
  test_experimentos.py       resumen, sensibilidad a q y ganancia, contra cifras calculadas a mano
  test_curvas.py             la regla de 1 ES con curvas sintéticas; --valores, --fijo y omitidos
  test_seleccion.py          la regla de D-23, con empate y sin él
  test_robustez.py           los folds hacia adelante, por sus índices, y el resumen
  test_evaluar_test.py       la evaluación única con datos sintéticos, sin abrir el test
  test_graficos.py           las figuras de análisis 01 a 03, con datos sintéticos
  test_graficos_2.py         las figuras 04 a 07 y la 02 final, también sin textos superpuestos
  test_graficos_presentacion.py
                             las figuras de la presentación
  test_numeros.py            el formato de las cifras, los macros y el verificador de G3
  test_entregar.py           el zip sobre un repositorio de juguete y la prueba de clon limpio
resultados/                  todo lo que produce el código, versionado (formato largo, N0-4)
resultados/eda/              reporte.txt, eda.html y las figuras del EDA
resultados/curvas/           una curva de validación por CSV
resultados/conclusiones.md   por qué rindió cada modelo, el hallazgo, errores, limitaciones y mejoras
figuras/                     figuras de análisis: 01 ablaciones, 02 modelos, 03 curvas, 04 y 05
                             robustez temporal, 06 ganancia, 07 sensibilidad al presupuesto
figuras/presentacion/        las figuras de la presentación, regeneradas para proyectar
informe/presentacion.tex     la presentación (beamer 16:9) -> presentacion.pdf
informe/guion.md             el guion de la defensa, con reloj y banco de preguntas
informe/slides-y-guion.pdf   el cuadernillo de ensayo: cada slide con su guion, generado desde el fuente
informe/numeros.tex, .md     generados por src/numeros.py; no se editan a mano
informe/resultados-test.tex  los números de test, generados por src/evaluar_test.py («?» hasta evaluar)
informe/README-entrega.md    el README que va dentro del zip del código
DECISIONES.md                cada decisión con su evidencia y la alternativa descartada (D-01, D-02, …)
GLOSARIO.md                  cada término y cada decisión en una línea, y el flujo de punta a punta
plan/                        PLAN.md, EXEC_STATE.md, SUGERENCIAS.md y consultas.md (no se entrega)
resumen-tp2/                 resumen del enunciado y del dataset para el grupo (PDF)
entregables/                 lo que se manda a la cátedra; lo genera src/entregar.py, no se edita
```

## Resultados

Validación cruzada de 5 folds sobre las 32 940 filas de train, con los hiperparámetros elegidos
(`resultados/cv_final_resumen.csv`); media ± desvío entre folds. El recall es llamando al 20 % de
cada fold con mayor puntaje.

| Modelo | AUC | Recall al 20 % | Brecha de AUC train − validación |
|---|---:|---:|---:|
| **Random Forest** (profundidad 8, 200 árboles) | **0,795 ± 0,006** | **0,629 ± 0,018** | 0,030 |
| KNN (k = 801) | 0,784 ± 0,008 | 0,617 ± 0,016 | 0,006 |
| Naive Bayes categórico | 0,782 ± 0,007 | 0,614 ± 0,017 | 0,001 |
| SVM lineal (C = 0,001, pesos balanceados) | 0,774 ± 0,009 | 0,614 ± 0,018 | 0,002 |
| sin modelo | 0,500 | 0,200 | — |

Los cuatro quedan a menos de 0,03 de AUC entre sí y a 0,27–0,30 de «sin modelo». Ninguno entra en el
umbral de empate de D-23, 0,7927, así que gana RF. El test todavía no se evaluó (N0-1).

**El hallazgo: el modelo aprende la época** (D-25; `resultados/conclusiones.md`, §2). Hacia adelante
en el tiempo, entrenando con el pasado y validando con el bloque siguiente, RF cae de 0,795 a
0,558 ± 0,123. Parte del 0,795 es ordenar épocas: el 66,3 % de los pares «yes»–«no» cruza años, y
ahí el AUC es 0,864, contra 0,658 dentro del mismo año. En 2008 el modelo no sirve, y en 2009–2010
pierde entre 0,05 y 0,08 contra el barajado sobre las mismas filas. Sacar las variables
macroeconómicas no lo arregla: 0,556. El número de test, cuando se evalúe, valdrá para clientes
nuevos de estas campañas, no para una campaña futura.

## Próximos pasos

- Evaluación única del test (paso 5.3) y análisis de sus errores (5.4), después del 30/09 (N0-1).
- Ola 8: auditoría adversarial, entregables con prueba de clon limpio, gates G1 a G12 y dos ensayos
  cronometrados; el envío, antes del martes 06/10.

El plan completo, con fechas y la prueba de terminado de cada paso, está en `plan/PLAN.md`.

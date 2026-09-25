# Plan de implementación del TP2, de punta a punta

**Qué es.** El runbook que lleva el TP2 desde el estado actual (paso 1 hecho) hasta la entrega, con
el nivel del TP1. La entrega son dos archivos: la presentación en PDF y el código en zip. El plan se
puede ejecutar con `/workforce`, en olas de pasos atómicos que tienen cada uno su prueba de
terminado, o a mano en el mismo orden.

**Fecha límite: martes 06/10/2026**, 24 h antes de la clase de la 1.ª defensa, que es el miércoles
07/10 (TP2, p. 1). Si al grupo le toca la 2.ª fecha, el 14/10 (que sólo figura en el cronograma), esa
semana extra es margen para ensayar y pulir, no para agregar alcance.

**Estado al 25/09:**
- Paso 1 hecho y commiteado (`108ff99`): la partición, el EDA en texto sobre train, `folds()` y las
  decisiones D-01 a D-06.
- En curso: el EDA en HTML sobre train (`resultados/eda/eda.html`).
- De las olas 0 a 8 no se ejecutó nada.

## Resumen

| Ola | Qué resuelve | Punto del enunciado | Fecha | Lo que la cierra |
|---|---|---|---|---|
| 0 | Arranque: estado, consultas a la cátedra, infraestructura común, presupuesto de cómputo | — | sáb 26/09 | `plan/EXEC_STATE.md`, mensaje a la cátedra, `src/preproceso.py`, `src/modelos.py`, `src/metricas.py`, `src/estilo.py`, con sus tests |
| 1 | EDA con consecuencias: cada observación del EDA termina en una decisión medida | 1 | sáb 26 – dom 27/09 | `resultados/linea_base.csv`, `resultados/ablaciones.csv`, D-07 a D-17 |
| 2 | Métricas, presupuesto de llamadas y desbalance | 2.3 | dom 27/09 | D-19 a D-21 |
| 3 | Los cuatro clasificadores con validación cruzada | 2.1, 2.2 | dom 27/09 | `resultados/cv_modelos.csv`, D-18 |
| 4 | Curvas de validación, sobreajuste y subajuste | 3.1 | lun 28 – mar 29/09 | `resultados/curvas/`, `resultados/hiperparametros.json`, D-22 |
| 5 | Modelo final, robustez temporal y la única evaluación del test | 4 | jue 01/10, después de la clase del 30/09 | `resultados/modelo_elegido.json`, `resultados/evaluacion_test.json`, D-23 a D-25 |
| 6 | Conclusiones y hallazgo | 5 | mar 29/09 (borrador) – jue 01/10 | `resultados/conclusiones.md`, que alimenta el deck |
| 7 | Presentación, figuras de proyección, números generados y guion | todos | mar 29/09 – sáb 03/10 | `informe/presentacion.pdf`, `informe/guion.md` |
| 8 | Auditoría adversarial, entregables, gates y ensayos | — | sáb 03 – mar 06/10 | `entregables/TP2-grupoN-presentacion.pdf` y `entregables/TP2-grupoN-codigo.zip` |

**Camino crítico:** respuestas de la cátedra (30/09) → evaluación única del test (01/10) → deck
(02/10) → auditoría (03/10) → entregables y ensayos (04–05/10) → envío (06/10).

**Qué pasa si la cátedra responde distinto.** Las olas 1 a 4 no esperan esas respuestas:
- Desde la ola 1 se guardan todas las métricas candidatas: AUC, recall, precisión y $F_1$ en el
  presupuesto de llamadas, y la precisión promedio.
- Las dos variantes de Naive Bayes se miden igual, y `class_weight` también.

Si una respuesta cambia una decisión, lo que cambia es qué columna se lee, no qué hay que volver a
correr.

---

## 1. El estándar: qué hizo excelente al TP1 y cómo se repite

Las fuentes son la devolución oral del 02/09 y los tres «Consejos prácticos vistos en el TP1» del
enunciado (TP2, pp. 2–3). Las dos ya están convertidas en reglas en
`wiki/apuntes/guia-de-presentaciones.md`, en las familias A a E. La tabla dice cómo se aplica cada
regla en el TP2 y cómo se comprueba.

| Qué dijo la cátedra | Regla de la guía | Cómo se aplica en el TP2 | Cómo se comprueba |
|---|---|---|---|
| Gustó el formato narrativo | A1, A2, A5 | Títulos de slide que afirman o preguntan; secciones encadenadas; el clímax compara contra algo ya mostrado | Leer seguidos los títulos de frame (`\begin{frame}{…}` o `\frametitle`): ninguno es un rótulo |
| Gustaron las slides con poco texto y objetos grandes | B1, B2, B4, B5, B6 | Un objeto protagonista por slide; figuras regeneradas para proyección desde `src/graficos_presentacion.py`; listas de hasta 3 ítems | El PDF al 25 % sigue legible; ninguna figura del deck es la del EDA o la de análisis |
| Gustó el histograma del hallazgo al final de todo | A4, A6 | El hallazgo es lo último que se muestra (ola 6); todo lo calificable, limitaciones incluidas, va antes | En la tabla de cobertura, ningún punto del enunciado cae después del hallazgo |
| Gustó comparar cada número nuevo contra una línea de base | C1, C4 | Una sola línea de base, «sin modelo», con el mismo rótulo en todas las slides. Se calcula en validación cruzada con los mismos folds: AUC 0,5 y 20 % de los «yes» con el 20 % de las llamadas | `resultados/linea_base.csv` existe antes de la primera figura con métricas |
| **No gustó** comparar el número de test contra la línea de base | C2, C3, C6 | El número de test va solo, una vez y sin competidor. La comparación test–validación sólo se dice en voz alta | La slide de test no compara nada; el macro de test aparece como mucho dos veces en el cuerpo de las slides |
| **No gustó** decir «elegimos el IQR» sin explicar para qué | D1, D2, D3 | Cada criterio del EDA termina en su consecuencia escrita y con número, aunque sea nula («se probó y no cambió nada: ΔAUC = 0,001»). Incluye los atípicos por IQR que ya publica `resultados/eda/reporte.txt` (D-17) | Leyendo sólo el texto proyectado de cada sección se entiende qué pasó con los datos |
| Consejo: «qué datos se usan para cada etapa» (TP2, p. 3) | A3, C5 | La slide 3 muestra la tabla etapa → datos → qué decide, antes del primer número. En el guion, cada número declara su conjunto | La primera métrica aparece después de esa slide |
| Consejo: gráficas mejor que tablas, y los números en las slides (TP2, p. 3) | B3 | Resultados como gráficas con el valor anotado encima; en el deck no hay tablas de números salvo la matriz de confusión | Contar las tablas de `presentacion.tex` |
| Consejo: 10 minutos, no más (TP2, p. 3) | E1, E2, E3 | Guion con reloj, puntos de control y orden de recorte; la teoría se nombra en una pasada. Las preguntas duran 10 minutos (en el TP1 eran 8), así que el banco pasa de 14 a al menos 20 respuestas | El esqueleto suma 9:15; palabras / 2,7 ≤ 9:45; dos ensayos cronometrados ≤ 10:00 |

**También se repite lo que el TP1 hizo bien sin que la cátedra lo nombrara**, porque es lo que
sostuvo el nivel:
- un `DECISIONES.md` con la alternativa descartada de cada decisión;
- 65 tests;
- números de test generados por código;
- análisis de evidencia, como ablaciones y sensibilidad a k;
- un README de entrega con el mapa enunciado → código;
- verificación independiente de las cifras.

**Lo que el TP1 dejó como deuda se corrige desde el arranque:**
- **Todos** los números proyectados salen de macros generados por código (guía C7); en el TP1, sólo
  los de test. Los del guion se comprueban contra `informe/numeros.md`.
- `slides-y-guion.pdf` se genera desde el fuente; en el TP1 se armó a mano el día de la defensa.
- El preámbulo del deck se reusa sin código muerto (`\piefig` y los macros sin uso; guía TP1-L).
- Los conteos de slides y los tiempos de ejecución se comprueban antes de escribirlos en el README.

## 2. Inventario de entregables

Como en el TP1, se entregan **dos archivos**: presentación y código (TP2, p. 1). Todo lo demás es
respaldo interno.

| Pieza | Ruta | ¿Se entrega? | En el TP1 |
|---|---|---|---|
| Presentación | `entregables/TP2-grupoN-presentacion.pdf`, compilada desde `informe/presentacion.tex` | Sí | 22 frames, 54 páginas con los revelados, beamer 16:9 |
| Código | `entregables/TP2-grupoN-codigo.zip`, con carpeta raíz `TP2-grupoN-codigo/`: README de entrega, `requirements.txt`, `data/`, `src/`, `tests/` y `resultados/` | Sí | README, `data/`, `src/` y `tests/`, sin resultados |
| README de entrega | Dentro del zip | Sí | Resultado en la cabecera, qué hay adentro, cómo correrlo, **dónde se resuelve cada punto del enunciado**, resultado y hallazgo |
| Guion | `informe/guion.md` | No | 767 líneas, reloj y 14 preguntas |
| Cuadernillo de ensayo | `informe/slides-y-guion.pdf`, generado | No | Hecho a mano |
| Informe de respaldo | `informe/informe.pdf` | No. **Opcional** (paso 7.7) | 22 páginas, respaldo para las preguntas |
| Bitácora | `DECISIONES.md` y `GLOSARIO.md` | Se decide en la ola 8 | Fuera del zip |
| EDA en HTML | `resultados/eda/eda.html` | Sí, dentro de `resultados/` | No existía |

**Un cambio respecto del TP1:** `resultados/` entra al zip. Las curvas de la SVM tardan, y así quien
corrige puede ver cada número sin volver a correr nada.

## 3. Decisiones

### Cerradas en el paso 1

| ID | Decisión |
|---|---|
| D-01 | Duplicados eliminados antes de partir |
| D-02 | Partición 80/20 estratificada |
| D-03 | Partición aleatoria, no temporal |
| D-04 | El test no se abre hasta el punto 4 |
| D-05 | `duration` fuera del modelo |
| D-06 | `folds()` barajado |

### Reservadas para las olas 1 a 5

Los números son fijos: cada paso escribe **sólo** las suyas. Una propuesta que no cambia nada se
registra igual, con su número (guía D3).

| ID | Tema | Propuesta por defecto | Evidencia que la cierra | Ola |
|---|---|---|---|---|
| D-07 | Momento en que opera el modelo | Antes de la llamada k de la campaña. Se conocen los datos del cliente, su historial, el contexto económico del mes, el canal y la fecha de la llamada, y el número de llamada (`campaign` «includes last contact», names.txt, atributo 12). No se conoce la duración. Que `campaign` dependa del resultado (tras un «yes» no se vuelve a llamar) va como limitación | `bank-additional-names.txt`, atributos 8 a 12; TP2, p. 1 | 1 |
| D-08 | `unknown` | Categoría propia, sin imputar | ΔAUC contra imputar por la moda | 1 |
| D-09 | `default` | Indicadora «unknown / no»; en train hay 2 «yes» | ΔAUC | 1 |
| D-10 | Contacto previo | `previous > 0` como indicadora; `pdays` sale, o queda sólo para los contactados | ΔAUC. En train hay 3.302 filas con `pdays = 999` y `previous ≥ 1` | 1 |
| D-11 | Categorías raras | Fusión explícita, fijada con el EDA de train: `illiterate` → `basic.4y` y `marital` `unknown` → `married`, la moda. `job` `unknown` queda como está (0,8 %), `default` `yes` lo resuelve D-09 y `dec` no se toca por su tasa distintiva. `OneHotEncoder(min_frequency=…)` no sirve: cada columna tiene un solo nivel raro, así que sólo le cambia el nombre (verificado con scikit-learn 1.9, G-01) | ΔAUC | 1 |
| D-12 | Escalado | `StandardScaler` en KNN y SVM; RF sin escalar; en NB, según la variante | ΔAUC de KNN sin escalar. En la SVM lo exige la teoría (Clase 8, slide 54) | 1 |
| D-13 | `campaign` | `log1p` para KNN y SVM | ΔAUC | 1 |
| D-14 | Bloque macro | Comparar las 5 variables contra 1 o 2 (`euribor3m`, `nr.employed`), sobre todo para NB y KNN | ΔAUC por modelo | 1 |
| D-15 | `month` y `day_of_week` | One-hot; `day_of_week` sale si no aporta | ΔAUC | 1 |
| D-16 | Edad en NB | En tramos para NB; sin transformar para el resto | ΔAUC de NB | 1 |
| D-17 | Valores atípicos (IQR, reporte del EDA, §5) | No se elimina ninguna fila. `campaign` se trata en D-13, y RF no es sensible a ellos. Se escribe cuántas filas marca el IQR y que no se borra ninguna | Conteo por variable y tasa de «yes» entre los marcados | 1 |
| D-18 | Variante de Naive Bayes | GaussianNB contra CategoricalNB, con las numéricas discretizadas dentro del pipeline | AUC en CV; consulta 4 | 3 |
| D-19 | Métricas | AUC para comparar y elegir, y recall en el presupuesto de D-20 para medir el uso real | Clase 4: costo desigual de los errores (slides 26–27 y 41–44), precisión y recall (30–31), ROC y AUC (45–51). Consulta 1 | 2 |
| D-20 | Presupuesto de llamadas | Se llama al q = 20 % de la lista con mayor puntaje. Es un supuesto de negocio, se declara, y la sensibilidad a q = 10 % y 30 % queda en la reserva | Ver §5, ola 2 | 2 |
| D-21 | Desbalance | No se remuestrea; `class_weight` es un hiperparámetro opcional | Consulta 3 | 2 |
| D-22 | Elección de hiperparámetros | El valor con mayor AUC medio de validación. Si la curva es plana dentro de 1 error estándar, el más simple | Curvas de la ola 4 | 4 |
| D-23 | Modelo final | El de mayor AUC de validación. Si hay empate dentro de 1 error estándar, el de mayor recall en el presupuesto, comparable porque todos llaman al mismo 20 % | `modelo_elegido.json` | 5 |
| D-24 | Evaluación del test | Una sola vez: AUC con intervalo del 95 % por bootstrap sobre las predicciones de test, y recall llamando al 20 % de la lista de test con mayor puntaje | `evaluacion_test.json` | 5 |
| D-25 | Robustez temporal | Validación hacia adelante dentro de train con varios cortes (`TimeSeriesSplit` sobre train ordenado por fecha). Se mide con todas las variables y sin las que codifican la época: primero el bloque macro, después el bloque macro y `month` | `robustez_temporal.csv` | 5 |

### Consultas a la cátedra (ola 0; las envía el grupo)

Van en un solo mensaje, antes de la clase del 30/09. Cada una tiene un valor por defecto, así que
ninguna bloquea el trabajo.

| # | Pregunta | En la wiki | Si no responden |
|---|---|---|---|
| 1 | ¿$F_1$ cuenta entre «las métricas vistas en clase»? | 4-nonies | AUC y recall en el presupuesto |
| 2 | Con `duration`, ¿alcanza con excluirla y justificarlo, o esperan la comparación con y sin ella? ¿Las variables del último contacto (`month`, `day_of_week`, `contact`, `campaign`) se consideran disponibles antes de llamar? | 4-decies | Excluir `duration` y mostrar su techo, ya medido en la ola 1; el resto queda como en D-07 |
| 3 | ¿Esperan que se trate el desbalance, o alcanza con métricas acordes? | 4-undecies | Métricas acordes; `class_weight` como hiperparámetro opcional |
| 4 | ¿Qué Naive Bayes esperan para datos numéricos y categóricos a la vez? | 4-duodecies | Se comparan dos variantes y queda la mejor en CV |
| 5 | El archivo está ordenado por fecha: ¿partición aleatoria o temporal? | 4-terdecies | Aleatoria (D-03), con la validación temporal como limitación (D-25) |
| 6 | ¿Qué día defiende el grupo? ¿La clase del 30/09 es la de KNN? | 4-septies | Se planifica para el 07/10; KNN se apoya en Mitchell, cap. 8, y en Alpaydin, cap. 8 |

---

## 4. Arquitectura del código

Se siguen las convenciones del TP1:
- identificadores en español;
- cada módulo corre como `python -m src.X`;
- tests con `assert` y sin pytest, que corren con `python -m tests.test_X` y terminan en
  `TODOS LOS TESTS OK`;
- semilla 42.

```
src/
  datos.py                 hecho: carga, duplicados, partición; cargar_train() y cargar_test()
  validacion.py            hecho: folds()
  evidencia_particion.py   hecho: las cifras de D-02 y D-03
  eda.py, eda_html.py      EDA sobre train, en texto y en HTML
  estilo.py                ola 0: paleta y rcParams, tomados de tp1/src/graficos.py:21-25
  preproceso.py            ola 0: transformaciones propias y ColumnTransformer por modelo
  modelos.py               ola 0: un Pipeline por clasificador, con sus grillas
  metricas.py              ola 0: AUC, recall/precisión/F1 en el presupuesto, precisión promedio
  ablaciones.py            ola 1: línea de base, y con y sin cada decisión del EDA
  experimentos.py          ola 3: los cuatro modelos en CV
  curvas.py                ola 4: curvas de validación
  seleccion.py             ola 5: modelo final -> modelo_elegido.json, sin tocar el test
  robustez.py              ola 5: validación hacia adelante dentro de train
  evaluar_test.py          ola 5: el ÚNICO módulo que llama a cargar_test()
  numeros.py               ola 7: resultados -> informe/numeros.tex e informe/numeros.md
  graficos.py              figuras de análisis -> figuras/
  graficos_presentacion.py figuras de proyección -> figuras/presentacion/
  entregar.py              ola 8: arma los dos archivos de entregables/
tests/
  test_datos.py, test_validacion.py   hechos (falta el cierre TODOS LOS TESTS OK, paso 0.8)
  test_preproceso.py       sin fuga: cada transformación se ajusta sólo con el fold de entrenamiento
  test_metricas.py         recall en el presupuesto y AUC, contra casos de juguete calculados a mano
  test_modelos.py          prueba rápida: cada pipeline entrena y predice con 500 filas
  test_aislamiento.py      nadie fuera de la lista blanca llama a cargar_test(), cargar() ni RUTA_TEST
  test_paleta.py           portado del TP1: contraste y daltonismo
resultados/                versionado, en formato largo
informe/                   presentacion.tex, numeros.tex y numeros.md (generados), resultados-test.tex (generado), guion.md
figuras/, figuras/presentacion/
plan/                      PLAN.md, EXEC_STATE.md, SUGERENCIAS.md (no se entregan)
```

**Contrato de resultados.** Los CSV de validación cruzada van en formato largo: una fila por
`modelo, configuracion, fold, conjunto (train o validacion), metrica, valor`. Así una sola función
dibuja cualquier curva, y `numeros.py` saca cualquier cifra sin reprocesar. Se guardan siempre las
cinco métricas candidatas: AUC, precisión promedio, y recall, precisión y $F_1$ en el presupuesto.

**Sin fuga por construcción.** Todo el preprocesamiento vive **dentro** del `Pipeline` y se vuelve a
ajustar en cada fold; el enunciado pide ajustarlo «dentro del pipeline correspondiente» (TP2, p. 1).
El parámetro `cv=` recibe siempre `folds()` (D-06).

---

## 5. Runbook por olas

Cada paso tiene una prueba de terminado ejecutable. Un paso está hecho cuando pasa esa prueba, no
cuando alguien dice que lo terminó.

### Ola 0 — Arranque (sáb 26/09)

| Paso | Qué | Archivos | Terminado cuando |
|---|---|---|---|
| 0.1 | Reconciliación: comprobar que el estado real coincide con este plan (commits `0a9b311` y `108ff99`, D-01 a D-06, EDA en HTML) y abrir el registro de estado | `plan/EXEC_STATE.md`, `plan/SUGERENCIAS.md` | EXEC_STATE tiene una fila por paso de este runbook |
| 0.2 | Redactar el mensaje a la cátedra con las seis preguntas de §3 | `plan/consultas.md` | El grupo lo envía; es una acción del grupo |
| 0.3 | `estilo.py`, nuevo, con la paleta de `tp1/src/graficos.py:21-25`: azul `#2A78D6` = «no», naranja `#EB6834` = «yes», los mismos neutros. `test_paleta.py`, portado del TP1 | `src/estilo.py`, `tests/test_paleta.py` | `python -m tests.test_paleta` termina en TODOS LOS TESTS OK |
| 0.4 | `preproceso.py` y `modelos.py`: una fábrica de pipelines que recibe como parámetros las opciones de D-08 a D-16 | `src/preproceso.py`, `src/modelos.py`, `tests/test_preproceso.py`, `tests/test_modelos.py` | Los tests pasan, incluido el de no fuga |
| 0.5 | `metricas.py`: AUC, precisión promedio, y recall, precisión y $F_1$ llamando al q % con mayor puntaje de cada fold de validación | `src/metricas.py`, `tests/test_metricas.py` | Los casos de juguete coinciden con el cálculo a mano |
| 0.6 | Presupuesto de cómputo: medir un ajuste por modelo y en cada extremo de las grillas, incluidos los kernels lineal y polinómico de la SVM con C alto. `SVC(kernel="linear")` escala mal: medido sobre 8.000 filas, tarda 6,5 s con C = 1 y 62 s con C = 10. Si no entra, se usa `LinearSVC(loss="hinge")` y se declara | `resultados/costos.csv` | Cada grilla de la ola 4 tiene su costo estimado, y ninguna pasa de 45 minutos |
| 0.7 | Test de aislamiento. Lista blanca: `src/datos.py`, que define las funciones; `src/evidencia_particion.py`, que mide la partición; `tests/test_datos.py`, que la verifica; y `src/evaluar_test.py`. Fuera de esa lista, ninguna llamada a `cargar_test(`, `cargar(` ni `RUTA_TEST` | `tests/test_aislamiento.py` | Pasa sobre el repo actual y falla si se agrega una llamada prohibida en otro módulo |
| 0.8 | Cerrar `test_datos` y `test_validacion` con el mensaje TODOS LOS TESTS OK | `tests/` | Las dos suites lo imprimen |

La medición exploratoria del 25/09 (un fold de 26.352 filas, en una máquina de 11 núcleos) sirve
sólo para el presupuesto. La SVM con RBF y C = 1 tarda 16,5 s en entrenar y 2,8 s en predecir. RF,
con 300 árboles, tarda 0,7 s; KNN predice en 0,2 s, y NB tarda menos de 0,1 s.

### Ola 1 — EDA con consecuencias (punto 1)

El enunciado pide que las observaciones del EDA tengan «alguna consecuencia sobre el
preprocesamiento, la selección de variables o la interpretación de los resultados». También sugiere
«comparar el desempeño del modelo con y sin ciertas variables problemáticas» (TP2, p. 1). Esta ola lo
hace de forma sistemática: cada observación se convierte en una ablación medida.

| Paso | Qué | Archivos | Terminado cuando |
|---|---|---|---|
| 1.1 | EDA en HTML sobre train (en curso) y lectura con el grupo | `src/eda_html.py`, `resultados/eda/eda.html` | El grupo leyó la tabla «Qué hacer con esto» |
| 1.2 | Línea de base «sin modelo» (`DummyClassifier`) en los mismos folds: AUC 0,5; recall en el presupuesto igual a q (20 %); exactitud de decir siempre «no», 88,7 % | `src/ablaciones.py`, `resultados/linea_base.csv` | Existe antes de cualquier figura con métricas (guía C4) |
| 1.3 | D-07: escribir el momento en que opera el modelo y qué variables existen en ese momento. Es la base de todo el análisis de fuga, no sólo de `duration` | `DECISIONES.md` (lo escribe el orquestador) | Cada una de las 20 predictoras está clasificada como disponible o no antes de llamar |
| 1.4 | Ablaciones A1 a A12 (lista debajo de la tabla) | `src/ablaciones.py`, `resultados/ablaciones.csv` | Cada variante está medida en los modelos que le corresponden, con 5 folds y ΔAUC pareado por fold |
| 1.5 | Regla de adopción: se adopta una variante si sube el AUC medio más que el desvío de las diferencias pareadas, o si es neutra y simplifica. Se escriben D-08 a D-17 con su número y su consecuencia, también las nulas | `DECISIONES.md` | Cada decisión tiene su ΔAUC por modelo y una frase de consecuencia |
| 1.6 | Figura resumen: ΔAUC por ablación y por modelo, en puntos con barra de desvío | `src/graficos.py`, `figuras/` | Se entiende sin texto |

**Las ablaciones del paso 1.4.** La configuración de referencia es NB gaussiano, SVM con RBF y
C = 1, KNN con k = 15 y RF con 300 árboles. Cada variante cambia una sola cosa:

| Variante | Qué cambia | En qué modelos |
|---|---|---|
| A1 | Agrega `duration`, como techo | Todos |
| A2 | Cambia `pdays` por `previous > 0` | Todos |
| A3 | `default` como indicadora | Todos |
| A4 | Funde las categorías raras según D-11 | Todos |
| A5 | `log1p` sobre `campaign` | KNN y SVM |
| A6 | Reduce el bloque macro | Todos |
| A7 | **Sin** el bloque macro. No se adopta: alimenta el hallazgo | Todos |
| A8 | Sin el bloque macro y sin `month`, que también codifica la época: en train, el 100 % de las filas de septiembre, el 91 % de octubre y el 94 % de diciembre caen en el último 20 % por fecha | Todos |
| A9 | Sin `day_of_week` | Todos |
| A10 | Edad en tramos | NB |
| A11 | `unknown` imputado por la moda, como contraste de D-08 | Todos |
| A12 | Sin escalar | KNN |

Son unos 200 ajustes. Los de la SVM se llevan casi todo el tiempo: alrededor de 20 minutos.

### Ola 2 — Métricas, presupuesto de llamadas y desbalance (punto 2.3)

**D-19, las métricas: AUC y recall en el presupuesto de llamadas.** Las dos salen de la Clase 4, y
la justificación tiene tres partes:
- **AUC:** mide si el modelo ordena bien a los clientes, que es exactamente «priorizar recursos de
  marketing» (TP2, p. 1). No depende del umbral y no premia decir siempre «no» (Clase 4, slides
  45–51).
- **Recall en el presupuesto:** qué parte de los clientes que contratarían se alcanza con las
  llamadas que se pueden hacer. Cuenta el error que más cuesta, un cliente perdido. La cátedra dice
  que el umbral «depende del costo de cada error» (Clase 4, slides 41–44).
- **Exactitud:** sólo sirve para mostrar por qué no se usa. Decir siempre «no» ya acierta el 88,7 %.

**D-20, el umbral: un presupuesto fijo, no la regla de la Clase 4.** La Clase 4 propone el punto de
la ROC más cercano a (0,1) (slide 52). No se usa por dos razones:
- Esa regla es simétrica, y la justificación del recall es asimétrica. La wiki registra la misma
  tensión en la página de la Clase 4, §6.3.
- Con un umbral distinto por modelo, los recalls no se pueden comparar.

En cambio, llamar al 20 % de cada lista con mayor puntaje:
- deja a todos los modelos con el mismo número de llamadas;
- no requiere estimar un umbral, así que no hay nada que se ajuste antes de tiempo;
- y se aplica igual a la lista de test.

El 20 % es un supuesto y se declara como tal.

| Paso | Qué | Terminado cuando |
|---|---|---|
| 2.1 | Escribir D-19 con las citas de arriba. Si la cátedra acepta $F_1$ (consulta 1), se lee la columna que ya está guardada | D-19 cita los slides y dice qué métrica decide qué |
| 2.2 | Escribir D-20, con la sensibilidad a q = 10 % y 30 % en la reserva de preguntas | Tabla de sensibilidad en `resultados/` |
| 2.3 | D-21, el desbalance: por defecto no se trata, y `class_weight="balanced"` se mide en RF y SVM | Queda registrada con su número, aunque sea nula |

### Ola 3 — Los cuatro clasificadores (puntos 2.1 y 2.2)

| Paso | Qué | Archivos | Terminado cuando |
|---|---|---|---|
| 3.1 | Pipelines definitivos, con las decisiones de la ola 1 | `src/modelos.py` | `test_modelos` pasa |
| 3.2 | D-18: GaussianNB contra CategoricalNB, con las numéricas discretizadas por cuantiles dentro del pipeline y Laplace α = 1, como en la Clase 6, slide 46 | `resultados/cv_modelos.csv`, `DECISIONES.md` | La variante elegida tiene su ΔAUC |
| 3.3 | Validación cruzada de los cuatro modelos con su configuración por defecto: las cinco métricas, en train y en validación, por fold | `src/experimentos.py`, `resultados/cv_modelos.csv` | 4 modelos × 5 folds × 2 conjuntos |
| 3.4 | Figura: AUC de validación de los cuatro contra la línea de base, en puntos con ±1 desvío y el valor anotado | `figuras/` | Se entiende sin texto |

### Ola 4 — Hiperparámetros (punto 3.1)

El protocolo es el de la Clase 7, slide 50: medir en train y en validación con K-fold, leer la
brecha entre las dos y graficar la curva.

| Paso | Qué | Grilla | Terminado cuando |
|---|---|---|---|
| 4.1 | RF: `max_depth`, que es la curva de sobreajuste para discutir en la presentación; y `n_estimators`, que llega a una meseta porque no es un eje de complejidad (Clase 7, slide 61) | Profundidad ∈ {2, 4, 6, 8, 10, 12, 15, 20, 25, sin límite}; árboles ∈ {10, 25, 50, 100, 200, 400, 800} | `resultados/curvas/rf_*.csv`, con train y validación |
| 4.2 | KNN: `n_neighbors`, con `weights` uniforme y por distancia (Mitchell §8.2.1). Aquí sobreajusta un k pequeño, al revés que en los otros ejes | k ∈ {1, 3, 5, 9, 15, 25, 41, 61, 101, 151, 201} | `resultados/curvas/knn_*.csv` |
| 4.3 | SVM: `C` con kernel RBF, y comparación de kernels (lineal, polinómico, RBF) en el mejor C, con el costo medido en el paso 0.6. Se usa la convención de scikit-learn: un C grande regulariza menos (Clase 8, slide 49; C-26) | C ∈ {0,001; 0,01; 0,1; 0,3; 1; 3; 10; 30; 100}, recortada según el paso 0.6 | `resultados/curvas/svm_*.csv` |
| 4.4 | D-22: elegir cada valor y escribir la discusión de sobreajuste y subajuste sobre la curva de RF, y sobre la de KNN si hay tiempo | `resultados/hiperparametros.json`, `DECISIONES.md` | Cada valor elegido tiene su número de validación y su brecha |
| 4.5 | Figuras de las tres curvas, con revelado en tres pasos como la curva en U del TP1: train, después validación, después bandas y zonas | `figuras/`, `figuras/presentacion/` | Se leen al 25 % |
| 4.6 | *(Opcional)* Grilla conjunta de C y γ para RBF (Clase 8, slides 48–49), si el costo lo permite | — | — |

Si un valor de la grilla pasa el presupuesto de cómputo, se saca y se declara: no hay recortes en
silencio.

### Ola 5 — Modelo final y la única evaluación del test (punto 4)

**Esta ola arranca después de la clase del 30/09**, con las respuestas de la cátedra ya recibidas.
Evaluar el test y después cambiar la métrica obligaría a evaluarlo de nuevo. El TP1 lo hizo tres
veces, y cada reevaluación le resta independencia al test (decisiones D-26 y D-29 del TP1).

| Paso | Qué | Archivos | Terminado cuando |
|---|---|---|---|
| 5.1 | Comparar los cuatro modelos ya ajustados, en los mismos folds y contra la línea de base. D-23 | `src/seleccion.py`, `resultados/modelo_elegido.json` | El JSON tiene el modelo, los hiperparámetros y las cinco métricas medias ± desvío |
| 5.2 | D-25: validación hacia adelante dentro de train, con `TimeSeriesSplit(n_splits=5)` sobre train ordenado por fecha. Tres variantes: todas las variables, sin el bloque macro, y sin el bloque macro ni `month`. Se compara con la validación cruzada barajada | `src/robustez.py`, `resultados/robustez_temporal.csv` | Se hizo sin tocar el test |
| 5.3 | Evaluación única (detalle debajo de la tabla) | `src/evaluar_test.py`, `resultados/evaluacion_test.json`, `informe/resultados-test.tex` | Hay un commit inmediato. Una segunda corrida exige escribir `si` y queda registrada en DECISIONES |
| 5.4 | Análisis de errores sobre la matriz de test. Un falso negativo es un cliente que habría contratado y no se llama; un falso positivo, una llamada de más | `resultados/conclusiones.md` | — |

**Qué hace la evaluación única (paso 5.3):**
- Reentrena con todo train (Clase 2, slides 88–89).
- Predice el test **una vez**.
- Calcula el AUC con su intervalo del 95 % por bootstrap sobre esas predicciones.
- Calcula el recall llamando al 20 % de la lista de test con mayor puntaje, y la matriz de confusión
  de ese corte.

### Ola 6 — Conclusiones y hallazgo (punto 5)

El borrador de los pasos 6.1 y 6.3 se escribe el 29/09 sin números de test, y se cierra el 01/10.

| Paso | Qué | Terminado cuando |
|---|---|---|
| 6.1 | Tabla de por qué cada modelo rindió lo que rindió, con los cuatro ejemplos que da el enunciado: variables muy correlacionadas, distribuciones no gaussianas, diferencias de escala y relaciones no lineales (TP2, p. 2). Lista debajo de la tabla | Cada afirmación tiene un número de la ola 1, 3 o 4 |
| 6.2 | Elegir el hallazgo con evidencia (candidatos debajo de la tabla) | El hallazgo elegido tiene una figura que se entiende sola |
| 6.3 | Limitaciones y mejoras (lista debajo de la tabla) | Las dos listas son cortas y están ordenadas por impacto |

**Qué cruza cada fila de la tabla del paso 6.1:**
- NB, contra la independencia condicional y la normalidad. Tres de las cinco variables macro tienen
  entre sí r de 0,91 a 0,97.
- KNN, contra las escalas distintas y las 62 columnas que deja el one-hot.
- SVM, contra la escala y el costo de cómputo.
- RF, que no depende de nada de eso y capta relaciones no lineales, como la U de la edad, y también
  interacciones como mes × macro.

**Candidatos a hallazgo (paso 6.2):**
- **H1, «el modelo aprende la época»**, es el candidato por defecto. Se muestra con la validación
  hacia adelante del paso 5.2 y la figura del tiempo del EDA.
- **H2:** el techo de `duration` contra el modelo realista, ya medido en A1.
- **H3:** cuántos «yes» se alcanzan según qué parte de la lista se llama. Es la curva de ganancia,
  calculada fuera de fold en la ola 3.

**Limitaciones y mejoras (paso 6.3).**

Limitaciones:
- la época: el test es de la misma época que train;
- los hiperparámetros elegidos sobre la misma validación cruzada que los reporta;
- el presupuesto del 20 % como supuesto;
- el desbalance sin tratar;
- los `unknown`;
- `duration` fuera del modelo;
- que `campaign` dependa del resultado.

Mejoras:
- validación temporal como esquema principal;
- costo explícito por tipo de error;
- calibración de probabilidades;
- más historial del cliente.

### Ola 7 — Presentación y guion

El deck arranca el 29/09 con los números de test en modo marcador, como en el TP1, que ya tenía ese
interruptor en `resultados-test.tex`. Se completa cuando existe `evaluacion_test.json`.

| Paso | Qué | Terminado cuando |
|---|---|---|
| 7.1 | Deck sobre el preámbulo del TP1, sin el código muerto: Fira Sans, la paleta, el índice de secciones en la cabecera, la barra de progreso, `n/N` al pie, un `\note` por frame, `\heroe` y `\figgrande` | Compila sin warnings, y el `.log` se lee antes de borrar los auxiliares |
| 7.2 | `numeros.py`: todos los números del deck como macros generados (`informe/numeros.tex`), y la misma lista legible para el guion (`informe/numeros.md`) | Un script comprueba que el cuerpo de las slides no tiene dígitos sueltos, salvo una lista permitida, y que cada número del guion figura en `numeros.md` |
| 7.3 | Figuras de proyección en 16:9, sin título, con la leyenda arriba y con revelados | Ninguna es la misma imagen que una figura de análisis |
| 7.4 | Guion con las convenciones del TP1 (lista debajo de la tabla) | Palabras / 2,7 ≤ 9:45 |
| 7.5 | `slides-y-guion.pdf`, generado desde el fuente | Se regenera con un solo comando |
| 7.6 | README de entrega, README del repo y GLOSARIO | El README de entrega tiene el mapa enunciado → módulo → slide |
| 7.7 | *(Opcional)* Informe de respaldo, sólo si sobra un día | — |

**Las convenciones del guion (paso 7.4):**
- una tabla de cobertura;
- un encabezado por slide, con su ventana de tiempo y su cantidad de palabras;
- las marcas `[→]`, a 2,7 palabras por segundo;
- puntos de control y un orden de recorte;
- los bloques «A aclarar» y quién habla en cada slide;
- un banco de al menos 20 preguntas, que incluya la teoría de los cuatro modelos citada por slide.

**Esqueleto propuesto:** 20 frames que suman 9:15 y dejan 30 s de margen sobre el tope de 9:45. Las
X se completan con los números reales.

| # | Sección | Título provisional (afirma o pregunta) | Punto del enunciado | Tiempo |
|---|---|---|---|---|
| 1 | — | Portada | — | 0:05 |
| 2 | El problema | Predecir quién contrata un plazo fijo, antes de llamar. Sin modelo, decir siempre «no» acierta el 88,7 % | 0 | 0:35 |
| 3 | Qué datos deciden | Train decide, test sólo mide | 1, 2.2, consejo de la p. 3 | 0:35 |
| 4 | EDA | La duración de la llamada predice el resultado, pero no existe antes de llamar | 1 (fuga) | 0:35 |
| 5 | EDA | El archivo está ordenado por fecha, y la tasa de «yes» se multiplica por X | 1 | 0:30 |
| 6 | EDA | 999 no significa «nunca contactado» | 1 | 0:25 |
| 7 | EDA | Dentro de cada fold, en este orden: el pipeline, con la consecuencia de cada decisión | 1, 2.2 | 0:35 |
| 8 | Métricas | Dos métricas que no premian decir siempre «no» | 2.3 | 0:35 |
| 9 | Modelos | Sin ajustar, X supera a la línea de base por Y | 2.1 | 0:30 |
| 10 | Hiperparámetros | La curva de RF: desde qué profundidad sobreajusta | 3.1 | 0:40 |
| 11 | Hiperparámetros | La curva de KNN | 3.1 | 0:25 |
| 12 | Hiperparámetros | La curva de la SVM | 3.1 | 0:25 |
| 13 | Modelo final | Después de ajustar, X gana por Y en validación | 4 | 0:25 |
| 14 | Modelo final | ¿Qué esperar en clientes nuevos de la misma época? El número de test, solo | 4 | 0:25 |
| 15 | Modelo final | Qué errores comete: la matriz de confusión de test, llamando al 20 % | 5 | 0:30 |
| 16 | Conclusiones | Por qué ganó X y perdió Y | 5 | 0:40 |
| 17 | Conclusiones | Limitaciones y qué mejoraríamos | 5 | 0:30 |
| 18–19 | Hallazgo | El modelo aprende la época (H1, en dos pasos) | 5 | 0:50 |
| 20 | — | ¿Preguntas? | — | — |

Todo lo calificable, limitaciones incluidas, va antes del hallazgo (guía A4), y el hallazgo es lo
último que se muestra (A6). La slide 14 dice «de la misma época» porque el hallazgo muestra a
continuación qué pasa hacia adelante en el tiempo. Si en el punto de control de la slide 13 el reloj
va atrasado, se recorta el segundo paso del hallazgo, nunca las limitaciones.

### Ola 8 — Auditoría, entregables, gates y ensayos

| Paso | Qué | Terminado cuando |
|---|---|---|
| 8.1 | Auditoría adversarial en cuatro lentes, cada una con su corrección (lista debajo de la tabla) | Los hallazgos altos están corregidos y el resto quedó en SUGERENCIAS |
| 8.2 | `python -m src.entregar` arma el zip y el PDF. Prueba de clon limpio: descomprimir en una carpeta temporal, instalar `requirements.txt` y correr los tests y el pipeline rápido | Pasa desde cero |
| 8.3 | Los gates de §6, sobre los entregables del paso 8.2 | Todos en verde |
| 8.4 | Dos ensayos cronometrados del grupo; con ellos se ajusta el orden de recorte. Si algo cambia, se vuelven a correr 8.2 y 8.3 | Los dos duran ≤ 10:00 |
| 8.5 | Envío; es una acción del grupo | Hecho antes del martes 06/10, 24 h antes de la clase |

**Las cuatro lentes de la auditoría (paso 8.1):**
1. la guía A–E, slide por slide;
2. metodología y fuga, en el código;
3. un recálculo independiente de cada número proyectado;
4. un «tribunal» que hace 20 preguntas y las contrasta con el banco del guion.

---

## 6. Gates antes de enviar

| Gate | Prueba |
|---|---|
| G1 Tests | Todas las suites terminan en TODOS LOS TESTS OK |
| G2 Test aislado | `test_aislamiento` pasa, y `evaluacion_test.json` registra una sola corrida |
| G3 Números | Todo número del deck sale de `numeros.tex` o de `resultados-test.tex`; todo número del guion figura en `numeros.md` |
| G4 Títulos | Ningún título de frame es un rótulo |
| G5 Cobertura | Cada punto del enunciado tiene su slide, antes del hallazgo |
| G6 Reloj | Palabras / 2,7 ≤ 9:45, y los dos ensayos ≤ 10:00 |
| G7 Legibilidad | Con el PDF al 25 %, el rótulo más pequeño se lee |
| G8 Paleta | `test_paleta` pasa |
| G9 Clon limpio | Pasa el paso 8.2 |
| G10 Idioma | Sin voseo en el deck, el guion ni los README (`grep`) |
| G11 Línea de base | Toda slide con métricas de validación muestra «sin modelo» con el mismo rótulo; la slide de test no la muestra |
| G12 Consecuencias | Cada sección del EDA termina en una frase proyectada que nombra un efecto |

## 7. Calendario

| Fecha | Qué | Quién |
|---|---|---|
| vie 25/09 | Plan y EDA en HTML | Claude |
| sáb 26/09 | Ola 0 y envío de las consultas; arranca la ola 1 | Claude; el grupo envía el mensaje |
| dom 27/09 | Cierre de la ola 1; olas 2 y 3 | Claude; el grupo revisa D-07 a D-21 |
| lun 28/09 | Ola 4, con las curvas de la SVM en segundo plano | Claude |
| mar 29/09 | Cierre de la ola 4. Borrador de la ola 6 sin test; esqueleto del deck con marcadores | Claude y el grupo |
| mié 30/09 | Clase (¿KNN?) y respuestas de la cátedra. Los ajustes sólo cambian qué columna se lee | El grupo |
| jue 01/10 | Ola 5, con la evaluación única del test; cierre de la ola 6 | Claude y el grupo |
| vie 02/10 | Ola 7: deck, figuras de proyección, números y guion | Claude y el grupo |
| sáb 03/10 | Cierre de la ola 7 (cuadernillo y README); auditoría y correcciones (8.1) | Claude |
| dom 04/10 | Entregables y clon limpio (8.2); gates (8.3); primer ensayo | Claude y el grupo |
| lun 05/10 | Segundo ensayo, ajustes y reconstrucción de los entregables si algo cambió | El grupo |
| mar 06/10 | Margen hasta la hora de envío; envío, 24 h antes de la clase (confirmar la hora) | El grupo |
| mié 07/10 | Defensa, o el 14/10 | El grupo |

El margen real es la mañana del 06/10 y lo que sobre del 05/10. Si al grupo le toca el 14/10, se
suma una semana de ensayos y revisión, sin alcance nuevo.

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| La cátedra pide $F_1$ u otra métrica | Las cinco métricas candidatas se guardan desde la ola 1, y el test se evalúa sólo a partir del 01/10 |
| La clase del 30/09 no es la de KNN, o dice algo distinto | KNN se arma con Mitchell y Alpaydin, y su slide se cierra después del 30/09 |
| La SVM con kernel lineal o C alto no termina | El costo se mide en el paso 0.6; se usa `LinearSVC` o se recorta la grilla, y se declara |
| H1 no se sostiene porque el efecto temporal es pequeño | Se pasa a H2 o H3, ya medidas en las olas 1 y 3 |
| El deck pasa de 10 minutos | El esqueleto suma 9:15, el orden de recorte está escrito en el guion, y todo agregado se paga con un recorte (guía E3) |
| Un número del deck o del guion no coincide con el código | Macros generados, `numeros.md`, y recálculo independiente en el paso 8.1 |
| El grupo no llega a ensayar | Los ensayos se reparten entre el 04/10 y el 05/10 |

## 9. Cómo ejecutarlo con /workforce

- **Modelos:** sesión en Opus, y workers, verificadores y auditores también en Opus. El rigor se
  gradúa con `effort`.
- **Archivos reservados al orquestador:** `DECISIONES.md`, `README.md`, `plan/EXEC_STATE.md`,
  `resultados/modelo_elegido.json` e `informe/numeros.tex`. Los workers devuelven el texto de su
  decisión como dato.
- **Olas 1, 3 y 4:** un worker por modelo, dueño exclusivo de su módulo o de su CSV.
- **Por unidad:** construir → verificar → corregir, sin una segunda verificación; el control final
  son las pruebas de terminado.
- **Ola 7:** un redactor para el deck y otro para el guion, en secuencia (el guion depende del deck),
  y un crítico.
- **Ola 8:** cuatro auditores (paso 8.1) y un corrector.
- **Commits:** uno por paso, sin trailers de coautoría. Nada se sube a GitHub sin que se pida.

**Fuera de alcance:**
- actualizar la wiki con el TP2, que se hace después junto con los avisos pendientes de `raw/TP2/`;
- modelos que no sean los cuatro del enunciado, incluido el *gradient boosting*.

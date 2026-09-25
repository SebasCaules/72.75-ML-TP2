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

| **D-07** | **El modelo opera antes de cada llamada a un cliente en la campaña actual.** En ese momento se conocen sus datos personales y crediticios (`bank-additional-names.txt`, en adelante names.txt, atributos 1 a 7), su historial de campañas previas (atributos 13 a 15), el contexto económico del período (atributos 16 a 20), el canal y la fecha, mes y día de la semana, de la llamada que se va a hacer (atributos 8 a 10) y qué número de llamada es en esta campaña, `campaign` (atributo 12). No se conoce la duración (atributo 11; D-05). De las 20 predictoras, 19 están disponibles, `campaign` con una reserva, y `duration` no (tabla siguiente). `contact`, `month`, `day_of_week` y `campaign` quedan sujetas a la consulta 2: si la cátedra no responde, se usan como se indica aquí; si responde que no están disponibles, salen del modelo y D-07 se revisa. D-07 decide sólo qué existe antes de llamar; qué variables entran por desempeño lo deciden D-08 a D-17 | El enunciado llama fuga de datos (*data leakage*) a la información que «no debería estar disponible antes de realizar la llamada», y dice que ese es el momento en que operaría el modelo; como insumo admite información de antes o de durante la campaña (TP2, p. 1). El criterio es, entonces, qué existe antes de marcar, no en qué bloque lo pone el diccionario: names.txt agrupa `contact`, `month`, `day_of_week` y `duration` como datos del último contacto (atributos 8 a 11), pero sólo `duration` se produce en la llamada; el canal y la fecha se eligen antes (inferencia). `campaign` «includes last contact» (atributo 12): antes de cada llamada ya se sabe qué número de llamada es. **Limitación:** cada fila guarda el último contacto del cliente en la campaña, y tras un «yes» no se vuelve a llamar (inferencia), así que el valor final de `campaign` depende de `y`. En train, su media es 2,053 en los «yes» y 2,627 en los «no», con la misma mediana, 2 (`resultados/eda/reporte.txt`, §4): es compatible con esa inferencia, aunque no la prueba, porque también es compatible con que los clientes menos interesados requieran más llamadas. Lo mismo vale, en menor grado, para el canal y la fecha, que también son los de la última llamada (inferencia) |

**Las 20 predictoras antes de llamar (D-07).** Es la prueba de terminado del paso 1.3 del plan.

| Variable | Disponible antes de llamar | Motivo |
|---|---|---|
| `age` | sí | Dato del cliente (names.txt, atributo 1) |
| `job` | sí | Dato del cliente (names.txt, atributo 2) |
| `marital` | sí | Dato del cliente (names.txt, atributo 3) |
| `education` | sí | Dato del cliente (names.txt, atributo 4) |
| `default` | sí | Dato crediticio que el banco ya tiene (names.txt, atributo 5). El «unknown» del 20,82 % de train (reporte.txt, §2) también se conoce antes de llamar: es falta de dato, no fuga (D-08, D-09) |
| `housing` | sí | Dato crediticio del cliente (names.txt, atributo 6) |
| `loan` | sí | Dato crediticio del cliente (names.txt, atributo 7) |
| `contact` | sí | Canal de la última llamada, celular o fijo (names.txt, atributo 8). Que se elija antes de marcar es inferencia, y por eso queda sujeta a la consulta 2 |
| `month` | sí | Mes de la última llamada (names.txt, atributo 9). Que se conozca antes de marcar es inferencia; sujeta a la consulta 2. Si además delata la época, es un problema de robustez temporal (D-25), no de fuga |
| `day_of_week` | sí | Día de la semana de la última llamada (names.txt, atributo 10). Misma inferencia; sujeta a la consulta 2 |
| `duration` | **no** | Se conoce sólo cuando la llamada termina, y entonces `y` ya se sabe (names.txt, atributo 11). En train, las 4 filas con `duration` = 0 son «no», y la tasa de «yes» va de 0 % en el decil más corto (≤ 59 s) a 46,05 % en el más largo (reporte.txt, §8). Fuera del modelo (D-05) |
| `campaign` | sí (con reserva) | Antes de cada llamada ya se sabe qué número de llamada es: cuenta las llamadas de esta campaña, incluida la última (names.txt, atributo 12). La reserva es un sesgo de selección: cada fila guarda el total al cierre de la campaña, y tras un «yes» no se vuelve a llamar (inferencia), así que su valor final depende de `y`. Se usa con esa limitación; sujeta a la consulta 2 |
| `pdays` | sí | Días desde el último contacto de una campaña anterior (names.txt, atributo 13). El 999 (96,34 % de train) es un centinela, y 3 302 filas con 999 tienen `previous` ≥ 1 (reporte.txt, §2): es un problema de codificación, no de momento (D-10) |
| `previous` | sí | Contactos anteriores a esta campaña (names.txt, atributo 14) |
| `poutcome` | sí | Resultado de la campaña anterior, no de la actual (names.txt, atributo 15). Es coherente con `previous`: ninguna fila tiene `previous` = 0 y `poutcome` distinto de «nonexistent» (reporte.txt, §2) |
| `emp.var.rate` | sí | Tasa de variación del empleo, indicador trimestral publicado por el Banco de Portugal (names.txt, atributo 16 y §4). El enunciado incluye el contexto económico entre los datos de entrada (TP2, p. 1). names.txt no dice con qué rezago se asignó el valor a cada fila; se supone el último publicado al llamar (inferencia). Como `month`, delata la época: es un problema de robustez temporal (D-03, D-25), no de fuga |
| `cons.price.idx` | sí | Índice de precios al consumidor, indicador mensual (names.txt, atributo 17). Mismas dos observaciones que `emp.var.rate` |
| `cons.conf.idx` | sí | Índice de confianza del consumidor, indicador mensual (names.txt, atributo 18). Mismas dos observaciones |
| `euribor3m` | sí | Tasa Euribor a 3 meses, indicador diario (names.txt, atributo 19). Mismas dos observaciones |
| `nr.employed` | sí | Número de empleados, indicador trimestral (names.txt, atributo 20). Mismas dos observaciones |

> **Para el guion de la defensa.** Antes de cada llamada, el modelo ordena a quién llamar a
> continuación, así que sólo puede usar lo que el banco sabe antes de marcar: los datos del
> cliente, cómo respondió en campañas anteriores, la situación económica y los datos de la llamada
> que va a hacer (por qué medio, en qué mes y qué día de la semana, y qué número de llamada es en
> esta campaña). Lo que no conoce es la duración, y cuando la conoce ya sabe si el cliente aceptó:
> por eso sale `duration`. La reserva es `campaign`: los datos guardan sólo la última llamada y, es
> de suponer, a quien acepta ya no se lo llama, así que ese número depende en parte del resultado.

## 3. Validación cruzada

| # | Decisión | Por qué |
|---|---|---|
| **D-06** | La validación cruzada usa **`folds()`** (`src/validacion.py`): `StratifiedKFold(5, shuffle=True, random_state=42)`. **Nunca `cv=5` a secas** | `cv=5` en `cross_val_score`, `GridSearchCV` o `validation_curve` equivale a `StratifiedKFold(5)` **sin barajar**. Como train está ordenado por fecha, cada fold quedaría en una época: la misma proporción de «yes» en todos, pero el euribor medio iría de 4,87 en el primer fold a 1,10 en el último. Con `folds()`, los cinco quedan entre 3,58 y 3,64. `k = 5` es provisional, igual que en el TP1: se revisa en el punto 2.2 |

## 4. Métricas, presupuesto de llamadas y desbalance (punto 2.3)

| # | Decisión | Por qué |
|---|---|---|
| **D-19** | **Dos métricas: el AUC para comparar y elegir modelos, y el recall en el presupuesto de llamadas (D-20) para medir el uso real.** Las dos salen de la Clase 4. Se calculan siempre, además, la precisión promedio y la precisión y $F_1$ en el presupuesto (`src/metricas.py`), por si la cátedra pide otra (consulta 1): cambiar de métrica cambia qué columna se lee, no qué se vuelve a correr. La exactitud sólo aparece para mostrar por qué no se usa | **AUC:** mide si el modelo ordena bien a los clientes, que es exactamente «priorizar recursos de marketing» (TP2, p. 1); no depende del umbral y no premia decir siempre «no» (Clase 4, slides 45–51). **Recall en el presupuesto:** qué parte de los clientes que contratarían se alcanza con las llamadas que se pueden hacer; cuenta el error que más cuesta, un cliente perdido. La cátedra dice que el umbral «depende del costo de cada error» (Clase 4, slides 41–44), y que tres modelos con la misma exactitud cometen errores distintos (slides 26–27); precisión y recall, slides 30–31. **Exactitud:** decir siempre «no» ya acierta el 88,73 % de train (`resultados/linea_base.csv`) |
| **D-20** | **El umbral es un presupuesto fijo: se llama al 20 % de cada lista con mayor puntaje**, no el punto de la ROC más cercano a (0,1) de la Clase 4, slide 52. El 20 % es un supuesto de negocio y se declara como tal; la sensibilidad a 10 % y 30 % queda en `resultados/sensibilidad_q_*.csv` como reserva para las preguntas (se mide después del paso 3.3, N0-10). Los empates en el corte se reparten en proporción (`src/metricas.py`, `en_presupuesto`) | La regla del slide 52 es simétrica entre los dos errores, y la justificación del recall es asimétrica; la wiki registra la misma tensión (Clase 4, §6.3). Con un umbral distinto por modelo, los recalls no se pueden comparar; con el presupuesto, todos los modelos hacen el mismo número de llamadas, no hay nada que estimar antes de tiempo, y la regla se aplica igual a la lista de test (D-24). Sin modelo, llamar al 20 % al azar alcanza el 20 % de los «yes»: es la línea de base del recall |
| **D-21** | **El desbalance (11,27 % de «yes») no se trata: ni remuestreo ni pesos por defecto.** `class_weight="balanced"` se mide como hiperparámetro opcional en RF y en la SVM en la ola 4 (`pesos_clase`), y se adopta sólo si mejora el AUC de validación | Las métricas de D-19 ya no premian decir siempre «no», y el presupuesto de D-20 no depende del umbral, que es lo que el desbalance distorsiona. Ninguna clase dio tratamiento del desbalance (consulta 3), así que remuestrear sería agregar un método sin fuente de cátedra. Medir los pesos como hiperparámetro deja la decisión en los datos y no en el supuesto |

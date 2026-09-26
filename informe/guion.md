# Guion de la defensa — TP2 · Grupo 7 · 07/10/2026 (o 14/10) · 10 + 10 minutos

Guion hablado de la presentación (`presentacion.pdf`: 20 slides y 4 de respaldo), slide por slide,
con reloj y marcas de avance. Sigue las notas de orador (`\note{}`) de `presentacion.tex`, con ajustes
de oralidad, y está recortado a lo que efectivamente se dice en 10 minutos. Grupo 7: Andrés Cortese
(64612) y Sebastián Caules (64331). Hora y aula: a confirmar.

**El test todavía no se evaluó (N0-1).** Sus números valen «?» en `informe/resultados-test.tex` y en
la sección «Números de test» de `informe/numeros.md`, y se completan solos después del 30/09 con
`python3 -m src.evaluar_test` y `python3 -m src.numeros`. Las slides 14 y 15 tienen aquí sus dos
variantes, como en el deck: la de la defensa, con el test evaluado, y la que se ensaya mientras siga
cerrado.

## El criterio de reparto: la teoría se nombra, los resultados se defienden

Lo que ya se dictó en la teórica —qué hacen train, validación y test, qué es una validación cruzada,
qué es fuga de datos, qué miden el AUC y el recall, qué es sobreajuste— **se nombra en una pasada y
no se desarrolla**. El tribunal lo sabe: repetirlo no suma nota y gasta el único recurso escaso, que
son diez minutos. Lo que sí hay que defender es lo que este trabajo hizo con eso: qué decidió el
análisis exploratorio y con qué número, por qué ganó Random Forest, qué esperamos del test y qué no
mide.

La regla práctica al ensayar: **si una frase se entendería igual en cualquier TP de la materia, es
teoría y va rápido; si sólo tiene sentido con nuestros datos adelante, es resultado y se dice
completa.**

| Bloque | Slides | Esqueleto (plan, §5) | Guion | |
| --- | --- | --- | --- | --- |
| Marco (portada, problema, cierre) | 1, 2, 20 | 0:40 | 0:40 | 0:00 |
| Teoría que se nombra (qué datos deciden, métricas) | 3, 8 | 1:10 | 1:05 | −0:05 |
| **EDA con consecuencias** | 4–7 | **2:05** | **1:55** | −0:10 |
| **Resultados: modelos, curvas, final, errores, conclusiones** | 9–17 | **4:30** | **4:25** | −0:05 |
| **Hallazgo** | 18–19 | **0:50** | **0:55** | +0:05 |
| **Total** | 1–20 | **9:15** | **9:00** | −0:15 |

El guion dura 9:00, 15 segundos menos que el esqueleto. Los cinco segundos que salen de la teoría
—el AUC de la slide 8 se nombra en una frase— van al hallazgo, que es lo último que se ve. Los otros
15 salen de no decir en voz alta cifras que la pantalla no muestra (guía B3): los deciles de
`duration` en la slide 4, las brechas de la 9, la mejora de los pesos en la 12 y los «yes» del
primer bloque en la 19 pasaron a las aclaraciones. Todo lo que se agregó respecto de las notas del
deck se pagó con un recorte: el propósito del IQR en la slide 7 (la crítica del TP1) y las
declaraciones de conjunto salieron de no leer en voz alta números que ya están anotados en
pantalla, y que en la slide 19 el 0,426 invierte el orden, de los «yes» del primer bloque.

## Reglas de la cátedra que este guion respeta (TP2, pp. 1 y 3)

- **10 minutos de presentación y 10 de preguntas** (TP2, p. 1), y el consejo lo repite: «10
  minutos, no más» (p. 3). El tope de este guion es 9:45 hablado; el resto es para los cambios de
  orador y las pulsaciones.
- **Presentación y código se mandan 24 horas antes** de la clase (p. 1): el martes 06/10, o el 13/10
  si al grupo le toca el 14/10.
- **Qué datos se usan en cada etapa** (consejo «Importante», p. 3). La slide 3 lo proyecta antes del
  primer número (etapa → datos → qué decide), y en este guion cada número hablado dice de qué
  conjunto sale.
- **Gráficas mejor que tablas, con el valor anotado** (p. 3). El deck no tiene tablas de números salvo
  la matriz de confusión, que es un dibujo. El guion no lee tablas: señala el valor anotado.
- **Los números en las slides** (p. 3). Todo número de resultado que se dice está en pantalla,
  generado por código: anotado sobre la figura o, en las curvas de las slides 10 y 11, como el punto
  que se señala; y cada diferencia que se dice es entre dos valores que se ven. La única excepción es
  la comparación del test con validación de la slide 14, que la guía C6 manda decir y no proyectar.
  Lo hablado sólo agrega números de apoyo (descriptivos de train y recuentos), y todos están en
  `informe/numeros.md`. Los resultados que la pantalla no muestra —las brechas de la slide 9, la
  mejora de los pesos en la 12— no se dicen: quedan en las aclaraciones.
- Lo que el enunciado pide decir en la presentación: **sobreajuste y subajuste en una curva** (punto
  3.1, p. 2: slide 10), y los métodos **no se describen**, pero los resultados se discuten sabiendo
  cómo funcionan (punto 2.1, p. 2: slide 16 y el banco de preguntas).

## Convenciones

- `[→]` = avanzar un overlay (una pulsación). La cantidad de `[→]` de cada slide coincide con los
  overlays del PDF (25 en total, comprobado contra sus 49 páginas); el último estado queda en
  pantalla mientras se termina de hablar.
- Ritmo de referencia: **2,7 palabras por segundo** (unas 162 por minuto), el del guion del TP1: sus
  22 encabezados declaran 1 632 palabras para un reloj de 10:00, es decir, 2,72 por segundo
  (contadas con la regla de «La suma», al final, son 1 606: 2,68 por segundo). El presupuesto de
  cada slide sale de dividir sus palabras por 2,7, y la ventana de cada encabezado es ese cálculo
  acumulado, redondeado a 5 segundos.
- **Lo que va entre `>` se dice; lo que está en `#### A aclarar` no.** Cada slide cierra con
  aclaraciones y preguntas probables sobre *esa* slide: es material para los 10 minutos de preguntas,
  no texto hablado. Es la reserva «Si preguntan» de las notas del deck, ampliada.
- **Quién habla**, en la línea que sigue a cada encabezado. Se alterna por bloques y el cambio de
  orador no se anuncia: quien entra empieza con la frase que engancha su bloque.
- **Cada número hablado figura, tal cual, en `informe/numeros.md`** (gate G3): coma decimal y espacio
  común de miles, como esa tabla, para que se pueda buscar. Un Δ se dice sin su signo cuando el verbo
  ya lo dice («le suma 0,013», «pierden 0,023»). **Y declara su conjunto** (guía C5):
  train, validación (en los 5 folds o fuera de fold), hacia adelante dentro de train, o test
  (pendiente). Los números de las aclaraciones y de las preguntas siguen la misma regla.
- **El número de test no se compara con «sin modelo» ni con otro modelo.** La única comparación
  permitida es contra validación, hablada y nunca proyectada, como chequeo de coherencia (guía C6);
  mientras el test siga cerrado, esa comparación vale «?».
- **Cada afirmación teórica cita la teórica** con clase y slide, y sólo con las citas que ya están en
  `DECISIONES.md` o en `plan/PLAN.md`: Clase 2, slides 88–89; Clase 4, slides 26–27, 30–31, 41–44,
  45–51 y 52; Clase 6, slide 46; Clase 7, slides 50 y 61; Clase 8, slides 48–49 y 54; Mitchell
  §8.2.1 y Alpaydin §8.3, p. 192. Lo que no sale de la cátedra —la regla de un error estándar, por
  ejemplo— está marcado como tal, y lo que es inferencia nuestra dice «inferencia». Una cita,
  verificada contra los slides, todavía no está registrada y lleva la marca «cita pendiente de
  registrar»: Clase 3, slides 98–100, el sobreajuste a la validación y sus cuatro defensas (slide 10
  y pregunta 11). Se registra en `DECISIONES.md` antes de entregar, y hasta entonces no entra en lo
  hablado.

## Quién habla

| Bloque | Slides | Quién | Duración |
| --- | --- | --- | --- |
| El problema y los datos | 1–3 | Andrés | 0:00 → 1:10 |
| EDA con consecuencias | 4–7 | Sebastián | 1:10 → 3:05 |
| Métricas, modelos e hiperparámetros | 8–12 | Andrés | 3:05 → 5:25 |
| Modelo final, errores, conclusiones y limitaciones | 13–17 | Sebastián | 5:25 → 8:00 |
| Hallazgo y cierre | 18–20 | Andrés | 8:00 → 9:00 |

Cada uno habla unos cuatro minutos y medio: Andrés 725 palabras (4:29) y Sebastián 732 (4:31). Son
cuatro cambios de orador. En las preguntas responde primero quien presentó ese bloque, y el otro
completa. El reparto es una propuesta: si se cambia, se cambia por bloque entero, nunca a mitad de un
bloque.

## Cobertura de la consigna

| Punto del enunciado | Dónde se dice | Minuto |
| --- | --- | --- |
| 0. Dataset y problema: predecir `y` para priorizar llamadas | Slide 2 | 0:05–0:35 |
| Consejo «Importante»: qué datos se usan en cada etapa | Slide 3 (y cada número del guion dice su conjunto) | 0:35–1:10 |
| 1. Partir antes de transformar; lo aprendido, dentro del pipeline | Slides 3 y 7 | 0:35–1:10 y 2:30–3:05 |
| 1. EDA: balance de clases | Slide 2 | 0:05–0:35 |
| 1. EDA: distribuciones y relación con `y`; diferencias entre quienes aceptan y quienes no | Slides 4–6 | 1:10–2:30 |
| 1. EDA: fuga de datos (qué existe antes de llamar) | Slide 4 (`campaign`, en las aclaraciones de la 2) | 1:10–1:35 |
| 1. EDA: faltantes y categorías raras | Slide 7 (`unknown` queda como categoría; dos categorías fundidas) | 2:30–3:05 |
| 1. Cada observación del EDA con su consecuencia | Slides 4–7, cada una con su veredicto | 1:10–3:05 |
| 2.1 Naive Bayes, SVM, KNN y RF con scikit-learn | Slide 9 (y 13) | 3:35–4:05 |
| 2.2 k-fold sólo sobre train, sin fuga | Slides 3 y 7 | 0:35–1:10 y 2:30–3:05 |
| 2.3 Dos métricas justificadas | Slide 8 | 3:05–3:35 |
| 3.1 Un hiperparámetro por modelo, con curvas de validación | Slides 10–12 | 4:05–5:25 |
| 3.1 Sobreajuste y subajuste en una curva | Slide 10 (y 11) | 4:05–4:40 |
| 4. El modelo elegido | Slide 13 | 5:25–5:50 |
| 4. El desempeño esperado en datos nuevos | Slide 14 | 5:50–6:20 |
| 5. Qué errores son los relevantes | Slide 15 | 6:20–6:45 |
| 5. Qué modelos funcionaron mejor y por qué; los supuestos de cada uno | Slide 16 | 6:45–7:25 |
| 5. Limitaciones y mejoras | Slide 17 | 7:25–8:00 |
| *(no calificable)* El hallazgo: el modelo aprende la época | Slides 18–19 | 8:00–8:55 |

**Gate G5: ningún punto calificable cae después del hallazgo.** Todo lo que la consigna califica,
limitaciones incluidas, termina en la slide 17, a los 8:00. El hallazgo empieza después y no responde
ningún punto que no esté ya respondido: le pone números a la primera limitación. Si el tribunal
corta en la 17, el trabajo está completo.

---

## El guion

### 1 · Portada — 0:00 → 0:05 (~12 palabras)

**Habla: Andrés** (bloque 1, slides 1–3).

> Buenas tardes. Grupo siete: Trabajo Práctico dos, clasificación supervisada sobre *Bank
> Marketing*.

#### A aclarar
- **El dataset es el del enunciado** (TP2, p. 1): el *Bank Marketing* de UCI con los indicadores
  macroeconómicos; 41 188 filas, 20 predictoras y la clase `y`, con campañas de mayo de 2008 a
  noviembre de 2010 (`bank-additional-names.txt`). El 41 188 de la portada es el archivo entero; todo
  lo que sigue sale de train, salvo que se diga otra cosa.
- **Los cuatro clasificadores son de scikit-learn**, como pide el punto 2.1.

### 2 · Decir siempre «no» ya acierta el 88,7 % — 0:05 → 0:35 (~87 palabras)

**Habla: Andrés.**

> Un banco ofrece un plazo fijo por teléfono. Antes de marcar, ¿este cliente va a contratar? Si lo
> sabemos, llamamos primero a los más probables.
> **[→]** El modelo sólo puede usar lo que el banco sabe antes de la llamada: el cliente, su historial
> en campañas anteriores, el contexto económico y los datos de la llamada que va a hacer.
> **[→]** Y la clase está muy desbalanceada: en train, sólo el 11,3 % dijo que sí. Decir siempre «no»
> ya acierta el 88,7 %, así que aquí la exactitud no sirve.

#### A aclarar
- **Cuántos son, en train:** 3 711 «yes» de 32 940 filas; 7,9 «no» por cada «yes».
- **Por qué 19 predictoras y no 20.** Sale `duration`, que es fuga (slide 4; D-05). Las otras 19 se
  revisaron una por una y existen antes de marcar (D-07).
- **¿`campaign` existe antes de llamar?** Sí: es el número de llamada que se va a hacer, contando esa
  (`bank-additional-names.txt`, atributo 12). La reserva es que cada fila guarda el total al cierre de
  la campaña y, tras un «yes», no se vuelve a llamar (inferencia), así que su valor final depende en
  parte del resultado. En train, la media es 2,053 en los «yes» y 2,627 en los «no», con la misma
  mediana, 2: compatible con esa inferencia, pero no la prueba. Va como limitación (D-07).
- **¿`month`, `day_of_week` y `contact`?** Son de la llamada que se va a hacer y se eligen antes de
  marcar (inferencia; es la consulta 2 a la cátedra).
- **Por qué el modelo ordena en lugar de clasificar con un umbral.** El enunciado habla de «priorizar
  recursos de marketing» (TP2, p. 1): lo que el banco necesita es a quién llamar primero.
- **La exactitud engaña con clases desbalanceadas:** modelos con la misma exactitud cometen errores
  muy distintos (Clase 4, slides 26–27; D-19).

### 3 · Train decide; test sólo mide, y una sola vez — 0:35 → 1:10 (~92 palabras)

**Habla: Andrés.** Es la regla de juego antes del primer número (guía A3) y el consejo «Importante»
del enunciado. Se dice completa pero rápido: son definiciones que el tribunal ya conoce. El segundo
paso cuenta, sin esconderla, la primera exploración sobre el archivo completo (D-04).

> Primero, quitamos 12 duplicados exactos: si una copia cae en train y otra en test, el test deja de
> ser independiente. Después, 80/20 estratificado por la clase.
> **[→]** El análisis exploratorio también decide, así que lee sólo train, como las ablaciones. Una
> primera exploración del archivo completo se rehízo sobre train antes de decidir nada.
> **[→]** La validación cruzada de 5 folds, también: con ella elegimos modelo e hiperparámetros.
> **[→]** Y el test no decide nada: se abre una vez, al final, para estimar el desempeño. Por eso cada
> número dice de qué conjunto sale.

#### A aclarar
- **Las cifras de la partición:** 41 176 filas sin duplicados (D-01; ningún par duplicado tiene
  etiquetas distintas); train 32 940 filas (80 %) y test 8 236 (20 %), con semilla 42 (D-02).
- **¿Por qué estratificar, si no se vio en clase?** Sin estratificar, la proporción de «yes» del test
  cambia con la semilla: en 400 particiones con semillas distintas, su desvío es de 0,31 puntos, entre
  10,4 % y 12,1 %. Estratificando, train y test quedan con el mismo 11,3 % (D-02). La cifra sale de
  `src/evidencia_particion.py`, que mide la partición y no elige nada del modelo.
- **¿Qué exploración corrió sobre el archivo completo?** La primera versión del análisis
  exploratorio, en el estado inicial del repositorio (commit `0a9b311`). Se rehízo sobre train, con
  `src/eda.py`, antes de tomar ninguna decisión (D-04).
- **¿Cómo se garantiza que nada mire el test?** Todo lo anterior al punto 4 lee sólo train con
  `cargar_train()`. El único módulo que evalúa un modelo en test es `src/evaluar_test.py`; otros
  cuatro archivos pueden nombrar el test o el CSV completo, y ninguno elige nada del modelo: la
  partición, su verificación, la medición de la partición misma y el propio test de aislamiento.
  `tests/test_aislamiento.py` falla si cualquier otro `.py` los nombra (D-04; pregunta 2).
- **El tamaño de cada vuelta de la validación cruzada, dentro de train:** 26 352 filas para ajustar y
  6 588 para validar.
- **Qué se evalúa en test:** Random Forest reentrenado con todo train, que es lo que pide la cátedra
  una vez elegido el modelo (Clase 2, slides 88–89).
- **El orden importa** porque el enunciado lo pide: partir «antes de aplicar cualquier
  transformación que pueda producir data leakage» (TP2, p. 1).

### 4 · La duración predice el «yes», pero no existe antes de llamar — 1:10 → 1:35 (~66 palabras)

**Habla: Sebastián** (bloque 2, slides 4–7). Entra con la primera decisión del EDA.

> Primera decisión del análisis exploratorio: la fuga. `duration` es la duración de la última
> llamada: se conoce al colgar, cuando el resultado ya se sabe. Medimos cuánto pesa, en validación:
> con `duration` los cuatro saltan, y Random Forest pasa de 0,770 a 0,939. Es un techo que no existe
> antes de llamar, y el propio diccionario del dataset pide descartarla.
> **[→]** Consecuencia: `duration` queda fuera del modelo.

#### A aclarar
- **Con qué modelos se midió.** Con la configuración de referencia de las ablaciones (A0: Naive Bayes
  gaussiano, SVM con RBF y C = 1, KNN con k = 15, Random Forest de 300 árboles), en los mismos 5 folds:
  es la ablación A1. Los cuatro saltos, en AUC de validación: NB gaussiano de 0,768 a 0,831, SVM de
  0,701 a 0,907, KNN de 0,755 a 0,911 y RF de 0,770 a 0,939 (+0,170).
- **¿Sola, cuánto da?** Como puntaje, sin ajustar nada, `duration` da AUC 0,818 sobre train.
- **Cuánto predice, en train** (la figura sólo muestra el AUC, así que no se dice): el decil más
  corto llega a 59 segundos y tiene 0,0 % de «yes»; el más largo empieza en 551 segundos y tiene
  46,1 %. Las 4 filas con `duration` = 0 son todas «no».
- **El diccionario lo dice textual:** `duration` sólo sirve como referencia y hay que descartarla para
  un modelo predictivo realista (`bank-additional-names.txt`, atributo 11; D-05).
- **¿Qué otras variables se revisaron por fuga?** Las 20, una por una: 19 existen antes de marcar, y
  `campaign` con la reserva de la slide 2 (D-07).
- **¿Por qué medirla, si de todos modos sale?** Porque el enunciado sugiere comparar el modelo con y
  sin las variables problemáticas (TP2, p. 1), y porque el techo dice cuánto cuesta no tenerla.

### 5 · El archivo va por fecha: el «yes» sube de 1,9 % a 50,9 % — 1:35 → 2:05 (~79 palabras)

**Habla: Sebastián.**

> Segunda: el archivo está ordenado por fecha, de mayo de 2008 a noviembre de 2010, y la tasa de
> «yes» no es estable. En train, en bloques de 2 000 filas, sube de 1,9 % a 50,9 %, mientras el
> euribor se desploma.
> **[→]** Consecuencia: partimos al azar —el enunciado plantea k-fold y no menciona el tiempo— y
> barajamos los folds, para que ninguno quede en una sola época. Lo que eso cuesta va como limitación,
> y es el hallazgo del final.

#### A aclarar
- **Por año, en train:** 4,9 % de «yes» en 2008, 19,2 % en 2009 y 51,9 % en 2010, 10,5 veces más. Por
  bloques, el último tiene 27,1 veces la tasa del primero.
- **El euribor, en train:** 4,86 de media en el primer bloque y 0,71 en el más bajo (panel inferior de
  la figura).
- **¿De dónde sale el año?** El CSV no lo trae: cada vez que el mes retrocede, empieza un año nuevo
  (año inferido, como en `resultados/eda/eda.html`).
- **¿Por qué no partir por fecha?** El enunciado plantea k-fold y no menciona el tiempo, y con el
  último 20 % del archivo como test se entrenaría en una época y se evaluaría en otra: train tendría
  6,4 % de «yes» y test 30,8 % (D-03). Es la consulta 5 a la cátedra.
- **¿Qué pasaba sin barajar?** `StratifiedKFold` sin barajar, que es lo que hace pasarle un número a
  `cv`, deja cada fold en una época: en train, euribor medio de 4,87 en el primero y 1,10 en el último;
  barajando, todos entre 3,58 y 3,64 (D-06).
- **`month` también fecha las filas:** en train, el 100,0 % de las de septiembre, el 90,9 % de las de
  octubre y el 94,5 % de las de diciembre caen en el último 20 % por fecha. Por eso la ablación A8 saca
  el bloque macro **y** `month` (slides 18–19).

### 6 · 999 no significa «nunca contactado» — 2:05 → 2:30 (~67 palabras)

**Habla: Sebastián.**

> Tercera: `pdays`. El diccionario dice que 999 es «no contactado antes», y es el 96,3 % de train.
> Pero 3 302 filas con 999 sí tuvieron contactos previos, y los que tienen días registrados contratan
> el 64,6 %.
> **[→]** En validación, sacarla empeora a Naive Bayes y a KNN, y en los otros dos no cambia nada. Se
> queda: escalado o en un árbol, el 999 funciona como un corte.

#### A aclarar
- **¿Cuánto empeora sin `pdays`?** En AUC de validación, pareado por fold (ablación A2): NB gaussiano
  −0,0012 con desvío 0,0003, y KNN −0,0019 con desvío 0,0007: más que su desvío. SVM +0,0023 (desvío
  0,0037) y RF +0,0003 (desvío 0,0022): neutros (D-10).
- **Cuántos tienen días registrados:** 1 207 filas de train con `pdays` < 999. Entre los que tienen
  999, el «yes» es 9,2 %.
- **La barra del medio** (999 con contactos previos) tiene su tasa anotada en la figura, pero esa
  tasa no es un macro de `numeros.md`: se señala en pantalla, no se dicta.
- **¿Y una indicadora de contacto previo?** Ya está: `poutcome` «nonexistent» equivale a
  `previous` = 0, y duplicarla pesaría doble en Naive Bayes y en las distancias de KNN (D-10).
- **¿Y el historial, sirve?** Con `poutcome` = success, en train el «yes» es 66,0 %.
- **Esas 1 207 filas son justo los atípicos de `pdays` por IQR:** por eso no se borran (slide 7).

### 7 · Todo se ajusta dentro de cada fold, en este orden — 2:30 → 3:05 (~101 palabras)

**Habla: Sebastián.** Las cajas entran en el orden en que el pipeline las ejecuta. El último overlay
es la consecuencia del IQR, que es exactamente lo que la cátedra marcó como inconcluso en el TP1:
se dice para qué sirvió, con su número, y que no se borró nada.

> La partición: 5 folds estratificados y barajados, para que cada uno mezcle las épocas.
> **[→]** Dentro de cada fold, reglas fijas: sale `duration`, `default` pasa a indicadora, se funden
> dos categorías raras y `unknown` queda como categoría; quedan 58 columnas.
> **[→]** Después, la codificación de cada modelo: escalado para KNN y la SVM, deciles para Naive
> Bayes, ajustados sólo con el fold de entrenamiento.
> **[→]** Los cuatro, en los mismos folds.
> **[→]** Y los atípicos por IQR: sirven para diagnosticar, no para filtrar. Marcan 6 731 filas de
> train, y entre los de `pdays` el 64,6 % dijo que sí: son señal, no se borra ninguna.

#### A aclarar
- **¿Por qué `default` como indicadora?** En train sólo 2 filas dicen «yes» (D-09). Mejora a RF en
  +0,0029 de AUC de validación (desvío 0,0021; ablación A3), es neutra en los otros tres, y pasa de 62
  a 60 columnas.
- **Las categorías raras** (D-11): `illiterate` se funde con `basic.4y`, y `marital` «unknown» con
  `married`, la moda. Es neutra o levemente mejor (NB gaussiano +0,0008 de AUC de validación;
  ablación A4) y simplifica. Con las dos, 58 columnas.
- **¿Y Random Forest?** One-hot sin escalar: no mide distancias (D-12). **¿Y Naive Bayes?** El
  categórico: las numéricas en 10 cortes por deciles aprendidos en cada fold, las categóricas como
  códigos, y Laplace con α = 1 (Clase 6, slide 46; D-18); para qué sirve, pregunta 16.
- **¿Cómo se decidió cada paso?** Con doce ablaciones, cada una con y sin una sola decisión, en los
  mismos folds; se adopta lo que sube el AUC de validación más que su desvío, o lo neutro que
  simplifica (respaldo R3; la A1 es la de la slide 4).
- **Los atípicos, completos** (D-17, en train): 6 731 filas, el 20,4 %, tienen algún valor fuera de
  1,5·IQR. El «yes» entre los atípicos de `pdays` es 64,6 %; de `age`, 46,5 %; de `previous`, 26,6 %;
  y de `campaign`, 4,5 %, contra 11,3 % en total. Borrarlos sacaría a los que más contratan.
- **¿Y `unknown`?** Queda como categoría (D-08): imputar por la moda es neutro (RF +0,0011 en
  validación) y borra una señal: en train, con `default` «unknown» el «yes» es 5,3 %, contra 12,8 %
  cuando se conoce.
- **¿KNN sin escalar no da más?** Sí: +0,0084 de AUC de validación (desvío 0,0067), pero por la razón
  equivocada; es la pregunta 8 del banco (D-12).
- **`campaign` y la edad, sin transformar** (AUC de validación): `log1p` sobre `campaign` es nulo
  (A5: KNN +0,0008, SVM −0,0006; D-13), y la edad en tramos suma columnas (70) sin mejorar al NB
  gaussiano (A10: +0,0001; D-16).
- **Sin fuga por construcción:** todo lo aprendido vive dentro del `Pipeline` y se vuelve a ajustar
  en cada fold, que es lo que pide el enunciado (TP2, p. 1); lo verifica `tests/test_preproceso.py`.

### 8 · Dos métricas que no premian decir siempre «no» — 3:05 → 3:35 (~83 palabras)

**Habla: Andrés** (bloque 3, slides 8–12). Es teoría de la Clase 4: se nombra, no se desarrolla. Lo
propio es el 20 % y por qué no se usa la exactitud.

> Falta la vara: dos métricas de la Clase 4. El AUC mide si el modelo ordena bien a quién llamar, sin
> depender de un umbral; sin modelo vale 0,5. Con ella comparamos y elegimos, siempre en validación.
> **[→]** La segunda es el uso real: si el banco llama al 20 % de la lista con mayor puntaje, ¿qué
> parte de los que contratarían alcanza? Cuenta el error caro, el cliente perdido; sin modelo vale
> 0,200.
> **[→]** La exactitud, no: decir siempre «no» ya da el 88,7 %.

#### A aclarar
- **Las citas:** el AUC, Clase 4, slides 45–51; el umbral «depende del costo de cada error», slides
  41–44; tres modelos con la misma exactitud cometen errores distintos, slides 26–27; precisión y
  recall, slides 30–31 (D-19).
- **Lo que el AUC no hace:** no premia decir siempre «no» (un puntaje constante da 0,5) y no depende
  del umbral. Por eso sirve para ordenar a quién llamar, que es priorizar recursos (TP2, p. 1).
- **«Sin modelo»** es un `DummyClassifier` que les da a todos el mismo puntaje, medido en los mismos
  folds de validación: AUC 0,500 y recall 0,200, porque llamar al 20 % al azar alcanza al 20 % de los
  «yes». Es el único rótulo de la línea de base en todo el TP.
- **¿Por qué no el umbral de la Clase 4, el punto de la ROC más cercano a (0,1)** (slide 52)? Es
  simétrico entre los dos errores, y aquí no cuestan lo mismo; y con un umbral por modelo los recalls
  no se comparan. Con el presupuesto, todos hacen las mismas llamadas: 1 318 por fold de validación
  (D-20). En test serán 1 647.
- **¿Por qué el 20 %?** Es un supuesto de negocio, declarado (D-20). Con el 10 % o el 30 %, Random
  Forest sigue primero en validación (respaldo R2; pregunta 9).
- **¿Y $F_1$?** Se guarda de todos modos: en validación, RF 0,454 y sin modelo 0,144. Es la consulta
  1 a la cátedra; si la acepta, cambia qué columna se lee, no qué se corre.
- **¿Y la precisión?** Se calcula, pero no decide: en validación, sin modelo es 0,113, la tasa de
  «yes».
- **¿Y el desbalance?** No se remuestrea: estas dos métricas ya no premian el «no» (D-21; pregunta 10).
- **Los empates en el corte** se reparten en proporción (`src/metricas.py`, `en_presupuesto`).

### 9 · Sin ajustar, gana Naive Bayes: 0,782 contra 0,500 sin modelo — 3:35 → 4:05 (~74 palabras)

**Habla: Andrés.** Son tres los que sobreajustan: también KNN con k = 15, que es el puente a la
slide 11, donde sobreajusta el k pequeño.

> Primero, los modelos con hiperparámetros de referencia: en AUC de validación, todos quedan muy por
> encima de sin modelo. El mejor es Naive Bayes en su versión categórica: discretizar las numéricas
> en deciles le suma 0,013 sobre el gaussiano, porque las numéricas están lejos de ser normales. Y
> tres sobreajustan: Random Forest sin límite de profundidad, la SVM con RBF y KNN con k = 15. Lo
> muestran, y lo corrigen, las curvas que siguen.

#### A aclarar
- **La referencia:** Random Forest de 300 árboles sin límite de profundidad; KNN con k = 15; SVM con
  RBF y C = 1; Naive Bayes con 10 cortes y Laplace α = 1 (Clase 6, slide 46; pregunta 16).
- **Categórico contra gaussiano** (D-18), en validación: 0,782 ± 0,007 contra 0,769 ± 0,009; fold a
  fold, +0,013 ± 0,0024, entre +0,010 y +0,016: gana en los cinco. En recall al 20 %, +0,048.
- **«Están lejos de ser normales», en train:** `pdays` es casi binaria por el centinela (999 en el
  96,3 % de las filas), `campaign` tiene cola larga (asimetría 4,89, máximo 56) y la edad forma una U
  (D-18).
- **¿Por qué RF da 0,770 en la slide 4 y 0,772 aquí?** Los dos son AUC de validación, pero la 4 es la
  referencia de las ablaciones (A0), antes de adoptar A3 y A4; aquí ya están. Lo mismo la SVM (0,701
  y 0,702) y el NB gaussiano (0,768 y 0,769); KNN da 0,755 en las dos.
- **Las brechas train − validación** (la figura sólo muestra validación, así que no se dicen): RF,
  train 1,000 y brecha 0,228; SVM con RBF, train 0,901 y brecha 0,200; KNN con k = 15, train 0,871 y
  brecha 0,116; NB categórico, 0,001. Las curvas de las slides 10 a 12 muestran cada sobreajuste.
- **Cuánto es «muy por encima»:** el mejor supera a sin modelo por 0,282 de AUC de validación.

### 10 · ¿Desde qué profundidad sobreajusta Random Forest? — 4:05 → 4:40 (~98 palabras)

**Habla: Andrés.** Es el punto 3.1 que el enunciado pide decir en la presentación: sobreajuste y
subajuste en una curva. Tres pasos, con el mismo encuadre: no apurarla.

> Esta es la curva para discutir sobreajuste: la profundidad máxima de los árboles. En train el AUC
> sube siempre, y sin límite llega a 1,000: los árboles memorizan.
> **[→]** La validación, en cambio, sube hasta profundidad 10 y después baja, hasta 0,772 sin límite:
> la brecha se abre a 0,228. Eso es sobreajuste. En el otro extremo, con profundidad 2, train y
> validación casi empatan, y los dos quedan bajos: el modelo es demasiado simple, subajuste.
> **[→]** Elegimos 8: queda dentro de un error estándar del máximo y es más simple. En validación da
> 0,795, con una brecha de 0,030.

#### A aclarar
- **¿Por qué no 10, si es el máximo?** En AUC de validación, 0,797 contra 0,795: la diferencia es
  menor que un error estándar del mejor, 0,0029. Quedan dentro 8, 10 y 12 (el umbral es 0,7939), y
  entre ellos la regla elige el más simple (D-22).
- **¿Y por qué no la profundidad seis?** Queda fuera del umbral por muy poco: en la práctica empata
  con 8 (inferencia), y la regla, aplicada tal cual, toma 8. Atención: a tres decimales, la de seis
  se lee 0,794 y parece entrar; su valor con cuatro decimales tiene el macro pendiente en
  `numeros.md`, así que no se dice de memoria. La banda de la figura es ± 1 desvío entre folds, no el
  error estándar, que es ese desvío dividido por la raíz de 5; el umbral no está dibujado, y a simple
  vista la de seis parece adentro.
- **¿De dónde sale la regla?** No está en las slides de la cátedra: es la forma operativa de lo que
  dice la Clase 3, slides 98–99 (cita pendiente de registrar): elegir el mejor entre muchas
  estimaciones con ruido también elige ruido favorable. Viene de los árboles CART de Breiman y otros,
  y está en Hastie, Tibshirani y Friedman, *The Elements of Statistical Learning*, §7.10.1: fuera de
  la bibliografía de la materia. También la usamos en el TP1.
- **El protocolo** es el de la Clase 7, slide 50: el AUC en train y en validación con los 5 folds, la
  brecha entre los dos y la curva.
- **La curva se midió con 300 árboles**, los de la referencia; el modelo final tiene 200, y con
  profundidad 8 da el mismo 0,795 de validación.
- **¿Y el número de árboles?** 200 (N0-11). Con profundidad 8 la curva de validación es plana: 25
  árboles dan 0,793 y 800 dan 0,796, y la regla elegía 25. Pero más árboles no suman complejidad: no
  cambian el sesgo y sólo bajan la varianza (Clase 7, slide 61). Es la pregunta 12 del banco.
- **¿Pesos de clase?** +0,0006 de AUC de validación, con un error estándar de 0,0024: despreciable
  (N0-12). Pregunta 13.
- **El subajuste, en números:** con profundidad 2, train 0,783 y validación 0,782, brecha 0,001; los
  dos por debajo del 0,795 que la validación alcanza con 8.
- **Otros puntos de validación** (la figura los tiene todos): profundidad 4, 0,789; 12, 0,795; 15,
  0,793; 20, 0,784; 25, 0,776.

### 11 · En KNN sobreajusta el k pequeño: se elige k = 801 — 4:40 → 5:05 (~62 palabras)

**Habla: Andrés.**

> En KNN el eje va al revés. Con un vecino, train da 0,988 y validación 0,619: cada cliente es su
> propio vecino, sobreajuste puro. Al sumar vecinos la brecha se cierra, y desde 151 todo queda dentro
> de un error estándar del mejor; tomamos el más suave, 801, con una brecha de 0,006. Ponderar por
> distancia queda peor en todo el rango.

#### A aclarar
- **El elegido, en validación:** k = 801 da 0,784, anotado en la figura con su brecha.
- **¿Dónde está el máximo?** En k = 601, también con 0,784 de AUC de validación; la curva es una
  meseta desde 301. El umbral de un error estándar es 0,7806 (error estándar del mejor, 0,0037).
- **¿No es enorme k = 801?** Cada fold entrena con 26 352 filas y el «yes» es el 11,3 % de train:
  estimar bien una proporción baja pide muchos vecinos (inferencia). Y está en el borde de la grilla
  a propósito: pregunta 21.
- **¿Y ponderar por distancia?** (Mitchell §8.2.1). Con `weights = distance`, train da 1,000, porque
  cada punto de train se encuentra a sí mismo, y validación 0,770 con k = 801, el que elige la misma
  regla: brecha 0,230. Con más de un vecino queda por debajo de la uniforme en todo el rango (con uno,
  las dos coinciden).
- **La brecha con un vecino es 0,369.** El eje es logarítmico porque la grilla va de 1 a 801.
- **¿Por qué train da 0,988 y no 1,000 con un vecino?** Porque en train hay filas con las mismas 19
  predictoras y distinta respuesta: a distancia cero, el vecino de una puede ser su gemela con la otra
  clase. Cuántas son tiene el macro pendiente en `numeros.md` (pregunta 7).
- **Escalar no es opcional en KNN:** mide distancias (Clase 8, slide 54; Alpaydin §8.3, p. 192).

### 12 · La SVM queda lineal y muy regularizada: C = 0,001 — 5:05 → 5:25 (~62 palabras)

**Habla: Andrés.**

> En la SVM exploramos C y el kernel, con pesos balanceados, que aquí sí ayudan. Con kernel lineal
> la curva de C es plana, sin sobreajuste, y tomamos 0,001, el más regularizado.
> **[→]** Con RBF, en cambio, un C grande sobreajusta: train sube casi a uno y la validación cae. Y
> ni en su mejor C el RBF supera al lineal: queda lineal.

#### A aclarar
- **¿Qué es C?** La convención de scikit-learn: un C grande castiga más las violaciones del margen y
  regulariza menos (Clase 8, slide 49). Pregunta 17.
- **Los kernels con C = 0,001**, en validación: lineal 0,774, polinómico 0,774 y RBF 0,768. Empatar no
  alcanza para elegir el polinómico: el lineal es más simple.
- **Los pesos** (no están en la figura, así que no se dicen): con RBF y C = 1, balancear lleva el AUC
  de validación de 0,702 a 0,770 (+0,068). Cambian el orden, no sólo el umbral: pregunta 10.
- **¿Por qué `LinearSVC`?** La SVM lineal de libsvm con C = 10 no terminó en 600 segundos (D-22).
  `LinearSVC` con pérdida *hinge* es la misma SVM lineal de margen blando, sin kernel, resuelta con
  liblinear en lugar de libsvm; lo único que cambia es que liblinear también regulariza el término
  independiente.
- **Con kernel lineal, 0,001 y 0,01 empatan** en 0,774 de validación: se toma el más regularizado. La
  brecha queda en 0,002. 0,001 es el menor C de la grilla: pregunta 21.
- **El mejor RBF**, en AUC de validación, es C = 0,01 con pesos, 0,772; sin pesos, C = 0,1, 0,706.
- **¿En la figura el RBF no queda por encima del lineal?** Entre C = 0,1 y 1, apenas, comparados en
  el mismo C; pero su máximo de validación, 0,772, no alcanza el 0,774 del lineal elegido.
- **¿Y γ?** No se ajustó: el kernel final es lineal. La grilla conjunta de C y γ (Clase 8, slides
  48–49) quedó como opcional.

### 13 · Ajustados, gana Random Forest por 0,011 de AUC en validación — 5:25 → 5:50 (~60 palabras)

**Habla: Sebastián** (bloque 4, slides 13–17). Entra con el resultado de la selección.

> Con los hiperparámetros elegidos gana Random Forest: en validación, 0,795 de AUC y 0,629 de recall
> llamando al 20 %. Le siguen KNN, Naive Bayes y la SVM, en ese orden, y ninguno queda dentro de un
> error estándar de Random Forest. Pero importa la escala: entre el mejor y el peor hay 0,021; entre
> Random Forest y sin modelo, 0,295.

#### A aclarar
- **Los cuatro, en AUC de validación** (anotados en la figura): Random Forest 0,795, KNN 0,784, Naive
  Bayes 0,782 y SVM 0,774.
- **El umbral de empate:** 0,7927, que es el AUC de validación de Random Forest menos su error
  estándar, 0,0025 (D-23). No entra ninguno, así que decide el AUC solo; si hubiera empate,
  desempataba el recall.
- **¿Sobreajusta?** Brecha de 0,030: train 0,825 y validación 0,795.
- **Los desvíos entre folds de validación** de Random Forest: ± 0,006 en AUC y ± 0,018 en recall.
- **Los recalls de los otros tres**, en validación: KNN 0,617, Naive Bayes 0,614 y SVM 0,614.
- **¿Cuánto sumó ajustar?** En AUC de validación: Random Forest +0,023, KNN +0,029 y SVM +0,072.
- **¿Y la precisión?** Llamando al 20 %, Random Forest acierta 0,354 de las llamadas; sin modelo,
  0,113 (validación).

### 14 · ¿Qué AUC esperar en clientes nuevos de la misma época? — 5:50 → 6:20 (~80 palabras)

**Habla: Sebastián.** Es la única slide que no compara nada (guía C2, C3): ni «sin modelo» ni otro
modelo, ni en pantalla ni en voz alta. La comparación con validación se dice, no se proyecta (C6).
Los «?» se leen de `informe/numeros.md`, sección «Números de test», después de evaluar.

> ¿Qué esperar en clientes nuevos? Lo dice el test: Random Forest reentrenado con todo train, sobre
> las 8 236 filas de test, una sola vez. Da ? de AUC, con un intervalo del 95 % de ?. Contra
> validación, la diferencia es ?, frente a un desvío entre folds de 0,006. Y atención a qué mide: el
> test sale de las mismas campañas que train, así que estima clientes nuevos de esa época, no una
> campaña futura. Eso es el hallazgo.

**Mientras el test siga cerrado** (ensayos hasta el 01/10), en lugar de lo anterior:

> ¿Qué esperar en clientes nuevos? Lo va a decir el test: Random Forest reentrenado con todo train,
> sobre las 8 236 filas de test, una sola vez. Todavía no lo abrimos: lo haremos después de las
> respuestas de la cátedra. Y atención a qué va a medir: el test sale de las mismas campañas que
> train, así que estima clientes nuevos de esa época, no una campaña futura. Eso es el hallazgo.

El encabezado cuenta la variante más larga.

#### A aclarar
- **¿Por qué esperar para abrir el test?** Si una respuesta de la cátedra cambiara una variable o la
  métrica, habría que evaluar el test dos veces, y cada evaluación le quita independencia (N0-1).
  Pregunta 1.
- **El intervalo:** bootstrap estratificado sobre las predicciones de test, 2 000 remuestreos y
  percentiles del 95 %; no se reentrena nada (D-24).
- **¿Por qué no aparece «sin modelo»?** Porque el test no compite con nada: sólo mide. Es exactamente
  lo que la cátedra marcó en el TP1: comparar el número de test contra la línea de base parecía una
  decisión.
- **Cómo leer la diferencia con validación:** contra el desvío entre folds de validación, 0,006, y
  contra el intervalo de test. Dentro de ± 0,006 es ruido, en cualquier sentido; si además da algo
  más bajo, es coherente con el optimismo de haber elegido los hiperparámetros en la misma validación
  que se reporta (slide 17), que no medimos (pregunta 11). Una diferencia mayor que un desvío no la
  explica el ruido, y difícilmente ese optimismo (inferencia): obliga a revisar la partición o el
  pipeline, y se dice (pregunta 30). En ningún caso se cambia el modelo después: lo que se promete es
  el número de test.
- **El test se evalúa una sola vez:** `\nevaluaciones` tiene que valer 1 en `resultados-test.tex`
  (gate G2).
- **El reentrenamiento con todo train** es el paso de la Clase 2, slides 88–89.

### 15 · El error caro es el «yes» que no se llama — 6:20 → 6:45 (~72 palabras)

**Habla: Sebastián.**

> ¿Qué errores comete? Es la matriz de test, llamando al 20 % de la lista. Un falso positivo es una
> llamada de más; un falso negativo, un cliente que habría contratado y no se llama: ese es el caro.
> **[→]** Quedan ? «yes» de test sin llamar. En validación, los perdidos eran sobre todo de 2008 y
> sin historial.
> **[→]** El recall de test, llamando al 20 %, es ?.

**Mientras el test siga cerrado**, la matriz de la slide es la de validación, y se dice:

> ¿Qué errores comete? Con el test cerrado, la matriz es la de validación: los puntajes fuera de
> fold, llamando al 20 % de la lista. Un falso positivo es una llamada de más; un falso negativo, un
> cliente que habría contratado y no se llama: ese es el caro.
> **[→]** Quedan 1 382 «yes» sin llamar, sobre todo de 2008 y sin historial.
> **[→]** Aun así, cada «yes» cuesta 2,8 llamadas, contra 8,9 sin modelo.

El encabezado cuenta la variante más larga.

#### A aclarar
- **¿Qué es fuera de fold (OOF)?** Cada fila de train puntuada por el modelo del fold en el que no
  entrenó: toda train, sin fuga.
- **La matriz de validación completa:** 2 329 «yes» alcanzados, 1 382 perdidos, 4 259 llamadas de
  más y 24 970 bien descartados; 3 711 «yes» en total.
- **¿Por qué la lista completa y no fold por fold?** La matriz junta los puntajes fuera de fold de
  los 5 folds en una sola lista y la corta en su 20 %, 6 588 llamadas: es el corte de la curva de
  ganancia y el del test (1 647 llamadas). `resultados/conclusiones.md`, §3, usa el mismo corte y da
  las mismas cuatro celdas.
- **¿Quiénes son los perdidos?** En validación (`conclusiones.md`, §3): tres de cada cuatro son de
  2008, un año en el que el modelo casi no llama, y casi todos no tienen campaña previa. En cambio,
  a los que tuvieron éxito en una campaña anterior los llama a todos.
- **El costo por «yes», en validación:** 2,8 llamadas es (2 329 + 4 259) / 2 329; sin modelo, 8,9 es
  1 / 0,113, la inversa de la tasa de «yes».
- **Por qué el falso negativo es el caro:** una llamada de más cuesta una llamada; un cliente perdido
  es un plazo fijo que no se contrata. Es el costo desigual de la Clase 4, slides 41–44.

### 16 · Por qué ganó Random Forest y quedó última la SVM — 6:45 → 7:25 (~107 palabras)

**Habla: Sebastián.** Son los cuatro ejemplos del enunciado (TP2, p. 2): variables muy
correlacionadas y distribuciones no gaussianas (Naive Bayes), diferencias de escala (KNN, SVM) y
relaciones no lineales (RF, SVM).

> ¿Por qué rindió cada uno? Por sus supuestos. Random Forest no supone escala ni forma, y sus cortes
> captan relaciones no lineales, como la U de la edad: contratan más los más jóvenes y los mayores.
> **[→]** KNN mide distancias y necesita escalar: sin escalar, en train, `pdays` tiene el 86,6 % de la
> varianza.
> **[→]** Naive Bayes supone independencia, y en train el bloque macro la viola, con correlaciones de
> 0,91 a 0,97; como las numéricas no son normales, el categórico le gana al gaussiano.
> **[→]** La SVM quedó lineal, y un puntaje lineal no dobla la U. Aun así, entre los cuatro hay sólo
> 0,021 de AUC en validación.

#### A aclarar
- **¿Por qué el bloque macro ayuda a Naive Bayes si viola la independencia?** Sacarlo cuesta −0,023
  de AUC de validación en el gaussiano (A7; D-14), y 0,033 en el categórico. Es un resultado contra la
  intuición, probablemente porque fecha a los clientes (inferencia). Pregunta 15.
- **¿Entonces violar la independencia no tuvo ningún efecto?** Lo tuvo en las probabilidades, no en
  el orden (inferencia). Contar tres veces la época las lleva a los extremos —por eso Naive Bayes se
  puntúa con log-odds: la probabilidad llegaba a 1 exacto (`src/metricas.py`)—, pero el AUC sólo mide
  el orden, y ahí el bloque ayuda. Calibrar está entre las mejoras (`conclusiones.md`, §5). Cuántos
  quedan con probabilidad casi 1, y cuántos de ellos contratan, tiene macros pendientes en
  `numeros.md`.
- **¿La U de la edad es real?** Se mezcla con la época: casi todos los mayores de sesenta son de
  2009–2010 (`conclusiones.md`, §1), y Random Forest es el que más pierde sin macro ni `month`:
  −0,093 de AUC de validación (A8).
- **¿La SVM con RBF no capta la U?** Podría, pero con γ sin ajustar no la aprovechó (inferencia): con
  C = 0,001, en validación, el RBF da 0,768 y el lineal 0,774. Que sea la causa de que quede última
  también es inferencia.
- **Las escalas, en train:** desvío de `pdays` 186,6 y de `nr.employed` 72,4; el mayor desvío de las
  numéricas es 374 veces el menor.
- **Las distribuciones no gaussianas, en train:** asimetría de `campaign` 4,89; `pdays` = 999 en el
  96,3 %; la edad en U.
- **Por qué un puntaje lineal no dobla la U:** la edad entra en años (D-16), y un puntaje lineal en
  la edad sube o baja, nunca las dos cosas. Que eso explique el último puesto es inferencia.

### 17 · Limitaciones: el test mide estas campañas, no una futura — 7:25 → 8:00 (~100 palabras)

**Habla: Sebastián.** Antes del hallazgo, siempre (guía A4, A7). Nunca se recorta.

> Tres limitaciones, por impacto. La época: el test sale de las mismas campañas, así que vale para
> clientes nuevos de esas campañas, no para una futura. Llamar al 20 % es un supuesto: con el 10 % o
> el 30 %, Random Forest sigue primero en validación. Y elegimos los hiperparámetros con la misma
> validación que los reporta: el máximo sale algo optimista.
> **[→]** Lo primero que haríamos es validar hacia adelante en el tiempo, que es como se usaría el
> modelo. Después, sumar historial: el 86,3 % de train no tiene campaña previa. Y un costo por tipo de
> error, en lugar del 20 %.

#### A aclarar
- **¿Cuánto optimismo?** No lo medimos: haría falta una validación cruzada anidada, y aquí ese papel
  lo cumple el test (pregunta 11). La regla de un error estándar lo atenúa, porque no elige el
  máximo: profundidad 8 y no 10 (`conclusiones.md`, §4).
- **¿Y dentro de una campaña?** El 0,795 y el 0,629 de validación ordenan una lista que mezcla
  2008–2010; en los pares del mismo año, el AUC fuera de fold es 0,658 (pregunta 27).
- **La sensibilidad al presupuesto**, en validación: Random Forest alcanza 0,446, 0,629 y 0,705 de los
  «yes» llamando al 10 %, 20 % y 30 %, contra 0,100, 0,200 y 0,300 sin modelo; su precisión baja de
  0,502 a 0,265 (respaldo R2).
- **¿Y el desbalance?** Pesos balanceados en Random Forest: +0,0006 de AUC de validación,
  despreciable (N0-12).
- **¿Y `unknown`?** Imputar es neutro: Random Forest +0,0011 de AUC de validación (D-08).
- **¿Y `campaign`?** Puede depender del resultado (D-07); el efecto no se midió.
- **¿Y `duration`?** Con ella, Random Forest llega a 0,939 de AUC de validación, pero no existe antes
  de llamar (slide 4).
- **Otra mejora:** calibrar las probabilidades, porque un costo por tipo de error necesita
  probabilidades y hoy sólo cuenta el orden (inferencia; `conclusiones.md`, §5).

### 18 · Entrenado con el pasado, Random Forest cae de 0,795 a 0,558 — 8:00 → 8:30 (~77 palabras)

**Habla: Andrés** (bloque 5, slides 18–20). Entra enganchado con la última frase de Sebastián:
validar hacia adelante.

> El hallazgo. Validamos hacia adelante dentro de train: cada fold entrena con el pasado y valida con
> el bloque siguiente. Barajados, los cuatro están entre 0,774 y 0,795; hacia adelante, ninguno pasa
> de 0,598, y Random Forest cae a 0,558.
> **[→]** ¿Por qué tanto? Parte del 0,795 es distinguir años: en los puntajes fuera de fold, el 66,3 %
> de los pares «yes»–«no» son de años distintos, y ahí el AUC es 0,864; en los del mismo año, 0,658.

#### A aclarar
- **¿Cómo se validó hacia adelante?** `TimeSeriesSplit` con 5 folds sobre train ordenado por fecha:
  cada fold valida con un bloque de 5 490 filas y entrena con todo lo anterior, de 5 490 a 27 450
  filas (D-25). Todo dentro de train: el test no participa.
- **¿Por qué el desvío hacia adelante es tan grande** (± 0,123 en Random Forest)? Porque el modelo
  sirve en dos bloques y en tres no: 0,664 y 0,711 en 2009–2010, y de 0,426 a 0,500 en 2008, donde
  el primer bloque entrena con 159 «yes». No es la tasa de «yes» de cada bloque, que va de 4,7 % a
  35,2 %: el AUC no depende de ella, porque la ROC calcula cada tasa dentro de su clase (Clase 4,
  slides 45–51). Lo que cambia es cuánto se deja ordenar cada bloque: barajado, en las mismas filas,
  de 0,519 a 0,584 en 2008, y 0,748 y 0,762 en 2009–2010.
- **¿Y para elegir a quién llamar dentro de una campaña?** Pregunta 27.
- **Los cuatro, hacia adelante:** Random Forest 0,558, KNN 0,539, Naive Bayes 0,598 y SVM 0,541;
  barajados, en validación, 0,795, 0,784, 0,782 y 0,774. Random Forest pierde 0,237. En recall al
  20 %, hacia adelante da 0,243.
- **La cuenta de los pares:** el AUC es la proporción de pares «yes»–«no» bien ordenados; el de los
  pares del mismo año es el de cada año ponderado por sus pares, y el de años distintos sale por
  diferencia (`conclusiones.md`, «Cómo se calculó»). Por año, fuera de fold: 2008, 0,598; 2009,
  0,759; 2010, 0,749.
- **Por qué los pares entre años son fáciles:** en train, 2008 tiene 4,9 % de «yes» y 2010, 51,9 %;
  basta con reconocer la época para ordenarlos bien (inferencia).

### 19 · En 2008 no sirve; en 2009–2010 pierde, pero no cae al azar — 8:30 → 8:55 (~71 palabras)

**Habla: Andrés.** Es lo último que se muestra (guía A6). Si el reloj va atrasado, el segundo paso
se reduce a la frase de recorte del «Reloj de ensayo».

> Bloque a bloque: en los tres bloques de 2008, hacia adelante no pasa de 0,5, y en el tercero,
> 0,426, el orden incluso se invierte.
> **[→]** Pero barajado, en esas mismas filas, tampoco pasa de 0,584: en 2008 el modelo no sirve ni
> mezclando épocas. En los dos bloques de 2009–2010, en cambio, pierde 0,084 y 0,051: una caída real,
> pero no al azar. Por eso el test vale para estas campañas.

#### A aclarar
- **¿Es porque las variables macro fechan las filas?** No: hacia adelante, sin ellas da 0,556, y sin
  ellas ni `month`, 0,547 (respaldo R4). Barajado, en validación, esas mismas variantes quedan en
  0,772 y 0,736, es decir, pierden 0,023 y 0,059. No es sólo que fechen a los clientes: la relación
  entre las variables y el «yes» cambia con el tiempo (inferencia).
- **¿Por qué no validar así desde el principio?** El enunciado plantea k-fold; un esquema temporal
  habría entrenado con 2008, con 4,9 % de «yes» en train, para predecir 2010, con 51,9 % (D-25). Es
  la primera mejora.
- **Hacia adelante, bloque a bloque** (anotado en la figura): 0,489, 0,500, 0,426, 0,664 y 0,711.
- **¿0,426 es azar?** No: es peor que el azar. Un AUC por debajo de 0,5 ordena al revés, el «mal»
  clasificador de la Clase 4 (slides 45–51). Es la señal más fuerte de que la relación entre las
  variables y el «yes» cambia con el tiempo (inferencia). No le calculamos un intervalo, así que no
  decimos a cuántos errores estándar queda de 0,5.
- **Barajado en las mismas filas** (el AUC fuera de fold de Random Forest en las filas de cada
  bloque): 0,536, 0,519, 0,584, 0,748 y 0,762. Hacia adelante pierde 0,047, 0,019, 0,158, 0,084 y
  0,051.
- **Los bloques:** mayo–julio, julio–agosto y agosto–noviembre de 2008; noviembre de 2008 a mayo de
  2009; mayo de 2009 a noviembre de 2010. El «yes» de cada uno: 4,7 %, 6,5 %, 5,6 %, 12,7 % y 35,2 %.
  Los «yes» con que entrena cada fold: 159, 419, 778, 1 084 y 1 780; el primero, con apenas 159, es
  el de 0,489.
- **Qué significa para el banco:** el número de test valdrá para clientes nuevos de estas campañas.
  Para una campaña futura, la mejor referencia que tenemos es la validación hacia adelante de
  2009–2010, 0,664 y 0,711 (inferencia).

### 20 · ¿Preguntas? — 8:55 → 9:00 (~7 palabras)

**Habla: Andrés.**

> Gracias; quedamos a disposición para las preguntas.

#### A aclarar
- **Esta slide queda puesta durante las preguntas.** Después vienen las cuatro de respaldo, fuera del
  recorrido: R1, la curva de ganancia; R2, la sensibilidad al presupuesto; R3, las ablaciones del
  análisis exploratorio; R4, el hallazgo sin las variables de época.
- **Los respaldos, por si hay que buscar un número:** `informe/numeros.md` (cada cifra con su
  fuente), `resultados/conclusiones.md`, `DECISIONES.md` y los CSV de `resultados/`.

---

## Los 10 minutos de preguntas

Respuestas cortas, con el dato al frente y su conjunto declarado. Entre corchetes, a qué slide
conviene volver y quién responde primero: el que presentó ese bloque. Las citas de clase son sólo
las que ya están en `DECISIONES.md` o en `plan/PLAN.md`, salvo la que dice «cita pendiente de
registrar» (ver «Convenciones»).

**Para volver a una slide.** El pie dice «n/20» y la página del PDF no coincide, por los overlays.
Página de la última variante de cada slide: 1 → 1 · 2 → 4 · 3 → 8 · 4 → 10 · 5 → 12 · 6 → 14 ·
7 → 19 · 8 → 22 · 9 → 23 · 10 → 26 · 11 → 27 · 12 → 29 · 13 → 30 · 14 → 31 · 15 → 34 · 16 → 38 ·
17 → 40 · 18 → 42 · 19 → 44 · 20 → 45 · R1 → 46 · R2 → 47 · R3 → 48 · R4 → 49.

### El test y los datos

**1. ¿Por qué no evaluaron el test todavía?** [slide 14 · Sebastián] — Porque las consultas a la
cátedra podían cambiar una variable, la métrica o la partición, y evaluar antes de saberlo obligaba
a evaluar dos veces: cada evaluación le quita independencia al test (N0-1). En el TP1 se evaluó más
de una vez sobre la misma partición y quedó como limitación; aquí no se repite. El módulo está
construido y probado sin abrir el test, corre una sola vez después del 30/09 y registra cuántas
veces corrió. *Si en la defensa siguiera cerrado:* la única estimación disponible es la de
validación, 0,795, y sale algo optimista; cuánto, no lo medimos (pregunta 11).

**2. ¿Cómo garantizan que ninguna decisión miró el test?** [slide 3 · Andrés] — Todo lo anterior al
punto 4 lee sólo train con `cargar_train()`. El único módulo que evalúa un modelo en test es
`src/evaluar_test.py`. Otros cuatro archivos pueden nombrar el test o el CSV completo, y ninguno
elige nada del modelo: `src/datos.py`, que hace la partición; `tests/test_datos.py`, que verifica que
train y test sean disjuntos y reconstruyan el dataset; `src/evidencia_particion.py`, que mide la
partición misma sobre el archivo completo (D-02, D-03); y el propio `tests/test_aislamiento.py`, que
nombra lo que busca. Fuera de esa lista blanca de cinco archivos, `tests/test_aislamiento.py` falla
si cualquier `.py` nombra el test o el CSV completo (D-04). El análisis exploratorio que decide corre
sobre train: una primera versión corrió sobre el CSV completo y se rehízo sobre train antes de decidir
nada (D-04). Además, todo lo que se aprende de los datos —escalado, deciles, codificación— vive dentro
del `Pipeline` y se ajusta en cada fold (`tests/test_preproceso.py`).

**3. ¿Por qué partición aleatoria, si el archivo está ordenado por fecha?** [slides 5 y 19 ·
Sebastián] — El enunciado plantea k-fold y no menciona el tiempo, y con el último 20 % del archivo
como test se entrenaría en una época y se evaluaría en otra: train tendría 6,4 % de «yes» y test
30,8 % (D-03). Lo que cuesta no está escondido: está medido en las slides 18 y 19, y por eso la
slide 14 dice «de la misma época». Es la consulta 5 a la cátedra.

**4. ¿Por qué estratificaron, si no se vio en clase?** [slide 3 · Andrés] — Porque sin estratificar
la proporción de «yes» del test cambia con la semilla: en 400 particiones, con un desvío de 0,31
puntos, entre 10,4 % y 12,1 %. El efecto es pequeño, pero estratificar no cuesta nada: train y test
quedan con el mismo 11,3 % (D-02).

**5. ¿Por qué `duration` fuera, si es la mejor variable?** [slide 4 · Sebastián] — Porque se conoce
al colgar, cuando `y` ya se sabe, y el propio diccionario pide descartarla para un modelo realista
(`bank-additional-names.txt`, atributo 11; D-05). Con ella Random Forest llega a 0,939 de AUC de validación, pero ese número no
existe antes de llamar, que es cuando opera el modelo (TP2, p. 1).

**6. ¿Hay otras variables con fuga?** [slide 2 · Andrés] — La que tiene reserva es `campaign`:
cuenta la llamada que se va a hacer, pero cada fila guarda el total al cierre de la campaña, y tras
un «yes» no se vuelve a llamar (inferencia). En train, la media es 2,053 en los «yes» y 2,627 en los
«no», con la misma mediana, 2: compatible, no concluyente; el efecto no se midió (D-07). `month`,
`day_of_week` y `contact` son de la llamada que se va a hacer (inferencia): es la consulta 2. Y los
indicadores trimestrales, `emp.var.rate` y `nr.employed`: `bank-additional-names.txt` no dice con
qué rezago se asignó el valor a cada fila, y suponemos el último publicado al llamar (inferencia;
D-07). Que delaten la época es un problema de robustez temporal, no de fuga (D-25).

**7. ¿El mismo cliente puede estar en train y en test?** [slide 3 · Andrés] — No se puede descartar.
El dataset no trae identificador de cliente (`bank-additional-names.txt` omite atributos por
privacidad), así que no se puede partir por cliente, y D-01 sólo quita los duplicados exactos. Sin
`duration`, en train se repiten combinaciones de las 19 predictoras, algunas con respuestas
distintas: el mismo cliente en dos campañas, o clientes distintos con los mismos datos. Si un
cliente cae en train y en test, el test sale algo optimista; no lo podemos medir, y es una
limitación más. Esas repeticiones explican también el 0,988 de KNN con un vecino en train
(slide 11). Cuántas filas son tiene el macro pendiente en `numeros.md`.

### Decisiones discutibles

**8. KNN sin escalar da más. ¿Por qué no lo usaron?** [slide 7 · Sebastián] — Da +0,0084 de AUC de
validación con desvío 0,0067, y pasaba nuestra regla de adopción: lo rechazamos por juicio, y está
declarado (D-12). Mejora por la razón equivocada: sin escalar, `pdays` y `nr.employed` son el 86,6 %
y el 13,0 % de la varianza de las numéricas de train, así que la distancia mide historial de contacto
y época —`nr.employed` es un reloj—. Es evidencia del hallazgo, no una mejora del modelo; y la
teoría pide escalar a los métodos de distancia (Clase 8, slide 54; Alpaydin §8.3, p. 192).

**9. ¿De dónde sale el 20 %?** [slide 8, respaldo R2 · Andrés] — Es un supuesto de negocio,
declarado (D-20). Lo que no depende del supuesto es el orden: en validación, llamando al 10 % o al
30 %, Random Forest sigue primero, con recall 0,446 y 0,705 contra 0,100 y 0,300 sin modelo; los
otros tres cambian de orden entre sí. No usamos el umbral de la Clase 4, el punto de la ROC más
cercano a (0,1) (slide 52), porque es simétrico entre los dos errores y aquí no cuestan lo mismo
(slides 41–44); y con un umbral por modelo los recalls no se comparan.

**10. ¿Por qué no trataron el desbalance, por ejemplo remuestreando?** [slides 8 y 12 · Andrés] — No
remuestreamos: las dos métricas ya no premian decir siempre «no» (D-19, D-21), y remuestrear sería
agregar un método que no se dio en clase (consulta 3). Pero el desbalance no sólo mueve el umbral:
puede sesgar el ajuste, y por eso los pesos de clase se midieron como hiperparámetro (D-21). En la
SVM, C es el costo de violar el margen (Clase 8, slide 49), y con 7,9 «no» por cada «yes» ese costo
lo dominan los «no»; `balanced` multiplica C por el peso de cada clase, así que cada «yes» pesa 7,9
veces más. Si los pesos sólo movieran el umbral, el AUC no cambiaría; en validación sube 0,068 con
RBF y C = 1, y en el mejor C de cada una, de 0,706 sin pesos a 0,772 con pesos: cambian el orden, y
se adoptan. En Random Forest sube sólo 0,0006 (N0-12): en cada hoja, el peso reescala la proporción
de «yes», casi como mover el umbral (inferencia).

**11. ¿No sobreajustaron la validación con tantas decisiones?** [slide 17 · Sebastián] — Algo, y
está declarado como limitación: es el sobreajuste a la validación de la Clase 3, slides 98–100 (cita
pendiente de registrar). Elegir el mejor entre muchas estimaciones con ruido también elige ruido
favorable, y las ablaciones, las curvas y los modelos de referencia usaron los mismos 5 folds: el
0,795 sale optimista, y no medimos cuánto. De las cuatro defensas del slide 100 usamos tres:
validación cruzada (D-06); búsquedas acotadas, porque las ablaciones y las grillas estaban en el
plan —lo agregado después, como la extensión de KNN hasta 801 y las curvas de C con pesos, queda
registrado en `hiperparametros.json`— y la regla de un error estándar no elige el máximo (D-22); y
un test intacto que se abre una vez (D-04). La cuarta, la validación cruzada anidada, no la
corrimos: la cátedra la presenta como avanzada, y aquí el test cumple el papel de la validación
externa. Lo que se promete es el número de test.

**12. ¿Por qué 200 árboles, si la regla de un error estándar daba 25?** [slide 10 · Andrés] — Porque
el número de árboles no es un eje de complejidad sino de estabilidad: más árboles no cambian el sesgo
y sólo bajan la varianza (Clase 7, slide 61), así que «el más simple» no aplica. Con profundidad 8 la
curva es plana en validación: 25 árboles dan 0,793 y 800 dan 0,796; 200 está en la meseta, con 0,795,
y da puntajes menos gruesos para ordenar la lista. Es un desvío de la regla, y está declarado
(N0-11).

**13. ¿Por qué Random Forest sin pesos de clase, si la regla elegía `balanced`?** [slide 10 · Andrés]
— Porque la mejora es +0,0006 de AUC de validación, con un error estándar de 0,0024: despreciable por
su tamaño. La regla del desbalance adopta los pesos sólo si mejoran más que el ruido (D-21), y sin
pesos el modelo es el de la Clase 7 tal cual (N0-12).

**14. ¿Por qué no validaron hacia adelante como esquema principal?** [slide 19 · Andrés] — Porque el
enunciado plantea k-fold, y un esquema temporal habría entrenado con 2008, con 4,9 % de «yes» en
train, para predecir 2010, con 51,9 % (D-25). Lo medimos dentro de train como limitación y es la
primera mejora: es como se usaría el modelo. Si la cátedra responde otra cosa en la consulta 5, el
código ya lo tiene (`src/robustez.py`).

### La teoría de los cuatro modelos

**15. ¿Qué supone Naive Bayes y cómo lo afectan estos datos?** [slides 9 y 16 · Sebastián] —
Independencia de las variables dada la clase. El bloque macro la viola: en train, tres de sus cinco
variables tienen correlaciones de 0,91 a 0,97, y aun así sacarlo le cuesta −0,023 de AUC de
validación al gaussiano (D-14), probablemente porque fecha a los clientes (inferencia). El gaussiano
supone además una normal por clase, y en train `pdays` es casi binaria, `campaign` tiene asimetría
4,89 y la edad forma una U. Por eso el del TP es el categórico: deciles aprendidos en cada fold y
Laplace con α = 1 (Clase 6, slide 46). Le gana al gaussiano por 0,013 de AUC de validación, y en los
cinco folds (D-18). Si repreguntan dónde se nota la violación: aclaraciones de la slide 16.

**16. ¿Para qué la corrección de Laplace? ¿Dónde les aparecía un cero?** [slides 7 y 9 · Sebastián]
— Un valor que en el fold de entrenamiento nunca aparece con «yes» tendría probabilidad 0 y anularía
el producto de Naive Bayes. Laplace suma α = 1 a cada conteo: (conteo + 1) / (N + k), con k los
valores posibles del atributo (Clase 6, slide 46). En los cinco folds barajados ningún valor
presente en el entrenamiento tiene conteo cero en una clase: el único caso, `default` = yes, 2 filas
de train y las dos «no», lo quitó D-09. Ahí α = 1 sólo suaviza, y no se ajustó porque el punto 3.1
no lo pide para Naive Bayes (D-22). Hacia adelante sí importa: el bloque siguiente trae valores
nunca vistos, como meses nuevos, con conteo cero en las dos clases. Sin α, el log-odds quedaría
indefinido; con α = 1, cada uno empuja hacia el «yes», porque la clase con menos filas tiene el
denominador menor (inferencia).

**17. ¿Qué hace C en la SVM, y por qué quedó en 0,001?** [slide 12 · Andrés] — C es el costo de las
violaciones del margen: con la convención de scikit-learn, un C grande castiga más y regulariza menos
(Clase 8, slide 49). Con kernel lineal la curva es plana en validación, 0,001 y 0,01 empatan en
0,774, y la regla toma el más regularizado. Con RBF, un C grande lleva train casi a uno y la
validación cae: sobreajuste.

**18. ¿Por qué lineal y no RBF? ¿Y γ?** [slide 12 · Andrés] — Comparados con C = 0,001 en
validación: lineal 0,774, polinómico 0,774 y RBF 0,768; el RBF en su mejor C, 0,01, llega a 0,772.
γ no se ajustó: la grilla conjunta de C y γ (Clase 8, slides 48–49) quedó como opcional una vez que el
kernel final fue lineal. La lineal va con `LinearSVC` porque la de libsvm con C = 10 no terminó en
600 segundos (D-22): es la misma SVM lineal de margen blando, resuelta con liblinear, y lo único que
cambia es que liblinear también regulariza el término independiente.

**19. ¿Por qué hay que escalar para KNN y la SVM, y no para Random Forest?** [slides 7 y 16 ·
Sebastián] — KNN y la SVM miden distancias, y sin escalar las decide la variable de mayor varianza
(Clase 8, slide 54; Alpaydin §8.3, p. 192): en train, `pdays` tiene desvío 186,6 y `nr.employed`
72,4. Un árbol corta cada variable por separado, así que la escala no le cambia nada (D-12).

**20. ¿No es enorme k = 801? ¿Y la ponderación por distancia?** [slide 11 · Andrés] — En KNN
sobreajusta el k pequeño: con un vecino, train 0,988 y validación 0,619. Desde 151 todo queda dentro
de un error estándar del mejor, y se toma el más suave: 801, con 0,784 de validación y brecha 0,006.
Con 26 352 filas por fold y 11,3 % de «yes» en train, estimar bien una proporción baja pide muchos
vecinos (inferencia). Por distancia (Mitchell §8.2.1), cada punto de train se encuentra a sí mismo:
train 1,000 y validación 0,770 con k = 801, el que elige la misma regla; con más de un vecino queda
por debajo de la uniforme en todo el rango.

**21. ¿k = 801 y C = 0,001 no quedaron en el borde de sus grillas?** [slides 11 y 12 · Andrés] — Sí,
los dos, y no por la misma razón. En KNN, el máximo quedó primero en el borde de una grilla más
corta, así que la extendimos hasta 801: es la extensión que declara la pregunta 11. Con ella, el
máximo, 601, quedó adentro, y la curva de validación es una meseta desde 301, entre 0,783 y 0,784.
801 está en el borde porque la regla elige el más suave de los que empatan, y la meseta llega hasta
el final de la grilla: un k mayor probablemente también empataría, sin cambiar el AUC (inferencia).
En la SVM, 0,001 es el menor C de la grilla: con kernel lineal, 0,001 y 0,01 empatan en 0,774 de
validación, y la regla toma el más regularizado. Un C menor sería una SVM lineal aún más
regularizada, y no la probamos: no sabemos si seguiría empatando o empezaría a subajustar.

**22. ¿Por qué Random Forest no sobreajusta con más árboles, pero sí con más profundidad?** [slide
10 · Andrés] — La profundidad es la complejidad de cada árbol: sin límite, train llega a 1,000 y
validación baja a 0,772, brecha 0,228. Los árboles, en cambio, se promedian: bajan la varianza sin
cambiar el sesgo (Clase 7, slide 61). El protocolo de las curvas —train y validación con los mismos
folds, la brecha como lectura del sobreajuste— es el de la Clase 7, slide 50.

**23. ¿Qué significa un AUC de 0,795?** [slide 8 · Andrés] — La probabilidad de que, tomando al azar
un «yes» y un «no» de validación, el modelo le dé más puntaje al «yes». Sin modelo es 0,5. No depende
del umbral y no premia decir siempre «no» (Clase 4, slides 45–51). Llamando al 20 % de la lista, eso
se traduce en 0,629 de recall de validación. Esa lista mezcla 2008–2010, y ordenar dentro de un mismo
año es más difícil (pregunta 27).

**24. ¿Los modelos son realmente distinguibles?** [slide 13 · Sebastián] — Random Forest supera a KNN
por 0,011 de AUC de validación, con un error estándar de 0,0025: el umbral de empate es 0,7927 y no
entra ninguno (D-23). Pero conviene decir la escala: entre los cuatro hay 0,021, y entre Random Forest
y sin modelo, 0,295. Y el error estándar supone folds independientes, cuando sus train se superponen
en tres cuartos: el real es algo mayor (inferencia).

**25. ¿Por qué 5 folds?** [slide 3 · Andrés] — Es el esquema del TP1 (D-06): cada vuelta entrena con
26 352 filas de train y valida con 6 588, y el desvío entre folds del AUC de validación de Random
Forest es 0,006, así que la estimación ya es estable. No comparamos otros k. Lo que sí importa es que
los folds estén barajados: sin barajar, cada uno quedaba en una época (slide 5).

### El hallazgo

**26. ¿El hallazgo no invalida el modelo?** [slides 18–19 · Andrés] — No invalida el número de test:
dice qué mide. El test es una muestra al azar de las mismas campañas, así que vale para clientes
nuevos de esas campañas, ordenados en una lista que las mezcla (pregunta 27). Hacia
adelante, en los dos bloques de 2009–2010, Random Forest conserva 0,664 y 0,711: pierde, pero no cae
al azar. En 2008 no servía ni barajado: 0,598 fuera de fold.

**27. Si el banco elige a quién llamar dentro de una campaña, ¿cuánto valen el AUC y el recall?**
[slide 18 · Andrés] — Menos. El 0,795 y el 0,629 de validación ordenan una lista que mezcla
2008–2010, y parte de su valor es reconocer la época: en los pares «yes»–«no» del mismo año, el AUC
fuera de fold es 0,658 (D-25, lectura (a)); por año, 0,598, 0,759 y 0,749. Dentro de un mismo mes,
más cerca de una campaña, el orden es más pobre todavía. Esa cifra y el recall repartiendo el 20 %
dentro de cada año tienen sus macros pendientes en `numeros.md`: no se dicen de memoria, y la vara
de ese recall es el 0,200 del azar, no el 0,629. El test es la misma mezcla (D-03): tendrá el mismo
sesgo.

**28. ¿No es sólo que las variables económicas delatan la fecha?** [respaldo R4 · Andrés] — No:
hacia adelante, sin el bloque macro Random Forest da 0,556, y sin el bloque macro ni `month`, 0,547,
casi lo mismo que con todas, 0,558. Barajado, en validación, sacarlas sí cuesta: quedan en 0,772 y
0,736, es decir, pierden 0,023 y 0,059. La relación entre las variables y el «yes» cambia con el
tiempo (inferencia).

**29. ¿Qué harían distinto con más tiempo?** [slide 17 · Sebastián] — Por impacto: validar hacia
adelante como esquema principal, que mide una campaña futura; más historial del cliente, porque el
86,3 % de train no tiene campaña previa; un costo por tipo de error en lugar del 20 % fijo (Clase 4,
slides 41–44); y calibrar las probabilidades para poder usar ese costo (inferencia).

**30. ¿Y si el test da bastante distinto que validación?** [slide 14 · Sebastián] — Depende de
cuánto. Se lee contra el desvío entre folds de validación, 0,006, y contra el intervalo del 95 % del
test. Una diferencia dentro de ± 0,006 es ruido, en cualquier sentido: es lo que ya varía el AUC de
un fold a otro. Si da algo más bajo, puede sumarse el optimismo que declaramos en la slide 17, que no
medimos (pregunta 11); difícilmente explique una diferencia grande (inferencia). Una
diferencia mayor que un desvío obliga a revisar la partición o el pipeline, buscando qué separa al
test de train, y se dice tal cual. En ningún caso se cambia el modelo después de abrirlo: lo que se
promete es el número de test. Y en ningún caso se lo compara con «sin modelo».

### Las consultas a la cátedra

**31. ¿Qué le consultaron a la cátedra, y qué pasa si responde distinto?** [— · los dos] — Seis
consultas, cada una con un valor por defecto para no bloquear el trabajo (`plan/consultas.md`): si
$F_1$ cuenta entre las métricas vistas; si con `duration` alcanza con excluirla, y si `month`,
`day_of_week`, `contact` y `campaign` se consideran disponibles antes de llamar; si esperan que se
trate el desbalance; qué Naive Bayes esperan con datos mixtos; si la partición debe ser aleatoria o
temporal; y qué día defendemos. Desde el principio se guardaron las cinco métricas candidatas y las
dos variantes de Naive Bayes, así que una respuesta distinta cambia qué columna se lee, no qué se
vuelve a correr. Las que más moverían el trabajo son la de las variables del último contacto —la
variante sin ellas está anotada como pendiente— y la de la partición. **Antes de la defensa,
actualizar esta respuesta con lo que la cátedra haya contestado.**

---

## Reloj de ensayo

| Slide | Título corto | Empieza | Dura | `[→]` | Bloque | Quién |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Portada | 0:00 | 0:05 | 0 | marco | Andrés |
| 2 | El problema | 0:05 | 0:30 | 2 | marco | Andrés |
| 3 | Qué datos deciden | 0:35 | 0:35 | 3 | teoría | Andrés |
| 4 | `duration`, fuera | 1:10 | 0:25 | 1 | **EDA** | Sebastián |
| 5 | El archivo va por fecha | 1:35 | 0:30 | 1 | **EDA** | Sebastián |
| 6 | El 999 de `pdays` | 2:05 | 0:25 | 1 | **EDA** | Sebastián |
| 7 | El pipeline y el IQR | 2:30 | 0:35 | 4 | **EDA** | Sebastián |
| 8 | Dos métricas | 3:05 | 0:30 | 2 | teoría | Andrés |
| 9 | Modelos sin ajustar | 3:35 | 0:30 | 0 | **resultados** | Andrés |
| 10 | Curva de Random Forest | 4:05 | 0:35 | 2 | **resultados** | Andrés |
| 11 | Curva de KNN | 4:40 | 0:25 | 0 | **resultados** | Andrés |
| 12 | Curva de la SVM | 5:05 | 0:20 | 1 | **resultados** | Andrés |
| 13 | Modelo final | 5:25 | 0:25 | 0 | **resultados** | Sebastián |
| 14 | El test | 5:50 | 0:30 | 0 | **resultados** | Sebastián |
| 15 | Los errores | 6:20 | 0:25 | 2 | **resultados** | Sebastián |
| 16 | Por qué rindió cada uno | 6:45 | 0:40 | 3 | **resultados** | Sebastián |
| 17 | Limitaciones y mejoras | 7:25 | 0:35 | 1 | **resultados** | Sebastián |
| 18 | Hallazgo: la caída | 8:00 | 0:30 | 1 | **hallazgo** | Andrés |
| 19 | Hallazgo: bloque a bloque | 8:30 | 0:25 | 1 | **hallazgo** | Andrés |
| 20 | ¿Preguntas? | 8:55 | 0:05 | 0 | marco | Andrés |

**Puntos de control al ensayar.**

- Al terminar la **slide 3** tienen que haber pasado **1:10**: la regla de juego está dicha y
  todavía no se mostró ningún resultado.
- Al terminar la **slide 7** tienen que haber pasado **3:05**. Es el control más importante: el EDA
  está cerrado con sus cuatro consecuencias, y todo lo anterior es marco, teoría y datos. Si pasaron
  más de 3:20, se aplican ya los recortes 3 y 4, que caen en las slides 9 y 11, y se da por hecho el
  recorte 1. Nunca se recorta en la 10.
- Al terminar la **slide 13** tienen que haber pasado **5:50**: el modelo está elegido y lo que queda
  del punto 4 es el test. Si pasaron más de 6:05, se aplica el recorte 1; si más de 6:20, también el 2.
- Al terminar la **slide 17** tienen que haber pasado **8:00**: todo lo calificable está dicho (gate
  G5). Lo que queda es el hallazgo, que es lo mejor del trabajo pero no un punto de la consigna. Si
  pasaron más de 8:15, el hallazgo va con los recortes 1 y 2.

**Orden de recorte, si hay que recortar en vivo.** Es un orden de prioridad: lo primero que se
sacrifica es lo último que se dice.

1. **Primero, el segundo paso del hallazgo.** En la slide 19, el segundo overlay se reduce a una
   frase: «Barajado, en esas filas tampoco pasa de 0,584; en 2009–2010 pierde, pero no cae al azar.
   Por eso el test vale para estas campañas.» De 46 a 24 palabras: unos 8 segundos.
2. **Después, el segundo overlay de la 18:** «Parte del 0,795 es distinguir años: entre años
   distintos el AUC es 0,864; en los del mismo año, 0,658.» Los dos números están en el veredicto
   de la pantalla. De 37 a 19 palabras: unos 7 segundos.
3. **En la slide 9, sin el porqué del categórico:** «El mejor es Naive Bayes, en su versión
   categórica.» El número queda en pantalla y la razón, en las aclaraciones. Unos 7 segundos.
4. **En la slide 11, sin la ponderación por distancia:** se omite «Ponderar por distancia queda peor
   en todo el rango»; la curva punteada queda en pantalla, y la explicación, en las aclaraciones y en
   la pregunta 20. Unos 3 segundos.

**No se recortan nunca:** las limitaciones (slide 17); la slide 3, qué datos deciden (el consejo
«Importante»); la 10, sobreajuste y subajuste (pedido por el punto 3.1); la 14, el número de test; y
la 15 y la 16, los errores y los supuestos (punto 5). El hallazgo se acorta, pero no se omite: la
última slide de contenido tiene que ser una figura (guía A6), y con los recortes 1 y 2 dura de todos
modos unos 40 segundos.

**La suma (gate G6).** Las palabras habladas se cuentan como tokens separados por espacios, sin las
marcas `[→]` ni los signos sueltos, y cada «?» del test cuenta como una palabra; en las slides 14 y
15 cuenta la variante más larga. Aplicada al guion del TP1, esta cuenta da 1 606 palabras, 26 menos
(el 1,6 %) que las 1 632 que declaran sus 22 encabezados: siete coinciden y los otros quince declaran
entre 1 y 4 palabras de más. Contado con esta regla, el TP1 dice 1 606 palabras en los 10:00 de su
reloj, 2,68 por segundo: redondeado, el mismo 2,7.

- **1 457 palabras / 2,7 = 539,6 s = 9:00 ≤ 9:45.** Pasa, con 45 segundos de margen. Con las
  ventanas redondeadas a 5 segundos, el reloj termina en 9:00, 15 segundos antes que el esqueleto del
  plan. A 2,68 por segundo, 9:04.
- **Con los números leídos completos: entre 9:21 y 9:26.** Los números de este guion tardan más en
  decirse que los del TP1, porque muchos son AUC con tres decimales. Se expandió cada número escrito
  con cifras a las palabras con que se lee («0,795» son seis: «cero coma setecientos noventa y
  cinco»), y cada «?» del test, como un número de tres decimales: cada palabra de este guion equivale
  a 1,19 palabras leídas, y cada una del TP1, a 1,15. A la velocidad de lectura que 2,7 por segundo
  le supone al TP1, este guion dura 9:21; si el TP1 se dijo en los 10:00 de su reloj, 9:26, 19
  segundos por debajo del tope. Hay 84 números hablados, el 5,8 % de las palabras (en el TP1, 93,
  también el 5,8 %).
- **Lo que decide son los dos ensayos cronometrados** (paso 8.4 del plan, gate G6): tienen que
  durar 10:00 o menos. Si un ensayo pasa de 9:45, se aplica el orden de recorte de arriba, en ese
  orden.

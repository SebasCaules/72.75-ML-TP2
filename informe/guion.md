# Guion de la defensa — TP2 · Grupo 7 · 07/10/2026 (o 14/10) · 10 + 10 minutos

Guion hablado de la presentación (`presentacion.pdf`: 20 slides), slide por slide,
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
primer bloque en la 19, que tampoco se usan al responder (N0-15). Todo lo que se agregó respecto
de las notas del deck se pagó con un recorte: el propósito del IQR en la slide 7 (la crítica del
TP1) y las declaraciones de conjunto salieron de no leer en voz alta números que ya están anotados
en pantalla. La slide 16 sigue las fichas corregidas del deck —un supuesto y un efecto medido por
modelo, sin la U de la edad como causa— y conserva su ventana: la pérdida de Random Forest sin las
variables de época ocupa el lugar de la U, y la ficha de KNN se dice casi textual.

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
  `informe/numeros.md`. Las cifras que la pantalla no muestra —las brechas de la slide 9, cuánto
  suman los pesos en la 12— no se dicen, y al responder no se usa nada que no esté en las 20 slides
  (N0-15). La charla sí afirma algunas cosas que no se ven, como que los pesos ayudan en la 12 o la
  primera exploración de la 3: si el tribunal repregunta por ellas, la respuesta preparada está en
  «Lo que se dice y no se ve», al final del banco.
- Lo que el enunciado pide decir en la presentación: **sobreajuste y subajuste en una curva** (punto
  3.1, p. 2: slide 10), y los métodos **no se describen**, pero los resultados se discuten sabiendo
  cómo funcionan (punto 2.1, p. 2: slide 16 y el banco de preguntas).

## Convenciones

- `[→]` = avanzar un overlay (una pulsación). La cantidad de `[→]` de cada slide coincide con los
  overlays del PDF (25 en total, comprobado contra sus 45 páginas); el último estado queda en
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
  (pendiente). Los números de las aclaraciones y de las preguntas siguen la misma regla y, además,
  tienen que verse en la slide que citan.
- **Las preguntas se contestan sólo con las 20 slides (N0-15).** Cada aclaración, cada pregunta del
  banco y cada «Si preguntan» de las notas del deck usa sólo lo que se ve en la slide que cita: su
  texto, sus figuras y los valores, ejes, leyendas y tarjetas anotados, en cualquiera de sus pasos
  (y, en las slides 14 y 15, también en su variante con el test evaluado). Se permite explicar la
  teoría de lo que se ve, con las citas de la cátedra de abajo, y razonar sin agregar resultados.
  No se usa ningún número, resultado ni hecho del trabajo que no esté en pantalla, ni un número
  derivado que no esté impreso (se nombran los dos valores que se ven), ni se remite a archivos del
  repositorio como respaldo de una respuesta. No hay slides de respaldo.
- **El número de test no se compara con «sin modelo» ni con otro modelo.** La única comparación
  permitida es contra validación, hablada y nunca proyectada, como chequeo de coherencia (guía C6);
  mientras el test siga cerrado, esa comparación vale «?».
- **Cada afirmación teórica cita la teórica** con clase y slide, y sólo con las citas que ya están en
  `DECISIONES.md` o en `plan/PLAN.md`: Clase 2, slides 88–89; Clase 3, slides 98–100 (el sobreajuste
  a la validación y sus cuatro defensas, en D-22); Clase 4, slides 26–27, 30–31, 41–44, 45–51 y 52;
  Clase 6, slide 46; Clase 7, slides 50 y 61; Clase 8, slides 48–49 y 54; Mitchell §8.2.1 y
  Alpaydin §8.3, p. 192. Lo que no sale de la cátedra —la regla de un error estándar, por ejemplo—
  está marcado como tal, y lo que es inferencia nuestra dice «inferencia».

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
- **El 41 188 de la portada es el archivo entero:** la slide 3 lo muestra como «las 41 188 del
  archivo», antes de quitar los 12 duplicados exactos. Desde la slide 2, cada número dice de qué
  conjunto sale, con la tabla de la slide 3: train, validación o test.
- **El dataset es el del enunciado** (TP2, p. 1): el *Bank Marketing*, contactos de campañas
  telefónicas de un banco, de 2008 a 2010, como dice la portada.

### 2 · Decir siempre «no» ya acierta el 88,7 % — 0:05 → 0:35 (~87 palabras)

**Habla: Andrés.**

> Un banco ofrece un plazo fijo por teléfono. Antes de marcar, ¿este cliente va a contratar? Si lo
> sabemos, llamamos primero a los más probables.
> **[→]** El modelo sólo puede usar lo que el banco sabe antes de la llamada: el cliente, su historial
> en campañas anteriores, el contexto económico y los datos de la llamada que va a hacer.
> **[→]** Y la clase está muy desbalanceada: en train, sólo el 11,3 % dijo que sí. Decir siempre «no»
> ya acierta el 88,7 %, así que aquí la exactitud no sirve.

#### A aclarar
- **Cuántos son:** 32 940 filas de train (slide 3), con 11,3 % de «yes» y 88,7 % de «no» (la barra
  de la figura). Decir siempre «no» acierta el 88,7 %: la exactitud no distingue un modelo de nada.
- **¿Qué variable no entra?** `duration`: se conoce al colgar, cuando el resultado ya se sabe (slide
  4; D-05), y el pipeline la saca en las reglas fijas (slide 7, «sin duration»). Las demás son las de
  las cuatro cajas: lo que el banco sabe antes de marcar.
- **¿`campaign` existe antes de llamar?** Sí: es el «n.º de llamada» de la cuarta caja, la llamada que
  se va a hacer, contando esa. La reserva es una inferencia: si cada fila guarda el total de la
  campaña y, tras un «yes», no se vuelve a llamar, su valor final depende en parte del resultado. El
  efecto no se midió (D-07).
- **¿`month`, `day_of_week` y `contact`?** Son el mes, el día y el canal de la llamada que se va a
  hacer (la cuarta caja: «canal, mes, día»), y se eligen antes de marcar (inferencia).
- **Por qué el modelo ordena en lugar de clasificar con un umbral.** El enunciado habla de «priorizar
  recursos de marketing» (TP2, p. 1): lo que el banco necesita es a quién llamar primero, y por eso
  las dos métricas de la slide 8 miden el orden.
- **La exactitud engaña con clases desbalanceadas:** modelos con la misma exactitud cometen errores
  muy distintos (Clase 4, slides 26–27; D-19).

### 3 · Train decide; test sólo mide, y una sola vez — 0:35 → 1:10 (~92 palabras)

**Habla: Andrés.** Es la regla de juego antes del primer número (guía A3) y el consejo «Importante»
del enunciado. Se dice completa pero rápido: son definiciones que el tribunal ya conoce. El segundo
paso cuenta, sin esconderla, la primera exploración sobre el archivo completo (D-04).

> Primero, quitamos 12 duplicados exactos: si una copia cae en train y otra en test, el test deja de
> ser independiente. Después, 80/20 estratificado por la clase.
> **[→]** El análisis exploratorio que decide y las ablaciones leen sólo train: una primera
> exploración usó el archivo completo, y la rehicimos sobre train antes de decidir nada.
> **[→]** La validación cruzada de 5 folds, también: con ella elegimos modelo e hiperparámetros.
> **[→]** Y el test no decide nada: se abre una vez, al final, para estimar el desempeño. Por eso cada
> número dice de qué conjunto sale.

#### A aclarar
- **Las cifras de la partición**, en la barra: 41 176 filas sin los 12 duplicados exactos; train
  32 940 filas (80 %) y test 8 236 (20 %), estratificado por `y`.
- **¿Por qué estratificar, si no se vio en clase?** Con 11,3 % de «yes» (slide 2), una partición al
  azar sin estratificar puede dejar al test con otra proporción de «yes» sólo por la semilla, y el
  test se parecería menos a train por azar. Estratificar por `y` mantiene la misma proporción en los
  dos, y no cuesta nada.
- **¿Cómo se garantiza que nada mire el test?** Por diseño, y la tabla lo fija: el análisis
  exploratorio, las ablaciones y la validación cruzada leen sólo train; el test se usa una vez, al
  final, y no decide nada. Lo que se aprende de los datos se ajusta dentro de cada fold, sólo con el
  de entrenamiento (slide 7; pregunta 2).
- **Cada vuelta de la validación cruzada**, dentro de train, valida con uno de los 5 folds y ajusta
  con los otros cuatro.
- **Qué se evalúa en test:** el modelo elegido, Random Forest (slide 13), reentrenado con todo train.
  El reentrenamiento no está en pantalla: lo dice la charla de la slide 14, y es el paso que pide la
  cátedra una vez elegido el modelo (Clase 2, slides 88–89).
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
- **Con qué se midió:** los cuatro modelos de la figura, con y sin `duration`, en AUC de validación,
  media de 5 folds (el eje). Los saltos, anotados: Random Forest de 0,770 a 0,939, KNN de 0,755 a
  0,911, SVM de 0,701 a 0,907 y Naive Bayes gaussiano de 0,768 a 0,831.
- **¿Por qué es fuga?** La duración de la llamada se conoce al colgar, cuando el «yes» ya se sabe, y
  el modelo decide antes de marcar (slide 2): ahí no existe. Por eso la leyenda la llama techo (D-05).
- **¿Por qué el AUC antes de la slide de métricas?** Porque ya es la métrica que decide: aquí se
  nombra, con su «sin modelo» en 0,5 (la línea punteada), y en la slide 8 se justifica junto con el
  recall.
- **¿Y las otras variables?** Son las de las cuatro cajas de la slide 2, que el banco conoce antes de
  marcar; `campaign`, el «n.º de llamada», con la reserva de la slide 2 (D-07).
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
- **Lo que muestra la figura, en train:** la tasa de «yes» por bloques de 2 000 filas, en el orden del
  archivo. En 2008 queda baja y pareja, siempre por debajo del 20 % del eje; la subida es de 2009 y
  2010, hasta 50,9 %. La tasa por año no está en la figura.
- **El euribor**, en el panel de abajo: por encima de 4 durante casi todo 2008 y por debajo de 2
  desde el cambio a 2009. El contexto económico fecha las filas.
- **¿De dónde sale el año?** No viene en los datos, y por eso la figura dice «inferido». Se deduce de
  lo que se ve: el archivo va por fecha (el eje) y el mes sí es una variable (slide 2: «canal, mes,
  día»), así que cada vez que el mes retrocede empieza un año nuevo (inferencia desde el orden).
- **¿Por qué no partir por fecha?** El enunciado plantea k-fold y no menciona el tiempo, y con el
  último 20 % del archivo como test (la proporción de la slide 3) se entrenaría en una época y se
  evaluaría en otra: train quedaría casi entero en la época de tasa baja y test, en el final, donde la
  tasa sube (D-03).
- **¿Qué pasaba sin barajar?** Los folds se armarían en el orden del archivo, y cada uno quedaría en
  una época de la figura. Barajados, «cada fold mezcla las épocas» (slide 7; D-06).

### 6 · 999 no significa «nunca contactado» — 2:05 → 2:30 (~67 palabras)

**Habla: Sebastián.**

> Tercera: `pdays`. El diccionario dice que 999 es «no contactado antes», y es el 96,3 % de train.
> Pero 3 302 filas con 999 sí tuvieron contactos previos, y los que tienen días registrados contratan
> el 64,6 %.
> **[→]** En validación, sacarla empeora a Naive Bayes y a KNN, y en los otros dos no cambia nada. Se
> queda: escalado o en un árbol, el 999 funciona como un corte.

#### A aclarar
- **Las tres barras, en train**, cada una con su tasa de «yes» anotada: los nunca contactados (999 y
  `previous` = 0), los 3 302 con 999 pero con contactos previos, y los 1 207 con días registrados
  (`pdays` < 999), que contratan el 64,6 %. Las tasas de las dos primeras barras se señalan en
  pantalla, no se dictan.
- **Por qué 999 no es «nunca contactado»:** la barra del medio son filas con 999 y `previous` ≥ 1,
  es decir, contactadas antes, y su tasa queda por encima de la de los nunca contactados.
- **¿Por qué se queda?** Porque separa grupos que contratan muy distinto. Escalado o en un árbol, el
  999 funciona como un corte, que es lo que dice el veredicto: el árbol lo aísla en una rama, y
  escalado queda como un extremo aparte (D-10).
- **¿Y una indicadora de contacto previo?** Ya la da el historial, que entra al modelo (slide 2:
  «campañas anteriores y su resultado»), y `previous`, que separa los dos grupos del 999, cuenta los
  contactos previos. Duplicarla pesaría doble en Naive Bayes y en las distancias de KNN.
- **¿Y los atípicos?** Con casi todas las filas en 999 (las dos primeras barras), el rango
  intercuartílico de `pdays` es cero: las 1 207 filas con días registrados son justo sus atípicos, y
  son las que más contratan. Por eso no se borran (slide 7).

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
- **¿Qué hacen las reglas fijas?** Son las mismas en todos los folds y no aprenden nada de los datos:
  sacan `duration` (slide 4) y funden categorías; las columnas pasan de 62 a 58 (la caja).
- **¿Qué codificación lleva cada modelo?** «Según el modelo», dice la caja. KNN y la SVM miden
  distancias y necesitan escalar (Clase 8, slide 54; Alpaydin §8.3, p. 192); un árbol corta cada
  variable por separado, así que a Random Forest la escala no le cambia nada (D-12); y el Naive Bayes
  categórico trabaja con valores discretos, así que las numéricas se discretizan (D-18). Para los
  ceros del categórico, la corrección de Laplace: pregunta 13.
- **¿Cómo se decidió cada paso?** Con ablaciones sobre train (slide 3: «EDA y ablaciones»): cada una
  compara el modelo con y sin una decisión, como la de `duration` en la slide 4, con el AUC de
  validación, media de 5 folds (su eje).
- **Los atípicos** (D-17): la regla del IQR marca 6 731 filas de train con algún valor alejado, y no
  se borra ninguna. Entre los de `pdays`, que son las 1 207 filas con días registrados, el «yes» es
  64,6 % (slide 6), contra 11,3 % en todo train (slide 2): borrarlos sacaría a los que más contratan.
  Sirven para diagnosticar, no para filtrar.
- **¿Y KNN sin escalar?** Sin escalar, en train, `pdays` es el 86,6 % de la varianza (slide 16): la
  distancia mediría casi sólo el historial de contacto, no clientes parecidos (pregunta 16).
- **¿Por qué barajar los folds?** Porque el archivo va por fecha (slide 5): sin barajar, cada fold
  quedaría en una época; barajados, «cada fold mezcla las épocas».
- **Sin fuga por construcción:** lo que se aprende de los datos —escalas, cortes— se ajusta dentro de
  cada fold, sólo con el de entrenamiento (la caja punteada), que es lo que pide el enunciado (TP2,
  p. 1).

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
- **Lo que el AUC no hace:** no premia decir siempre «no» (un puntaje constante da 0,500) y no
  depende del umbral. Por eso sirve para ordenar a quién llamar, que es priorizar recursos (TP2,
  p. 1).
- **«Sin modelo»** es un puntaje igual para todos: no ordena, así que su AUC es 0,500, y llamar al
  20 % de la lista al azar alcanza al 20 % de los «yes»: recall 0,200. Es el mismo rótulo en todas
  las slides que comparan con una línea de base; la slide 14 no compara, y la slide 15 tampoco
  cuando muestra el test.
- **¿Por qué no el umbral de la Clase 4, el punto de la ROC más cercano a la esquina superior
  izquierda** (slide 52)? Es simétrico entre los dos errores, y aquí no cuestan lo mismo; y con un
  umbral por modelo los recalls no se comparan. Con el 20 % de la lista, todos los modelos hacen las
  mismas llamadas (D-20).
- **¿Por qué el 20 %?** Es un supuesto de negocio, declarado como limitación (slide 17), y la mejora
  propuesta es un costo por tipo de error en lugar del 20 % fijo (pregunta 8).
- **¿Y $F_1$ o la precisión?** Con el 20 % fijo, las llamadas ya están dadas: precisión, recall y
  $F_1$ crecen los tres con los «yes» alcanzados, así que ordenan igual a los modelos (cuenta nuestra,
  con las definiciones de la Clase 4, slides 30–31). Sin modelo, la precisión es la proporción de
  «yes», el 11,3 % de train (slide 2).
- **¿Y el desbalance?** No se remuestrea: el pipeline de la slide 7 no tiene ese paso, y estas dos
  métricas no premian el «no» (D-21; pregunta 9).

### 9 · Sin ajustar, gana Naive Bayes: 0,782 contra 0,500 sin modelo — 3:35 → 4:05 (~74 palabras)

**Habla: Andrés.** Son tres los que sobreajustan: también KNN con k = 15, que es el puente a la
slide 11, donde sobreajusta el k pequeño.

> Primero, los modelos con hiperparámetros de referencia: en AUC de validación, todos quedan muy por
> encima de sin modelo. El mejor es Naive Bayes en su versión categórica: discretizar las numéricas
> en deciles le suma 0,013 sobre el gaussiano, porque las numéricas están lejos de ser normales. Y
> tres sobreajustan: Random Forest sin límite de profundidad, la SVM con RBF y KNN con k = 15. Lo
> muestran, y lo corrigen, las curvas que siguen.

#### A aclarar
- **Qué es «sin ajustar»:** cada modelo con valores de referencia de sus hiperparámetros, antes de
  las curvas de las slides 10–12, que eligen uno por modelo.
- **Categórico contra gaussiano** (D-18): 0,782 contra 0,769 de AUC de validación. El gaussiano supone
  una normal por clase, y las numéricas están lejos de serlo: `pdays`, por ejemplo, es casi binaria,
  con casi todas las filas en 999 (slide 6). El categórico discretiza las numéricas y no supone
  ninguna forma (Clase 6, slide 46; pregunta 13).
- **¿Cuánto sobreajustan?** La figura sólo muestra validación; la brecha con train se ve en las
  curvas de las slides 10–12: Random Forest sin límite de profundidad, con train en el tope del eje;
  KNN en k = 15, una de las marcas del eje de la slide 11, con train bien por encima de validación;
  la SVM con RBF y un C grande.
- **¿Por qué Random Forest da 0,770 en la slide 4 y 0,772 aquí?** Son dos mediciones de validación
  distintas. La de la slide 4 es del análisis exploratorio (su sección en la barra es «EDA», y en la
  slide 3 el EDA decide «qué variables entran, y cómo»); la de la 9 viene después del pipeline de la
  slide 7. Que la primera sea anterior a ese pipeline completo es inferencia por el orden de las
  slides, y qué cambió exactamente entre una y otra no está en pantalla. Lo mismo la SVM, 0,701 y
  0,702, y el Naive Bayes gaussiano, 0,768 y 0,769; KNN da 0,755 en las dos.
- **«Muy por encima» de sin modelo:** el mejor, 0,782, y el peor, la SVM, 0,702, contra 0,500 (el
  título).
- **Las barras** son ± 1 desvío entre los 5 folds (el eje).

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
- **¿Por qué 8 y no 10, si ahí la validación es un poco más alta?** Por la regla que dice la charla:
  no se elige el máximo, sino el más simple de los que quedan dentro de un error estándar de él
  (D-22). Cerca del máximo la curva es casi plana, y 8 queda dentro.
- **¿Y por qué no 6, que se ve casi igual?** Si 6 hubiera quedado dentro de ese umbral, la regla lo
  habría elegido, porque es más simple. La estrella en 8 dice que quedó fuera, por poco: en la
  figura, 6 está apenas por debajo de 8. Con la banda no se puede ver dónde pasa el umbral (la viñeta
  siguiente).
- **La banda no es el error estándar:** es ± 1 desvío entre los 5 folds; el error estándar es ese
  desvío dividido por la raíz de 5, más angosto, y el umbral de la regla no está dibujado.
- **¿De dónde sale la regla?** No está en las slides de la cátedra. La Clase 3 advierte que elegir el
  mejor entre muchas estimaciones con ruido también elige ruido favorable (slides 98–100), y la
  regla lo atenúa porque no elige el máximo (D-22). Viene de los árboles CART de Breiman y otros, y
  está en Hastie, Tibshirani y Friedman, *The Elements of Statistical Learning*, §7.10.1: fuera de la
  bibliografía de la materia.
- **El protocolo** es el de la Clase 7, slide 50: el AUC en train y en validación con los 5 folds, la
  brecha entre los dos y la curva.
- **La brecha** es la distancia entre train y validación en la misma profundidad: con 8, 0,030
  (anotada). A la derecha se abre: train llega al tope del eje, 1,00, y la validación baja
  («sobreajuste →»).
- **El subajuste:** con profundidad 2, train y validación casi coinciden, y los dos quedan por debajo
  del 0,795 de la profundidad 8: el modelo es demasiado simple («← subajuste»).
- **¿Y el número de árboles?** No es un eje de complejidad: más árboles no cambian el sesgo y sólo
  bajan la varianza (Clase 7, slide 61; pregunta 19).

### 11 · En KNN sobreajusta el k pequeño: se elige k = 801 — 4:40 → 5:05 (~62 palabras)

**Habla: Andrés.**

> En KNN el eje va al revés. Con un vecino, train da 0,988 y validación 0,619: cada cliente es su
> propio vecino, sobreajuste puro. Al sumar vecinos la brecha se cierra, y desde 151 todo queda dentro
> de un error estándar del mejor; tomamos el más suave, 801, con una brecha de 0,006. Ponderar por
> distancia queda peor en todo el rango.

#### A aclarar
- **El elegido, en validación:** k = 801, con AUC 0,784 y brecha con train 0,006 (anotados).
- **¿Dónde está el máximo?** La validación forma una meseta en los k grandes: desde 151 casi no
  cambia. Entre los que empatan con el mejor, la regla de un error estándar de la slide 10 toma el
  más suave, el de más vecinos.
- **¿No es enorme k = 801?** Cada fold entrena con cuatro quintos de las 32 940 filas de train (slide
  3), y el «yes» es el 11,3 % (slide 2): estimar bien una proporción baja pide muchos vecinos
  (inferencia). Y está en el borde de la grilla: pregunta 18.
- **¿Y ponderar por distancia?** (Mitchell §8.2.1). Es la línea punteada: en train queda en 1,0,
  porque cada punto de train se encuentra a sí mismo a distancia cero; en validación, con más de un
  vecino, queda por debajo de la uniforme en todo el rango (con uno, las dos coinciden).
- **El eje es logarítmico** porque la grilla va de 1 a 801.
- **¿Por qué train no llega a 1,0 con un vecino?** Si cada fila de train fuera única, con un vecino se
  encontraría a sí misma y train daría 1,0. Que quede apenas por debajo indica filas con las mismas
  predictoras y distinta respuesta, que empatan a distancia cero (inferencia; pregunta 7).
- **Escalar no es opcional en KNN:** mide distancias (Clase 8, slide 54; Alpaydin §8.3, p. 192).

### 12 · La SVM queda lineal y muy regularizada: C = 0,001 — 5:05 → 5:25 (~62 palabras)

**Habla: Andrés.**

> En la SVM exploramos C y el kernel, con pesos balanceados, que aquí sí ayudan. Con kernel lineal
> la curva de C es plana, sin sobreajuste, y tomamos 0,001, el más regularizado.
> **[→]** Con RBF, en cambio, un C grande sobreajusta: train sube casi a uno y la validación cae. Y
> ni en su mejor C el RBF supera al lineal: queda lineal.

#### A aclarar
- **¿Qué es C?** El costo de las violaciones del margen: un C grande castiga más y regulariza menos,
  como dice el eje (Clase 8, slide 49; pregunta 14).
- **El elegido:** C = 0,001, con AUC 0,774 de validación y brecha con train 0,002 (anotados). Con
  kernel lineal la curva es plana: 0,001 y 0,01 quedan a la misma altura, y se toma el más
  regularizado. Es el menor C del eje: pregunta 18.
- **El RBF sobreajusta con C grande:** la línea punteada de train sube casi a uno y la de validación
  cae («sobreajuste con RBF →»).
- **¿El RBF no mejora al lineal en ningún C?** Su mejor valor de validación, 0,772 en su mejor C
  (slide 16), no alcanza el 0,774 del lineal elegido.
- **¿Y γ?** La curva del RBF varía sólo C, con γ fijo (el eje). El kernel elegido es el lineal, que no
  tiene γ; la grilla conjunta de C y γ es la de la Clase 8, slides 48–49.

### 13 · Ajustados, gana Random Forest por 0,011 de AUC en validación — 5:25 → 5:50 (~60 palabras)

**Habla: Sebastián** (bloque 4, slides 13–17). Entra con el resultado de la selección.

> Con los hiperparámetros elegidos gana Random Forest: en validación, 0,795 de AUC y 0,629 de recall
> llamando al 20 %. Le siguen KNN, Naive Bayes y la SVM, en ese orden, y ninguno queda dentro de un
> error estándar de Random Forest. Pero importa la escala: entre el mejor y el peor hay 0,021; entre
> Random Forest y sin modelo, 0,295.

#### A aclarar
- **Los cuatro, en AUC de validación** (anotados en la figura): Random Forest 0,795, KNN 0,784, Naive
  Bayes 0,782 y SVM 0,774. En recall de validación, llamando al 20 % de la lista: 0,629, 0,617, 0,614
  y 0,614.
- **La escala:** los cuatro van de 0,774 a 0,795, y sin modelo es 0,5 (la línea punteada); Random
  Forest le saca 0,011 a KNN (el título).
- **¿Y si hubiera empate?** No lo hubo: ninguno quedó dentro de un error estándar de Random Forest
  (lo dice la charla, con la regla de la slide 10), y Random Forest es primero también en recall:
  0,629 contra 0,617 de KNN (anotados).
- **¿Sobreajusta?** Su brecha con train es 0,030 (slide 10, la profundidad 8).
- **¿Cuánto sumó ajustar?** En AUC de validación, contra la slide 9: Random Forest de 0,772 a 0,795,
  KNN de 0,755 a 0,784 y la SVM de 0,702 a 0,774. Naive Bayes no tiene curva en las slides 10–12:
  queda en 0,782.
- **¿Y la precisión?** Con el 20 % fijo ordena a los modelos igual que el recall (aclaraciones de la
  slide 8). Mientras el test siga cerrado, se ve además como su inversa en la slide 15: 2,8 llamadas
  por cada «yes» alcanzado, contra 8,9 sin modelo (validación). Con el test evaluado, la slide 15
  muestra en su lugar el recall de test, y esos dos valores ya no están en pantalla.

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
- **¿Por qué una sola vez?** Si el test se mira más de una vez y algo cambia en el medio, deja de ser
  independiente: pasa a decidir, y la tabla de la slide 3 dice que no decide nada. Por eso se evalúa
  una vez («test, una sola vez», en esa tabla), con todo lo demás cerrado (N0-1; pregunta 1).
  Mientras siga cerrado, la slide dice además cuándo: «después del 30/09».
- **Qué mide:** Random Forest reentrenado con todo train, sobre las 8 236 filas de test (slide 3):
  clientes nuevos de las mismas campañas, no de una campaña futura (slide 17). El reentrenamiento no
  está en pantalla: lo dice la charla, y es el paso de la Clase 2, slides 88–89.
- **El intervalo** (con el test evaluado, en la slide): cuánto variaría el AUC de test con otra
  muestra de clientes nuevos de la misma época; es la incertidumbre de medir sobre un solo test.
- **¿Por qué no aparece «sin modelo»?** Porque el test no compite con nada: «sólo estima el
  desempeño» (slide 3). Una línea de base al lado lo haría parecer una comparación que decide algo.
- **Cómo leer la diferencia con validación** (se dice, no se proyecta): se nombran los dos valores, el
  AUC de test y el 0,795 de validación (slide 13), y la distancia se lee contra la variación entre
  folds de validación (la barra de ± 1 desvío de la slide 18) y contra el intervalo de test. Dentro de
  esa variación es ruido, en cualquier sentido; si además da algo más bajo, es coherente con el
  optimismo de haber elegido los hiperparámetros con la misma validación (slide 17), que no medimos
  (pregunta 10). Una diferencia mayor ya no la explica el ruido: obliga a revisar la partición o el
  pipeline, y se dice (pregunta 27). En ningún caso se cambia el modelo después: lo que se promete es
  el número de test.

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
- **¿Qué es fuera de fold (OOF)?** (Con el test cerrado: la slide dice «validación (OOF)».) Cada
  fila de train puntuada por el modelo del fold en el que no entrenó: toda train, sin fuga.
- **La matriz de validación** (con el test cerrado): 2 329 «yes» alcanzados, 1 382 perdidos, 4 259
  llamadas de más y 24 970 bien descartados, llamando al 20 % de la lista completa, 6 588 llamadas.
- **¿Por qué la lista completa y no fold por fold?** (Con el test cerrado: la slide dice «la lista
  completa».) Los puntajes fuera de fold de los 5 folds se juntan en una sola lista y se corta su
  20 %: es el mismo corte que se aplica al test, el 20 % de su lista. Con el test evaluado, la slide
  corta el 20 % de la lista de test («llamando al 20 % de la lista»), y la pregunta no se plantea.
- **El costo por «yes», en validación** (con el test cerrado): 2,8 llamadas es
  (2 329 + 4 259) / 2 329, las llamadas sobre los «yes» alcanzados; sin modelo, 8,9 es la inversa de
  la proporción de «yes», el 11,3 % de train (slide 2). Con el test evaluado, la slide muestra en su
  lugar el recall de test: el costo por «yes» de test no está impreso y no se calcula en vivo; se
  nombran las llamadas y los «yes» alcanzados de la matriz.
- **¿Quiénes son los perdidos?** La slide no los desglosa, y se dice así. Lo que se ve lo hace
  esperable (inferencia): en 2008 el modelo ordena poco aun barajado, de 0,519 a 0,584 (slide 19), y
  el 86,3 % de train no tiene campaña previa (slide 17).
- **Por qué el falso negativo es el caro:** una llamada de más cuesta una llamada; un cliente perdido
  es un plazo fijo que no se contrata. Es el costo desigual de la Clase 4, slides 41–44.
- **Con el test evaluado**, la matriz es la de test, con el 20 % de su lista, y al lado su recall; no
  se compara con «sin modelo», como en la slide 14.

### 16 · ¿Por qué ganó Random Forest y quedó última la SVM? — 6:45 → 7:25 (~107 palabras)

**Habla: Sebastián.** Son los cuatro ejemplos del enunciado (TP2, p. 2): variables muy
correlacionadas y distribuciones no gaussianas (Naive Bayes), diferencias de escala (KNN, SVM) y
relaciones no lineales (RF, SVM). Como las fichas del deck, cada modelo lleva un supuesto y un
efecto medido, y ninguna frase da una causa del orden como probada: la U de la edad se mezcla con la
época, y no se midió qué pierde cada modelo sin la edad.

> ¿Por qué rindió cada uno? Por sus supuestos. Random Forest no supone escala ni forma, y es el que
> más pierde sin las variables de época: 0,059 en validación. Que gane por reconocer la época es
> inferencia.
> **[→]** KNN mide distancias: sin escalar, en train, `pdays` tiene el 86,6 % de la varianza.
> **[→]** Naive Bayes supone independencia, y en train el bloque macro la viola, con correlaciones de
> 0,91 a 0,97; como las numéricas no son normales, el categórico le gana al gaussiano.
> **[→]** La SVM quedó lineal: el RBF no la mejora. Por qué queda última no lo medimos; entre los
> cuatro hay sólo 0,021 de AUC en validación.

#### A aclarar
- **¿Qué pierde Random Forest sin las variables de época?** Sin el bloque macro ni `month`, 0,059 de
  AUC de validación (la tarjeta), y es el que más pierde de los cuatro (D-25). Que gane por reconocer
  la época es inferencia; lo retoma el hallazgo (slides 18–19). No medimos importancias de variables.
- **¿Por qué también `month`?** En un archivo ordenado por fecha (slide 5), el mes también fecha las
  filas (inferencia).
- **¿Por qué el bloque macro no arruina a Naive Bayes, si viola la independencia?** Tres de sus
  variables tienen, en train, correlaciones de 0,91 a 0,97: es casi la misma información tres veces.
  Eso lleva las probabilidades a los extremos, pero no necesariamente cambia el orden, que es lo que
  mide el AUC (inferencia; pregunta 12).
- **¿Por qué queda última la SVM?** No lo medimos. Lo que se ve: el RBF, que podría captar relaciones
  no lineales, no mejora al lineal, 0,772 de AUC de validación en su mejor C contra 0,774 (el pie de
  la slide); y la distancia con los demás es pequeña: los cuatro van de 0,774 a 0,795.
- **Las escalas:** sin escalar, en train, `pdays` sola es el 86,6 % de la varianza; la distancia de
  KNN miraría casi sólo esa variable. Por eso se escala dentro de cada fold (slide 7).
- **Las distribuciones no gaussianas:** `pdays` es casi binaria, con casi todas las filas en 999
  (slide 6); por eso el Naive Bayes del TP es el categórico, que le gana al gaussiano (slide 9).

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
- **¿Cuánto optimismo?** No lo medimos: haría falta una validación cruzada anidada (Clase 3, slides
  98–100; D-22), y aquí ese papel lo cumple el test, que se abre una vez (slide 3; pregunta 10). La
  regla de un error estándar lo atenúa, porque no elige el máximo: profundidad 8 y no 10 (slide 10).
- **¿Y dentro de una campaña?** El 0,795 y el 0,629 de validación (slide 13) ordenan una lista que
  mezcla las épocas; en los pares «yes»–«no» del mismo año, el AUC baja a 0,658 (slide 18; pregunta
  24).
- **¿Por qué validar hacia adelante es lo primero?** Porque es como se usaría el modelo: entrenado con
  el pasado, para una campaña futura. Lo que cambia se ve en el hallazgo (slides 18–19).
- **«Más historial»:** el 86,3 % de train no tiene campaña previa, y el historial separa mucho: con
  días registrados, el «yes» es 64,6 % (slide 6), contra 11,3 % en todo train (slide 2).
- **¿Y `duration`?** Con ella, Random Forest llega a 0,939 de AUC de validación, pero no existe antes
  de llamar (slide 4).
- **Otra mejora:** calibrar las probabilidades, porque un costo por tipo de error necesita
  probabilidades y hoy las dos métricas sólo miden el orden (slide 8; inferencia).

### 18 · Entrenado con el pasado, Random Forest cae de 0,795 a 0,558 — 8:00 → 8:30 (~77 palabras)

**Habla: Andrés** (bloque 5, slides 18–20). Entra enganchado con la última frase de Sebastián:
validar hacia adelante.

> El hallazgo. Validamos hacia adelante dentro de train: cada fold entrena con el pasado y valida con
> el bloque siguiente. Barajados, los cuatro están entre 0,774 y 0,795; hacia adelante, ninguno pasa
> de 0,598, y Random Forest cae a 0,558.
> **[→]** ¿Por qué tanto? Parte del 0,795 es distinguir años: en los puntajes fuera de fold, el 66,3 %
> de los pares «yes»–«no» son de años distintos, y ahí el AUC es 0,864; en los del mismo año, 0,658.

#### A aclarar
- **¿Cómo se validó hacia adelante?** Con train ordenado por fecha y partido en bloques: cada fold
  valida con un bloque y entrena con todos los anteriores (la leyenda de la slide 19), y son los cinco
  bloques de esa slide (D-25). Todo dentro de train: el test no participa (slide 3).
- **Los cuatro, en AUC de validación:** barajados, 0,795, 0,784, 0,782 y 0,774; hacia adelante,
  Random Forest 0,558, KNN 0,539, Naive Bayes 0,598 y SVM 0,541.
- **¿Por qué no Naive Bayes, si hacia adelante da 0,598?** El modelo se elige con la validación
  cruzada de la slide 3, barajada (slide 7), y ahí gana Random Forest (slide 13). Hacia adelante, las
  barras de ± 1 desvío de los cuatro se superponen (el eje dice «media ± 1 desvío entre folds»), así
  que no hay un ganador claro. Validar así es la primera mejora (slide 17).
- **¿Por qué la barra hacia adelante es tan ancha?** Porque el modelo sirve en dos bloques y en tres
  no (slide 19): los tres de 2008 dan de 0,426 a 0,500, y los de 2009–2010, 0,664 y 0,711. No es la
  tasa de «yes» de cada bloque, que va de 4,7 % a 35,2 %: el AUC no depende de ella, porque la ROC
  calcula cada tasa dentro de su clase (Clase 4, slides 45–51). Y no es sólo entrenar con poco
  pasado: 2008 cuesta ordenarlo aun barajado, en las mismas filas, de 0,519 a 0,584, contra 0,748 y
  0,762 en 2009–2010.
- **¿Y para elegir a quién llamar dentro de una campaña?** Es la pregunta 24.
- **La cuenta de los pares:** el AUC es la proporción de pares «yes»–«no» bien ordenados (Clase 4,
  slides 45–51). El veredicto separa los pares por año, con los puntajes barajados de validación:
  0,864 entre años distintos y 0,658 dentro del mismo año. No es entrenar en un año y validar en
  otro: eso lo mide la validación hacia adelante.
- **Por qué los pares entre años son fáciles:** la tasa de «yes» cambia mucho con la época, de 1,9 %
  a 50,9 % en los bloques de la slide 5; basta con reconocer la época para ordenar bien un «yes» de
  2010 frente a un «no» de 2008 (inferencia).

### 19 · En 2008 no sirve; en 2009–2010 pierde, pero no cae al azar — 8:30 → 8:55 (~71 palabras)

**Habla: Andrés.** Es lo último que se muestra (guía A6). Si el reloj va atrasado, el segundo paso
se reduce a la frase de recorte del «Reloj de ensayo».

> Bloque a bloque: en los tres bloques de 2008, hacia adelante no pasa de 0,5, y en el tercero,
> 0,426, el orden incluso se invierte.
> **[→]** Pero barajado, en esas mismas filas, tampoco pasa de 0,584: en 2008 el modelo no sirve ni
> mezclando épocas. En los dos bloques de 2009–2010, en cambio, pierde 0,084 y 0,051: una caída real,
> pero no al azar. Por eso el test vale para estas campañas.

#### A aclarar
- **Hacia adelante, bloque a bloque** (anotado en la figura): 0,489, 0,500, 0,426, 0,664 y 0,711.
- **Barajado, en las mismas filas** (el AUC fuera de fold de Random Forest en las filas de cada
  bloque): 0,536, 0,519, 0,584, 0,748 y 0,762. Bloque a bloque, hacia adelante contra barajado:
  0,489 y 0,536; 0,500 y 0,519; 0,426 y 0,584; 0,664 y 0,748; 0,711 y 0,762.
- **Los bloques** (las etiquetas del eje): mayo–julio, julio–agosto y agosto–noviembre de 2008;
  noviembre de 2008 a mayo de 2009; mayo de 2009 a noviembre de 2010. El «yes» de cada uno: 4,7 %,
  6,5 %, 5,6 %, 12,7 % y 35,2 %.
- **¿0,426 es azar?** Queda por debajo de 0,5, así que en ese bloque el orden sale invertido: un AUC
  por debajo de 0,5 ordena al revés, el «mal» clasificador de la Clase 4 (slides 45–51). Pero es un
  solo bloque y sin intervalo, así que no decimos a cuántos errores estándar queda de 0,5: es un
  indicio, no una prueba de que sea peor que el azar.
- **¿Es sólo que las variables macro fechan las filas?** En parte. Si el modelo sólo reconociera la
  época, frente a una nueva ordenaría como el azar, cerca de 0,5, y eso dan los dos primeros bloques
  de 2008, 0,489 y 0,500. En el tercero da 0,426, por debajo de sin modelo: el orden se invierte, y
  eso sugiere que la relación entre las variables y el «yes» cambia con el tiempo (inferencia). Es un
  solo bloque y sin intervalo: un indicio, no una prueba. Barajado, sacar el bloque macro y `month` sí
  le cuesta a Random Forest 0,059 (slide 16): reconocer la época ayuda cuando las épocas se mezclan.
- **¿Por qué no validar así desde el principio?** El enunciado plantea k-fold; un esquema temporal
  habría entrenado con la época de tasa baja para predecir la de tasa alta: los bloques de 2008 tienen
  de 4,7 % a 6,5 % de «yes», y el último, 35,2 % (D-25). Es la primera mejora (slide 17).
- **Qué significa para el banco:** el número de test valdrá para clientes nuevos de estas campañas.
  Para una campaña futura, la mejor referencia que tenemos es la validación hacia adelante de
  2009–2010, 0,664 y 0,711 (inferencia).

### 20 · ¿Preguntas? — 8:55 → 9:00 (~7 palabras)

**Habla: Andrés.**

> Gracias; quedamos a disposición para las preguntas.

#### A aclarar
- **Esta slide queda puesta durante las preguntas.** No hay slides de respaldo: cada pregunta se
  contesta volviendo a la slide que corresponde (el mapa de páginas está al principio del banco) y
  sólo con lo que ella muestra (N0-15).

---

## Los 10 minutos de preguntas

Respuestas cortas, con el dato al frente y su conjunto declarado, y sólo con lo que muestra la slide
a la que se vuelve (N0-15). Entre corchetes, a qué slide volver y quién responde primero: el que
presentó ese bloque. Las citas de clase son sólo las de «Convenciones». Si repreguntan por algo
que la charla dice y la pantalla no muestra, se dice que no está en pantalla y se contesta con lo que
sí se ve, sin ninguna cifra que no esté en la slide.

**Para volver a una slide.** El pie dice «n/20» y la página del PDF no coincide, por los overlays.
Página de la última variante de cada slide: 1 → 1 · 2 → 4 · 3 → 8 · 4 → 10 · 5 → 12 · 6 → 14 ·
7 → 19 · 8 → 22 · 9 → 23 · 10 → 26 · 11 → 27 · 12 → 29 · 13 → 30 · 14 → 31 · 15 → 34 · 16 → 38 ·
17 → 40 · 18 → 42 · 19 → 44 · 20 → 45.

### El test y los datos

**1. ¿Por qué no evaluaron el test todavía?** [slide 14 · Sebastián] — Porque se evalúa una sola vez
(slides 3 y 14), y una vez abierto ya no se puede cambiar nada sin que deje de ser independiente: se
abre con todo lo demás cerrado, después del 30/09 (N0-1). *Si en la defensa siguiera cerrado:* la
única estimación disponible es la de validación, 0,795 (slide 13), y sale algo optimista (slide 17);
cuánto, no lo medimos (pregunta 10).

**2. ¿Cómo garantizan que ninguna decisión miró el test?** [slide 3 · Andrés] — Por diseño, y la
tabla lo muestra: el análisis exploratorio, las ablaciones y la validación cruzada leen sólo train;
el test se usa una vez, al final, y no decide nada (D-04). Además, lo que se aprende de los datos
—escalas, cortes— se ajusta dentro de cada fold, sólo con el de entrenamiento (slide 7). *Mientras
siga cerrado*, la slide 14 lo muestra pendiente: se evalúa una vez, después del 30/09.

**3. ¿Por qué partición aleatoria, si el archivo está ordenado por fecha?** [slides 5 y 19 ·
Sebastián] — El enunciado plantea k-fold y no menciona el tiempo, y con el último 20 % del archivo
como test (la proporción de la slide 3) se entrenaría en una época y se evaluaría en otra: la tasa
de «yes» va de 1,9 % a 50,9 % a lo largo del archivo (slide 5; D-03). Lo que cuesta no está
escondido: está medido en las slides 18 y 19, y por eso la slide 14 dice «de la misma época».

**4. ¿Por qué estratificaron, si no se vio en clase?** [slide 3 · Andrés] — Porque con 11,3 % de
«yes» (slide 2), una partición al azar sin estratificar puede dejar al test con otra proporción de
«yes» sólo por la semilla. Estratificar por `y` mantiene la misma en train y en test, y no cuesta
nada (D-02).

**5. ¿Por qué `duration` fuera, si es la mejor variable?** [slide 4 · Sebastián] — Porque se conoce
al colgar, cuando `y` ya se sabe, y el modelo decide antes de llamar (slide 2; D-05). Con ella Random
Forest llega a 0,939 de AUC de validación, pero ese número no existe antes de llamar, que es cuando
opera el modelo (TP2, p. 1): la figura lo llama techo.

**6. ¿Hay otras variables con fuga?** [slide 2 · Andrés] — Las demás son lo que el banco sabe antes
de marcar: las cuatro cajas de la slide. La que tiene reserva es `campaign`, el «n.º de llamada»:
cuenta la llamada que se va a hacer, pero si cada fila guarda el total de la campaña y, tras un
«yes», no se vuelve a llamar, su valor final depende en parte del resultado (inferencia); el efecto
no se midió (D-07). El mes, el día y el canal son de la llamada que se va a hacer, y se eligen antes
de marcar (inferencia). Y el contexto económico delata la época (slide 5), pero eso es un problema
de robustez temporal, no de fuga: es el hallazgo (slides 18 y 19; D-25).

**7. ¿El mismo cliente puede estar en train y en test?** [slide 3 · Andrés] — No se puede descartar:
la partición es de filas, 80/20 estratificado, y sólo se quitaron los 12 duplicados exactos (D-01).
Si un cliente cae en train y en test, el test sale algo optimista; no lo podemos medir, y es una
limitación más. Filas con los mismos datos y distinta respuesta explicarían también que KNN con un
vecino no llegue a 1,0 en train (slide 11; inferencia).

### Decisiones discutibles

**8. ¿De dónde sale el 20 %?** [slides 8 y 17 · Andrés] — Es un supuesto de negocio, declarado como
limitación (slide 17; D-20): el banco llama a una parte fija de la lista, y con el 20 % todos los
modelos hacen las mismas llamadas, así que los recalls se comparan. La mejora que proponemos es un
costo por tipo de error en lugar del 20 % fijo. No usamos el umbral de la Clase 4, el punto de la
ROC más cercano a la esquina superior izquierda (slide 52), porque es simétrico entre los dos errores
y aquí no cuestan lo mismo (slides 41–44); y con un umbral por modelo los recalls no se comparan.

**9. ¿Por qué no trataron el desbalance, por ejemplo remuestreando?** [slide 8 · Andrés] — No
remuestreamos: el pipeline de la slide 7 no tiene ese paso, porque las dos métricas ya no premian
decir siempre «no» (D-19, D-21): un puntaje igual para todos da 0,500 de AUC y 0,200 de recall,
aunque acierte el 88,7 %. Con ellas, el desbalance no engaña a la comparación. Y una corrección que
sólo moviera el umbral no cambiaría el AUC, que no depende del umbral (Clase 4, slides 45–51).

**10. ¿No sobreajustaron la validación con tantas decisiones?** [slide 17 · Sebastián] — Algo, y está
declarado como limitación: «hiperparámetros elegidos con la misma validación». Es el sobreajuste a
la validación de la Clase 3, slides 98–100: elegir el mejor entre muchas estimaciones con ruido
también elige ruido favorable, así que el 0,795 de validación (slide 13) sale optimista, y no
medimos cuánto. De las cuatro defensas del slide 100 usamos tres: validación cruzada (slide 3);
búsquedas acotadas, una grilla corta por modelo, con la regla de un error estándar, que no elige el
máximo (slides 10–12; D-22); y un test intacto que se abre una vez (slide 3; D-04). La cuarta, la
validación cruzada anidada, no la corrimos: el test intacto ya cumple el papel de la evaluación
externa. Lo que se promete es el número de test.

**11. ¿Por qué no validaron hacia adelante como esquema principal?** [slide 19 · Andrés] — Porque el
enunciado plantea k-fold, y un esquema temporal habría entrenado con la época de tasa baja para
predecir la de tasa alta: los bloques de 2008 tienen de 4,7 % a 6,5 % de «yes», y el último, 35,2 %
(D-25). Lo medimos dentro de train como limitación, y es la primera mejora: es como se usaría el
modelo (slide 17).

### La teoría de los cuatro modelos

**12. ¿Qué supone Naive Bayes y cómo lo afectan estos datos?** [slides 9 y 16 · Sebastián] —
Independencia de las variables dada la clase. El bloque macro la viola: tres de sus variables
tienen, en train, correlaciones de 0,91 a 0,97 (slide 16), casi la misma información tres veces. Eso
lleva las probabilidades a los extremos, pero el AUC mide el orden (inferencia). El gaussiano supone
además una normal por clase, y `pdays` es casi binaria, con casi todas las filas en 999 (slide 6).
Por eso el del TP es el categórico, que discretiza las numéricas y le gana al gaussiano: 0,782
contra 0,769 de AUC de validación (slide 9; D-18). Si repreguntan dónde se nota la violación:
aclaraciones de la slide 16.

**13. ¿Para qué la corrección de Laplace?** [slides 7 y 9 · Sebastián] — En el Naive Bayes
categórico, un valor que en el fold de entrenamiento nunca aparece con «yes» tendría probabilidad 0
y anularía el producto, sin importar el resto de las variables. Laplace suma α a cada conteo:
(conteo + α) / (N + kα), con k los valores posibles del atributo (Clase 6, slide 46). Así ningún
valor queda con probabilidad cero, y la corrección pesa más cuantas menos filas tiene un valor.

**14. ¿Qué hace C en la SVM, y por qué quedó en 0,001?** [slide 12 · Andrés] — C es el costo de las
violaciones del margen: un C grande castiga más y regulariza menos, como dice el eje (Clase 8, slide
49). Con kernel lineal la curva es plana en validación: 0,001 y 0,01 quedan a la misma altura, y la
regla toma el más regularizado, 0,001, con 0,774 y brecha 0,002 (anotados). Con RBF, un C grande
lleva train casi a uno y la validación cae: sobreajuste.

**15. ¿Por qué lineal y no RBF? ¿Y γ?** [slides 12 y 16 · Andrés] — Porque el RBF no lo mejora: en su
mejor C da 0,772 de AUC de validación, contra 0,774 del lineal elegido (slide 16), y el lineal es más
simple. γ quedó fijo: la curva del RBF varía sólo C. Ajustarlo es la grilla conjunta de C y γ de la
Clase 8, slides 48–49; con γ fijo, el RBF no superó al lineal.

**16. ¿Por qué hay que escalar para KNN y la SVM, y no para Random Forest?** [slides 7 y 16 ·
Sebastián] — KNN y la SVM miden distancias, y sin escalar las decide la variable de mayor varianza
(Clase 8, slide 54; Alpaydin §8.3, p. 192): sin escalar, en train, `pdays` sola es el 86,6 % de la
varianza (slide 16), así que el vecino más cercano sería el de historial de contacto parecido, no el
cliente parecido. Un árbol corta cada variable por separado, así que la escala no le cambia nada
(D-12). Por eso la caja dice «codificar y escalar», «según el modelo» (slide 7).

**17. ¿No es enorme k = 801? ¿Y la ponderación por distancia?** [slide 11 · Andrés] — En KNN
sobreajusta el k pequeño: con un vecino, train queda cerca de 1,0 y validación apenas por encima de
0,6 (la figura). Al sumar vecinos la brecha se cierra, la validación forma una meseta en los k
grandes, y se toma el más suave: 801, con 0,784 de validación y brecha 0,006 (anotados). Con 11,3 %
de «yes» en train (slide 2), estimar bien una proporción baja pide muchos vecinos (inferencia). Por
distancia (Mitchell §8.2.1), la línea punteada: en train queda en 1,0, porque cada punto de train se
encuentra a sí mismo, y en validación, con más de un vecino, queda por debajo de la uniforme en todo
el rango.

**18. ¿k = 801 y C = 0,001 no quedaron en el borde de sus grillas?** [slides 11 y 12 · Andrés] — Sí,
los dos, y no por la misma razón. En KNN, la validación es una meseta en los k grandes, y la regla
elige el más suave de los que empatan: la meseta llega hasta el final de la grilla, 801, así que un
k mayor probablemente también empataría, sin cambiar el AUC (inferencia). En la SVM, 0,001 es el
menor C del eje: con kernel lineal la curva es plana, y la regla toma el más regularizado. Un C menor
sería una SVM lineal aún más regularizada, y no la probamos: no sabemos si seguiría empatando o
empezaría a subajustar.

**19. ¿Por qué Random Forest no sobreajusta con más árboles, pero sí con más profundidad?** [slide
10 · Andrés] — La profundidad es la complejidad de cada árbol: sin límite, train llega al tope del
eje y la validación baja («sobreajuste →»). Los árboles, en cambio, se promedian: bajan la varianza
sin cambiar el sesgo (Clase 7, slide 61). El protocolo de las curvas —train y validación con los
mismos folds, la brecha como lectura del sobreajuste— es el de la Clase 7, slide 50.

**20. ¿Qué significa un AUC de 0,795?** [slides 8 y 13 · Andrés] — La probabilidad de que, tomando al
azar un «yes» y un «no» de validación, el modelo le dé más puntaje al «yes». Sin modelo es 0,500. No
depende del umbral y no premia decir siempre «no» (Clase 4, slides 45–51). Llamando al 20 % de la
lista, eso se traduce en 0,629 de recall de validación. Esa lista mezcla las épocas, y ordenar
dentro de un mismo año es más difícil: 0,658 en los pares del mismo año (slide 18; pregunta 24).

**21. ¿Los modelos son realmente distinguibles?** [slide 13 · Sebastián] — Random Forest es el
primero en las dos métricas: 0,795 de AUC y 0,629 de recall de validación, contra 0,784 y 0,617 de
KNN, el segundo (D-23). Pero conviene decir la escala: los cuatro van de 0,774 a 0,795, y sin modelo
es 0,5 (la línea punteada). Y el error estándar con el que la charla descarta un empate supone folds
independientes, cuando sus train se superponen en tres cuartos: el real es algo mayor (inferencia).

**22. ¿Por qué 5 folds?** [slide 3 · Andrés] — Cada vuelta valida con uno de los 5 folds de train y
ajusta con los otros cuatro (la tabla: «train, en 5 folds»; D-06). Las barras de ± 1 desvío entre
folds de la slide 9 son cortas frente a la distancia a sin modelo, así que la estimación ya es
estable (inferencia). No comparamos otros k. Lo que sí importa es que los folds estén barajados: sin
barajar, cada uno quedaría en una época (slides 5 y 7).

### El hallazgo

**23. ¿El hallazgo no invalida el modelo?** [slides 18–19 · Andrés] — No invalida el número de test:
dice qué mide. El test es de las mismas campañas (slide 17), así que vale para clientes nuevos de
esas campañas, ordenados en una lista que las mezcla (pregunta 24). Hacia adelante, en los dos
bloques de 2009–2010, Random Forest conserva 0,664 y 0,711: pierde, pero no cae al azar. En 2008 no
servía ni barajado: de 0,519 a 0,584 en esos bloques.

**24. Si el banco elige a quién llamar dentro de una campaña, ¿cuánto valen el AUC y el recall?**
[slide 18 · Andrés] — Menos. El 0,795 y el 0,629 de validación (slide 13) ordenan una lista que
mezcla las épocas, y parte de su valor es reconocer la época: en los pares «yes»–«no» del mismo año
el AUC baja a 0,658, contra 0,864 en los de años distintos (D-25, lectura (a)). El recall dentro de
una campaña no está en las slides; también baja (inferencia), y su vara sigue siendo el 0,200 de sin
modelo (slide 8). El test es la misma mezcla (slide 17): tendrá el mismo sesgo.

**25. ¿No es sólo que las variables económicas delatan la fecha?** [slides 19 y 16 · Andrés] — En
parte, sí. Barajado, sacar el bloque macro y `month` le cuesta a Random Forest 0,059 (slide 16):
reconocer la época ayuda cuando las épocas se mezclan. Hacia adelante, si el modelo sólo reconociera
la época, frente a una nueva ordenaría como el azar, cerca de 0,5, y eso dan los dos primeros
bloques de 2008, 0,489 y 0,500 (slide 19). En el tercero da 0,426, por debajo de sin modelo: el orden
se invierte, y eso sugiere que además la relación entre las variables y el «yes» cambia con el
tiempo (inferencia). Pero es un solo bloque y sin intervalo: un indicio, no una prueba.

**26. ¿Qué harían distinto con más tiempo?** [slide 17 · Sebastián] — Por impacto, lo de la slide:
validar hacia adelante como esquema principal, que mide una campaña futura; más historial del
cliente, porque el 86,3 % de train no tiene campaña previa; y un costo por tipo de error en lugar
del 20 % fijo (Clase 4, slides 41–44). Para usar ese costo harían falta probabilidades calibradas:
hoy las dos métricas sólo miden el orden (slide 8; inferencia).

**27. ¿Y si el test da bastante distinto que validación?** [slide 14 · Sebastián] — Depende de
cuánto. Se nombran los dos valores, el AUC de test de la slide y el 0,795 de validación (slide 13), y
la distancia se lee contra la variación entre folds de validación (la barra de la slide 18) y contra el
intervalo de test. Dentro de esa variación es ruido, en cualquier sentido. Si da algo más bajo,
puede sumarse el optimismo que declaramos en la slide 17, que no medimos (pregunta 10), y por eso no
sirve para descartar nada: una diferencia mayor que esa variación obliga a revisar la partición o el
pipeline, buscando qué separa al test de train, y se dice tal cual. En ningún caso se cambia el
modelo después de abrirlo: lo que se promete es el número de test. Y en ningún caso se lo compara con
«sin modelo» (slide 3: el test no decide nada).

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
   la pregunta 17. Unos 3 segundos.

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
- **Con los números leídos completos: entre 9:23 y 9:28.** Los números de este guion tardan más en
  decirse que los del TP1, porque muchos son AUC con tres decimales. Se expandió cada número escrito
  con cifras a las palabras con que se lee («0,795» son seis: «cero coma setecientos noventa y
  cinco»), y cada «?» del test, como un número de tres decimales: cada palabra de este guion equivale
  a 1,19 palabras leídas, y cada una del TP1, a 1,15. A la velocidad de lectura que 2,7 por segundo
  le supone al TP1, este guion dura 9:23; si el TP1 se dijo en los 10:00 de su reloj, 9:28, 17
  segundos por debajo del tope. Hay 85 números hablados, el 5,8 % de las palabras (en el TP1, 93,
  también el 5,8 %).
- **Lo que decide son los dos ensayos cronometrados** (paso 8.4 del plan, gate G6): tienen que
  durar 10:00 o menos. Si un ensayo pasa de 9:45, se aplica el orden de recorte de arriba, en ese
  orden.

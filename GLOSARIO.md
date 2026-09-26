# Glosario y decisiones

Qué significa cada término que aparece en el repo, y por qué se decidió cada cosa. Una línea por
ítem. Las cifras son de validación sobre train y salen de `DECISIONES.md`,
`resultados/conclusiones.md` e `informe/numeros.md`; ninguna es de test, que todavía no se evaluó
(N0-1).

---

## 1. Términos

### Evaluación

| Término | Qué es | Por qué aparece aquí |
|---|---|---|
| **AUC** | Área bajo la curva ROC: la probabilidad de que un «yes» elegido al azar reciba más puntaje que un «no» elegido al azar. 0,5 es azar; 1, el orden perfecto | Compara y elige modelos (D-19): mide si el modelo ordena bien a quién llamar y no depende del umbral. RF: 0,795 en validación |
| **recall al 20 %** | De los clientes que contratarían, la parte que queda dentro del 20 % de la lista con mayor puntaje | La segunda métrica (D-19): cuenta el error caro, el cliente perdido. RF: 0,629; sin modelo, 0,200 |
| **precisión** | De las llamadas hechas, la parte que termina en «yes» | Se calcula siempre, pero no decide: RF 0,354 al 20 %, contra 0,113 sin modelo, que es la tasa de «yes» |
| **exactitud** | La proporción de aciertos, sin distinguir clases | No se usa: decir siempre «no» ya acierta el 88,7 % de train |
| **precisión promedio y $F_1$** | El área bajo la curva de precisión y recall; la media armónica de precisión y recall | Se guardan por si la cátedra pide otra métrica (consulta 1): cambiarla cambia qué columna se lee, no qué se corre |
| **presupuesto de llamadas (q)** | La parte de la lista a la que se llama: el 20 % con mayor puntaje | Es el umbral (D-20): todos los modelos hacen las mismas llamadas, 1 318 por fold de validación y 1 647 en test. Es un supuesto de negocio |
| **línea de base «sin modelo»** | Un `DummyClassifier` que les da a todos el mismo puntaje, medido en los mismos folds | Todo número de validación se compara contra ella: AUC 0,5 y recall 0,200, porque llamar al 20 % al azar alcanza al 20 % de los «yes». Es el único rótulo de la línea de base en el TP |
| **error estándar (ES)** | El desvío entre los 5 folds dividido por √5 | La vara de la regla de 1 ES: el del AUC de RF es 0,0025 |
| **regla de 1 ES** | Entre los valores que quedan dentro de 1 ES del mejor, elegir el más simple | D-22: profundidad 8 (0,7954) y no 10 (0,7968); k = 801; C = 0,001. Supone folds independientes, y sus train se superponen en tres cuartos, así que el ES real es mayor (inferencia) |
| **sesgo del mínimo** (aquí, del máximo) | El máximo de varias estimaciones ruidosas sale optimista aunque cada una sea insesgada | Los hiperparámetros se eligen en la misma validación cruzada que los reporta: el 0,795 puede estar inflado, del orden de un ES (inferencia). Por eso lo que se promete es el número de test |
| **brecha train − validación** | La diferencia entre el AUC en el fold de entrenamiento y en el de validación | La firma del sobreajuste: RF sin límite de profundidad, 0,228; con profundidad 8, 0,030; KNN con k = 1, 0,369 |

### Datos

| Término | Qué es | Por qué aparece aquí |
|---|---|---|
| **fuga de datos (*data leakage*)** | Usar información que no existe cuando el modelo opera, o que viene de validación o de test | Dos frentes: `duration`, que se conoce al terminar la llamada (D-05, D-07), y las transformaciones ajustadas fuera del fold: todo va dentro del `Pipeline` (`tests/test_preproceso.py`) |
| **centinela 999** | El valor de `pdays` que, según el diccionario, marca a quien no fue contactado en una campaña anterior | Lo tiene el 96,3 % de train, pero no significa «nunca contactado»: 3 302 filas con 999 tienen `previous` ≥ 1. Queda como está (D-10) |
| **`unknown`** | El nivel de faltante de seis categóricas | Queda como categoría (D-08): en `default`, «unknown» tiene 5,3 % de «yes», contra 12,8 % cuando se conoce |
| **one-hot** | Una columna 0/1 por nivel de una categórica | Para NB gaussiano, SVM, KNN y RF; el NB categórico usa códigos. El preprocesamiento pasa de 62 columnas en la referencia a 58 con las decisiones del EDA (D-09, D-11) |
| **escalado** | Restar la media y dividir por el desvío (`StandardScaler`), ajustado en cada fold | Sólo en KNN y SVM (D-12): sin él, `pdays` y `nr.employed` son el 86,6 % y el 13,0 % de la varianza de las numéricas |
| **categoría rara** | Un nivel con menos del 1 % de las filas de train | `illiterate` se funde con `basic.4y`, y `marital` «unknown» con `married`, que es la moda (D-11) |
| **atípico por IQR** | Un valor fuera de [Q1 − 1,5·IQR, Q3 + 1,5·IQR] | 6 731 filas de train (20,4 %), y ninguna se borra (D-17): borrarlas sacaría a los que más contratan, como el 64,6 % de «yes» entre los atípicos de `pdays` |
| **año inferido** | El año de cada fila: el CSV está ordenado por fecha y no trae el año, así que cada vez que el mes retrocede empieza un año nuevo | La base del hallazgo: «yes» 4,9 % en 2008, 19,2 % en 2009 y 51,9 % en 2010 (`eda_html.anio_inferido()`) |
| **bloque macro** | Las cinco variables del contexto económico: `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m` y `nr.employed` | Entran completas (D-14), pero fechan las filas: sacarlas cuesta AUC barajado y no arregla la validación hacia adelante (D-25) |

### Validación

| Término | Qué es | Por qué aparece aquí |
|---|---|---|
| **train, validación y test** | Train: las 32 940 filas que deciden todo. Validación: el fold que queda afuera en cada vuelta. Test: las 8 236 filas reservadas | Train ajusta, validación compara y elige, test mide una sola vez (D-02, D-04) |
| **k-fold** | Partir train en k bloques y, k veces, entrenar con k − 1 y validar con el que queda | k = 5: cada vuelta entrena con 26 352 filas y valida con 6 588 |
| **StratifiedKFold barajado** | Folds con la misma proporción de «yes» y con filas de todas las épocas: `folds()` | D-06: `cv=5` a secas no baraja y, con el archivo ordenado por fecha, armaría un fold por época |
| **OOF (fuera de fold)** | El puntaje de cada fila de train dado por el modelo que no la vio | La sensibilidad al presupuesto, la curva de ganancia y el análisis de errores salen de ahí sin volver a ajustar (N0-5) |
| **validación hacia adelante** | Entrenar con el pasado y validar con el bloque siguiente, cinco veces (`TimeSeriesSplit`) | El hallazgo (D-25): RF pasa de 0,795 barajado a 0,558 ± 0,123 hacia adelante |
| **ablación** | Una variante que cambia una sola cosa respecto de la referencia A0, medida en los mismos folds | A2 a A12 deciden D-08 a D-16, y A1 mide el techo de `duration`, con el ΔAUC pareado fold a fold: se adopta lo que mejora más que su desvío, o lo neutro que simplifica |
| **curva de validación** | El AUC en train y en validación para cada valor de un hiperparámetro | El punto 3.1 (D-22): la de profundidad de RF muestra el subajuste y el sobreajuste |

### Modelos

| Término | Qué es | Por qué aparece aquí |
|---|---|---|
| **Naive Bayes categórico** | Supone que las variables son independientes dada la clase; las numéricas se discretizan en deciles dentro del pipeline | El NB del TP (D-18): 0,782 ± 0,007 |
| **Naive Bayes gaussiano** | Una normal por clase para cada numérica | La comparación: 0,769 ± 0,009. La edad en U, `pdays` casi binaria y `campaign` con cola larga no son gaussianas |
| **corrección de Laplace (α)** | Sumar α a cada conteo para que ningún nivel tenga probabilidad cero | α = 1 (Clase 6, slide 46) |
| **SVM lineal y RBF** | El hiperplano de máximo margen, en las variables (lineal) o en el espacio de un kernel (RBF, polinómico) | La final es lineal (`LinearSVC`, pérdida *hinge*): con C = 0,001, lineal 0,7741 y RBF 0,7681 (D-22) |
| **C** | El costo de las violaciones del margen; con la convención de scikit-learn, un C grande regulariza menos (Clase 8, slide 49) | C = 0,001, el más regularizado de la meseta: 0,001 y 0,01 empatan (D-22) |
| **pesos de clase** | `class_weight="balanced"`: cada error pesa en proporción inversa a la frecuencia de su clase | Se adoptan en la SVM (+0,068 con RBF y C = 1) y no en RF (+0,0006, despreciable; N0-12) |
| **KNN** | Clasificar por los k vecinos más cercanos de train | Segundo en validación: 0,784 ± 0,008 |
| **k** | Cuántos vecinos votan (`n_neighbors`) | k = 801, el mayor de la meseta dentro de 1 ES (D-22). Aquí sobreajusta el k pequeño: con k = 1, validación 0,619 y train 0,988 |
| **uniform y distance** | Todos los vecinos votan igual, o cada uno pesa en proporción inversa a su distancia | Uniforme: por distancia, la curva queda por debajo para todo k > 1 y memoriza train (1,000) |
| **Random Forest** | Muchos árboles de decisión, cada uno sobre una muestra con reemplazo de las filas y con variables sorteadas en cada corte; decide el promedio | El modelo final (D-23): 0,795 ± 0,006 |
| **max_depth** | La profundidad máxima de cada árbol: el eje de complejidad | 8 (D-22). Con 2, subajuste: 0,782 en validación y 0,783 en train; sin límite, sobreajuste: 0,772 y 1,000 |
| **n_estimators** | Cuántos árboles | 200 (N0-11): más árboles bajan la varianza pero no cambian el sesgo, y la curva es plana desde 100 |

---

## 2. Decisiones en una línea

Cada una tenía una alternativa razonable; el detalle, con la cifra y la cita, está en `DECISIONES.md`.

| # | Decisión | Por qué |
|---|---|---|
| **D-01** | Quitar los 12 duplicados exactos antes de partir: 41 188 → 41 176 filas | Una copia en train y otra en test le quitan independencia al test; ningún par tiene etiquetas distintas |
| **D-02** | Partición 80/20 estratificada por `y`, barajada, semilla 42: train 32 940 (3 711 «yes»), test 8 236 (928) | Sin estratificar, el % de «yes» del test cambia con la semilla: en 400 semillas, desvío de 0,31 pp y extremos de 10,4 % y 12,1 % |
| **D-03** | Partición aleatoria, no temporal | Con el último 20 % como test, train tendría 6,38 % de «yes» y test 30,83 %; el costo, la época, va como limitación (D-25) |
| **D-04** | El test no se abre hasta el punto 4: todo lo anterior lee `cargar_train()` | Si una decisión mira el test, el número del punto 4 sale optimista; `tests/test_aislamiento.py` lo hace cumplir |
| **D-05** | `duration` queda fuera del modelo | Se conoce al terminar la llamada, cuando `y` ya se sabe: 0 % de «yes» en el decil más corto y 46 % en el más largo; con ella, RF llegaría a 0,939 (A1) |
| **D-06** | Toda validación cruzada usa `folds()`, `StratifiedKFold(5)` barajado; nunca `cv=5` | `cv=5` no baraja: el euribor medio iría de 4,87 en el primer fold a 1,10 en el último; con `folds()`, de 3,58 a 3,64 |
| **D-07** | El modelo opera antes de cada llamada: 19 de las 20 predictoras están disponibles, `campaign` con reserva | El enunciado define la fuga por lo que no se conoce antes de llamar; `campaign` puede depender del resultado (media 2,053 en «yes» y 2,627 en «no») |
| **D-08** | `unknown` queda como categoría, sin imputar (A11 no se adopta) | Imputar por la moda es neutro (RF +0,0011 ± 0,0034) y borra una señal: 5,3 % de «yes» con `default` «unknown» contra 12,8 % |
| **D-09** | `default` pasa a una indicadora «unknown / no» (A3) | Sólo 2 filas de train tienen `default` = yes; mejora RF (+0,0029 ± 0,0021), es neutra en los otros tres y saca dos columnas |
| **D-10** | `pdays` queda con el 999 (A2 no se adopta) | Sacarlo empeora NB (−0,0012 ± 0,0003) y KNN (−0,0019 ± 0,0007); con `pdays` < 999 la tasa de «yes» es 64,6 %, contra 9,2 % |
| **D-11** | Las categorías raras se funden: `illiterate` con `basic.4y`, y `marital` «unknown» con `married`, la moda (A4) | Neutro o marginalmente mejor en los cuatro (NB +0,0008 ± 0,0006) y simplifica: 62 → 60 columnas |
| **D-12** | Escalado sólo en KNN y SVM, dentro del pipeline (A12 no se adopta) | Miden distancias. Sin escalar, KNN sube +0,0084 ± 0,0067 por la razón equivocada: pasa a agrupar por historial de contacto y por época |
| **D-13** | `campaign` sin transformar (A5 no se adopta) | `log1p` no cambia nada: KNN +0,0008 ± 0,0026, SVM −0,0006 ± 0,0020 |
| **D-14** | El bloque macro entra completo, las cinco variables (A6 y A7 no se adoptan) | Reducirlo empeora NB (−0,0041) y KNN (−0,0083), y sacarlo cuesta entre 0,02 y 0,05 de AUC. En NB ayuda aunque viola la independencia: r de 0,91 a 0,97 |
| **D-15** | `month` y `day_of_week` se quedan, en one-hot (A9 no se adopta) | Sin `day_of_week` la SVM pierde 0,0074 ± 0,0021; sin `month` ni el bloque macro, RF pierde 0,093: `month` codifica la época |
| **D-16** | La edad entra en años, no en tramos (A10 no se adopta) | En NB gaussiano es neutro (+0,0001 ± 0,0022) y agrega 8 columnas; la U de la edad la captan los cortes de RF |
| **D-17** | Los atípicos por IQR se diagnostican y no se borran: 6 731 filas (20,4 %) | Son señal: entre los atípicos de `pdays` hay 64,6 % de «yes», y entre los de `age`, 46,5 %, contra 11,3 % global |
| **D-18** | El Naive Bayes es el categórico: deciles aprendidos en cada fold y Laplace con α = 1 | 0,782 ± 0,007 contra 0,769 ± 0,009 del gaussiano; fold a fold, +0,0132 ± 0,0024 |
| **D-19** | Dos métricas: el AUC para comparar y elegir, el recall al 20 % para medir el uso real | El AUC mide el orden y no premia decir siempre «no»; el recall cuenta el cliente perdido; la exactitud de decir siempre «no» ya es 88,7 % |
| **D-20** | El umbral es un presupuesto: se llama al 20 % de cada lista con mayor puntaje | Todos los modelos hacen las mismas llamadas y no hay umbral que estimar. Con 10 % y 30 %, RF alcanza 0,446 y 0,705 de los «yes» |
| **D-21** | El desbalance no se trata; los pesos de clase se miden como un hiperparámetro más | Las métricas de D-19 no premian decir siempre «no», y el presupuesto no depende del umbral. Los pesos entran en la SVM (+0,068) y no en RF (+0,0006) |
| **D-22** | Cada hiperparámetro, el de mayor AUC de validación o, dentro de 1 ES, el más simple: RF con profundidad 8; KNN con k = 801, uniforme; SVM lineal con C = 0,001 y pesos balanceados | RF: 8 (0,7954) queda dentro de 1 ES de 10 (0,7968), con brecha 0,030. KNN: meseta desde 301. SVM: con C = 0,001, lineal 0,7741 contra RBF 0,7681 |
| **N0-11** | RF con 200 árboles, no los 25 que daba la regla | El número de árboles estabiliza y no agrega complejidad (Clase 7, slide 61); la curva es plana desde 100 |
| **N0-12** | RF sin pesos de clase, aunque la regla daba «balanced» | Suman +0,0006; fold a fold, +0,0006 ± 0,0009, que no supera su desvío (la regla de las ablaciones) y es despreciable. Sin pesos, el modelo es el de la Clase 7 tal cual |
| **D-23** | El modelo final es Random Forest: el mayor AUC de validación; dentro de 1 ES, desempataría el recall | RF 0,795 ± 0,006 contra KNN 0,784, NB 0,782 y SVM 0,774; el umbral de empate, 0,7927, no lo alcanza ningún otro |
| **D-24** | El test se evalúa una sola vez: reentrenar con todo train, AUC con intervalo del 95 % por bootstrap y recall llamando al 20 % de la lista de test | Evaluar dos veces le resta independencia al test. Queda para después del 30/09, cuando responda la cátedra (N0-1); su fila de `DECISIONES.md` se escribe con la evaluación |
| **D-25** | La validación hacia adelante se mide dentro de train y se reporta como limitación, no como esquema principal | Hacia adelante RF cae de 0,795 a 0,558 ± 0,123; el test, aleatorio de la misma mezcla, mide clientes nuevos de esas campañas, no una campaña futura |

---

## 3. El flujo de punta a punta

```text
41 188 filas × 21 columnas (bank-additional-full.csv, ordenado por fecha: mayo de 2008 a noviembre de 2010)
  └─ quitar los 12 duplicados exactos (D-01) ................................. 41 176
       └─ partición 80/20 estratificada por y, semilla 42 (D-02, D-03)
            ├─ TRAIN 32 940 (3 711 «yes») ─ EDA, ablaciones y curvas: todo sale de aquí (D-04)
            │    └─ 5 folds barajados (D-06); en cada fold, un Pipeline: Derivadas → codificación → modelo
            │         ├─ referencia: NB, SVM, KNN y RF contra «sin modelo» (D-18 a D-21)
            │         ├─ curvas y regla de 1 ES → RF prof. 8 y 200 árboles; KNN k = 801; SVM lineal, C = 0,001 (D-22)
            │         ├─ final: RF, AUC 0,795 ± 0,006 y recall al 20 % 0,629 → el modelo elegido (D-23)
            │         └─ hacia adelante (TimeSeriesSplit): RF 0,558 ± 0,123 → aprende la época (D-25)
            └─ TEST 8 236 (928 «yes») ...... intacto hasta `python -m src.evaluar_test` (D-24, N0-1)
```

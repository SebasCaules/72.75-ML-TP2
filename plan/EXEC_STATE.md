# Estado de ejecución del TP2

Registro vivo del runbook de `plan/PLAN.md`. Se actualiza al cerrar cada paso, no al final de la
sesión. Estados: TODO, DOING, DONE, BLOCKED. Un paso está DONE cuando pasa su prueba de terminado.

## Pasos

| Paso | Estado | Commit | Notas |
|---|---|---|---|
| 0.1 Reconciliación y registro de estado | DONE | 64567aa | Ver «Gaps de la reconciliación» |
| 0.2 Mensaje a la cátedra | DONE (borrador) | a0e6e5e | `plan/consultas.md`; el envío es del grupo |
| 0.3 `estilo.py` y `test_paleta.py` | DONE | f1113e7 | |
| 0.4 `preproceso.py` y `modelos.py` | DONE | 02ef7ff, 1e50b4e | Verificado; 9 hallazgos corregidos en 1e50b4e |
| 0.5 `metricas.py` | DONE | de6efb6, 1e50b4e | NB puntúa por log-odds; recall NaN sin positivos |
| 0.6 Presupuesto de cómputo | DONE | (este) | `resultados/costos.csv` y `costos_grillas.csv`: todas las grillas entran; la SVM lineal de libsvm con C = 10 excede 600 s y queda fuera, la lineal va por liblinear |
| 0.7 Test de aislamiento del test | DONE | 8c04854, 1e50b4e | Busca nombres, no llamadas; todo el repo |
| 0.8 Cierre TODOS LOS TESTS OK en las suites del paso 1 | DONE | de6efb6 | |
| 1.1 EDA en HTML sobre train | DONE (falta la lectura del grupo) | fdc110c | 16 gráficos; publicado como página privada |
| 1.2 Línea de base «sin modelo» | DONE | 94de6af | `resultados/linea_base.csv`: AUC 0,5, recall_q 0,2001, exactitud 0,8873 |
| 1.3 D-07, momento en que opera el modelo | DONE | (este) | Redactor y refutador en la ola A; 8 hallazgos aplicados por N0 al integrar |
| 1.4 Ablaciones A1 a A12 | DONE | 2ecee39 | 4 modelos × hasta 12 variantes × 5 folds; `resultados/ablaciones_resumen.csv` |
| 1.5 D-08 a D-17 | DONE | 2ecee39 | Adoptadas A3 (default indicadora) y A4 (raras fundidas); A12 rechazada por juicio (D-12); el resto nulo o peor |
| 1.6 Figura de ablaciones | DONE | c20bab6 | `figuras/01-ablaciones.png`, `01b` |
| 2.1 D-19, métricas | DONE | 9a6347f | |
| 2.2 D-20, presupuesto de llamadas | DONE (texto); tabla de sensibilidad en curso | 9a6347f | `experimentos --sensibilidad-q` corre tras el 3.3 (N0-10) |
| 2.3 D-21, desbalance | DONE (texto); `pesos_clase` en curso | 9a6347f | Curvas rf/svm `pesos_clase` corriendo |
| 3.1 Pipelines definitivos | DONE | 2ecee39 | `OPCIONES_FINALES = Opciones(default="indicadora", raras=True)`, 58 columnas |
| 3.2 D-18, variante de Naive Bayes | DONE | 2519d9b | Categórico: +0,0132 ± 0,0024 sobre el gaussiano |
| 3.3 CV de los cuatro modelos | DONE | 2519d9b, 0be05cb | `cv_referencia_*` (5 modelos) y `cv_final_*` (4 modelos con los hiperparámetros elegidos) |
| 3.4 Figura de los cuatro modelos | DONE | 9552f04 | `02` (final) y `02b` (referencia) |
| 4.1 Curvas de RF | DONE | f476258, d694d40 | max_depth (a 300 árboles), n_estimators y pesos (a profundidad 8) |
| 4.2 Curvas de KNN | DONE | d694d40 | Grilla extendida a 801; uniform y distance |
| 4.3 Curvas de la SVM | DONE | d694d40 | C con RBF (sin y con pesos), kernels, C con lineal |
| 4.4 D-22, hiperparámetros | DONE | d694d40 | Con los desvíos N0-11 (200 árboles) y N0-12 (RF sin pesos) |
| 4.5 Figuras de las curvas | DONE | c20bab6, 9552f04 | 16 curvas; las de proyección con revelado van en la ola 7 |
| 4.6 Grilla C × γ (opcional) | TODO | | S-04; el kernel final es lineal, así que γ no aplica |
| 5.1 D-23, modelo final | DONE | 0be05cb | RF 0,7952 ± ES 0,0025; sin empate |
| 5.2 D-25, robustez temporal | DONE | 42f5aba | Hacia adelante, RF 0,558 ± 0,123; sin macro no lo arregla |
| 5.3 D-24, evaluación única del test | TODO | 9df4545 | Módulo construido y probado sin abrir el test; NO se ejecuta (N0-1). `informe/resultados-test.tex` con marcadores |
| 5.4 Análisis de errores | DONE (sobre validación) | c343cd3 | `conclusiones.md` §3; sobre test, después del 30/09 |
| 6.1 Tabla por modelo | DONE | c343cd3 | `resultados/conclusiones.md` §1 |
| 6.2 Hallazgo | DONE | c343cd3 | H1 en tres partes (a) ordenar épocas, (b) 2008 no sirve, (c) 2009–2010 pierde 0,05–0,08 |
| 6.3 Limitaciones y mejoras | DONE | c343cd3 | `conclusiones.md` §4 y §5 |
| 7.1 Deck | DONE | 61dff32 | 20 frames + 4 de respaldo, 49 páginas; G3 en verde; los medios del verificador se corrigen en la ola 8. Con N0-15, sin respaldo: 20 frames y 45 páginas |
| 7.2 `numeros.py` | DONE | 6ef5dac | 468 macros; `--verificar` para el gate G3 |
| 7.3 Figuras de proyección | DONE | 91a351f | 18 figuras en `figuras/presentacion/`, con revelados |
| 7.4 Guion | DONE | 8eb008b | 1 492 palabras, 9:13; 27 preguntas; medios pendientes en la ola 8 |
| 7.5 Cuadernillo de ensayo | DONE | 3287282 | `python -m src.cuadernillo`, 27 páginas; se regenera al cerrar la ola 8 |
| 7.6 README y GLOSARIO | DONE | 7aa844a | `informe/README-entrega.md`, README del repo, GLOSARIO |
| 7.7 Informe de respaldo (opcional) | TODO | | S-05 |
| 8.1 Auditoría adversarial | DONE | 47355bc, 540766d | 5 lentes: 80 hallazgos (34 altos o medios, 46 bajos); 30 confirmados y 4 refutados; correcciones por grupo y cierre de los pendientes cruzados, todo re-verificado |
| 8.2 Entregables y clon limpio | DONE | (este) | `python -m src.entregar --con-bitacora --probar`: zip de 177 archivos con DECISIONES y GLOSARIO (S-01), PDF idéntico a `informe/presentacion.pdf`; en el clon, 20 suites en verde |
| 8.3 Gates | DONE | (este) | G1 20/20; G2 test intacto (`evaluacion_test.json` no existe: N0-1); G3 `--verificar` OK; G4 títulos afirman o preguntan; G5 tabla de cobertura del guion, todo antes de la 18; G6 9:00 (1 606 palabras a 2,68/s); G7 figuras legibles al 25 % (verificadores de 7.3 y 8.1); G8 paleta OK; G9 clon limpio OK; G10 sin voseo ni coloquialismos; G11 «sin modelo» único rótulo, 18 apariciones, ninguna en la slide 14; G12 cada sección del EDA cierra con su consecuencia (L1) |
| 8.4 Ensayos | TODO | | Del grupo: dos ensayos cronometrados con `informe/slides-y-guion.pdf`; si pasa de 9:45, el orden de recorte del guion |
| 8.5 Envío | TODO | | Del grupo, antes del 06/10 (24 h antes de la clase); antes, evaluar el test (5.3) el 01/10 y regenerar deck, guion y cuadernillo |

## Gaps de la reconciliación (G-nn)

Estado real contra el plan, al 25/09. Cada gap se cierra antes de la ola 1 o tiene su paso.

| Gap | Qué no coincidía | Cómo se cierra | Estado |
|---|---|---|---|
| G-01 | El plan proponía agrupar las categorías raras con `OneHotEncoder(min_frequency=…)`. El EDA en HTML mostró que cada columna tiene un solo nivel con menos del 1 %, así que el encoder sólo les cambia el nombre | D-11 y A4 corregidas en el plan: fusión explícita de cada nivel raro | Cerrado |
| G-02 | `tests/test_datos.py` y `tests/test_validacion.py` no terminan en «TODOS LOS TESTS OK», que es lo que exige G1 | Paso 0.8 | Cerrado (de6efb6) |
| G-03 | No se conocen el número de grupo ni el día de defensa | Grupo 7 (N0-2); el día, consulta 6 | Cerrado el grupo; el día sigue abierto |
| G-04 | En esta máquina `python` no está en el PATH: los comandos se corren con `python3`. El README dice `python -m …`, como en el TP1 | S-07 | Abierto, no bloquea |

Coinciden con el plan: los commits `0a9b311` y `108ff99`; D-01 a D-06 en `DECISIONES.md`; la
partición en `data/particion/` con sus tests en verde; y `folds()` en `src/validacion.py`.

## Decisiones tomadas durante la ejecución (N0-n)

Cada decisión de contrato que no estaba en el plan, con su porqué.

| N0 | Decisión | Por qué |
|---|---|---|
| N0-1 | **El test no se evalúa en esta ejecución** (usuario, 25/09). Se construye `src/evaluar_test.py` y se prueba sin abrir el test; los pasos 5.3 y 5.4 quedan para después del 30/09. El deck y el guion llevan los números de test en modo marcador, como el TP1 | Las consultas a la cátedra pueden cambiar variables o partición; evaluar dos veces le resta independencia al test |
| N0-2 | Grupo 7: Andrés Cortese (64612) y Sebastián Caules (64331), como en el TP1 (usuario, 25/09). Entregables `TP2-grupo7-*` | Cierra G-03 y S-03 |
| N0-3 | `src/configuracion.py`, reservado al orquestador, define `OPCIONES_FINALES` e `HIPERPARAMETROS_FINALES`. Arrancan en la referencia y se actualizan al cerrar las olas 1 y 4. Todos los módulos de cómputo leen de ahí | Un solo lugar para las decisiones vigentes; los workers no deciden contrato |
| N0-4 | Contrato de resultados: CSV en formato largo con columnas `modelo, configuracion, fold, conjunto, metrica, valor` (más `variante` en las ablaciones), `configuracion` como JSON con claves ordenadas, folds 1 a 5, `conjunto` en {train, validacion}, siempre las cinco métricas de `src/metricas.py` | Una sola función dibuja cualquier curva y `numeros.py` saca cualquier cifra |
| N0-5 | Los puntajes fuera de fold (OOF) de cada modelo se guardan en `resultados/oof_<etiqueta>_<modelo>.csv` (`fila, fold, puntaje`) | Sirven para la sensibilidad a q (D-20), la curva de ganancia (H3) y el análisis de errores sin volver a ajustar |
| N0-6 | Cada módulo de cómputo escribe archivos propios por modelo (`<nombre>_<modelo>.csv`) y tiene un modo `--resumen` que los concatena | Permite correr los modelos en procesos paralelos sin pisarse |
| N0-7 | Los modelos de los agentes: Fable sólo para N0; todo worker, verificador y auditor en Opus con `effort: 'max'` (usuario, 25/09) | Política del usuario para esta ejecución |
| N0-8 | Los OOF de la referencia de las ablaciones se llaman `oof_ablacion_A0_<modelo>.csv`, no `oof_referencia_<modelo>.csv`: ese nombre es de `experimentos --etiqueta referencia`, que mide con `OPCIONES_FINALES`, y se pisaban. `ablaciones_resumen.csv` no trae filas A0: el AUC de A0 está en `auc_referencia` | Hallazgo del verificador de la ola A |
| N0-9 | Las ablaciones miden siempre contra `Opciones()` y `REFERENCIA`, sin leer `configuracion.py`: la base no se mueve después de la ola 1, y una corrida posterior reproduce D-08 a D-17 | Constructor de ablaciones; N0 lo confirma |
| N0-10 | El paso 2.2 (sensibilidad a q) pasa a después del 3.3: `experimentos --sensibilidad-q` exige `cv_referencia_<modelo>.csv` para no leer un OOF de otra corrida | Hallazgo del verificador de experimentos |
| N0-11 | RF con 200 árboles, no los 25 que daba la regla de 1 ES (DECISIONES.md §6) | El número de árboles es un eje de estabilidad, no de complejidad |
| N0-12 | RF sin pesos de clase (la regla daba «balanced» por +0,0006 ± 0,0009, positiva en los 5 folds) | D-21: sólo si la mejora vale la pena; ésta es despreciable por su tamaño, no por ruido |
| N0-13 | En `seleccion.py`, la configuración vigente (`src/configuracion.py`) prevalece sobre la propuesta mecánica de `hiperparametros.json`; el JSON sólo completa lo que la vigente no fija | Sin esto, N0-11 y N0-12 no se podían aplicar |
| N0-14 | `experimentos --etiqueta referencia` mide siempre con `REFERENCIA` de `src/modelos.py` (como las ablaciones, N0-9) y no con `configuracion.py`; así los comandos del README reconstruyen `cv_referencia_*` desde cero | Hallazgo L2-metodologia-04 de la auditoría; lo cambió el corrector del grupo código |
| N0-15 | Sin slides de respaldo: el deck queda en 20 slides (45 páginas) y toda respuesta preparada (aclaraciones del guion, banco de preguntas y «Si preguntan» de las notas) se contesta sólo con lo que se ve en esas 20 slides, más la teoría de lo que se ve con sus citas. Se quitan las cuatro figuras de reserva de `graficos_presentacion.py` (18 → 14) | Pedido del usuario, 26/09/2026: «todas las preguntas deben ser sobre las slides mostradas y se deberá contestar únicamente con esas» |

## Veredicto final

**Estado al 26/09/2026:** el TP2 está completo salvo lo que depende de una fecha o del grupo.

- Hecho y verificado: partición, EDA, ablaciones y D-07 a D-17, métricas y presupuesto (D-19 a D-21), Naive Bayes categórico (D-18), curvas e hiperparámetros (D-22, N0-11, N0-12), modelo final RF (D-23), robustez temporal (D-25), conclusiones y hallazgo H1, figuras de análisis y de proyección, deck de 20 frames (sin respaldo desde N0-15), guion de 8:30 con 27 preguntas que se contestan sólo con lo que muestran las slides, cuadernillo de ensayo, README de entrega, GLOSARIO, entregables con prueba de clon limpio, auditoría adversarial en cinco lentes con sus correcciones. Los 12 gates pasan.
- Pendiente con fecha: la evaluación única del test (5.3 y 5.4), después de la clase del 30/09 y de las respuestas de la cátedra (N0-1). Comando: `python -m src.evaluar_test` (abre el test una sola vez, escribe `evaluacion_test.json` y `resultados-test.tex`); después `python -m src.numeros`, recompilar el deck, completar las slides 14 y 15 del guion y regenerar el cuadernillo y los entregables.
- Pendiente del grupo: enviar `plan/consultas.md` a la cátedra (0.2), leer `resultados/eda/eda.html` (1.1), los dos ensayos (8.4) y el envío (8.5).
- Deuda registrada: `plan/SUGERENCIAS.md`, S-01 a S-15; ninguna bloquea la entrega.
- Riesgo principal: si la cátedra responde que las variables del último contacto no están disponibles antes de llamar (consulta 2), hay que medir la variante A13 (S-11) y revisar D-07; si pide partición temporal (consulta 5), cambia el esquema entero.

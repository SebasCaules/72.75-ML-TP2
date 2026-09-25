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
| 1.2 Línea de base «sin modelo» | TODO | | |
| 1.3 D-07, momento en que opera el modelo | TODO | | |
| 1.4 Ablaciones A1 a A12 | TODO | | |
| 1.5 D-08 a D-17 | TODO | | |
| 1.6 Figura de ablaciones | TODO | | |
| 2.1 D-19, métricas | TODO | | |
| 2.2 D-20, presupuesto de llamadas | TODO | | |
| 2.3 D-21, desbalance | TODO | | |
| 3.1 Pipelines definitivos | TODO | | |
| 3.2 D-18, variante de Naive Bayes | TODO | | |
| 3.3 CV de los cuatro modelos | TODO | | |
| 3.4 Figura de los cuatro modelos | TODO | | |
| 4.1 Curvas de RF | TODO | | |
| 4.2 Curvas de KNN | TODO | | |
| 4.3 Curvas de la SVM | TODO | | |
| 4.4 D-22, hiperparámetros | TODO | | |
| 4.5 Figuras de las curvas | TODO | | |
| 4.6 Grilla C × γ (opcional) | TODO | | S-04 |
| 5.1 D-23, modelo final | TODO | | No antes del 30/09 |
| 5.2 D-25, robustez temporal | TODO | | |
| 5.3 D-24, evaluación única del test | TODO | | No antes del 01/10 |
| 5.4 Análisis de errores | TODO | | |
| 6.1 Tabla por modelo | TODO | | |
| 6.2 Hallazgo | TODO | | |
| 6.3 Limitaciones y mejoras | TODO | | |
| 7.1 Deck | TODO | | |
| 7.2 `numeros.py` | TODO | | |
| 7.3 Figuras de proyección | TODO | | |
| 7.4 Guion | TODO | | |
| 7.5 Cuadernillo de ensayo | TODO | | |
| 7.6 README y GLOSARIO | TODO | | |
| 7.7 Informe de respaldo (opcional) | TODO | | S-05 |
| 8.1 Auditoría adversarial | TODO | | |
| 8.2 Entregables y clon limpio | TODO | | |
| 8.3 Gates | TODO | | |
| 8.4 Ensayos | TODO | | Del grupo |
| 8.5 Envío | TODO | | Del grupo |

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

## Veredicto final

(pendiente)

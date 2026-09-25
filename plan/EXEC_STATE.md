# Estado de ejecución del TP2

Registro vivo del runbook de `plan/PLAN.md`. Se actualiza al cerrar cada paso, no al final de la
sesión. Estados: TODO, DOING, DONE, BLOCKED. Un paso está DONE cuando pasa su prueba de terminado.

## Pasos

| Paso | Estado | Commit | Notas |
|---|---|---|---|
| 0.1 Reconciliación y registro de estado | DOING | | Ver «Gaps de la reconciliación» |
| 0.2 Mensaje a la cátedra | TODO | | Lo envía el grupo |
| 0.3 `estilo.py` y `test_paleta.py` | TODO | | |
| 0.4 `preproceso.py` y `modelos.py` | TODO | | |
| 0.5 `metricas.py` | TODO | | |
| 0.6 Presupuesto de cómputo | TODO | | |
| 0.7 Test de aislamiento del test | TODO | | |
| 0.8 Cierre TODOS LOS TESTS OK en las suites del paso 1 | TODO | | |
| 1.1 EDA en HTML sobre train | DONE | | Hecho el 25/09, antes de la ola 0. 16 gráficos; publicado como página privada. Falta que el grupo lo lea |
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
| G-02 | `tests/test_datos.py` y `tests/test_validacion.py` no terminan en «TODOS LOS TESTS OK», que es lo que exige G1 | Paso 0.8 | Abierto |
| G-03 | No se conocen el número de grupo ni el día de defensa | Los entregables se nombran `grupoN` hasta que el grupo lo confirme (S-03); el día, consulta 6 | Abierto, no bloquea |
| G-04 | En esta máquina `python` no está en el PATH: los comandos se corren con `python3`. El README dice `python -m …`, como en el TP1 | S-07 | Abierto, no bloquea |

Coinciden con el plan: los commits `0a9b311` y `108ff99`; D-01 a D-06 en `DECISIONES.md`; la
partición en `data/particion/` con sus tests en verde; y `folds()` en `src/validacion.py`.

## Decisiones tomadas durante la ejecución (N0-n)

Cada decisión de contrato que no estaba en el plan, con su porqué.

(ninguna todavía)

## Veredicto final

(pendiente)

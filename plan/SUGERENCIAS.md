# Sugerencias y arreglos pendientes (S-nn)

Un solo lugar para lo que no bloquea la ola en curso. Cada fila dice dónde se resuelve.

| S | Qué | Superficie | Origen | Cuándo se resuelve |
|---|---|---|---|---|
| S-01 | ¿`DECISIONES.md` y `GLOSARIO.md` van dentro del zip? En el TP1 quedaron afuera | Entregables | Plan, §2 | Resuelto en la ola 8 (L5-consistencia-03): sí entran. A diferencia del TP1, el README de entrega los lista en el árbol y remite a `DECISIONES.md` por los D-xx, y el código los cita; el comando de `README.md` usa `--con-bitacora`. Queda para el orquestador que sea el valor por defecto de `src/entregar.py` |
| S-02 | ¿Se repite la limpieza de comentarios del TP1 antes de entregar (commit `1cca310`)? El porqué de cada decisión ya vive en `DECISIONES.md` | `src/` | Relevamiento del TP1 | Ola 8, con el grupo |
| S-03 | Número de grupo e integrantes en el README y en los nombres de los entregables | README, `entregables/` | G-03 | Resuelto: grupo 7, Cortese y Caules (N0-2) |
| S-04 | Grilla conjunta de C y γ para la SVM con RBF (Clase 8, slides 48–49) | `src/curvas.py` | Paso 4.6 | Si sobra presupuesto de cómputo |
| S-05 | Informe de respaldo, como el del TP1 | `informe/` | Paso 7.7 | Si sobra un día |
| S-06 | Wiki: cerrar los avisos de copia a `raw/TP2/` y propagar los resultados del TP2 | `wiki/` | Pedido del usuario del 25/09 | Después de la entrega |
| S-07 | `python` o `python3` en los comandos del README | README | G-04 | Ola 7 |
| S-08 | `--rapido` acepta una `--salida` dentro de `resultados/` si se escribe con otras mayúsculas (macOS no distingue): comparar con `samefile` en ablaciones, experimentos, robustez y seleccion | `src/*.py` | Verificador de la ola A | Ola 8, transversal |
| S-09 | Tests que no fijan lo que sostienen: la tolerancia de `verificar_oof` (experimentos), el paso de `rapido` a `medir` (robustez), el IC del bootstrap y la conexión guarda-evaluación (evaluar_test) | `tests/` | Verificadores de la ola A | Ola 8, transversal |
| S-10 | `leer_largo` no conserva el último ulp de algunos flotantes; usar `float_precision="round_trip"` si hace falta un ida y vuelta exacto | `src/resultados.py` | Constructor de ablaciones | Si aparece una diferencia real |
| S-11 | Variante A13, «sin las variables del último contacto» (`contact`, `month`, `day_of_week`, `campaign`), por si la cátedra responde que no están disponibles (consulta 2). `Opciones` hoy no puede sacar `contact` ni `campaign` | `src/preproceso.py`, `src/ablaciones.py` | Redactor y verificador de D-07 | Cuando responda la cátedra |
| S-12 | `informe/resultados-test.tex` escribe la coma decimal sin llaves y el intervalo con «–»: dentro de `$…$` se imprime «0, 823». Usar `{,}` y `--`, o usar los macros sólo en modo texto (`\heroe`) | `src/evaluar_test.py` | Re-verificador de numeros | Antes de evaluar el test (ola 5, paso 5.3) |
| S-13 | Los «avisos» de `resultados/hiperparametros.json` (y por eso los de `modelo_elegido.json`) describen la propuesta mecánica (25 árboles, balanced) como «configuración final»; la vigente es la de `configuracion.py` (N0-13). Reformular el texto en `src/curvas.py` | `src/curvas.py` | Constructor de conclusiones | Ola 8, transversal |
| S-14 | Con S-01, `DECISIONES.md` y `GLOSARIO.md` viajan en el zip y remiten a piezas que no viajan: `N0-1`, `N0-5` y `N0-10` (definidas sólo en `plan/EXEC_STATE.md`), las consultas 1 a 5 (`plan/consultas.md`), pasos y olas del plan, la wiki (C-34 en D-10 y la tensión de la Clase 4 en D-20) y el commit `0a9b311` (D-04); `DECISIONES.md` trae además el bloque «Para el guion de la defensa». Decidir si se explican en una línea o se quitan | `DECISIONES.md`, `GLOSARIO.md`, `informe/README-entrega.md` | Corrector de docs, ola 8 | Ola 8, antes del paso 8.2, con el grupo |

# Sugerencias y arreglos pendientes (S-nn)

Un solo lugar para lo que no bloquea la ola en curso. Cada fila dice dónde se resuelve.

| S | Qué | Superficie | Origen | Cuándo se resuelve |
|---|---|---|---|---|
| S-01 | ¿`DECISIONES.md` y `GLOSARIO.md` van dentro del zip? En el TP1 quedaron afuera | Entregables | Plan, §2 | Ola 8 |
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

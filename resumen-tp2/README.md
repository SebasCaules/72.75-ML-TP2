# resumen-tp2

`resumen-tp2.pdf`: documento de lectura para el grupo del TP2 (72.75, 2C 2026). Resume qué pide el
enunciado, qué trae el dataset Bank Marketing y qué hay que decidir antes del envío del 06/10.
Creado el 25/09/2026 con la skill `/dossier`, perfil breve — texto normal — imágenes más — lector
equipo — formato digital. Rehecho el mismo día con la versión 0.3: los diagramas de la página 2
(carriles de entrenamiento y test) y de la página 3 (grupos de variables con `duration` después del
momento de predecir) son nuevos, y ya no lleva la foto de la página 1 del enunciado.

## Regenerar

Desde esta carpeta:

```
python3 graficos.py              # escribe fig/ leyendo ../data/raw/bank-additional-full.csv
python3 ~/.claude/skills/dossier/scripts/medir.py resumen-tp2.tex --paginas 6 --paginas-texto 1.8 --densa 550 --visuales-min 9
```

`medir.py` compila (los auxiliares van a `_build/`) y controla el largo, el texto y las
piezas visuales del perfil. Sin la skill a mano alcanza con `latexmk -lualatex resumen-tp2.tex`.

## Fuentes

- **TP2, p. N**: el enunciado, `../TP2-Clasificación supervisada.pdf`.
- **CSV**: `../data/raw/bank-additional-full.csv`. Todos los gráficos y las cifras que llevan
  esta fuente salen de un cálculo propio sobre ese archivo, hecho en `graficos.py`. Las cifras coinciden
  con las de la wiki de la materia (`wiki/fuentes/dataset-bank-marketing.md`), que un crítico
  independiente recalculó.
- Las cifras y gráficos de este resumen son del **CSV completo, test incluido**: describen el
  dataset, no justifican decisiones del modelo, que salen sólo de train (`../DECISIONES.md`, D-04).
- **names §N**: `../data/raw/bank-additional-names.txt`, por sección.
- El año de cada fila es inferido, porque el archivo no lo trae: está ordenado por fecha
  (names §4) y cada vez que el mes retrocede empieza un año nuevo.
- Lo marcado como **propuesta** (las fechas intermedias de los próximos pasos) y como
  **inferencia** no sale de ninguna fuente.
- Paleta: la de `dossier.sty`. Las series usan la paleta validada de la skill dataviz: azul para
  «no» y naranja para «yes» en todo el documento.

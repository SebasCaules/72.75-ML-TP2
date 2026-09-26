# Números del TP2

Generado por `src/numeros.py` (`python3 -m src.numeros`); no editar a mano. Cada fila es un macro de `informe/numeros.tex`: en el deck se escribe el macro, nunca la cifra, y en el guion se escribe el valor de esta tabla. Todo número del guion tiene que figurar en la columna **Valor** (gate G3 del plan); los números de las otras columnas y de los títulos no cuentan. La columna **Fuente** dice de qué archivo y columna sale cada cifra.

Formato: coma decimal; tres decimales en las métricas (AUC, recall, precisión); cuatro en los errores estándar y en los umbrales de un error estándar; los Δ con signo, con tres decimales desde una centésima y cuatro por debajo; los porcentajes con un decimal. En LaTeX, la coma decimal va como `{,}`, los miles y el signo % van con espacio fino y el menos es `\signoMenos`, para que el macro se lea igual en texto y dentro de `$…$`; aquí, con coma, espacio común y «−», como en el guion.

Resultados leídos de `resultados/`.

## Pendientes

Estos macros valen «?» porque falta su entrada. El deck compila igual, y se completan solos al volver a correr `python3 -m src.numeros`.

- `resultados/evidencia_particion.json`: `\semillasSinEstratificar`, `\desvioPctYesTestSinEstratificar`, `\pctYesTestSinEstratificarMinimo`, `\pctYesTestSinEstratificarMaximo`, `\pctYesTrainTemporal`, `\pctYesTestTemporal`
- `resultados/evaluacion_test.json`: `\yesTest`, `\pctYesTest`, `\diferenciaAucTestValidacion`, `\diferenciaRecallTestValidacion`

## Slide 2 · El problema

| Macro | Valor | Fuente |
|---|---|---|
| `\filasCsv` | 41 188 | `data/raw/bank-additional-names.txt`, §5: Number of Instances |
| `\predictoras` | 20 | `data/raw/bank-additional-names.txt`, §6: Number of Attributes (sin y) |
| `\filasTrain` | 32 940 | train (`cargar_train()`): filas |
| `\yesTrain` | 3 711 | train (`cargar_train()`): filas con y = yes |
| `\noTrain` | 29 229 | train (`cargar_train()`): filas con y = no |
| `\pctYesTrain` | 11,3 % | train (`cargar_train()`): % de y = yes |
| `\exactitudSiempreNo` | 88,7 % | train (`cargar_train()`): % de y = no, la exactitud de decir siempre «no» |
| `\razonNoYes` | 7,9 | train (`cargar_train()`): filas «no» por cada «yes» |

## Slide 3 · Qué datos deciden

| Macro | Valor | Fuente |
|---|---|---|
| `\filasSinDuplicados` | 41 176 | `src/evaluar_test.py`: N_TOTAL, el CSV sin duplicados (D-01) |
| `\duplicados` | 12 | `data/raw/bank-additional-names.txt`, §5, menos N_TOTAL de `src/evaluar_test.py` (D-01) |
| `\filasTest` | 8 236 | `src/evaluar_test.py`: N_TOTAL menos filasTrain (D-02) |
| `\llamadasTest` | 1 647 | `src/metricas.py`: llamadas(filasTest), el presupuesto de D-20 |
| `\proporcionTrain` | 80 % | `src/datos.py`: 1 − PROP_TEST |
| `\proporcionTest` | 20 % | `src/datos.py`: PROP_TEST |
| `\kFolds` | 5 | `resultados/modelo_elegido.json`: k_folds |
| `\semilla` | 42 | `resultados/modelo_elegido.json`: semilla |
| `\filasFoldEntrenamiento` | 26 352 | `resultados/robustez_folds.csv`: barajado, fold 1, n_train |
| `\filasFoldValidacion` | 6 588 | `resultados/robustez_folds.csv`: barajado, fold 1, n_validacion |
| `\llamadasFoldValidacion` | 1 318 | `src/metricas.py`: llamadas(filasFoldValidacion) |
| `\llamadasTrain` | 6 588 | `src/metricas.py`: llamadas(filasTrain), el corte de la lista fuera de fold |
| `\semillasSinEstratificar` | ? | `resultados/evidencia_particion.json`: n_semillas, las particiones sin estratificar (D-02) (pendiente) |
| `\desvioPctYesTestSinEstratificar` | ? | `resultados/evidencia_particion.json`: sin_estratificar.desvio_pp, el desvío del % de «yes» del test entre semillas, en puntos porcentuales (D-02) (pendiente) |
| `\pctYesTestSinEstratificarMinimo` | ? | `resultados/evidencia_particion.json`: sin_estratificar.minimo_pct, el menor % de «yes» del test entre semillas (D-02) (pendiente) |
| `\pctYesTestSinEstratificarMaximo` | ? | `resultados/evidencia_particion.json`: sin_estratificar.maximo_pct, el mayor (D-02) (pendiente) |
| `\pctYesTrainTemporal` | ? | `resultados/evidencia_particion.json`: temporal.pct_yes_train, % de «yes» de train si el test fuera el último 20 % del archivo (D-03) (pendiente) |
| `\pctYesTestTemporal` | ? | `resultados/evidencia_particion.json`: temporal.pct_yes_test, el del test (D-03) (pendiente) |

## Slide 4 · EDA: la duración es fuga

| Macro | Valor | Fuente |
|---|---|---|
| `\durationDecilCorto` | 59 | train (`cargar_train()`): duration, borde superior del primer decil (s) |
| `\durationDecilLargo` | 551 | train (`cargar_train()`): duration, borde inferior del último decil (s) |
| `\pctYesDurationDecilCorto` | 0,0 % | train (`cargar_train()`): % de «yes» en el primer decil de duration |
| `\pctYesDurationDecilLargo` | 46,1 % | train (`cargar_train()`): % de «yes» en el último decil de duration |
| `\filasDurationCero` | 4 | train (`cargar_train()`): filas con duration = 0 (todas «no») |
| `\aucDurationSola` | 0,818 | train (`cargar_train()`): AUC de duration sola como puntaje |
| `\predictorasDisponibles` | 19 | `data/raw/bank-additional-names.txt`, §6, menos las predictoras que `src/datos.py` deja fuera del modelo (EXCLUIDAS sin fila: duration); D-05 y D-07 |
| `\aucRfConDuration` | 0,939 | `resultados/ablaciones_resumen.csv`: rf, A1, auc_variante (techo con duration) |
| `\aucNbGaussianoConDuration` | 0,831 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A1, auc_variante (techo con duration) |
| `\aucSvmConDuration` | 0,907 | `resultados/ablaciones_resumen.csv`: svm, A1, auc_variante (techo con duration) |
| `\aucKnnConDuration` | 0,911 | `resultados/ablaciones_resumen.csv`: knn, A1, auc_variante (techo con duration) |
| `\deltaAucDurationNbGaussiano` | +0,063 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A1, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucDurationNbGaussianoDesvio` | 0,0055 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A1, delta_desvio |
| `\deltaAucDurationSvm` | +0,206 | `resultados/ablaciones_resumen.csv`: svm, A1, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucDurationSvmDesvio` | 0,016 | `resultados/ablaciones_resumen.csv`: svm, A1, delta_desvio |
| `\deltaAucDurationKnn` | +0,156 | `resultados/ablaciones_resumen.csv`: knn, A1, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucDurationKnnDesvio` | 0,013 | `resultados/ablaciones_resumen.csv`: knn, A1, delta_desvio |
| `\deltaAucDurationRf` | +0,170 | `resultados/ablaciones_resumen.csv`: rf, A1, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucDurationRfDesvio` | 0,0072 | `resultados/ablaciones_resumen.csv`: rf, A1, delta_desvio |

## Slide 5 · EDA: el archivo está ordenado por fecha

| Macro | Valor | Fuente |
|---|---|---|
| `\anioInicio` | 2008 | `data/raw/bank-additional-names.txt`, §4: primer año del archivo |
| `\anioFin` | 2010 | `data/raw/bank-additional-names.txt`, §4: último año del archivo |
| `\filasBloqueEda` | 2 000 | `src/eda_html.py`: BLOQUE, filas del CSV original por bloque |
| `\pctYesPrimerBloqueEda` | 1,9 % | train (`cargar_train()`): % de «yes» en el primer bloque de BLOQUE filas |
| `\pctYesUltimoBloqueEda` | 50,9 % | train (`cargar_train()`): % de «yes» en el último bloque de BLOQUE filas |
| `\factorYesBloquesEda` | 27,1 | train (`cargar_train()`): % de «yes» del último bloque / del primero |
| `\pctYesDosMilOcho` | 4,9 % | train (`cargar_train()`): % de «yes» en 2008 (año inferido del orden de los meses, como eda.html) |
| `\pctYesDosMilNueve` | 19,2 % | train (`cargar_train()`): % de «yes» en 2009 (año inferido del orden de los meses, como eda.html) |
| `\pctYesDosMilDiez` | 51,9 % | train (`cargar_train()`): % de «yes» en 2010 (año inferido del orden de los meses, como eda.html) |
| `\factorYesAnios` | 10,5 | train (`cargar_train()`): % de «yes» de 2010 / de 2008 |
| `\euriborPrimerBloqueEda` | 4,86 | train (`cargar_train()`): euribor3m medio del primer bloque de BLOQUE filas |
| `\euriborMinimoBloqueEda` | 0,71 | train (`cargar_train()`): euribor3m medio del bloque más bajo |
| `\euriborFoldSinBarajarPrimero` | 4,87 | train (`cargar_train()`): euribor3m medio del fold 1 con StratifiedKFold(5) sin barajar, cv=5 (D-06) |
| `\euriborFoldSinBarajarUltimo` | 1,10 | train (`cargar_train()`): ídem, fold 5 (D-06) |
| `\euriborFoldBarajadoMinimo` | 3,58 | train (`cargar_train()`): euribor3m medio, el menor de los folds() (D-06) |
| `\euriborFoldBarajadoMaximo` | 3,64 | train (`cargar_train()`): ídem, el mayor (D-06) |
| `\pctSeptiembreUltimoTramo` | 100,0 % | train (`cargar_train()`): % de las filas de month = sep que caen en el último 20 % de train por fecha (A8) |
| `\pctOctubreUltimoTramo` | 90,9 % | train (`cargar_train()`): % de las filas de month = oct que caen en el último 20 % de train por fecha (A8) |
| `\pctDiciembreUltimoTramo` | 94,5 % | train (`cargar_train()`): % de las filas de month = dec que caen en el último 20 % de train por fecha (A8) |

## Slide 6 · EDA: 999 no es «nunca contactado»

| Macro | Valor | Fuente |
|---|---|---|
| `\pdaysCentinela` | 999 | `src/eda_html.py`: PDAYS_CENTINELA, el pdays de quien no fue contactado en una campaña anterior (D-10) |
| `\pctPdaysCentinela` | 96,3 % | train (`cargar_train()`): % de filas con pdays = 999 |
| `\filasCentinelaConPrevio` | 3 302 | train (`cargar_train()`): filas con pdays = 999 y previous >= 1 |
| `\filasPdaysContactados` | 1 207 | train (`cargar_train()`): filas con pdays < 999 |
| `\pctYesPdaysContactados` | 64,6 % | train (`cargar_train()`): % de «yes» con pdays < 999 |
| `\pctYesPdaysCentinela` | 9,2 % | train (`cargar_train()`): % de «yes» con pdays = 999 |
| `\pctPoutcomeInexistente` | 86,3 % | train (`cargar_train()`): % de filas con poutcome = nonexistent |
| `\pctYesPoutcomeExito` | 66,0 % | train (`cargar_train()`): % de «yes» con poutcome = success |
| `\deltaAucSinPdaysNbGaussiano` | −0,0012 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A2, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinPdaysNbGaussianoDesvio` | 0,0003 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A2, delta_desvio |
| `\deltaAucSinPdaysSvm` | +0,0023 | `resultados/ablaciones_resumen.csv`: svm, A2, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinPdaysSvmDesvio` | 0,0037 | `resultados/ablaciones_resumen.csv`: svm, A2, delta_desvio |
| `\deltaAucSinPdaysKnn` | −0,0019 | `resultados/ablaciones_resumen.csv`: knn, A2, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinPdaysKnnDesvio` | 0,0007 | `resultados/ablaciones_resumen.csv`: knn, A2, delta_desvio |
| `\deltaAucSinPdaysRf` | +0,0003 | `resultados/ablaciones_resumen.csv`: rf, A2, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinPdaysRfDesvio` | 0,0022 | `resultados/ablaciones_resumen.csv`: rf, A2, delta_desvio |

## Slide 7 · EDA: el pipeline y la consecuencia de cada decisión

| Macro | Valor | Fuente |
|---|---|---|
| `\columnasReferencia` | 62 | `src/preproceso.py`: columnas del one-hot con Opciones(), la referencia de las ablaciones |
| `\columnasFinales` | 58 | `src/preproceso.py`: columnas del one-hot con OPCIONES_FINALES (`src/configuracion.py`) |
| `\columnasDuration` | 63 | `resultados/ablaciones_resumen.csv`: A1, columnas |
| `\columnasSinPdays` | 61 | `resultados/ablaciones_resumen.csv`: A2, columnas |
| `\columnasDefaultIndicadora` | 60 | `resultados/ablaciones_resumen.csv`: A3, columnas |
| `\columnasRaras` | 60 | `resultados/ablaciones_resumen.csv`: A4, columnas |
| `\columnasLogCampaign` | 62 | `resultados/ablaciones_resumen.csv`: A5, columnas |
| `\columnasMacroReducido` | 59 | `resultados/ablaciones_resumen.csv`: A6, columnas |
| `\columnasSinMacro` | 57 | `resultados/ablaciones_resumen.csv`: A7, columnas |
| `\columnasSinMacroNiMonth` | 47 | `resultados/ablaciones_resumen.csv`: A8, columnas |
| `\columnasSinDiaSemana` | 57 | `resultados/ablaciones_resumen.csv`: A9, columnas |
| `\columnasEdadTramos` | 70 | `resultados/ablaciones_resumen.csv`: A10, columnas |
| `\columnasImputarModa` | 56 | `resultados/ablaciones_resumen.csv`: A11, columnas |
| `\columnasSinEscalar` | 62 | `resultados/ablaciones_resumen.csv`: A12, columnas |
| `\aucReferenciaAblacionRf` | 0,770 | `resultados/ablaciones_resumen.csv`: rf, auc_referencia (A0) |
| `\aucReferenciaAblacionNbGaussiano` | 0,768 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, auc_referencia (A0) |
| `\aucReferenciaAblacionSvm` | 0,701 | `resultados/ablaciones_resumen.csv`: svm, auc_referencia (A0) |
| `\aucReferenciaAblacionKnn` | 0,755 | `resultados/ablaciones_resumen.csv`: knn, auc_referencia (A0) |
| `\aucRfSinMacro` | 0,742 | `resultados/ablaciones_resumen.csv`: rf, A7, auc_variante |
| `\aucNbGaussianoSinMacro` | 0,746 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A7, auc_variante |
| `\aucSvmSinMacro` | 0,697 | `resultados/ablaciones_resumen.csv`: svm, A7, auc_variante |
| `\aucKnnSinMacro` | 0,708 | `resultados/ablaciones_resumen.csv`: knn, A7, auc_variante |
| `\aucRfSinMacroNiMonth` | 0,676 | `resultados/ablaciones_resumen.csv`: rf, A8, auc_variante |
| `\aucNbGaussianoSinMacroNiMonth` | 0,718 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A8, auc_variante |
| `\aucSvmSinMacroNiMonth` | 0,653 | `resultados/ablaciones_resumen.csv`: svm, A8, auc_variante |
| `\aucKnnSinMacroNiMonth` | 0,679 | `resultados/ablaciones_resumen.csv`: knn, A8, auc_variante |
| `\deltaAucDefaultIndicadoraNbGaussiano` | −0,0003 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A3, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucDefaultIndicadoraNbGaussianoDesvio` | 0,0017 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A3, delta_desvio |
| `\deltaAucDefaultIndicadoraSvm` | +0,0004 | `resultados/ablaciones_resumen.csv`: svm, A3, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucDefaultIndicadoraSvmDesvio` | 0,0012 | `resultados/ablaciones_resumen.csv`: svm, A3, delta_desvio |
| `\deltaAucDefaultIndicadoraKnn` | +0,0000 | `resultados/ablaciones_resumen.csv`: knn, A3, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucDefaultIndicadoraKnnDesvio` | 0,0028 | `resultados/ablaciones_resumen.csv`: knn, A3, delta_desvio |
| `\deltaAucDefaultIndicadoraRf` | +0,0029 | `resultados/ablaciones_resumen.csv`: rf, A3, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucDefaultIndicadoraRfDesvio` | 0,0021 | `resultados/ablaciones_resumen.csv`: rf, A3, delta_desvio |
| `\deltaAucRarasNbGaussiano` | +0,0008 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A4, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucRarasNbGaussianoDesvio` | 0,0006 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A4, delta_desvio |
| `\deltaAucRarasSvm` | +0,0002 | `resultados/ablaciones_resumen.csv`: svm, A4, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucRarasSvmDesvio` | 0,0007 | `resultados/ablaciones_resumen.csv`: svm, A4, delta_desvio |
| `\deltaAucRarasKnn` | +0,0001 | `resultados/ablaciones_resumen.csv`: knn, A4, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucRarasKnnDesvio` | 0,0006 | `resultados/ablaciones_resumen.csv`: knn, A4, delta_desvio |
| `\deltaAucRarasRf` | −0,0000 | `resultados/ablaciones_resumen.csv`: rf, A4, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucRarasRfDesvio` | 0,0014 | `resultados/ablaciones_resumen.csv`: rf, A4, delta_desvio |
| `\deltaAucLogCampaignSvm` | −0,0006 | `resultados/ablaciones_resumen.csv`: svm, A5, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucLogCampaignSvmDesvio` | 0,0020 | `resultados/ablaciones_resumen.csv`: svm, A5, delta_desvio |
| `\deltaAucLogCampaignKnn` | +0,0008 | `resultados/ablaciones_resumen.csv`: knn, A5, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucLogCampaignKnnDesvio` | 0,0026 | `resultados/ablaciones_resumen.csv`: knn, A5, delta_desvio |
| `\deltaAucMacroReducidoNbGaussiano` | −0,0041 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A6, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucMacroReducidoNbGaussianoDesvio` | 0,0016 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A6, delta_desvio |
| `\deltaAucMacroReducidoSvm` | +0,0030 | `resultados/ablaciones_resumen.csv`: svm, A6, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucMacroReducidoSvmDesvio` | 0,0083 | `resultados/ablaciones_resumen.csv`: svm, A6, delta_desvio |
| `\deltaAucMacroReducidoKnn` | −0,0083 | `resultados/ablaciones_resumen.csv`: knn, A6, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucMacroReducidoKnnDesvio` | 0,0030 | `resultados/ablaciones_resumen.csv`: knn, A6, delta_desvio |
| `\deltaAucMacroReducidoRf` | +0,0005 | `resultados/ablaciones_resumen.csv`: rf, A6, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucMacroReducidoRfDesvio` | 0,0028 | `resultados/ablaciones_resumen.csv`: rf, A6, delta_desvio |
| `\deltaAucSinMacroNbGaussiano` | −0,023 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A7, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinMacroNbGaussianoDesvio` | 0,0037 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A7, delta_desvio |
| `\deltaAucSinMacroSvm` | −0,0037 | `resultados/ablaciones_resumen.csv`: svm, A7, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinMacroSvmDesvio` | 0,011 | `resultados/ablaciones_resumen.csv`: svm, A7, delta_desvio |
| `\deltaAucSinMacroKnn` | −0,048 | `resultados/ablaciones_resumen.csv`: knn, A7, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinMacroKnnDesvio` | 0,0080 | `resultados/ablaciones_resumen.csv`: knn, A7, delta_desvio |
| `\deltaAucSinMacroRf` | −0,028 | `resultados/ablaciones_resumen.csv`: rf, A7, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinMacroRfDesvio` | 0,0041 | `resultados/ablaciones_resumen.csv`: rf, A7, delta_desvio |
| `\deltaAucSinMacroNiMonthNbGaussiano` | −0,050 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A8, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinMacroNiMonthNbGaussianoDesvio` | 0,0046 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A8, delta_desvio |
| `\deltaAucSinMacroNiMonthSvm` | −0,048 | `resultados/ablaciones_resumen.csv`: svm, A8, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinMacroNiMonthSvmDesvio` | 0,0060 | `resultados/ablaciones_resumen.csv`: svm, A8, delta_desvio |
| `\deltaAucSinMacroNiMonthKnn` | −0,076 | `resultados/ablaciones_resumen.csv`: knn, A8, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinMacroNiMonthKnnDesvio` | 0,0073 | `resultados/ablaciones_resumen.csv`: knn, A8, delta_desvio |
| `\deltaAucSinMacroNiMonthRf` | −0,093 | `resultados/ablaciones_resumen.csv`: rf, A8, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinMacroNiMonthRfDesvio` | 0,0048 | `resultados/ablaciones_resumen.csv`: rf, A8, delta_desvio |
| `\deltaAucSinDiaSemanaNbGaussiano` | −0,0000 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A9, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinDiaSemanaNbGaussianoDesvio` | 0,0002 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A9, delta_desvio |
| `\deltaAucSinDiaSemanaSvm` | −0,0074 | `resultados/ablaciones_resumen.csv`: svm, A9, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinDiaSemanaSvmDesvio` | 0,0021 | `resultados/ablaciones_resumen.csv`: svm, A9, delta_desvio |
| `\deltaAucSinDiaSemanaKnn` | −0,0014 | `resultados/ablaciones_resumen.csv`: knn, A9, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinDiaSemanaKnnDesvio` | 0,0061 | `resultados/ablaciones_resumen.csv`: knn, A9, delta_desvio |
| `\deltaAucSinDiaSemanaRf` | −0,0034 | `resultados/ablaciones_resumen.csv`: rf, A9, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinDiaSemanaRfDesvio` | 0,0038 | `resultados/ablaciones_resumen.csv`: rf, A9, delta_desvio |
| `\deltaAucEdadTramosNbGaussiano` | +0,0001 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A10, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucEdadTramosNbGaussianoDesvio` | 0,0022 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A10, delta_desvio |
| `\deltaAucImputarModaNbGaussiano` | +0,0000 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A11, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucImputarModaNbGaussianoDesvio` | 0,0039 | `resultados/ablaciones_resumen.csv`: nb_gaussiano, A11, delta_desvio |
| `\deltaAucImputarModaSvm` | −0,0002 | `resultados/ablaciones_resumen.csv`: svm, A11, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucImputarModaSvmDesvio` | 0,0049 | `resultados/ablaciones_resumen.csv`: svm, A11, delta_desvio |
| `\deltaAucImputarModaKnn` | +0,0001 | `resultados/ablaciones_resumen.csv`: knn, A11, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucImputarModaKnnDesvio` | 0,0022 | `resultados/ablaciones_resumen.csv`: knn, A11, delta_desvio |
| `\deltaAucImputarModaRf` | +0,0011 | `resultados/ablaciones_resumen.csv`: rf, A11, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucImputarModaRfDesvio` | 0,0034 | `resultados/ablaciones_resumen.csv`: rf, A11, delta_desvio |
| `\deltaAucSinEscalarKnn` | +0,0084 | `resultados/ablaciones_resumen.csv`: knn, A12, delta_media (ΔAUC pareado contra A0) |
| `\deltaAucSinEscalarKnnDesvio` | 0,0067 | `resultados/ablaciones_resumen.csv`: knn, A12, delta_desvio |
| `\pctDefaultUnknown` | 20,8 % | train (`cargar_train()`): % de filas con default = unknown |
| `\pctYesDefaultUnknown` | 5,3 % | train (`cargar_train()`): % de «yes» con default = unknown |
| `\pctYesDefaultConocido` | 12,8 % | train (`cargar_train()`): % de «yes» con default conocido |
| `\filasDefaultYes` | 2 | train (`cargar_train()`): filas con default = yes (D-09) |
| `\mediaCampaignYes` | 2,053 | train (`cargar_train()`): media de campaign en los «yes» (D-07) |
| `\mediaCampaignNo` | 2,627 | train (`cargar_train()`): media de campaign en los «no» (D-07) |
| `\medianaCampaignYes` | 2 | train (`cargar_train()`): mediana de campaign en los «yes» (D-07) |
| `\medianaCampaignNo` | 2 | train (`cargar_train()`): mediana de campaign en los «no» (D-07) |
| `\maximoCampaign` | 56 | train (`cargar_train()`): máximo de campaign (D-13) |
| `\asimetriaCampaign` | 4,89 | train (`cargar_train()`): asimetría de campaign, pandas skew (D-13) |
| `\factorIqr` | 1,5 | `src/numeros.py`: FACTOR_IQR, la regla de Tukey de `resultados/eda/reporte.txt`, §5 (D-17) |
| `\filasAtipicasIqr` | 6 731 | train (`cargar_train()`): filas con algún valor fuera de 1,5·IQR en las 9 numéricas del modelo (D-17) |
| `\pctAtipicasIqr` | 20,4 % | train (`cargar_train()`): ídem, en % de las filas (D-17) |
| `\atipicosPrevious` | 4 509 | train (`cargar_train()`): filas con previous fuera de 1,5·IQR (D-17) |
| `\pctYesAtipicosPrevious` | 26,6 % | train (`cargar_train()`): % de «yes» entre los atípicos de previous (D-17) |
| `\atipicosCampaign` | 1 911 | train (`cargar_train()`): filas con campaign fuera de 1,5·IQR (D-17) |
| `\pctYesAtipicosCampaign` | 4,5 % | train (`cargar_train()`): % de «yes» entre los atípicos de campaign (D-17) |
| `\atipicosPdays` | 1 207 | train (`cargar_train()`): filas con pdays fuera de 1,5·IQR (D-17) |
| `\pctYesAtipicosPdays` | 64,6 % | train (`cargar_train()`): % de «yes» entre los atípicos de pdays (D-17) |
| `\atipicosAge` | 383 | train (`cargar_train()`): filas con age fuera de 1,5·IQR (D-17) |
| `\pctYesAtipicosAge` | 46,5 % | train (`cargar_train()`): % de «yes» entre los atípicos de age (D-17) |
| `\atipicosConsConfIdx` | 357 | train (`cargar_train()`): filas con cons.conf.idx fuera de 1,5·IQR (D-17) |
| `\pctYesAtipicosConsConfIdx` | 40,6 % | train (`cargar_train()`): % de «yes» entre los atípicos de cons.conf.idx (D-17) |

## Slide 8 · Métricas y presupuesto de llamadas

| Macro | Valor | Fuente |
|---|---|---|
| `\aucSinModelo` | 0,500 | `resultados/cv_final_resumen.csv`: sin_modelo, validacion, auc, media |
| `\aucAzar` | 0,5 | `resultados/cv_final_resumen.csv`: sin_modelo, validacion, auc, media, con un decimal |
| `\recallSinModelo` | 0,200 | `resultados/cv_final_resumen.csv`: sin_modelo, validacion, recall_q, media |
| `\precisionSinModelo` | 0,113 | `resultados/cv_final_resumen.csv`: sin_modelo, validacion, precision_q, media |
| `\apSinModelo` | 0,113 | `resultados/cv_final_resumen.csv`: sin_modelo, validacion, ap, media |
| `\fUnoSinModelo` | 0,144 | `resultados/cv_final_resumen.csv`: sin_modelo, validacion, f1_q, media |
| `\presupuesto` | 20 % | `src/metricas.py`: PRESUPUESTO (D-20) |
| `\presupuestoBajo` | 10 % | `resultados/sensibilidad_q_final.csv`: el menor q |
| `\presupuestoAlto` | 30 % | `resultados/sensibilidad_q_final.csv`: el mayor q |
| `\rotuloLineaBase` | sin modelo | `src/estilo.py`: ROTULO_LINEA_BASE (texto) |

## Slide 9 · Los modelos sin ajustar

| Macro | Valor | Fuente |
|---|---|---|
| `\aucNbReferencia` | 0,782 | `resultados/cv_referencia_resumen.csv`: nb_categorico, validacion, auc, media |
| `\aucNbReferenciaDesvio` | 0,007 | `resultados/cv_referencia_resumen.csv`: nb_categorico, validacion, auc, desvio |
| `\recallNbReferencia` | 0,614 | `resultados/cv_referencia_resumen.csv`: nb_categorico, validacion, recall_q, media |
| `\aucTrainNbReferencia` | 0,783 | `resultados/cv_referencia_resumen.csv`: nb_categorico, train, auc, media |
| `\brechaNbReferencia` | 0,001 | `resultados/cv_referencia_resumen.csv`: nb_categorico, brecha, auc, media (train − validación) |
| `\aucNbGaussianoVal` | 0,769 | `resultados/cv_referencia_resumen.csv`: nb_gaussiano, validacion, auc, media |
| `\aucNbGaussianoValDesvio` | 0,009 | `resultados/cv_referencia_resumen.csv`: nb_gaussiano, validacion, auc, desvio |
| `\recallNbGaussianoVal` | 0,566 | `resultados/cv_referencia_resumen.csv`: nb_gaussiano, validacion, recall_q, media |
| `\aucTrainNbGaussianoVal` | 0,771 | `resultados/cv_referencia_resumen.csv`: nb_gaussiano, train, auc, media |
| `\brechaNbGaussianoVal` | 0,002 | `resultados/cv_referencia_resumen.csv`: nb_gaussiano, brecha, auc, media (train − validación) |
| `\aucSvmRbfReferencia` | 0,702 | `resultados/cv_referencia_resumen.csv`: svm, validacion, auc, media |
| `\aucSvmRbfReferenciaDesvio` | 0,009 | `resultados/cv_referencia_resumen.csv`: svm, validacion, auc, desvio |
| `\recallSvmRbfReferencia` | 0,549 | `resultados/cv_referencia_resumen.csv`: svm, validacion, recall_q, media |
| `\aucTrainSvmRbfReferencia` | 0,901 | `resultados/cv_referencia_resumen.csv`: svm, train, auc, media |
| `\brechaSvmRbfReferencia` | 0,200 | `resultados/cv_referencia_resumen.csv`: svm, brecha, auc, media (train − validación) |
| `\aucKnnReferencia` | 0,755 | `resultados/cv_referencia_resumen.csv`: knn, validacion, auc, media |
| `\aucKnnReferenciaDesvio` | 0,012 | `resultados/cv_referencia_resumen.csv`: knn, validacion, auc, desvio |
| `\recallKnnReferencia` | 0,593 | `resultados/cv_referencia_resumen.csv`: knn, validacion, recall_q, media |
| `\aucTrainKnnReferencia` | 0,871 | `resultados/cv_referencia_resumen.csv`: knn, train, auc, media |
| `\brechaKnnReferencia` | 0,116 | `resultados/cv_referencia_resumen.csv`: knn, brecha, auc, media (train − validación) |
| `\aucRfReferencia` | 0,772 | `resultados/cv_referencia_resumen.csv`: rf, validacion, auc, media |
| `\aucRfReferenciaDesvio` | 0,006 | `resultados/cv_referencia_resumen.csv`: rf, validacion, auc, desvio |
| `\recallRfReferencia` | 0,605 | `resultados/cv_referencia_resumen.csv`: rf, validacion, recall_q, media |
| `\aucTrainRfReferencia` | 1,000 | `resultados/cv_referencia_resumen.csv`: rf, train, auc, media |
| `\brechaRfReferencia` | 0,228 | `resultados/cv_referencia_resumen.csv`: rf, brecha, auc, media (train − validación) |
| `\modeloMejorReferencia` | Naive Bayes | `resultados/cv_referencia_resumen.csv`: el de mayor AUC de validación entre los cuatro del enunciado (texto) |
| `\margenAucReferencia` | 0,282 | `resultados/cv_referencia_resumen.csv`: el mayor AUC de validación de los cuatro menos el de sin_modelo |
| `\deltaAucNbCategorico` | +0,013 | `resultados/cv_referencia.csv`: nb_categorico − nb_gaussiano, AUC de validación fold a fold, media (D-18) |
| `\deltaAucNbCategoricoDesvio` | 0,0024 | `resultados/cv_referencia.csv`: ídem, desvío |
| `\deltaAucNbCategoricoMinimo` | +0,010 | `resultados/cv_referencia.csv`: ídem, el menor fold |
| `\deltaAucNbCategoricoMaximo` | +0,016 | `resultados/cv_referencia.csv`: ídem, el mayor fold |
| `\deltaRecallNbCategorico` | +0,048 | `resultados/cv_referencia.csv`: nb_categorico − nb_gaussiano, recall_q de validación fold a fold, media (D-18) |
| `\deltaRecallNbCategoricoDesvio` | 0,0083 | `resultados/cv_referencia.csv`: ídem, desvío |
| `\vecinosKnnReferencia` | 15 | `resultados/cv_referencia_resumen.csv`: knn, configuracion, n_neighbors |
| `\arbolesRfReferencia` | 300 | `resultados/cv_referencia_resumen.csv`: rf, configuracion, n_estimators |
| `\cSvmReferencia` | 1 | `resultados/cv_referencia_resumen.csv`: svm, configuracion, C |
| `\cortesNb` | 10 | `resultados/cv_final_resumen.csv`: nb_categorico, configuracion, n_cortes (D-18, D-22) |
| `\alfaNb` | 1 | `resultados/cv_final_resumen.csv`: nb_categorico, configuracion, alpha, la corrección de Laplace (D-18) |

## Slide 10 · Curva de RF

| Macro | Valor | Fuente |
|---|---|---|
| `\profundidadRf` | 8 | `resultados/cv_final_resumen.csv`: rf, configuracion, max_depth |
| `\arbolesRf` | 200 | `resultados/cv_final_resumen.csv`: rf, configuracion, n_estimators |
| `\aucRfProfundidadElegida` | 0,795 | `resultados/hiperparametros.json`: curvas.rf_max_depth.auc_validacion_media |
| `\errorEstandarRfProfundidadElegida` | 0,0025 | `resultados/hiperparametros.json`: curvas.rf_max_depth.error_estandar |
| `\brechaRfProfundidadElegida` | 0,030 | `resultados/hiperparametros.json`: curvas.rf_max_depth.brecha |
| `\profundidadRfMejor` | 10 | `resultados/hiperparametros.json`: curvas.rf_max_depth.mejor |
| `\aucRfProfundidadMejor` | 0,797 | `resultados/hiperparametros.json`: curvas.rf_max_depth.auc_validacion_mejor |
| `\umbralUnoEsRfProfundidad` | 0,7939 | `resultados/hiperparametros.json`: curvas.rf_max_depth.umbral_1es |
| `\errorEstandarRfProfundidadMejor` | 0,0029 | `resultados/hiperparametros.json`: curvas.rf_max_depth.error_estandar_mejor |
| `\profundidadRfMinima` | 2 | `src/modelos.py`: GRILLAS, la menor max_depth de la grilla de RF (el extremo del subajuste) |
| `\arbolesRfRegla` | 25 | `resultados/hiperparametros.json`: curvas.rf_n_estimators_depth8.valor, lo que elegía la regla de 1 ES (N0-11) |
| `\arbolesRfMejor` | 800 | `resultados/hiperparametros.json`: curvas.rf_n_estimators_depth8.mejor (N0-11) |
| `\umbralUnoEsRfArboles` | 0,7930 | `resultados/hiperparametros.json`: curvas.rf_n_estimators_depth8.umbral_1es (N0-11) |
| `\aucRfProfundidadDos` | 0,782 | `resultados/curvas/rf_max_depth.csv`: punto 2, validacion, auc, media |
| `\aucRfProfundidadCuatro` | 0,789 | `resultados/curvas/rf_max_depth.csv`: punto 4, validacion, auc, media |
| `\aucRfProfundidadSeis` | 0,794 | `resultados/curvas/rf_max_depth.csv`: punto 6, validacion, auc, media |
| `\aucRfProfundidadOcho` | 0,795 | `resultados/curvas/rf_max_depth.csv`: punto 8, validacion, auc, media |
| `\aucRfProfundidadDiez` | 0,797 | `resultados/curvas/rf_max_depth.csv`: punto 10, validacion, auc, media |
| `\aucRfProfundidadDoce` | 0,795 | `resultados/curvas/rf_max_depth.csv`: punto 12, validacion, auc, media |
| `\aucRfProfundidadQuince` | 0,793 | `resultados/curvas/rf_max_depth.csv`: punto 15, validacion, auc, media |
| `\aucRfProfundidadVeinte` | 0,784 | `resultados/curvas/rf_max_depth.csv`: punto 20, validacion, auc, media |
| `\aucRfProfundidadVeinticinco` | 0,776 | `resultados/curvas/rf_max_depth.csv`: punto 25, validacion, auc, media |
| `\aucRfProfundidadSinLimite` | 0,772 | `resultados/curvas/rf_max_depth.csv`: punto None, validacion, auc, media |
| `\aucTrainRfProfundidadSinLimite` | 1,000 | `resultados/curvas/rf_max_depth.csv`: punto None, train, auc, media |
| `\brechaRfProfundidadSinLimite` | 0,228 | `resultados/curvas/rf_max_depth.csv`: punto None, train − validación |
| `\aucTrainRfProfundidadDos` | 0,783 | `resultados/curvas/rf_max_depth.csv`: punto 2, train, auc, media |
| `\brechaRfProfundidadDos` | 0,001 | `resultados/curvas/rf_max_depth.csv`: punto 2, train − validación |
| `\aucRfArbolesDiez` | 0,791 | `resultados/curvas/rf_n_estimators_depth8.csv`: punto 10, validacion, auc, media (N0-11) |
| `\aucRfArbolesVeinticinco` | 0,793 | `resultados/curvas/rf_n_estimators_depth8.csv`: punto 25, validacion, auc, media (N0-11) |
| `\aucRfArbolesCincuenta` | 0,795 | `resultados/curvas/rf_n_estimators_depth8.csv`: punto 50, validacion, auc, media (N0-11) |
| `\aucRfArbolesCien` | 0,795 | `resultados/curvas/rf_n_estimators_depth8.csv`: punto 100, validacion, auc, media (N0-11) |
| `\aucRfArbolesDoscientos` | 0,795 | `resultados/curvas/rf_n_estimators_depth8.csv`: punto 200, validacion, auc, media (N0-11) |
| `\aucRfArbolesCuatrocientos` | 0,796 | `resultados/curvas/rf_n_estimators_depth8.csv`: punto 400, validacion, auc, media (N0-11) |
| `\aucRfArbolesOchocientos` | 0,796 | `resultados/curvas/rf_n_estimators_depth8.csv`: punto 800, validacion, auc, media (N0-11) |
| `\aucRfSinPesos` | 0,795 | `resultados/curvas/rf_pesos_clase_depth8.csv`: punto None, validacion, auc, media (N0-12) |
| `\aucRfPesosBalanceados` | 0,796 | `resultados/curvas/rf_pesos_clase_depth8.csv`: punto balanced, validacion, auc, media (N0-12) |
| `\deltaAucPesosRf` | +0,0006 | `resultados/curvas/rf_pesos_clase_depth8.csv`: balanced − None (N0-12) |
| `\errorEstandarRfPesosBalanceados` | 0,0024 | `resultados/curvas/rf_pesos_clase_depth8.csv`: punto balanced, error estándar (N0-12) |

## Slide 11 · Curva de KNN

| Macro | Valor | Fuente |
|---|---|---|
| `\vecinosKnn` | 801 | `resultados/cv_final_resumen.csv`: knn, configuracion, n_neighbors |
| `\vecinosKnnMejor` | 601 | `resultados/hiperparametros.json`: curvas.knn_n_neighbors_uniform.mejor |
| `\aucKnnMejor` | 0,784 | `resultados/hiperparametros.json`: curvas.knn_n_neighbors_uniform.auc_validacion_mejor |
| `\vecinosKnnMinimoUnoEs` | 151 | `resultados/hiperparametros.json`: el menor de curvas.knn_n_neighbors_uniform.puntos_dentro_1es |
| `\umbralUnoEsKnn` | 0,7806 | `resultados/hiperparametros.json`: curvas.knn_n_neighbors_uniform.umbral_1es |
| `\errorEstandarKnnMejor` | 0,0037 | `resultados/hiperparametros.json`: curvas.knn_n_neighbors_uniform.error_estandar_mejor |
| `\vecinosKnnMinimo` | 1 | `src/modelos.py`: GRILLAS, el menor n_neighbors de la grilla de KNN (el extremo del sobreajuste) |
| `\vecinosKnnMeseta` | 301 | DECISIONES.md, D-22: el k desde el que la curva de KNN es una meseta (una lectura de la curva, no una regla); «?» si ya no está dentro de 1 ES |
| `\aucKnnUno` | 0,619 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 1, validacion, auc, media |
| `\aucKnnTres` | 0,696 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 3, validacion, auc, media |
| `\aucKnnCinco` | 0,724 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 5, validacion, auc, media |
| `\aucKnnNueve` | 0,741 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 9, validacion, auc, media |
| `\aucKnnQuince` | 0,755 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 15, validacion, auc, media |
| `\aucKnnVeinticinco` | 0,765 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 25, validacion, auc, media |
| `\aucKnnCuarentaYUno` | 0,769 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 41, validacion, auc, media |
| `\aucKnnSesentaYUno` | 0,773 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 61, validacion, auc, media |
| `\aucKnnCientoUno` | 0,779 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 101, validacion, auc, media |
| `\aucKnnCientoCincuentaYUno` | 0,782 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 151, validacion, auc, media |
| `\aucKnnDoscientosUno` | 0,783 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 201, validacion, auc, media |
| `\aucKnnTrescientosUno` | 0,783 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 301, validacion, auc, media |
| `\aucKnnCuatrocientosUno` | 0,784 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 401, validacion, auc, media |
| `\aucKnnSeiscientosUno` | 0,784 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 601, validacion, auc, media |
| `\aucKnnOchocientosUno` | 0,784 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 801, validacion, auc, media |
| `\aucTrainKnnUno` | 0,988 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 1, train, auc, media |
| `\brechaKnnUno` | 0,369 | `resultados/curvas/knn_n_neighbors_uniform.csv`: punto 1, train − validación |
| `\vecinosKnnDistancia` | 201 | `resultados/hiperparametros.json`: curvas.knn_n_neighbors_distance.valor (weights = distance) |
| `\aucKnnDistancia` | 0,769 | `resultados/hiperparametros.json`: curvas.knn_n_neighbors_distance.auc_validacion_media |
| `\aucTrainKnnDistancia` | 1,000 | `resultados/hiperparametros.json`: curvas.knn_n_neighbors_distance.auc_train_media |
| `\brechaKnnDistancia` | 0,231 | `resultados/hiperparametros.json`: curvas.knn_n_neighbors_distance.brecha |

## Slide 12 · Curva de la SVM

| Macro | Valor | Fuente |
|---|---|---|
| `\cSvm` | 0,001 | `resultados/cv_final_resumen.csv`: svm, configuracion, C |
| `\kernelSvm` | lineal | `resultados/cv_final_resumen.csv`: svm, configuracion, kernel (texto) |
| `\aucSvmBalanceada` | 0,770 | `resultados/curvas/svm_pesos_clase.csv`: punto balanced, validacion, auc, media (RBF, C = 1) |
| `\deltaAucPesosSvm` | +0,068 | `resultados/curvas/svm_pesos_clase.csv`: balanced − None (D-21, N0-12) |
| `\cSvmSinPesosMejor` | 0,1 | `resultados/hiperparametros.json`: curvas.svm_C.mejor (RBF, sin pesos) |
| `\aucSvmSinPesosMejor` | 0,706 | `resultados/hiperparametros.json`: curvas.svm_C.auc_validacion_mejor |
| `\cSvmBalanceadaMejor` | 0,01 | `resultados/hiperparametros.json`: curvas.svm_C_balanced.mejor (RBF, pesos balanceados) |
| `\aucSvmBalanceadaMejor` | 0,772 | `resultados/hiperparametros.json`: curvas.svm_C_balanced.auc_validacion_mejor |
| `\cSvmKernel` | 0,001 | `resultados/curvas/svm_kernel.csv`: C de la configuración en que se comparan los kernels |
| `\aucSvmKernelLineal` | 0,774 | `resultados/curvas/svm_kernel.csv`: punto linear, validacion, auc, media |
| `\aucSvmKernelPolinomico` | 0,774 | `resultados/curvas/svm_kernel.csv`: punto poly, validacion, auc, media |
| `\aucSvmKernelRbf` | 0,768 | `resultados/curvas/svm_kernel.csv`: punto rbf, validacion, auc, media |
| `\cSvmLinealMejor` | 0,01 | `resultados/hiperparametros.json`: curvas.svm_C_linear_balanced.mejor |
| `\aucSvmLinealMejor` | 0,774 | `resultados/hiperparametros.json`: curvas.svm_C_linear_balanced.auc_validacion_mejor |
| `\segundosLimiteCosto` | 600 | `resultados/costos.csv`: el límite de la corrida excedida (s) |
| `\cSvmCostoExcedido` | 10 | `resultados/costos.csv`: C de la corrida excedida (SVC lineal) |

## Slide 13 · El modelo final en validación

| Macro | Valor | Fuente |
|---|---|---|
| `\aucRfVal` | 0,795 | `resultados/cv_final_resumen.csv`: rf, validacion, auc, media |
| `\aucRfValDesvio` | 0,006 | `resultados/cv_final_resumen.csv`: rf, validacion, auc, desvio |
| `\recallRfVal` | 0,629 | `resultados/cv_final_resumen.csv`: rf, validacion, recall_q, media |
| `\recallRfValDesvio` | 0,018 | `resultados/cv_final_resumen.csv`: rf, validacion, recall_q, desvio |
| `\precisionRfVal` | 0,354 | `resultados/cv_final_resumen.csv`: rf, validacion, precision_q, media |
| `\apRfVal` | 0,462 | `resultados/cv_final_resumen.csv`: rf, validacion, ap, media |
| `\fUnoRfVal` | 0,454 | `resultados/cv_final_resumen.csv`: rf, validacion, f1_q, media |
| `\aucTrainRf` | 0,825 | `resultados/cv_final_resumen.csv`: rf, train, auc, media |
| `\brechaRf` | 0,030 | `resultados/cv_final_resumen.csv`: rf, brecha, auc, media (train − validación) |
| `\aucKnnVal` | 0,784 | `resultados/cv_final_resumen.csv`: knn, validacion, auc, media |
| `\aucKnnValDesvio` | 0,008 | `resultados/cv_final_resumen.csv`: knn, validacion, auc, desvio |
| `\recallKnnVal` | 0,617 | `resultados/cv_final_resumen.csv`: knn, validacion, recall_q, media |
| `\recallKnnValDesvio` | 0,016 | `resultados/cv_final_resumen.csv`: knn, validacion, recall_q, desvio |
| `\precisionKnnVal` | 0,348 | `resultados/cv_final_resumen.csv`: knn, validacion, precision_q, media |
| `\apKnnVal` | 0,429 | `resultados/cv_final_resumen.csv`: knn, validacion, ap, media |
| `\fUnoKnnVal` | 0,445 | `resultados/cv_final_resumen.csv`: knn, validacion, f1_q, media |
| `\aucTrainKnn` | 0,791 | `resultados/cv_final_resumen.csv`: knn, train, auc, media |
| `\brechaKnn` | 0,006 | `resultados/cv_final_resumen.csv`: knn, brecha, auc, media (train − validación) |
| `\aucNbVal` | 0,782 | `resultados/cv_final_resumen.csv`: nb_categorico, validacion, auc, media |
| `\aucNbValDesvio` | 0,007 | `resultados/cv_final_resumen.csv`: nb_categorico, validacion, auc, desvio |
| `\recallNbVal` | 0,614 | `resultados/cv_final_resumen.csv`: nb_categorico, validacion, recall_q, media |
| `\recallNbValDesvio` | 0,017 | `resultados/cv_final_resumen.csv`: nb_categorico, validacion, recall_q, desvio |
| `\precisionNbVal` | 0,346 | `resultados/cv_final_resumen.csv`: nb_categorico, validacion, precision_q, media |
| `\apNbVal` | 0,422 | `resultados/cv_final_resumen.csv`: nb_categorico, validacion, ap, media |
| `\fUnoNbVal` | 0,443 | `resultados/cv_final_resumen.csv`: nb_categorico, validacion, f1_q, media |
| `\aucTrainNb` | 0,783 | `resultados/cv_final_resumen.csv`: nb_categorico, train, auc, media |
| `\brechaNb` | 0,001 | `resultados/cv_final_resumen.csv`: nb_categorico, brecha, auc, media (train − validación) |
| `\aucSvmVal` | 0,774 | `resultados/cv_final_resumen.csv`: svm, validacion, auc, media |
| `\aucSvmValDesvio` | 0,009 | `resultados/cv_final_resumen.csv`: svm, validacion, auc, desvio |
| `\recallSvmVal` | 0,614 | `resultados/cv_final_resumen.csv`: svm, validacion, recall_q, media |
| `\recallSvmValDesvio` | 0,018 | `resultados/cv_final_resumen.csv`: svm, validacion, recall_q, desvio |
| `\precisionSvmVal` | 0,346 | `resultados/cv_final_resumen.csv`: svm, validacion, precision_q, media |
| `\apSvmVal` | 0,405 | `resultados/cv_final_resumen.csv`: svm, validacion, ap, media |
| `\fUnoSvmVal` | 0,442 | `resultados/cv_final_resumen.csv`: svm, validacion, f1_q, media |
| `\aucTrainSvm` | 0,776 | `resultados/cv_final_resumen.csv`: svm, train, auc, media |
| `\brechaSvm` | 0,002 | `resultados/cv_final_resumen.csv`: svm, brecha, auc, media (train − validación) |
| `\deltaAucAjusteRf` | +0,023 | `resultados/cv_final_resumen.csv` menos `resultados/cv_referencia_resumen.csv`: rf, validacion, auc, media |
| `\deltaAucAjusteKnn` | +0,029 | `resultados/cv_final_resumen.csv` menos `resultados/cv_referencia_resumen.csv`: knn, validacion, auc, media |
| `\deltaAucAjusteSvm` | +0,072 | `resultados/cv_final_resumen.csv` menos `resultados/cv_referencia_resumen.csv`: svm, validacion, auc, media |
| `\modeloFinal` | Random Forest | `resultados/modelo_elegido.json`: ranking, 1.º (texto) |
| `\modeloSegundoFinal` | KNN | `resultados/modelo_elegido.json`: ranking, 2.º (texto) |
| `\modeloUltimoFinal` | SVM | `resultados/modelo_elegido.json`: ranking, último (texto) |
| `\errorEstandarAucRf` | 0,0025 | `resultados/cv_final_resumen.csv`: rf, validacion, auc, error_estandar |
| `\umbralEmpateFinal` | 0,7927 | `resultados/modelo_elegido.json`: ranking, AUC del 1.º menos su error estándar (D-23) |
| `\modelosDentroUnoEs` | 0 | `resultados/modelo_elegido.json`: cuántos quedan dentro de 1 ES del mejor (empate_dentro_1es) |
| `\margenAucFinal` | 0,011 | `resultados/modelo_elegido.json`: ranking, AUC del 1.º menos el del 2.º |
| `\margenAucSinModeloFinal` | 0,295 | `resultados/modelo_elegido.json`: ranking, AUC del 1.º menos linea_base.auc |
| `\rangoAucFinal` | 0,021 | `resultados/modelo_elegido.json`: ranking, AUC del 1.º menos el del último |

## Slides 14 y 15 · Test y matriz de confusión

| Macro | Valor | Fuente |
|---|---|---|
| `\yesTest` | ? | `resultados/evaluacion_test.json`: n_yes_test (pendiente) |
| `\pctYesTest` | ? | `resultados/evaluacion_test.json`: n_yes_test / n_test (pendiente) |
| `\diferenciaAucTestValidacion` | ? | `resultados/evaluacion_test.json`: metricas.auc.valor menos `resultados/modelo_elegido.json`: metricas.validacion.auc.media (sólo en voz alta, guía C2) (pendiente) |
| `\diferenciaRecallTestValidacion` | ? | `resultados/evaluacion_test.json`: metricas.recall_q.valor menos `resultados/modelo_elegido.json`: metricas.validacion.recall_q.media (sólo en voz alta, guía C2) (pendiente) |
| `\vpRfVal` | 2 329 | `resultados/oof_final_rf.csv` con la y de train (`cargar_train()`): VP llamando al 20 % de la lista fuera de fold |
| `\fpRfVal` | 4 259 | `resultados/oof_final_rf.csv` con la y de train (`cargar_train()`): FP llamando al 20 % de la lista fuera de fold |
| `\fnRfVal` | 1 382 | `resultados/oof_final_rf.csv` con la y de train (`cargar_train()`): FN llamando al 20 % de la lista fuera de fold |
| `\vnRfVal` | 24 970 | `resultados/oof_final_rf.csv` con la y de train (`cargar_train()`): VN llamando al 20 % de la lista fuera de fold |

## Slide 16 · Por qué cada modelo rindió lo que rindió

| Macro | Valor | Fuente |
|---|---|---|
| `\correlacionMacroMinima` | 0,91 | train (`cargar_train()`): el menor \|r\| entre los pares del bloque macro con \|r\| >= 0,9 (D-14) |
| `\correlacionMacroMaxima` | 0,97 | train (`cargar_train()`): el mayor \|r\| entre esos pares (D-14) |
| `\desvioPdays` | 186,6 | train (`cargar_train()`): desvío estándar de pdays (D-12) |
| `\desvioNrEmployed` | 72,4 | train (`cargar_train()`): desvío estándar de nr.employed (D-12) |
| `\pctVarianzaPdays` | 86,6 % | train (`cargar_train()`): % de la varianza de las 9 numéricas del modelo que es de pdays (D-12) |
| `\pctVarianzaNrEmployed` | 13,0 % | train (`cargar_train()`): ídem, de nr.employed (D-12) |
| `\razonDesviosNumericas` | 374 | train (`cargar_train()`): el mayor desvío de las 9 numéricas del modelo / el menor |

## Slide 17 · Limitaciones: sensibilidad al presupuesto

| Macro | Valor | Fuente |
|---|---|---|
| `\recallRfDiez` | 0,446 | `resultados/sensibilidad_q_final.csv`: rf, q = 0.1, recall_q, media de los folds |
| `\recallRfTreinta` | 0,705 | `resultados/sensibilidad_q_final.csv`: rf, q = 0.3, recall_q, media de los folds |
| `\recallKnnDiez` | 0,414 | `resultados/sensibilidad_q_final.csv`: knn, q = 0.1, recall_q, media de los folds |
| `\recallKnnTreinta` | 0,690 | `resultados/sensibilidad_q_final.csv`: knn, q = 0.3, recall_q, media de los folds |
| `\recallNbDiez` | 0,411 | `resultados/sensibilidad_q_final.csv`: nb_categorico, q = 0.1, recall_q, media de los folds |
| `\recallNbTreinta` | 0,697 | `resultados/sensibilidad_q_final.csv`: nb_categorico, q = 0.3, recall_q, media de los folds |
| `\recallSvmDiez` | 0,415 | `resultados/sensibilidad_q_final.csv`: svm, q = 0.1, recall_q, media de los folds |
| `\recallSvmTreinta` | 0,695 | `resultados/sensibilidad_q_final.csv`: svm, q = 0.3, recall_q, media de los folds |
| `\recallSinModeloDiez` | 0,100 | `resultados/sensibilidad_q_final.csv`: sin_modelo, q = 0.1, recall_q, media de los folds |
| `\recallSinModeloTreinta` | 0,300 | `resultados/sensibilidad_q_final.csv`: sin_modelo, q = 0.3, recall_q, media de los folds |
| `\precisionRfDiez` | 0,502 | `resultados/sensibilidad_q_final.csv`: rf, q = 0.1, precision_q, media de los folds |
| `\precisionRfVeinte` | 0,354 | `resultados/sensibilidad_q_final.csv`: rf, q = 0.2, precision_q, media de los folds |
| `\precisionRfTreinta` | 0,265 | `resultados/sensibilidad_q_final.csv`: rf, q = 0.3, precision_q, media de los folds |

## Slides 18 y 19 · Hallazgo: el modelo aprende la época

| Macro | Valor | Fuente |
|---|---|---|
| `\aucRfAdelante` | 0,558 | `resultados/robustez_temporal_resumen.csv`: rf, hacia_adelante, todas, auc, media |
| `\aucRfAdelanteDesvio` | 0,123 | `resultados/robustez_temporal_resumen.csv`: rf, hacia_adelante, todas, auc, desvio |
| `\aucRfSinMacroAdelante` | 0,556 | `resultados/robustez_temporal_resumen.csv`: rf, hacia_adelante, sin_macro, auc, media |
| `\aucRfSinMacroAdelanteDesvio` | 0,110 | `resultados/robustez_temporal_resumen.csv`: rf, hacia_adelante, sin_macro, auc, desvio |
| `\aucRfSinMacroBarajado` | 0,772 | `resultados/robustez_temporal_resumen.csv`: rf, barajado, sin_macro, auc, media |
| `\aucRfSinMacroBarajadoDesvio` | 0,010 | `resultados/robustez_temporal_resumen.csv`: rf, barajado, sin_macro, auc, desvio |
| `\aucRfSinMacroNiMonthAdelante` | 0,547 | `resultados/robustez_temporal_resumen.csv`: rf, hacia_adelante, sin_macro_ni_month, auc, media |
| `\aucRfSinMacroNiMonthAdelanteDesvio` | 0,098 | `resultados/robustez_temporal_resumen.csv`: rf, hacia_adelante, sin_macro_ni_month, auc, desvio |
| `\aucRfSinMacroNiMonthBarajado` | 0,736 | `resultados/robustez_temporal_resumen.csv`: rf, barajado, sin_macro_ni_month, auc, media |
| `\aucRfSinMacroNiMonthBarajadoDesvio` | 0,009 | `resultados/robustez_temporal_resumen.csv`: rf, barajado, sin_macro_ni_month, auc, desvio |
| `\recallRfAdelante` | 0,243 | `resultados/robustez_temporal_resumen.csv`: rf, hacia_adelante, todas, recall_q, media |
| `\caidaAucRfAdelante` | 0,237 | `resultados/robustez_temporal_resumen.csv`: rf, barajado_menos_hacia_adelante, todas, auc, media |
| `\deltaAucSinMacroRfBarajado` | −0,023 | `resultados/robustez_temporal_resumen.csv`: rf, barajado, sin_macro menos todas, auc, media |
| `\deltaAucSinMacroNiMonthRfBarajado` | −0,059 | `resultados/robustez_temporal_resumen.csv`: rf, barajado, sin_macro_ni_month menos todas, auc, media |
| `\aucKnnAdelante` | 0,539 | `resultados/robustez_temporal_resumen.csv`: knn, hacia_adelante, todas, auc, media |
| `\aucKnnAdelanteDesvio` | 0,065 | `resultados/robustez_temporal_resumen.csv`: knn, hacia_adelante, todas, auc, desvio |
| `\aucKnnSinMacroAdelante` | 0,557 | `resultados/robustez_temporal_resumen.csv`: knn, hacia_adelante, sin_macro, auc, media |
| `\aucKnnSinMacroAdelanteDesvio` | 0,099 | `resultados/robustez_temporal_resumen.csv`: knn, hacia_adelante, sin_macro, auc, desvio |
| `\aucKnnSinMacroBarajado` | 0,754 | `resultados/robustez_temporal_resumen.csv`: knn, barajado, sin_macro, auc, media |
| `\aucKnnSinMacroBarajadoDesvio` | 0,009 | `resultados/robustez_temporal_resumen.csv`: knn, barajado, sin_macro, auc, desvio |
| `\aucKnnSinMacroNiMonthAdelante` | 0,542 | `resultados/robustez_temporal_resumen.csv`: knn, hacia_adelante, sin_macro_ni_month, auc, media |
| `\aucKnnSinMacroNiMonthAdelanteDesvio` | 0,091 | `resultados/robustez_temporal_resumen.csv`: knn, hacia_adelante, sin_macro_ni_month, auc, desvio |
| `\aucKnnSinMacroNiMonthBarajado` | 0,731 | `resultados/robustez_temporal_resumen.csv`: knn, barajado, sin_macro_ni_month, auc, media |
| `\aucKnnSinMacroNiMonthBarajadoDesvio` | 0,009 | `resultados/robustez_temporal_resumen.csv`: knn, barajado, sin_macro_ni_month, auc, desvio |
| `\recallKnnAdelante` | 0,215 | `resultados/robustez_temporal_resumen.csv`: knn, hacia_adelante, todas, recall_q, media |
| `\caidaAucKnnAdelante` | 0,245 | `resultados/robustez_temporal_resumen.csv`: knn, barajado_menos_hacia_adelante, todas, auc, media |
| `\deltaAucSinMacroKnnBarajado` | −0,031 | `resultados/robustez_temporal_resumen.csv`: knn, barajado, sin_macro menos todas, auc, media |
| `\deltaAucSinMacroNiMonthKnnBarajado` | −0,053 | `resultados/robustez_temporal_resumen.csv`: knn, barajado, sin_macro_ni_month menos todas, auc, media |
| `\aucNbAdelante` | 0,598 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, hacia_adelante, todas, auc, media |
| `\aucNbAdelanteDesvio` | 0,087 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, hacia_adelante, todas, auc, desvio |
| `\aucNbSinMacroAdelante` | 0,604 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, hacia_adelante, sin_macro, auc, media |
| `\aucNbSinMacroAdelanteDesvio` | 0,090 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, hacia_adelante, sin_macro, auc, desvio |
| `\aucNbSinMacroBarajado` | 0,749 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, barajado, sin_macro, auc, media |
| `\aucNbSinMacroBarajadoDesvio` | 0,012 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, barajado, sin_macro, auc, desvio |
| `\aucNbSinMacroNiMonthAdelante` | 0,566 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, hacia_adelante, sin_macro_ni_month, auc, media |
| `\aucNbSinMacroNiMonthAdelanteDesvio` | 0,101 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, hacia_adelante, sin_macro_ni_month, auc, desvio |
| `\aucNbSinMacroNiMonthBarajado` | 0,724 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, barajado, sin_macro_ni_month, auc, media |
| `\aucNbSinMacroNiMonthBarajadoDesvio` | 0,011 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, barajado, sin_macro_ni_month, auc, desvio |
| `\recallNbAdelante` | 0,287 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, hacia_adelante, todas, recall_q, media |
| `\caidaAucNbAdelante` | 0,184 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, barajado_menos_hacia_adelante, todas, auc, media |
| `\deltaAucSinMacroNbBarajado` | −0,033 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, barajado, sin_macro menos todas, auc, media |
| `\deltaAucSinMacroNiMonthNbBarajado` | −0,058 | `resultados/robustez_temporal_resumen.csv`: nb_categorico, barajado, sin_macro_ni_month menos todas, auc, media |
| `\aucSvmAdelante` | 0,541 | `resultados/robustez_temporal_resumen.csv`: svm, hacia_adelante, todas, auc, media |
| `\aucSvmAdelanteDesvio` | 0,078 | `resultados/robustez_temporal_resumen.csv`: svm, hacia_adelante, todas, auc, desvio |
| `\aucSvmSinMacroAdelante` | 0,563 | `resultados/robustez_temporal_resumen.csv`: svm, hacia_adelante, sin_macro, auc, media |
| `\aucSvmSinMacroAdelanteDesvio` | 0,122 | `resultados/robustez_temporal_resumen.csv`: svm, hacia_adelante, sin_macro, auc, desvio |
| `\aucSvmSinMacroBarajado` | 0,754 | `resultados/robustez_temporal_resumen.csv`: svm, barajado, sin_macro, auc, media |
| `\aucSvmSinMacroBarajadoDesvio` | 0,012 | `resultados/robustez_temporal_resumen.csv`: svm, barajado, sin_macro, auc, desvio |
| `\aucSvmSinMacroNiMonthAdelante` | 0,549 | `resultados/robustez_temporal_resumen.csv`: svm, hacia_adelante, sin_macro_ni_month, auc, media |
| `\aucSvmSinMacroNiMonthAdelanteDesvio` | 0,081 | `resultados/robustez_temporal_resumen.csv`: svm, hacia_adelante, sin_macro_ni_month, auc, desvio |
| `\aucSvmSinMacroNiMonthBarajado` | 0,726 | `resultados/robustez_temporal_resumen.csv`: svm, barajado, sin_macro_ni_month, auc, media |
| `\aucSvmSinMacroNiMonthBarajadoDesvio` | 0,009 | `resultados/robustez_temporal_resumen.csv`: svm, barajado, sin_macro_ni_month, auc, desvio |
| `\recallSvmAdelante` | 0,233 | `resultados/robustez_temporal_resumen.csv`: svm, hacia_adelante, todas, recall_q, media |
| `\caidaAucSvmAdelante` | 0,233 | `resultados/robustez_temporal_resumen.csv`: svm, barajado_menos_hacia_adelante, todas, auc, media |
| `\deltaAucSinMacroSvmBarajado` | −0,020 | `resultados/robustez_temporal_resumen.csv`: svm, barajado, sin_macro menos todas, auc, media |
| `\deltaAucSinMacroNiMonthSvmBarajado` | −0,048 | `resultados/robustez_temporal_resumen.csv`: svm, barajado, sin_macro_ni_month menos todas, auc, media |
| `\aucRfAdelanteFoldUno` | 0,489 | `resultados/robustez_temporal.csv`: rf, hacia_adelante, todas, fold 1, validacion, auc |
| `\pctYesBloqueUno` | 4,7 % | `resultados/robustez_folds.csv`: hacia_adelante, fold 1, pct_yes_validacion |
| `\pctYesEntrenamientoAdelanteUno` | 2,9 % | `resultados/robustez_folds.csv`: hacia_adelante, fold 1, pct_yes_train |
| `\filasEntrenamientoAdelanteUno` | 5 490 | `resultados/robustez_folds.csv`: hacia_adelante, fold 1, n_train |
| `\aucRfAdelanteFoldDos` | 0,500 | `resultados/robustez_temporal.csv`: rf, hacia_adelante, todas, fold 2, validacion, auc |
| `\pctYesBloqueDos` | 6,5 % | `resultados/robustez_folds.csv`: hacia_adelante, fold 2, pct_yes_validacion |
| `\pctYesEntrenamientoAdelanteDos` | 3,8 % | `resultados/robustez_folds.csv`: hacia_adelante, fold 2, pct_yes_train |
| `\filasEntrenamientoAdelanteDos` | 10 980 | `resultados/robustez_folds.csv`: hacia_adelante, fold 2, n_train |
| `\aucRfAdelanteFoldTres` | 0,426 | `resultados/robustez_temporal.csv`: rf, hacia_adelante, todas, fold 3, validacion, auc |
| `\pctYesBloqueTres` | 5,6 % | `resultados/robustez_folds.csv`: hacia_adelante, fold 3, pct_yes_validacion |
| `\pctYesEntrenamientoAdelanteTres` | 4,7 % | `resultados/robustez_folds.csv`: hacia_adelante, fold 3, pct_yes_train |
| `\filasEntrenamientoAdelanteTres` | 16 470 | `resultados/robustez_folds.csv`: hacia_adelante, fold 3, n_train |
| `\aucRfAdelanteFoldCuatro` | 0,664 | `resultados/robustez_temporal.csv`: rf, hacia_adelante, todas, fold 4, validacion, auc |
| `\pctYesBloqueCuatro` | 12,7 % | `resultados/robustez_folds.csv`: hacia_adelante, fold 4, pct_yes_validacion |
| `\pctYesEntrenamientoAdelanteCuatro` | 4,9 % | `resultados/robustez_folds.csv`: hacia_adelante, fold 4, pct_yes_train |
| `\filasEntrenamientoAdelanteCuatro` | 21 960 | `resultados/robustez_folds.csv`: hacia_adelante, fold 4, n_train |
| `\aucRfAdelanteFoldCinco` | 0,711 | `resultados/robustez_temporal.csv`: rf, hacia_adelante, todas, fold 5, validacion, auc |
| `\pctYesBloqueCinco` | 35,2 % | `resultados/robustez_folds.csv`: hacia_adelante, fold 5, pct_yes_validacion |
| `\pctYesEntrenamientoAdelanteCinco` | 6,5 % | `resultados/robustez_folds.csv`: hacia_adelante, fold 5, pct_yes_train |
| `\filasEntrenamientoAdelanteCinco` | 27 450 | `resultados/robustez_folds.csv`: hacia_adelante, fold 5, n_train |
| `\filasBloqueAdelante` | 5 490 | `resultados/robustez_folds.csv`: hacia_adelante, fold 1, n_validacion |

## Reserva · Curva de ganancia (H3)

| Macro | Valor | Fuente |
|---|---|---|
| `\gananciaRfDiez` | 44,6 % | `resultados/ganancia_final.csv`: rf, p = 10, pct_yes_alcanzado |
| `\gananciaRfVeinte` | 62,8 % | `resultados/ganancia_final.csv`: rf, p = 20, pct_yes_alcanzado |
| `\gananciaRfTreinta` | 70,4 % | `resultados/ganancia_final.csv`: rf, p = 30, pct_yes_alcanzado |
| `\gananciaRfCincuenta` | 81,7 % | `resultados/ganancia_final.csv`: rf, p = 50, pct_yes_alcanzado |
| `\gananciaKnnVeinte` | 61,7 % | `resultados/ganancia_final.csv`: knn, p = 20, pct_yes_alcanzado |
| `\gananciaNbVeinte` | 61,4 % | `resultados/ganancia_final.csv`: nb_categorico, p = 20, pct_yes_alcanzado |
| `\gananciaSvmVeinte` | 61,3 % | `resultados/ganancia_final.csv`: svm, p = 20, pct_yes_alcanzado |
| `\gananciaSinModeloVeinte` | 20,0 % | `resultados/ganancia_final.csv`: sin_modelo, p = 20, pct_yes_alcanzado |
| `\pctListaYesMitadRf` | 12 % | `resultados/ganancia_final.csv`: rf, el menor p con pct_yes_alcanzado >= 50 |
| `\pctListaYesOchentaRf` | 48 % | `resultados/ganancia_final.csv`: rf, el menor p con pct_yes_alcanzado >= 80 |

## Números de test (`informe/resultados-test.tex`)

Los escribe `src/evaluar_test.py`, no este módulo: el deck los toma de ese archivo. Se repiten aquí para que el guion tenga todos los números en un solo lugar. Mientras el test no se evalúe, valen «?».

| Macro | Valor | Fuente |
|---|---|---|
| `\auctest` | ? | `informe/resultados-test.tex` |
| `\aucic` | ? | `informe/resultados-test.tex` |
| `\recalltest` | ? | `informe/resultados-test.tex` |
| `\recallic` | ? | `informe/resultados-test.tex` |
| `\precisiontest` | ? | `informe/resultados-test.tex` |
| `\fitest` | ? | `informe/resultados-test.tex` |
| `\aptest` | ? | `informe/resultados-test.tex` |
| `\vptest` | ? | `informe/resultados-test.tex` |
| `\fptest` | ? | `informe/resultados-test.tex` |
| `\fntest` | ? | `informe/resultados-test.tex` |
| `\vntest` | ? | `informe/resultados-test.tex` |
| `\recallmatriztest` | ? | `informe/resultados-test.tex` |
| `\precisionmatriztest` | ? | `informe/resultados-test.tex` |
| `\empatadostest` | ? | `informe/resultados-test.tex` |
| `\llamadosempatadostest` | ? | `informe/resultados-test.tex` |
| `\ntest` | ? | `informe/resultados-test.tex` |
| `\llamadastest` | ? | `informe/resultados-test.tex` |
| `\nevaluaciones` | ? | `informe/resultados-test.tex` |

## Cifras que este módulo no mide

- **D-02** (la dispersión de la proporción de «yes» del test entre semillas) y **D-03** (la tasa de «yes» de train y de test con una partición por fecha) se miden sobre el CSV completo, test incluido, con `src/evidencia_particion.py`. Este módulo no puede leerlo (D-04, `tests/test_aislamiento.py`): sus macros (slide 3) leen `resultados/evidencia_particion.json`, con las claves `n_semillas`, `sin_estratificar.desvio_pp`, `sin_estratificar.minimo_pct`, `sin_estratificar.maximo_pct`, `temporal.pct_yes_train` y `temporal.pct_yes_test` (porcentajes en puntos), y valen «?» hasta que ese módulo escriba el archivo.
- `\vecinosKnnMeseta` (301) es una lectura de la curva que hace D-22, no el resultado de una regla: se copia de DECISIONES.md y vale «?» si ese punto deja de estar dentro de 1 error estándar del mejor.

"""Correr con: python -m tests.test_experimentos

La validación cruzada corre en modo rápido (3000 filas de train, un proceso) en un directorio
temporal. El resumen, la sensibilidad a q y la curva de ganancia se prueban además sobre datos
sintéticos, con cada cifra calculada a mano en el comentario que la acompaña.

src/configuracion.py arranca igual a la referencia, así que un módulo que la ignorara pasaría
cualquier test que corra con ella. Los tests que miran qué configuración se usó la reemplazan por
una distinta en las opciones y en los hiperparámetros, y la restauran al terminar.
"""

import contextlib
import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import experimentos
from src.configuracion import HIPERPARAMETROS_FINALES, MODELOS_FINALES
from src.datos import FILA, OBJETIVO, cargar_train
from src.experimentos import (
    ARBOLES_RAPIDO,
    BRECHA,
    CAMPOS_GANANCIA,
    CAMPOS_SENSIBILIDAD,
    FILAS_RAPIDO,
    LINEA_BASE,
    PORCENTAJES,
    Q_SENSIBILIDAD,
    SIN_MODELO,
    con_objetivo,
    con_sin_modelo,
    ganancia,
    hiperparametros_de,
    hiperparametros_vigentes,
    leer_oof,
    oof_sin_cv,
    resumir,
    resumir_tabla,
    sensibilidad_q,
    submuestra,
    tabla_validacion,
    verificar_oof,
)
from src.metricas import METRICAS, PRESUPUESTO, evaluar
from src.modelos import REFERENCIA, crear_modelo, separar_X_y
from src.preproceso import Opciones
from src.resultados import CAMPOS, CAMPOS_OOF, configuracion_json, validacion_cruzada
from src.validacion import K


def _rapidos():
    """Una variante de Naive Bayes y KNN entre los modelos vigentes: D-18 deja una sola variante."""
    vigentes = [m for m in MODELOS_FINALES if m in HIPERPARAMETROS_FINALES]
    return next(m for m in ("nb_gaussiano", "nb_categorico") if m in vigentes), "knn"


RAPIDOS = _rapidos()

# Distinta de la referencia en las opciones y en los hiperparámetros: un módulo que ignore
# cualquiera de las dos da otro OOF.
OPCIONES_PRUEBA = Opciones(pdays="sin")
HIPERPARAMETROS_PRUEBA = {"knn": {"n_neighbors": 5}}


def _cerca(a, b, tol=1e-9):
    return abs(a - b) < tol


def _en_silencio(funcion, *args):
    """Corre `funcion` sin mostrar lo que imprime: las tablas del modo rápido y los errores de
    argparse."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return funcion(*args)


def _salida_de(funcion, *args):
    """Lo que `funcion` imprime."""
    texto = io.StringIO()
    with contextlib.redirect_stdout(texto), contextlib.redirect_stderr(io.StringIO()):
        funcion(*args)
    return texto.getvalue()


@contextlib.contextmanager
def _reemplazar(**valores):
    """Reemplaza nombres globales de src/experimentos.py (la configuración vigente, DIR o
    cargar_train) y los restaura al salir, aunque el bloque falle."""
    anteriores = {nombre: getattr(experimentos, nombre) for nombre in valores}
    try:
        for nombre, valor in valores.items():
            setattr(experimentos, nombre, valor)
        yield
    finally:
        for nombre, valor in anteriores.items():
            setattr(experimentos, nombre, valor)


def _configuracion_de_prueba(**extra):
    return _reemplazar(OPCIONES_FINALES=OPCIONES_PRUEBA,
                       HIPERPARAMETROS_FINALES=HIPERPARAMETROS_PRUEBA,
                       MODELOS_FINALES=["knn"], **extra)


def _niega(argv):
    """La línea de comandos tiene que terminar con un código de error."""
    try:
        _en_silencio(experimentos.main, argv)
    except SystemExit as e:
        assert e.code not in (0, None), argv
    else:
        raise AssertionError(f"no falló: {argv}")


def _train_sintetico(filas, y):
    return pd.DataFrame({FILA: filas, OBJETIVO: np.where(np.asarray(y) == 1, "yes", "no")})


def _oof_knn(muestra, opciones, **hiperparametros):
    """El OOF de knn sobre `muestra`, calculado aquí con el corredor común y ordenado por fila."""
    X, y = separar_X_y(muestra)
    _, oof = validacion_cruzada(crear_modelo("knn", opciones, **hiperparametros), X, y,
                                filas=muestra[FILA], n_jobs=1)
    return oof.sort_values(FILA).reset_index(drop=True)


def _leer_oof_ordenado(ruta):
    return pd.read_csv(ruta, float_precision="round_trip").sort_values(FILA).reset_index(drop=True)


def _mismo_oof(a, b):
    return (len(a) == len(b) and (a[FILA].to_numpy() == b[FILA].to_numpy()).all()
            and (a["fold"].to_numpy() == b["fold"].to_numpy()).all()
            and np.allclose(a["puntaje"], b["puntaje"], rtol=0, atol=1e-12))


def _cv_de(oof, etiqueta, modelo):
    """El cv que respalda a un OOF sintético con `y`: las cinco métricas de cada fold, iguales en
    train y en validación."""
    filas = []
    for fold, parte in oof.groupby("fold"):
        medidas = evaluar(parte["y"].to_numpy(), parte["puntaje"].to_numpy())
        filas += [{"etiqueta": etiqueta, "modelo": modelo, "configuracion": "{}", "fold": fold,
                   "conjunto": conjunto, "metrica": m, "valor": medidas[m]}
                  for conjunto in ("train", "validacion") for m in METRICAS]
    return pd.DataFrame(filas)


def _tabla(etiqueta, modelo, auc_train, auc_validacion, configuracion="{}"):
    """Tabla larga sintética: el AUC indicado por fold y 0,5 en las otras cuatro métricas."""
    return pd.DataFrame([
        {"etiqueta": etiqueta, "modelo": modelo, "configuracion": configuracion, "fold": fold,
         "conjunto": conjunto, "metrica": metrica, "valor": auc if metrica == "auc" else 0.5}
        for fold, par in enumerate(zip(auc_train, auc_validacion), start=1)
        for conjunto, auc in zip(("train", "validacion"), par)
        for metrica in METRICAS])


def test_submuestra_estratificada_con_semilla_42(train):
    muestra = submuestra(train)
    esperada, _ = train_test_split(train, train_size=3000, stratify=train[OBJETIVO],
                                   random_state=42)
    pd.testing.assert_frame_equal(muestra, esperada.sort_values(FILA).reset_index(drop=True))
    # Estratificada: los «yes» de train en proporción, salvo el redondeo de una fila.
    yes = int((muestra[OBJETIVO] == "yes").sum())
    assert abs(yes - len(muestra) * (train[OBJETIVO] == "yes").mean()) <= 1
    print(f"ok  la submuestra del modo rápido son {len(muestra)} filas estratificadas con la "
          f"semilla 42 ({yes} «yes»), en el orden de la fila original")


def test_rapido_escribe_el_contrato(directorio, train):
    _en_silencio(experimentos.main, ["--rapido", "--modelo", *RAPIDOS, "--salida", str(directorio)])
    muestra = submuestra(train)
    for modelo in RAPIDOS:
        cv = pd.read_csv(directorio / f"cv_referencia_{modelo}.csv")
        assert list(cv.columns) == ["etiqueta"] + CAMPOS, modelo
        assert len(cv) == K * 2 * len(METRICAS) and set(cv["etiqueta"]) == {"referencia"}
        assert set(cv["configuracion"]) == {
            configuracion_json(hiperparametros_de(modelo, "referencia", rapido=True))}
        assert cv["valor"].between(0, 1).all()

        oof = pd.read_csv(directorio / f"oof_referencia_{modelo}.csv")
        assert list(oof.columns) == CAMPOS_OOF
        assert len(oof) == FILAS_RAPIDO == len(muestra) and oof[FILA].is_unique
        assert set(oof[FILA]) == set(muestra[FILA])
        assert sorted(oof["fold"].unique()) == list(range(1, K + 1))
        assert np.isfinite(oof["puntaje"]).all()
    print(f"ok  --rapido escribe el cv y el OOF de {' y '.join(RAPIDOS)} con las columnas del "
          f"contrato; cada OOF cubre las {FILAS_RAPIDO} filas una vez")


def test_rapido_resume_y_deriva(directorio):
    cv = pd.read_csv(directorio / "cv_referencia.csv")
    assert list(cv.columns) == ["etiqueta"] + CAMPOS
    assert len(cv) == len(RAPIDOS) * K * 2 * len(METRICAS)

    r = pd.read_csv(directorio / "cv_referencia_resumen.csv")
    for modelo in RAPIDOS:
        propio = r[r["modelo"] == modelo].set_index(["conjunto", "metrica"])
        assert len(propio) == 2 * len(METRICAS) + 1 and (propio["n"] == K).all()
        media = propio["media"]
        assert _cerca(media[(BRECHA, "auc")], media[("train", "auc")] - media[("validacion", "auc")])

    # Con q = 20 %, la sensibilidad reproduce fold a fold el presupuesto que midió
    # validacion_cruzada: la y que se cruza por `fila` es la de cada puntaje.
    s = pd.read_csv(directorio / "sensibilidad_q_referencia.csv")
    assert list(s.columns) == CAMPOS_SENSIBILIDAD
    assert set(s["modelo"]) == {*RAPIDOS, SIN_MODELO}
    assert len(s) == (len(RAPIDOS) + 1) * len(Q_SENSIBILIDAD) * K * 3
    validacion = cv[cv["conjunto"] == "validacion"]
    for modelo in RAPIDOS:
        for metrica in ("recall_q", "precision_q", "f1_q"):
            medido = validacion[(validacion["modelo"] == modelo) & (validacion["metrica"] == metrica)]
            propio = s[(s["modelo"] == modelo) & np.isclose(s["q"], PRESUPUESTO)
                       & (s["metrica"] == metrica)]
            assert np.allclose(propio.set_index("fold")["valor"].sort_index(),
                               medido.set_index("fold")["valor"].sort_index(), rtol=0, atol=1e-12)
    # Sin modelo, cada fold de 600 filas llama a 60, 120 o 180 clientes: el recall es q exacto.
    sin = s[(s["modelo"] == SIN_MODELO) & (s["metrica"] == "recall_q")]
    assert np.allclose(sin["valor"], sin["q"], rtol=0, atol=1e-12)

    g = pd.read_csv(directorio / "ganancia_referencia.csv")
    assert list(g.columns) == CAMPOS_GANANCIA
    assert set(g["modelo"]) == {*RAPIDOS, SIN_MODELO}
    for modelo, curva in g.groupby("modelo"):
        assert list(curva["p"]) == list(PORCENTAJES)
        assert (np.diff(curva["pct_yes_alcanzado"]) >= -1e-9).all()
        assert _cerca(curva["pct_yes_alcanzado"].iloc[-1], 100)
    # Sin modelo, sobre 3000 filas llamar al p % es llamar a 30·p clientes: alcanza el p % exacto.
    sin = g[g["modelo"] == SIN_MODELO]
    assert np.allclose(sin["pct_yes_alcanzado"], sin["p"], rtol=0, atol=1e-9)
    print("ok  el resumen trae la brecha del AUC; con q = 20 % la sensibilidad reproduce recall, "
          "precisión y F1 de cada fold; la ganancia crece hasta el 100 % y sin modelo es la diagonal")


def test_cada_etiqueta_usa_su_configuracion(directorio, train):
    muestra = submuestra(train)
    vigente = _oof_knn(muestra, OPCIONES_PRUEBA, **HIPERPARAMETROS_PRUEBA["knn"])
    referencia = _oof_knn(muestra, OPCIONES_PRUEBA)
    # El test discrimina: con las opciones de referencia el OOF es otro, y el de los
    # hiperparámetros vigentes no es el de REFERENCIA.
    for otro in (_oof_knn(muestra, Opciones(), **HIPERPARAMETROS_PRUEBA["knn"]), referencia):
        assert not np.allclose(otro["puntaje"], vigente["puntaje"])
    configuraciones = {
        "final": configuracion_json({**REFERENCIA["knn"], **HIPERPARAMETROS_PRUEBA["knn"]}),
        "referencia": configuracion_json(REFERENCIA["knn"]),
    }

    rapido, normal, de_referencia = (directorio / c for c in ("rapido", "normal", "referencia"))
    with _configuracion_de_prueba():
        texto = _salida_de(experimentos.main, ["--rapido", "--etiqueta", "final",
                                               "--salida", str(rapido)])
    assert f"Opciones vigentes: {OPCIONES_PRUEBA}" in texto
    assert "Hiperparámetros: los vigentes de src/configuracion.py." in texto
    # El modo normal, sobre la misma muestra en lugar de todo train.
    with _configuracion_de_prueba(cargar_train=lambda: muestra.copy()):
        texto = _salida_de(experimentos.main, ["--etiqueta", "final", "--n-jobs", "1",
                                               "--salida", str(normal)])
        assert "Etiqueta final: knn, 1 proceso, en" in texto and f"{OPCIONES_PRUEBA}" in texto
        # La etiqueta referencia no lee HIPERPARAMETROS_FINALES: mide REFERENCIA aunque la vigente
        # sea otra, así que rehacerla reproduce cv_referencia_* después de la ola 4.
        texto = _salida_de(experimentos.main, ["--etiqueta", "referencia", "--n-jobs", "1",
                                               "--salida", str(de_referencia)])
        assert "Hiperparámetros: REFERENCIA, de src/modelos.py." in texto
        assert f"Opciones vigentes: {OPCIONES_PRUEBA}" in texto

    # Sin --modelo corren los de MODELOS_FINALES y sólo ellos.
    assert sorted(p.name for p in rapido.iterdir()) == sorted([
        "cv_final_knn.csv", "oof_final_knn.csv", "cv_final.csv", "cv_final_resumen.csv",
        "sensibilidad_q_final.csv", "ganancia_final.csv"])
    assert sorted(p.name for p in normal.iterdir()) == ["cv_final_knn.csv", "oof_final_knn.csv"]
    assert sorted(p.name for p in de_referencia.iterdir()) == ["cv_referencia_knn.csv",
                                                               "oof_referencia_knn.csv"]
    for carpeta, etiqueta, esperado in ((rapido, "final", vigente), (normal, "final", vigente),
                                        (de_referencia, "referencia", referencia)):
        cv = pd.read_csv(carpeta / f"cv_{etiqueta}_knn.csv")
        assert set(cv["configuracion"]) == {configuraciones[etiqueta]}, carpeta.name
        assert set(cv["etiqueta"]) == {etiqueta}
        oof = _leer_oof_ordenado(carpeta / f"oof_{etiqueta}_knn.csv")
        assert _mismo_oof(oof, esperado), carpeta.name
    print("ok  sin --modelo corren los modelos de MODELOS_FINALES, con OPCIONES_FINALES; la "
          "etiqueta final usa HIPERPARAMETROS_FINALES sobre la referencia, en el modo rápido y en "
          "el normal, y la etiqueta referencia usa REFERENCIA aunque la vigente sea otra: cada OOF "
          "es el de su configuración")


def test_derivados_leen_solo_los_oof_de_su_cv(directorio, train):
    # `directorio` es el del modo normal del test anterior: el cv y el OOF de knn, etiqueta final.
    muestra = submuestra(train)
    # Un OOF de otra corrida con el nombre de la etiqueta: el de A0 (Opciones() y REFERENCIA), que
    # una versión anterior de src/ablaciones.py guardaba así.
    a0 = _oof_knn(muestra, Opciones())
    a0.to_csv(directorio / "oof_final_nb_gaussiano.csv", index=False)
    with _reemplazar(cargar_train=lambda: muestra.copy()):
        texto = _salida_de(experimentos.main, ["--etiqueta", "final", "--sensibilidad-q",
                                               "--ganancia", "--salida", str(directorio)])
    assert "Se ignora oof_final_nb_gaussiano.csv: no tiene su cv_final_<modelo>.csv" in texto
    assert [r.name for r in oof_sin_cv(directorio, "final")] == ["oof_final_nb_gaussiano.csv"]
    for nombre in ("sensibilidad_q_final.csv", "ganancia_final.csv"):
        assert set(pd.read_csv(directorio / nombre)["modelo"]) == {"knn", SIN_MODELO}, nombre

    # El OOF de knn pisado por el de A0: las métricas de su cv ya no lo respaldan.
    a0.to_csv(directorio / "oof_final_knn.csv", index=False)
    try:
        leer_oof(directorio, "final", muestra)
    except ValueError as e:
        assert "el OOF de knn no reproduce" in str(e), e
    else:
        raise AssertionError("aceptó un OOF que no reproduce las métricas de su cv")

    # Un cv sin su OOF.
    (directorio / "oof_final_knn.csv").unlink()
    with _reemplazar(cargar_train=lambda: muestra.copy()):
        _niega(["--etiqueta", "final", "--ganancia", "--salida", str(directorio)])
    print("ok  la sensibilidad y la ganancia ignoran, con un aviso, un OOF sin su cv; un OOF que "
          "no reproduce su cv o un cv sin OOF son un error")


def test_verificar_oof_reconoce_el_oof_de_otra_corrida():
    # Dos folds de 6 clientes, cada uno con dos «yes», los de 0,9 y 0,4.
    oof = pd.DataFrame({FILA: range(12), "fold": [1] * 6 + [2] * 6,
                        "puntaje": [0.9, 0.4, 0.8, 0.3, 0.2, 0.1] * 2, "y": [1, 1, 0, 0, 0, 0] * 2})
    cv = _cv_de(oof, "referencia", "m")
    verificar_oof("m", oof, cv)

    # En el fold 2 cambian de lugar el «yes» de 0,4 y el «no» de 0,3: el AUC pasa de 7/8 a 6/8 y
    # el AP, de 5/6 a 3/4. Con q = 20 % se llama sólo al de 0,9, así que recall, precisión y F1
    # no cambian: sólo el AUC y el AP lo delatan.
    otra = oof.copy()
    otra.loc[[7, 9], "puntaje"] = otra.loc[[9, 7], "puntaje"].to_numpy()
    try:
        verificar_oof("m", otra, cv)
    except ValueError as e:
        assert "el OOF de m no reproduce auc, ap de validación del fold 2" in str(e), e
    else:
        raise AssertionError("aceptó un OOF con otro orden en el fold 2")

    try:
        verificar_oof("m", oof, cv[cv["fold"] == 1])
    except ValueError:
        pass
    else:
        raise AssertionError("aceptó un cv al que le falta un fold del OOF")
    print("ok  un OOF sólo pasa si reproduce las cinco métricas de validación de cada fold de su "
          "cv: un par de filas que cambia de orden en un fold ya lo delata")


def test_resumen_con_linea_base_y_brecha_a_mano(directorio):
    directorio.mkdir(parents=True)
    # knn: brechas 0,10, 0,11 y 0,09 → media 0,10, desvío 0,01.
    # rf: brechas 0,20, 0,19 y 0,18 → media 0,19, desvío 0,01.
    _tabla("referencia", "knn", [0.80, 0.82, 0.84], [0.70, 0.71, 0.75]).to_csv(
        directorio / "cv_referencia_knn.csv", index=False)
    _tabla("referencia", "rf", [0.99, 0.99, 0.99], [0.79, 0.80, 0.81]).to_csv(
        directorio / "cv_referencia_rf.csv", index=False)
    # Lo que el resumen no debe tomar: otra etiqueta y un resumen anterior de la misma.
    _tabla("final", "knn", [0.9] * 3, [0.6] * 3).to_csv(directorio / "cv_final_knn.csv", index=False)
    pd.DataFrame({"etiqueta": ["referencia"], "modelo": ["anterior"], "media": [0.0]}).to_csv(
        directorio / "cv_referencia_resumen.csv", index=False)
    # La línea de base llega sin etiqueta y con una columna propia antes de las del contrato.
    base = _tabla("-", SIN_MODELO, [0.5] * 3, [0.5] * 3, '{"strategy": "prior"}')
    base.drop(columns="etiqueta").assign(variante="linea_base").to_csv(directorio / LINEA_BASE,
                                                                       index=False)

    _en_silencio(resumir, directorio, "referencia")
    cv = pd.read_csv(directorio / "cv_referencia.csv")
    assert list(cv.columns) == ["etiqueta"] + CAMPOS and set(cv["etiqueta"]) == {"referencia"}
    assert set(cv["modelo"]) == {"knn", "rf", SIN_MODELO}
    assert len(cv) == 3 * 3 * 2 * len(METRICAS)

    r = pd.read_csv(directorio / "cv_referencia_resumen.csv")
    x = r.set_index(["modelo", "conjunto", "metrica"])
    assert _cerca(x.loc[("knn", "validacion", "auc"), "media"], 0.72)  # (0,70 + 0,71 + 0,75) / 3
    for modelo, media in (("knn", 0.10), ("rf", 0.19), (SIN_MODELO, 0.0)):
        assert _cerca(x.loc[(modelo, BRECHA, "auc"), "media"], media)
        assert x.loc[(modelo, BRECHA, "auc"), "n"] == 3
    assert _cerca(x.loc[("knn", BRECHA, "auc"), "desvio"], 0.01)
    assert _cerca(x.loc[("knn", BRECHA, "auc"), "error_estandar"], 0.01 / np.sqrt(3))
    assert _cerca(x.loc[("rf", BRECHA, "auc"), "desvio"], 0.01)
    assert list(tabla_validacion(r).index) == ["rf", "knn", SIN_MODELO]
    print("ok  --resumen toma sólo los cv de su etiqueta, agrega la línea de base y calcula la "
          "brecha fold a fold como a mano; la tabla sale ordenada por AUC")


def test_resumen_rechaza_filas_repetidas_o_claves_vacias():
    tabla = _tabla("referencia", "knn", [0.8] * 3, [0.7] * 3)
    # Una fila de AP: groupby la contaría dos veces, o la descartaría sin avisar si le falta una
    # clave, y la brecha, que sólo mira el AUC, no lo notaría.
    ap = tabla.index[tabla["metrica"] == "ap"][0]
    repetida = pd.concat([tabla, tabla.loc[[ap]]], ignore_index=True)
    vacia = tabla.copy()
    vacia.loc[ap, "configuracion"] = np.nan
    for nombre, mala in (("repetida", repetida), ("vacía", vacia)):
        try:
            resumir_tabla(mala)
        except ValueError:
            pass
        else:
            raise AssertionError(f"resumió una tabla con una clave {nombre}")
    assert len(resumir_tabla(tabla)) == 2 * len(METRICAS) + 1
    print("ok  el resumen se niega a promediar filas repetidas o con claves vacías")


def test_guardas_de_la_linea_de_comandos(directorio, train):
    # resultados/ se reemplaza por un directorio temporal que ya tiene un cv de la etiqueta
    # referencia: el caso de --modelo con --resumen sólo lo puede negar su guarda, y si una guarda
    # fallara nada se escribiría en el resultados/ real. Con train reducido a la muestra, una
    # guarda rota tampoco correría la validación cruzada sobre todo train.
    falso, vacio = directorio / "resultados", directorio / "vacio"
    falso.mkdir(parents=True)
    vacio.mkdir()
    _tabla("referencia", "knn", [0.8] * K, [0.7] * K).to_csv(falso / "cv_referencia_knn.csv",
                                                             index=False)
    ayuda = io.StringIO()
    with contextlib.redirect_stdout(ayuda):
        try:
            experimentos.main(["--help"])
        except SystemExit as e:
            assert e.code == 0
    assert "--etiqueta" in ayuda.getvalue() and "--rapido" in ayuda.getvalue()

    casos = [
        ["--modelo", "knn"],                                                # falta la etiqueta
        ["--etiqueta", "intermedia"],                                       # etiqueta desconocida
        ["--etiqueta", "referencia", "--resumen", "--modelo", "knn"],       # --modelo con un derivado
        ["--etiqueta", "referencia", "--modelo", "knn", "--n-jobs", "0"],   # procesos sin sentido
        ["--etiqueta", "referencia", "--modelo", "knn", "--n-jobs", "-1"],
        ["--rapido", "--salida", str(falso)],                               # --rapido en resultados/
        ["--rapido", "--salida", str(falso / "rapido")],
        ["--etiqueta", "referencia", "--resumen", "--salida", str(vacio)],  # nada que resumir
        ["--etiqueta", "referencia", "--ganancia", "--salida", str(vacio)],
    ]
    muestra = submuestra(train)
    with _reemplazar(DIR=falso, cargar_train=lambda: muestra.copy()):
        for argv in casos:
            _niega(argv)
    assert sorted(p.name for p in falso.iterdir()) == ["cv_referencia_knn.csv"]
    assert not any(vacio.iterdir())
    print("ok  --help funciona; sin etiqueta, con --modelo en un modo derivado, con --n-jobs menor "
          "que 1, con --rapido dentro de resultados/ o sin archivos que leer, la línea de comandos "
          "se niega y no escribe nada")


def test_hiperparametros_de_cada_etiqueta():
    # final: los de configuracion.py pisan a los de la referencia; los que no nombran quedan como
    # están. referencia: REFERENCIA, aunque configuracion.py diga otra cosa o no nombre al modelo.
    antes = {m: dict(h) for m, h in REFERENCIA.items()}
    with _reemplazar(HIPERPARAMETROS_FINALES={"rf": {"max_depth": 8}}):
        assert hiperparametros_vigentes("rf") == {**REFERENCIA["rf"], "max_depth": 8}
        assert hiperparametros_de("rf", "final") == hiperparametros_vigentes("rf")
        arboles = min(REFERENCIA["rf"]["n_estimators"], ARBOLES_RAPIDO)
        assert hiperparametros_vigentes("rf", rapido=True) == {
            **REFERENCIA["rf"], "max_depth": 8, "n_estimators": arboles}
        assert hiperparametros_de("rf", "referencia") == REFERENCIA["rf"]
        assert hiperparametros_de("rf", "referencia", rapido=True) == {
            **REFERENCIA["rf"], "n_estimators": arboles}
        assert hiperparametros_de("knn", "referencia") == REFERENCIA["knn"]
        for malo in (lambda: hiperparametros_vigentes("knn"),
                     lambda: hiperparametros_de("rf", "intermedia")):
            try:
                malo()
            except ValueError:
                continue
            raise AssertionError("aceptó un modelo sin hiperparámetros en configuracion.py o una "
                                 "etiqueta desconocida")
    assert REFERENCIA == antes, "recortar los árboles del modo rápido no toca REFERENCIA"
    assert ARBOLES_RAPIDO <= 20
    # Sin --modelo corren todos los de MODELOS_FINALES: cada uno tiene que tener los suyos.
    for modelo in MODELOS_FINALES:
        hiperparametros_de(modelo, "final")
        hiperparametros_de(modelo, "referencia")
    print("ok  la etiqueta final usa los de configuracion.py sobre la referencia y la etiqueta "
          f"referencia, REFERENCIA; en modo rápido, RF con {ARBOLES_RAPIDO} árboles o menos; cada "
          "modelo de MODELOS_FINALES tiene los suyos")


def test_sensibilidad_q_calculada_a_mano():
    # Fold 1: 10 clientes, «yes» en los dos primeros.
    #   q = 10 %, 1 llamada: el de 0,9, un «yes». Recall 1/2, precisión 1, F1 2/3.
    #   q = 20 %, 2 llamadas: 0,9 («yes») y 0,8 («no»). Recall 1/2, precisión 1/2, F1 1/2.
    # Fold 2: 10 clientes, 3 «yes» (0,9, 0,85 y 0,4).
    #   q = 10 %, 1 llamada: el de 0,95, un «no». Recall 0, precisión 0, F1 0.
    #   q = 20 %, 2 llamadas: 0,95 («no») y 0,9 («yes»). Recall 1/3, precisión 1/2, F1 2/5.
    # Si se juntaran los folds, con q = 10 % se llamaría a 2 de 20 y el recall sería 1/5.
    y = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
    s = [0.9, 0.1, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.05] + [0.95, 0.9, 0.3, 0.85, 0.2, 0.1, 0.4,
                                                                0.5, 0.6, 0.7]
    filas = np.arange(100, 300, 10)
    oof = pd.DataFrame({FILA: filas, "fold": [1] * 10 + [2] * 10, "puntaje": s})
    # El train sintético va en orden inverso: la y sólo puede llegar por el cruce por `fila`.
    train = _train_sintetico(filas, y).iloc[::-1]
    t = sensibilidad_q({"m": con_objetivo(oof, train)}, qs=(0.10, 0.20))
    assert list(t.columns) == CAMPOS_SENSIBILIDAD and len(t) == 2 * 2 * 3
    esperado = {(1, 0.10): (1 / 2, 1, 2 / 3), (1, 0.20): (1 / 2, 1 / 2, 1 / 2),
                (2, 0.10): (0, 0, 0), (2, 0.20): (1 / 3, 1 / 2, 2 / 5)}
    valores = t.set_index(["fold", "q", "metrica"])["valor"]
    for (fold, q), (recall, precision, f1) in esperado.items():
        assert _cerca(valores[(fold, q, "recall_q")], recall), (fold, q)
        assert _cerca(valores[(fold, q, "precision_q")], precision), (fold, q)
        assert _cerca(valores[(fold, q, "f1_q")], f1), (fold, q)
    print("ok  sensibilidad a q: recall, precisión y F1 por fold coinciden con el cálculo a mano")


def test_con_objetivo_rechaza_filas_que_no_estan_en_train():
    train = _train_sintetico([1, 2, 3], [1, 0, 0])
    oof = pd.DataFrame({FILA: [1, 2, 4], "fold": [1, 1, 1], "puntaje": [0.1, 0.2, 0.3]})
    try:
        con_objetivo(oof, train)
    except ValueError:
        pass
    else:
        raise AssertionError("aceptó una fila del OOF que no está en train")
    print("ok  una fila del OOF que no está en train es un error, no una y vacía")


def test_leer_oof_exige_los_mismos_folds(directorio):
    directorio.mkdir(parents=True)
    # 12 clientes, un «yes» de cada tres: con las dos particiones, cada fold tiene «yes» y «no».
    y = [1, 0, 0] * 4
    train = _train_sintetico(range(12), y)
    base = pd.DataFrame({FILA: range(12), "puntaje": np.linspace(0, 1, 12), "y": y})
    for modelo, fold in (("knn", [1, 2] * 6), ("rf", [1] * 6 + [2] * 6)):
        oof = base.assign(fold=fold)
        oof[CAMPOS_OOF].to_csv(directorio / f"oof_referencia_{modelo}.csv", index=False)
        _cv_de(oof, "referencia", modelo).to_csv(directorio / f"cv_referencia_{modelo}.csv",
                                                 index=False)
    try:
        leer_oof(directorio, "referencia", train)
    except ValueError as e:
        assert "mismos folds" in str(e), e
    else:
        raise AssertionError("aceptó OOF con folds distintos")
    print("ok  los OOF de una etiqueta tienen que cubrir las mismas filas con los mismos folds")


def test_ganancia_sobre_un_oof_sintetico():
    # 100 clientes y 10 «yes»: con n = 100, llamar al p % es llamar a exactamente p clientes.
    # Los «yes» son las filas 0 a 4 (fold 1) y 50 a 54 (fold 2).
    n = 100
    i = np.arange(n)
    y = ((i < 5) | ((i >= 50) & (i < 55))).astype(int)
    fold = np.where(i < 50, 1, 2)
    base = pd.DataFrame({FILA: i, "fold": fold, "y": y})
    oofs = {
        # Los 10 «yes» con los 10 puntajes más altos.
        "perfecto": base.assign(puntaje=np.where(y == 1, 200.0 + i, i)),
        # Los 10 «yes» con los 10 más bajos.
        "invertido": base.assign(puntaje=np.where(y == 1, -200.0 + i, i)),
        # Perfecto dentro de cada fold, pero todo el fold 1 puntúa más alto que el fold 2. La lista
        # junta los dos folds: primero los 5 «yes» del fold 1, después sus 45 «no», después el
        # fold 2. Fold por fold se llegaría al 100 % con el 10 % de la lista; junto, al 55 %.
        "por_fold": base.assign(puntaje=1000.0 * (fold == 1) + 100.0 * y + i),
    }
    g = ganancia(con_sin_modelo(oofs)).set_index(["modelo", "p"])["pct_yes_alcanzado"]
    assert len(g) == 4 * len(PORCENTAJES)
    for p in PORCENTAJES:
        assert _cerca(g[("perfecto", p)], min(10 * p, 100)), p
        assert _cerca(g[("invertido", p)], max(0, 10 * (p - 90))), p
        assert _cerca(g[(SIN_MODELO, p)], p), p
        por_fold = 10 * p if p <= 5 else 50 if p <= 50 else min(50 + 10 * (p - 50), 100)
        assert _cerca(g[("por_fold", p)], por_fold), p
    print("ok  ganancia: el orden perfecto llega al 100 % en p = 10 %, el invertido no alcanza "
          "ningún «yes» hasta pasar el 90 %, sin modelo es la diagonal y la lista junta los folds")


def main():
    train = cargar_train()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        test_submuestra_estratificada_con_semilla_42(train)
        test_rapido_escribe_el_contrato(tmp / "rapido", train)
        test_rapido_resume_y_deriva(tmp / "rapido")
        test_cada_etiqueta_usa_su_configuracion(tmp / "configuracion", train)
        test_derivados_leen_solo_los_oof_de_su_cv(tmp / "configuracion" / "normal", train)
        test_resumen_con_linea_base_y_brecha_a_mano(tmp / "resumen")
        test_guardas_de_la_linea_de_comandos(tmp / "guardas", train)
        test_leer_oof_exige_los_mismos_folds(tmp / "oof")
    test_verificar_oof_reconoce_el_oof_de_otra_corrida()
    test_resumen_rechaza_filas_repetidas_o_claves_vacias()
    test_hiperparametros_de_cada_etiqueta()
    test_sensibilidad_q_calculada_a_mano()
    test_con_objetivo_rechaza_filas_que_no_estan_en_train()
    test_ganancia_sobre_un_oof_sintetico()
    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()

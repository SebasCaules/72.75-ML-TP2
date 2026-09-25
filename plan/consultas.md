# Consultas a la cátedra sobre el TP2

Borrador del mensaje del paso 0.2. Lo envía el grupo, antes de la clase del 30/09. Cada pregunta
tiene en `PLAN.md` (§3) un valor por defecto, así que el trabajo sigue aunque la respuesta tarde.

Antes de enviarlo, completar el número de grupo y los integrantes.

---

**Asunto:** TP2 — consultas sobre métricas, variables y partición (grupo N)

Buenas tardes:

Somos el grupo N (integrantes). Antes de avanzar con el TP2 queríamos hacerles algunas consultas:

1. **Métricas (punto 2.3).** ¿$F_1$ cuenta entre «las métricas vistas en clase»? En la Clase 4
   aparece nombrada, pero no encontramos su definición.
2. **Fuga de datos (punto 1).** `duration` no se conoce antes de hacer la llamada. ¿Alcanza con
   excluirla y justificarlo, o esperan que comparemos el modelo con y sin ella? Además, las
   variables del último contacto (`month`, `day_of_week`, `contact` y `campaign`, que según el
   diccionario incluye la última llamada), ¿las consideran disponibles antes de llamar?
3. **Desbalance.** Hay un 11 % de «yes». ¿Esperan que lo tratemos, por ejemplo con pesos de clase,
   o alcanza con elegir métricas acordes?
4. **Naive Bayes.** El dataset mezcla 10 variables numéricas y 10 categóricas. ¿Esperan alguna
   variante en particular, por ejemplo gaussiano o categórico con las numéricas discretizadas?
5. **Partición.** El archivo está ordenado por fecha, de mayo de 2008 a noviembre de 2010, y la tasa
   de «yes» cambia mucho con el tiempo. ¿Prefieren una partición aleatoria estratificada o una
   temporal?
6. **Fechas.** ¿Qué día nos toca defender, el 07/10 o el 14/10? ¿KNN se va a ver en la clase del
   30/09?

Muchas gracias.

Grupo N

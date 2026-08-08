# Calendario diario — Tesis MCD

> **Ventana:** 12 semanas full time. Inicio: miércoles 29 de julio de 2026 (Semana 0). Entrega estimada: viernes 23 de octubre de 2026.
> Ajustá por feriados. Al final está cómo comprimir a 8 semanas o estirar a 16.

---

## La regla del día

**Lo primero que hacés cada mañana es lanzar lo que tenga que correr en la GPU. Después leés o escribís mientras corre.**

Ese es el truco que hace que 12 semanas alcancen: las ~25 horas de GPU del plan no compiten con tu tiempo, corren en paralelo. Si esperás a que termine un experimento para ponerte a leer, el plan no entra.

Segunda regla: **nunca lances una corrida larga sin un smoke test de 20 minutos antes.** Perder una noche de GPU por un typo en el nombre de una columna es el error más caro y más evitable del proyecto.

---

## Semana 0 — Entender el reencuadre (mié 29 → vie 31 de julio)

Nada de contactar a nadie. Primero entendés vos.

| Día | Tarea | GPU |
|---|---|---|
| **Mié 29** | Leer **von Rueden, "Informed Machine Learning"** (arXiv:1903.12394), primera mitad. Ir anotando en qué casilla de la taxonomía cae cada uno de tus 4 brazos. | — |
| **Jue 30** | Terminar von Rueden. Escribir 1 página propia: *"mis cuatro métodos en la taxonomía de von Rueden"*. **Si no te sale esa página, no entendiste el reencuadre todavía** — es tu propio test, y además es el germen del Cap. 2.2. | — |
| **Vie 31** | Higiene del repo: sacar `pinn_model.py` de `archive/legacy_code/`, arreglar `requirements.txt`, mover `master_results.csv`, `pinn_factorial.csv` y `aggregate_metrics.csv` a `archive/`. | — |

---

## Semana 1 — Prueba de concepto (3 → 7 de agosto)

**El objetivo de esta semana es una sola cosa: averiguar con código si el reencuadre se sostiene.** Nada de infraestructura completa todavía; lo mínimo para responder la pregunta.

| Día | Tarea | GPU |
|---|---|---|
| **Lun 3** | Verificar que `pinn_model.py` (brazo de física blanda) corre en el estado actual del repo. 200 épocas de prueba. Documentar y arreglar lo que rompa. | ~1 h |
| **Mar 4** | Escribir un **mini-runner** (no el definitivo): solo 2 brazos —`wave_pinn` y `pinn_model`— con semilla fija por parámetro y salida a CSV con columnas `method` y `seed`. |  — |
| **Mié 5** | **Lanzar la prueba de concepto.** 2 posiciones (2000, 6000) × 3 niveles de ruido (0, 8000, 50000) × 3 semillas × 2 brazos = 36 corridas. `q_leak` fijo en 0.015. | 🔴 ~2.3 h |
| **Jue 6** | Analizar: calcular **sesgo** (media del error con signo) y **varianza** (desvío entre semillas) por brazo y por nivel de ruido. En paralelo, regenerar `loss_landscape.csv` con el wave-PINN actual. | 🟡 ~2 h |
| **Vie 7** | **DÍA DE DECISIÓN.** Escribir el veredicto (ver abajo). Leer **Karpatne, "Theory-Guided Data Science"** mientras lo pensás. | — |

### Cómo leer el resultado del viernes 7

| Lo que ves | Qué significa | Qué hacés |
|---|---|---|
| Wave-PINN: sesgo alto y ~constante con el ruido, varianza baja. Vanilla: sesgo bajo, varianza alta. | **El reencuadre está confirmado.** Tenés el trade-off sesgo-varianza medido sobre el eje del conocimiento del dominio, con tus propios datos. | Es tu tesis. Seguí con el plan y ahora sí buscá tutor. |
| Los dos brazos tienen el mismo sesgo de ~150 m. | El sesgo no viene de la inyección analítica sino de otro lado (el MOC, el dataset, el filtrado temporal). El reencuadre **sigue siendo válido** —la comparación de 4 brazos se sostiene igual— pero perdés esa demostración puntual. | Seguí, y movés el diagnóstico del sesgo a la semana 7 como estaba. |
| El brazo vanilla directamente no converge. | Problema de implementación, no conceptual. | Dedicá el lunes 10 a arreglarlo antes de seguir. |
| El paisaje de pérdida regenerado sigue plano. | Confirmás la no identificabilidad. **Es tu mejor figura.** | Subí el Experimento 3 de prioridad. |

> **Gate:** al final del viernes 7 sabés si el reencuadre tiene respaldo empírico. Recién con eso en la mano tiene sentido hablar con alguien.

---

## Semana 2 — Tutor (con evidencia) + fundamentos (10 → 14 de agosto)

| Día | Tarea | GPU |
|---|---|---|
| **Lun 10** | Con el resultado del viernes en la mano: reescribir título y abstract (opción B de `reencuadre_ciencia_datos.md`), actualizar `README.md`, cerrar el alcance por escrito. | — |
| **Mar 11** | Identificar 3 docentes candidatos (perfil: ML, aprendizaje profundo, metodología experimental, inferencia). Mandar los mails, **adjuntando el gráfico de sesgo-varianza de la semana 1**. Un pitch con un resultado adentro pesa distinto que uno con una idea. | — |
| **Mié 12** | **Ejercicio: PINN de calor 1D desde cero.** Sin copiar código. `∂u/∂t = α ∂²u/∂x²`, verificar contra la solución analítica. Es el día que más te va a rendir del mes. | — |
| **Jue 13** | Leer **[13] Raissi et al. 2019**, secciones 3-4, con foco en el problema **inverso**. Después del ejercicio de ayer se lee distinto. | — |
| **Vie 14** | Extender el mini-runner a `run_experiments.py` completo: los 4 brazos, resume, semillas. Smoke test 4 métodos × 2 escenarios × 2 semillas. | ~2 h |

> **Gate:** el runner funciona end-to-end y es reproducible. Sin esto, no lances nada la semana que viene.

---

## Semana 3 — Experimento 1 + primer capítulo escrito (17 → 21 de agosto)

| Día | Tarea | GPU |
|---|---|---|
| **Lun 17** | **Lanzar Experimento 1** (espectro de sesgo inductivo): 4 x_leak × 3 q_leak × 3 noise × 3 semillas, `n_sensors`=3, los 4 brazos. Dejarlo corriendo. | 🔴 ~13 h |
| **Mar 18** | Mientras corre: escribir **Cap. 2.1 — Sesgo inductivo y trade-off sesgo-varianza**. Ya tenés todo lo que hace falta de von Rueden y Karpatne. | 🔴 sigue |
| **Mié 19** | Escribir **Cap. 2.2 — El espectro de modelos**, reusando la página del martes 4. Chequear que Exp 1 avanza sin errores. | 🔴 sigue |
| **Jue 20** | Leer **Willard et al.** (arXiv:2003.04919) — intro y tabla de clasificación, saltear el catálogo. Leer **[15] Karniadakis 2021**. | ✅ termina |
| **Vie 21** | Primer análisis de Exp 1: tabla de media ± desvío por método. **Chequear la hipótesis del bias:** ¿el wave-PINN tiene bias alto y varianza baja, y la PINN vanilla al revés? | — |

> **Momento clave del proyecto.** Si el viernes se confirma la hipótesis del sesgo inductivo, tenés la tesis entera cerrada conceptualmente. Anotá el resultado con detalle.

---

## Semana 4 — Eficiencia de datos + identificabilidad (24 → 28 de agosto)

| Día | Tarea | GPU |
|---|---|---|
| **Lun 24** | **Lanzar Experimento 2** (curva de eficiencia de datos del LSTM): 10 %, 25 %, 50 %, 100 % del corpus × 3 semillas. | 🟡 ~1 h |
| **Mar 25** | Graficar Exp 2: error vs. cantidad de datos etiquetados, con PINN y baseline analítico como líneas horizontales. Identificar el punto de cruce. | — |
| **Mié 26** | **Regenerar `loss_landscape.csv` con el wave-PINN actual.** Verificar si el paisaje sigue plano. Averiguar de dónde sale el pico en 3000-3500 m. | 🟡 ~2 h |
| **Jue 27** | **Lanzar Experimento 3** (frontera de identificabilidad): 5 q_leak × 5 noise × 3 semillas, x_leak fijo en 6000, wave-PINN + analítico. | 🔴 ~5 h |
| **Vie 28** | Leer **[18] Wang, Teng & Perdikaris 2021** (patologías del flujo de gradientes). Es la justificación teórica de tu curriculum de 3 fases. | ✅ termina |

---

## Semana 5 — Ablation, sensores y diseño experimental escrito (31 ago → 4 sep)

| Día | Tarea | GPU |
|---|---|---|
| **Lun 31** | **Lanzar Experimento 4** (ablation wave-injection): wave-PINN vs. la misma red sin `P_sing`, 12 escenarios × 3 semillas. | 🟡 ~2 h |
| **Mar 1** | **Lanzar Experimento 5** (presupuesto de sensores): 4 n_sensors × 3 noise × 3 semillas, 2 brazos. | 🔴 ~4.5 h |
| **Mié 2** | Escribir **Cap. 3 — Banco de pruebas y diseño experimental**. El MOC va acá pero breve; la derivación completa va al apéndice. | 🔴 sigue |
| **Jue 3** | Escribir **Cap. 3.2-3.3**: diseño factorial, métricas, protocolo. Justificar el diseño reducido (por qué no cruzás todo con todo). | ✅ termina |
| **Vie 4** | Leer **[19] Wang, Yu & Perdikaris 2022** (NTK) — quedate con las conclusiones, es denso. Buffer. | — |

> **Gate:** los cinco experimentos corridos. A partir de acá el proyecto es análisis y escritura.

---

## Semana 6 — Los cuatro modelos + competencia (7 → 11 de septiembre)

| Día | Tarea | GPU |
|---|---|---|
| **Lun 7** | Escribir **Cap. 4.1-4.2**: baseline analítico (NPW/balance de masa) y LSTM. Explicitar la asimetría de caudalímetros. | — |
| **Mar 8** | Escribir **Cap. 4.3**: PINN con física en el loss. Conectar con [18] y [19]. | — |
| **Mié 9** | Escribir **Cap. 4.4**: tu wave-injection, presentada como residual learning. Sin vocabulario de física. | — |
| **Jue 10** | Leer **[23] Tariq et al. 2023 (SPE)** — tu competencia directa. ¿Asumen `q_leak` conocido? ¿Qué hacen distinto? | — |
| **Vie 11** | Escribir la subsección de posicionamiento: en qué se diferencia tu trabajo de [23]. Leer **[20] Krishnapriyan 2021**. | — |

---

## Semana 7 — Atacar el bias de 150 m (14 → 18 de septiembre)

| Día | Tarea | GPU |
|---|---|---|
| **Lun 14** | Diagnóstico del bias: comparar la onda analítica `P_sing` contra la onda del MOC punto a punto. Cuantificar la discrepancia. | 🟡 ~1 h |
| **Mar 15** | Probar corrección: ajustar el desfasaje temporal (0.125 s) o el suavizado Heaviside. Correr 6 escenarios de control. | 🟡 ~2 h |
| **Mié 16** | Si la corrección funciona, re-correr el subconjunto afectado de Exp 1. Si no, documentar el bias como costo del sesgo inductivo (que es igual de válido). | 🔴 ~4 h |
| **Jue 17** | Consolidar **todas** las figuras: espectro, curva de eficiencia, mapa de identificabilidad, ablation, sensores, paisaje de pérdida. | ✅ termina |
| **Vie 18** | Leer **[7] Shamloo & Haghighi 2009** y **[8] Liggett & Chen 1994** (problema inverso clásico). | — |

---

## Semana 8 — Capítulo de resultados (21 → 25 de septiembre)

| Día | Tarea |
|---|---|
| **Lun 21** | **Cap. 5.1** — Comparación del espectro completo. Tabla principal con media ± desvío. |
| **Mar 22** | **Cap. 5.2** — Eficiencia de datos y escasez de etiquetas. El argumento del corpus inexistente en la industria. |
| **Mié 23** | **Cap. 5.3** — Frontera de identificabilidad. Mapa de calor + paisaje de pérdida. |
| **Jue 24** | **Cap. 5.4** — Ablation de la wave-injection y presupuesto de sensores. |
| **Vie 25** | Revisión completa del Cap. 5: ¿cada figura tiene una conclusión escrita al lado? ¿hay alguna tabla sin interpretar? |

---

## Semana 9 — Discusión y apéndice CFD (28 sep → 2 de octubre)

| Día | Tarea |
|---|---|
| **Lun 28** | **Cap. 6.1** — Cuándo conviene cada nivel de prior. Es la respuesta a tu pregunta de investigación. |
| **Mar 29** | **Cap. 6.2** — Limitaciones: validez externa, supuestos físicos, alcance de los datos sintéticos. |
| **Mié 30** | **Apéndice A** — Derivación de Navier-Stokes a las ecuaciones de compatibilidad. |
| **Jue 1** | **Apéndice A** (cont.) — MOC, CFL=1, validación del simulador. Acá va todo el CFD. |
| **Vie 2** | **Cap. 7** — Conclusiones y trabajo futuro (XPINN, validación externa, fricción no estacionaria). |

---

## Semana 10 — Introducción y cierre del marco teórico (5 → 9 de octubre)

| Día | Tarea |
|---|---|
| **Lun 5** | Leer **[1] Murvay & Silea 2012**, **[2] Adegboye 2019**, **[3] Geiger 2006** (surveys de detección de fugas) para la introducción. |
| **Mar 6** | Leer **[9] API 1130**, **[10] Silva 1996**, **[11] Hochreiter & Schmidhuber 1997**, **[22] Cai 2021**. Completar justificación de baselines. |
| **Mié 7** | **Cap. 1 — Introducción.** Se escribe ahora, con el panorama cerrado. Incluir la motivación del tema. |
| **Jue 8** | Cerrar **Cap. 2** con las referencias que faltaban. Revisar que toda decisión de diseño esté conectada a una cita. |
| **Vie 9** | Resumen/abstract definitivo. Verificar coherencia con el título. |

---

## Semana 11 — Revisión integral (12 → 16 de octubre)

| Día | Tarea |
|---|---|
| **Lun 12** | Lectura completa de punta a punta, sin corregir, solo anotando. |
| **Mar 13** | Corregir lo anotado. Verificar consistencia de números entre texto, tablas y figuras. |
| **Mié 14** | Bibliografía: verificar cada cita (autores, año, volumen, páginas). Verificar que toda ref citada esté en la lista y viceversa. |
| **Jue 15** | Reproducibilidad: clonar el repo en limpio, seguir el README, verificar que corre. Arreglar lo que falle. |
| **Vie 16** | **Buffer.** Reservado para lo que se haya atrasado. No planifiques nada acá. |

---

## Semana 12 — Defensa y entrega (19 → 23 de octubre)

| Día | Tarea |
|---|---|
| **Lun 19** | Escribir las respuestas a las preguntas de defensa de `guia_de_estudio.md`, **literalmente**, sobre todo las incómodas. |
| **Mar 20** | Armar la presentación. Estructura: pregunta → espectro → experimentos → cuándo conviene cada método. |
| **Mié 21** | Poner a punto el dashboard Streamlit (`app/app.py`) como demo en vivo. |
| **Jue 22** | Ensayo completo de la defensa, cronometrado, en voz alta. Idealmente frente a alguien. |
| **Vie 23** | Últimos ajustes y **entrega**. |

---

## Cómo reescalar

**Si tenés 8 semanas:** eliminá el Experimento 5 (presupuesto de sensores) y la semana 7 (bias de 150 m — documentalo como limitación en vez de atacarlo). Fusioná las semanas 9 y 10. El núcleo defendible son los Experimentos 1, 2 y 3.

**Si tenés 16 semanas:** agregá 2 semanas de validación externa (contactar a los autores del rig de ITA, o adaptar BattLeDIM) y 2 semanas para convertir el trabajo en un paper corto. Con el Experimento 2 solo, hay material publicable.

**Si te atrasás:** el orden de sacrificio es Exp 5 → bias de 150 m → Exp 4 → dashboard. **Nunca** sacrifiques las semillas múltiples ni el Experimento 1.

---

## Checkpoints

| Fecha | Tiene que estar listo |
|---|---|
| **30 jul** | Tu página de "los 4 métodos en la taxonomía de von Rueden" escrita — tu test de que entendiste |
| **7 ago** | **Veredicto de la prueba de concepto:** el reencuadre tiene respaldo empírico o no |
| **11 ago** | Mails a tutores enviados, con el gráfico de sesgo-varianza adjunto |
| **14 ago** | Runner reproducible funcionando end-to-end |
| **21 ago** | Experimento 1 corrido y analizado — *el momento decisivo* |
| **4 sep** | Los 5 experimentos terminados |
| **25 sep** | Capítulo de resultados escrito |
| **9 oct** | Documento completo en borrador |
| **23 oct** | Entrega |

Si llegás al 21 de agosto sin el Experimento 1 analizado, recortá alcance ese mismo día en vez de esperar.

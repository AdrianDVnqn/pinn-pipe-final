# To-Do List Definitiva — Tesis MCD

> **Reencuadre aplicado:** el proyecto ya no es "una PINN para detectar fugas" sino **un estudio comparativo de sesgos inductivos para estimación de parámetros latentes bajo ruido y escasez de etiquetas**, con el ducto como caso de aplicación. Ver [`reencuadre_ciencia_datos.md`](./reencuadre_ciencia_datos.md) para el razonamiento completo.
>
> Última revisión: 2026-07-28. Reemplaza a la versión anterior de este archivo.

---

## Presupuesto de cómputo — leelo antes de planificar

Esto condiciona todo lo demás. Una corrida de PINN son ~225 s en tu RTX 3080 (según `benchmark_v2_results.csv`). El factorial completo con todas las variables sería:

```
4 x_leak × 3 q_leak × 3 noise × 4 n_sensors × 5 semillas = 720 corridas ≈ 45 h por brazo PINN
```

Con dos brazos PINN eso son **90 horas de GPU**. No es viable en paralelo con la cursada. La solución es un **diseño reducido**: no todas las variables se cruzan con todas. Presupuesto realista:

| Experimento | Diseño | Corridas | GPU aprox. |
|---|---|---|---|
| 1. Espectro de sesgo inductivo | 4 x_leak × 3 q_leak × 3 noise × **3 semillas**, `n_sensors` fijo en 3 | 108 × 2 brazos PINN | ~13 h |
| 2. Eficiencia de datos (LSTM) | 4 fracciones del corpus × 3 semillas | 12 (solo LSTM, barato) | ~1 h |
| 3. Frontera de identificabilidad | 5 q_leak × 5 noise × 3 semillas, `x_leak` fijo en 6000 | 75 (solo wave-PINN + analítico) | ~5 h |
| 4. Ablation wave-injection | 12 escenarios × 3 semillas | 36 | ~2 h |
| 5. Presupuesto de sensores | 4 n_sensors × 3 noise × 3 semillas, resto fijo | 36 × 2 brazos | ~4.5 h |
| | | **Total** | **~25 h** |

25 horas repartidas en dos o tres fines de semana es manejable. `benchmark_wave_pinn.py` ya tiene lógica de resume (saltea corridas hechas), así que podés cortar y retomar sin perder trabajo — mantené esa lógica al refactorizar.

> **Regla:** 3 semillas es el mínimo para reportar media ± desvío. Si tenés que recortar algo, recortá el grid, **nunca** las semillas. Un factorial chico con repeticiones vale más que uno grande sin ellas.

---

## 🔴 Fase 0 — Desbloquear el proyecto (esta semana)

Sin esto lo demás no importa. Es lo más barato y lo más urgente.

- [ ] **Conseguir tutor.** Usar el pitch de `reencuadre_ciencia_datos.md` (sección "Cómo pitchearlo a un tutor"). Perfil a buscar: ML / aprendizaje profundo / metodología experimental / inferencia estadística. **No** hace falta que sepa PINNs ni fluidos.
- [ ] **Reescribir el título y el abstract** con la opción B del reencuadre, antes de la primera reunión. Es lo único que el tutor va a leer para decidir.
- [ ] **Actualizar `README.md`** al framing nuevo. Hoy arranca con "Physics-Informed Neural Network... ecuaciones de flujo transitorio 1D" — eso es exactamente lo que espanta a un director de DS.

---

## 🔴 Fase 1 — Infraestructura experimental (antes de correr nada)

El problema de fondo hoy: los CSVs de `results/` son de fechas y versiones distintas del código, así que **ninguna comparación entre métodos es válida todavía**.

- [ ] **Rescatar `pinn_model.py` de `archive/legacy_code/`.** Verifiqué el código: usa `leak_source()` gaussiano y penaliza residuos de EDP en el loss, sin inyección analítica. O sea que **es el brazo "física blanda"**, el grupo de control del experimento central — no es código muerto.
- [ ] **Archivar los CSVs obsoletos.** `master_results.csv`, `pinn_factorial.csv` y `aggregate_metrics.csv` (11-12 jun) vienen del PINN roto anterior a las correcciones de `avances.md` — muestran ~2 km y 165 % de error constante. Moverlos a `archive/` para que nadie los confunda con resultados vigentes.
- [ ] **Escribir un runner único** (`run_experiments.py`) que corra los cuatro brazos sobre los mismos escenarios, con la misma lista de semillas, escribiendo a un CSV con columna `method` y `seed`. Sin esto no hay comparación defendible.
- [ ] **Fijar semillas explícitamente** en `torch`, `numpy` y en la generación de ruido. Hoy no hay control de semilla en ningún lado.
- [ ] **Arreglar `requirements.txt`** — falta `torch` y `streamlit`. Hoy el repo no es reproducible siguiendo el README.
- [ ] **Decidir qué pasa con `xpinn_model.py`.** Está en el repo sin rol en ningún benchmark. O entra como quinto brazo, o se declara trabajo futuro y se archiva. Código muerto sin explicación resta.

---

## 🟠 Fase 2 — Los experimentos

### Experimento 1 — Espectro de sesgo inductivo *(el central)*

Los cuatro brazos, mismos escenarios, mismas semillas, misma sesión:

| Nivel de prior | Implementación |
|---|---|
| Total (analítico puro) | `baseline_mass_balance.py` |
| Alto (física en arquitectura) | `wave_pinn.py` |
| Medio (física en el loss) | `pinn_model.py` (rescatado) |
| Nulo (data-driven) | `baseline_lstm.py` |

- [ ] Correr el diseño reducido (108 corridas por brazo PINN).
- [ ] Reportar **media ± desvío** de error de localización, error de `q_leak`, y tiempo de inferencia.
- [ ] **Explicitar la asimetría de información:** el baseline de balance de masa tiene caudalímetros en ambos extremos, así que `q_leak = Q_in − Q_out` es casi trivial para él. Las PINNs infieren `q_leak` **solo desde presión**. Sin esta aclaración tu resultado de `q_leak` parece malo cuando en realidad estás resolviendo un problema más difícil.

### Experimento 2 — Eficiencia de datos y escasez de etiquetas *(el más vendedor)*

El argumento más fuerte que tenés y todavía no usás. El LSTM necesita un corpus de fugas etiquetadas (`LeakDataset` con splits train/val/test); las PINNs se ajustan por instancia, sin histórico. **En la industria ese corpus no existe**: las fugas son raras, caras, y nadie las provoca a propósito.

- [ ] Entrenar el LSTM con 10 %, 25 %, 50 % y 100 % del corpus.
- [ ] Graficar error vs. cantidad de datos etiquetados, con PINN y baseline analítico como líneas horizontales (no dependen del corpus).
- [ ] Identificar el **punto de cruce**. Ese gráfico solo es probablemente la figura más citable de la tesis.

### Experimento 3 — Frontera de identificabilidad

Acá es donde va a parar el fallo que antes era una crisis: con `q=0.005` + `noise=50000` el modelo colapsa a ~1985 m sin importar la posición real (errores de 20-60 % en `benchmark_v2_results.csv`). Reencuadrado, no es un bug: es el límite de identificabilidad del parámetro.

- [ ] Barrido fino de tamaño de fuga × SNR → mapa de calor de error por método.
- [ ] Usar `scan_loss_landscape.py` en los casos que fallan para mostrar **por qué** falla (paisaje de pérdida sin mínimo distinguible vs. mínimo local espurio).
- [ ] Redactarlo como caracterización de la región de operación válida de cada familia de métodos, no como limitación vergonzante.

### Experimento 4 — Ablation de la wave-injection

Sin esto, la contribución arquitectónica central no tiene evidencia. Nota: el Experimento 1 ya te da media ablation (wave-PINN vs. PINN con física en el loss), así que esto es el complemento fino.

- [ ] Comparar wave-PINN vs. la misma red **sin** el término `P_sing`.
- [ ] Métricas: error final, velocidad de convergencia, estabilidad frente al fenómeno de Gibbs.

### Experimento 5 — Presupuesto de sensores

`avances.md` menciona soporte para 2, 3, 5 y 11 sensores, pero `benchmark_wave_pinn.py` no itera sobre esa variable y `benchmark_v2_results.csv` no tiene la columna.

- [ ] Agregar `n_sensors` al runner y correr el diseño reducido.
- [ ] Encuadrarlo como **diseño de adquisición / presupuesto de features**, no como detalle de instrumentación.

### Pendiente técnico transversal

- [ ] **Corregir el bias sistemático de ~150 m.** Es demasiado constante (146-175 m en todas las corridas) para presentarlo como "aceptable en ingeniería" — un jurado va a preguntar por qué no lo calibrás. Si lo corregís, tu PINN pasa de perder contra el NPW en ruido bajo a empatarle, y le sigue ganando en ruido alto. Mejora la historia entera.

---

## 🟡 Fase 3 — Lectura

### Nuevas referencias que exige el reencuadre *(prioridad máxima — hoy no están en `marco_teorico_referencias.md`)*

El marco teórico actual tiene 23 refs, todas de PINNs o de fluidos. **Falta toda la literatura del eje conceptual nuevo**, que es justamente la que le va a hablar a tu tutor de DS:

- [ ] **von Rueden et al., "Informed Machine Learning — A Taxonomy and Survey of Integrating Prior Knowledge into Learning Systems"**, IEEE TKDE (DOI 10.1109/TKDE.2021.3079836; preprint arXiv:1903.12394). **La referencia clave del reencuadre:** propone exactamente la taxonomía de *dónde* se inyecta el conocimiento (arquitectura vs. función de pérdida vs. datos) que estructura tu comparación de cuatro brazos.
- [ ] **Karpatne et al., "Theory-Guided Data Science: A New Paradigm for Scientific Discovery from Data"**, IEEE TKDE 29(10), 2318-2331, 2017. El paper fundacional del paradigma híbrido teoría+datos.
- [ ] **Willard et al., "Integrating Physics-Based Modeling with Machine Learning: A Survey"**, arXiv:2003.04919, 2020. Taxonomía complementaria, muy citada.
- [ ] Buscar una referencia clásica de **sesgo inductivo** y otra de **trade-off sesgo-varianza** para el capítulo 2.1 (Mitchell 1980 sobre la necesidad de sesgos; Geman, Bienenstock & Doursat 1992 sobre el dilema sesgo/varianza — verificar la cita exacta antes de incluirla).

### Referencias del marco teórico existente, reordenadas

- [ ] **[23] Tariq et al. 2023 (SPE)** — competencia directa. Sigue siendo lectura obligatoria para posicionar tu aporte.
- [ ] **[18] Wang, Teng & Perdikaris 2021** y **[19] Wang, Yu & Perdikaris 2022** — justifican el curriculum de 3 fases. Necesarias antes de escribir metodología.
- [ ] **[20] Krishnapriyan et al. 2021** — modos de falla; alimenta directo el Experimento 3.
- [ ] **[15] Karniadakis et al. 2021** (Nature Rev. Physics) — puente natural entre el eje "informed ML" y el eje PINNs.
- [ ] **[7] Shamloo & Haghighi 2009** y **[8] Liggett & Chen 1994** — el problema inverso con optimización clásica, para encuadrar tu baseline.
- [ ] Las refs de las secciones 1 y 4 del marco teórico ([1][2][3], [9][10][11][12]) se leen al final, sirven para la introducción y la justificación de baselines.

---

## 🟢 Fase 4 — Escritura

Estructura nueva (detalle completo en `reencuadre_ciencia_datos.md`). El orden de escritura sugerido:

- [ ] **Cap. 2.1-2.2** — Sesgo inductivo y el espectro de modelos. Escribilo primero: define la tesis y es lo que valida tu tutor.
- [ ] **Cap. 4** — Los cuatro modelos comparados. Presentar la wave-injection como *residual learning* (modelo analítico = base learner, MLP = corrección del residuo). Sin física, en tres líneas.
- [ ] **Cap. 3** — Banco de pruebas y diseño experimental. El MOC va acá, **breve**.
- [ ] **Apéndice A** — Toda la derivación CFD (Navier-Stokes → ecuaciones de compatibilidad → MOC → validación del simulador). Acá va a parar lo que hoy asusta al tutor: no lo perdés, lo ubicás.
- [ ] **Cap. 5** — Resultados, uno por experimento.
- [ ] **Cap. 6** — Discusión: cuándo conviene cada nivel de prior. Validez externa (ver abajo).
- [ ] **Cap. 1** — Introducción, al final, cuando el panorama esté cerrado.

---

## 🔵 Fase 5 — Opcionales de alto retorno

- [ ] **Validez externa con datos externos.** No resuelve el problema científico central pero blinda la crítica de "probaste solo con tus propios datos". Opciones, de más a menos parecida a tu física: (a) pedirle los datos crudos a los autores del rig experimental de ITA en PVC (Soares, Covas & Reis 2011); (b) [BattLeDIM](https://zenodo.org/records/4017659) / [LeakDB](https://github.com/KIOS-Research/LeakDB), descargables ya, pero son redes de distribución de agua, no ducto único. Documentar honestamente las diferencias de escala, fluido y topología.
- [ ] **Dashboard Streamlit** (`app/app.py`) como demo para la defensa oral. No aporta al documento escrito, aporta muchísimo a la presentación.

---

## Orden de ataque, en una línea

**Tutor y título → rescatar `pinn_model.py` y armar el runner con semillas → Experimento 1 → Experimento 2 → leer von Rueden y Karpatne → empezar a escribir el Cap. 2 → Experimentos 3, 4 y 5 en paralelo con la escritura.**

Sources:
- [Informed Machine Learning – A Taxonomy and Survey (IEEE Xplore)](https://ieeexplore.ieee.org/document/9429985/)
- [Informed Machine Learning (arXiv:1903.12394)](https://arxiv.org/abs/1903.12394)
- [Theory-Guided Data Science (ResearchGate)](https://www.researchgate.net/publication/311925961_Theory-guided_Data_Science_A_New_Paradigm_for_Scientific_Discovery)
- [BattLeDIM dataset (Zenodo)](https://zenodo.org/records/4017659)
- [LeakDB (GitHub)](https://github.com/KIOS-Research/LeakDB)

# Reencuadre: de proyecto de CFD a proyecto de Ciencia de Datos

> Objetivo: que la tesis sea dirigible y evaluable por un docente de MCD que **no sepa nada de golpe de ariete ni de PINNs**, sin tirar a la basura una sola línea del código ya escrito.

---

## La idea central

Hoy la tesis se presenta así: *"construí un simulador de flujo transitorio y una PINN para resolver el problema inverso de las ecuaciones de golpe de ariete"*. Eso suena a mecánica computacional, y un docente de ciencia de datos no se siente capacitado para dirigirlo.

El reencuadre es este: **el simulador MOC no es la contribución, es el banco de pruebas.** Y tener un banco de pruebas con ground truth exacto y controlable es un **lujo metodológico** que casi ningún proyecto de ciencia de datos tiene. La mayoría de los trabajos de DS no pueden responder "¿cuánto se equivocó mi modelo *realmente*?" porque no conocen la verdad. Vos sí. Esa es la fortaleza que hay que poner adelante, no esconder.

Con eso, el proyecto deja de ser sobre tuberías y pasa a ser sobre una pregunta central de machine learning:

> **¿Cuánto conocimiento del dominio conviene incorporar a un modelo, y en qué forma, cuando los datos son ruidosos y las etiquetas escasas?**

El ducto es el caso de estudio. La pregunta es de ciencia de datos.

---

## Por qué esto casi no te da trabajo extra

Porque **ya tenés implementados los cuatro puntos del espectro** de sesgo inductivo. Están dispersos en el repo y etiquetados como "baselines" y "código viejo", cuando en realidad son los cuatro brazos del experimento central:

| Grado de conocimiento físico incorporado | Cómo se incorpora | Archivo | Estado |
|---|---|---|---|
| **Total** (modelo analítico puro, cero aprendizaje) | Fórmula cerrada de triangulación NPW | `baseline_mass_balance.py` | ✅ Corrido |
| **Alto** (física en la *arquitectura*, restricción dura) | Inyección de la solución analítica de onda | `wave_pinn.py` | ✅ Corrido |
| **Medio** (física en la *función de pérdida*, restricción blanda) | Residuos de EDP penalizados en el loss | `archive/legacy_code/pinn_model.py` | ⚠️ Existe, sin correr en el benchmark actual |
| **Nulo** (puramente data-driven) | LSTM supervisada sobre datos etiquetados | `archive/legacy_code/baseline_lstm.py` | ✅ Corrido (versión vieja) |

Verifiqué el código: `pinn_model.py` usa `leak_source()` con un término gaussiano y penaliza los residuos de la EDP en el loss, **sin** la inyección analítica — o sea, es exactamente el brazo "física blanda". Y `baseline_lstm.py` tiene `LeakDataset` con splits train/val/test y `train_lstm()`, o sea es un aprendiz supervisado clásico que necesita un corpus etiquetado.

Ese `pinn_model.py` que archivaste como "código legacy" es en realidad **el grupo de control de tu experimento principal**. Sacalo de `archive/`.

---

## Traducción del vocabulario

El mismo trabajo, contado en el idioma que tu jurado evalúa. Esto no es cosmética: cambia qué preguntas te van a hacer.

| Hoy decís (CFD) | Deberías decir (DS) |
|---|---|
| Simulador MOC de flujo transitorio | Proceso generador de datos con ground truth conocido y controlable |
| Resolver el problema inverso de la EDP | Estimación de parámetros latentes a partir de observaciones ruidosas |
| `x_leak`, `q_leak` | Parámetros latentes a inferir; se analizan sesgo y varianza del estimador |
| Inyección de solución analítica | Sesgo inductivo estructural — la red aprende el **residuo** sobre un modelo base analítico (es *residual learning*: el modelo físico es el base learner, la MLP corrige lo que le falta) |
| Ruido del sensor (Pa) | Degradación controlada de la relación señal-ruido |
| Cantidad de sensores (2, 3, 5, 11) | Presupuesto de adquisición / diseño experimental sobre features |
| Fugas chicas indetectables con ruido alto | **Frontera de identificabilidad** del parámetro |
| Curriculum de 3 fases | Estrategia de optimización para un objetivo multi-término con gradientes en conflicto |
| El modelo colapsa a ~1985 m | Modo de fallo caracterizado: por debajo de cierto SNR el parámetro deja de ser identificable |

Fijate que la última fila convierte tu peor resultado en un hallazgo.

---

## Títulos posibles

**Opción A (la más "DS pura"):**
> *Sesgo inductivo versus datos: estudio comparativo de modelos analíticos, data-driven y physics-informed para la estimación de parámetros latentes bajo ruido y escasez de etiquetas*

**Opción B (equilibrada, la que recomiendo):**
> *¿Cuánta física conviene incorporar a un modelo de aprendizaje? Un estudio comparativo de sesgos inductivos aplicado a la localización de fugas en ductos*

**Opción C (conserva visible la aplicación industrial):**
> *Detección y localización de fugas en ductos: comparación sistemática de cuatro niveles de conocimiento del dominio bajo degradación de señal y escasez de datos etiquetados*

La B lidera con la pregunta de ML y deja la aplicación como complemento — que es exactamente el balance que necesitás para conseguir tutor sin perder la relevancia industrial.

---

## Estructura de capítulos reencuadrada

Compará con el orden actual de `marco_teorico_referencias.md`: la física baja de nivel, la metodología experimental sube.

1. **Introducción** — El problema de ML: cuánto prior incorporar. Escasez de etiquetas en contextos industriales. Recién al final, el caso de aplicación.
2. **Marco teórico**
   - 2.1 Sesgo inductivo y el trade-off sesgo-varianza (concepto de ML, no de física)
   - 2.2 El espectro: modelos mecanísticos → híbridos → puramente data-driven
   - 2.3 PINNs como caso de física en la función de pérdida — refs [13], [15], [22]
   - 2.4 Dificultades de optimización multi-objetivo — refs [18], [19], [20]
   - 2.5 Problemas inversos e identificabilidad
3. **Banco de pruebas y diseño experimental** ← *acá va todo el CFD, comprimido*
   - 3.1 El proceso generador de datos (MOC) — **breve**, con la derivación completa en apéndice
   - 3.2 Diseño factorial: posición × tamaño × SNR × nº de sensores × semilla
   - 3.3 Métricas y protocolo de evaluación
4. **Los cuatro modelos comparados** — un apartado por brazo del espectro; tu wave-injection se presenta acá como *aporte arquitectónico*, explicado en términos de residual learning
5. **Resultados**
   - 5.1 Comparación del espectro completo
   - 5.2 Eficiencia de datos y escasez de etiquetas
   - 5.3 Frontera de identificabilidad
   - 5.4 Ablation: ¿aporta la inyección analítica?
6. **Discusión y limitaciones** — cuándo conviene cada nivel de prior; validez externa
7. **Conclusiones**

**Apéndice A:** derivación de Navier-Stokes a las ecuaciones de compatibilidad, MOC, validación del simulador.

Ese apéndice es clave: es donde va a parar todo el CFD que hoy asusta al tutor. No lo perdés — lo ponés donde corresponde y le decís al tutor "esto está validado, no necesitás auditarlo".

---

## Los tres experimentos que hacen DS a la tesis

Todos baratos, todos reutilizan código existente.

### 1. Espectro completo de sesgo inductivo (el experimento central)

Los cuatro modelos, sobre los **mismos escenarios, mismas semillas, misma sesión**. Hoy tus CSVs son de fechas distintas y versiones distintas del código, así que la comparación no es válida.

- Métricas: error de localización (media ± desvío sobre semillas), error de `q_leak`, tiempo de inferencia, y **cuántos datos etiquetados necesita cada uno**.
- Resultado esperable según lo que ya viste: gana el analítico con señal limpia, gana el physics-informed con ruido alto, el LSTM queda atrás salvo que le des mucho dato. Ese cruce **es** la tesis.

### 2. Eficiencia de datos / escasez de etiquetas ← *el argumento más fuerte que tenés y todavía no usás*

Hay una asimetría enorme entre los métodos que hoy no está explicitada:

- El **LSTM necesita un corpus de fugas etiquetadas** para entrenar (`LeakDataset`, splits train/val/test).
- La **PINN se ajusta por instancia**: no necesita ningún histórico de fugas previas, solo la señal del evento actual y las ecuaciones.

En la industria real **no existen corpus de fugas etiquetadas** — las fugas son raras, caras y nadie las provoca a propósito. O sea que el método data-driven necesita justo lo que nunca vas a tener.

**Experimento:** entrenar el LSTM con 10%, 25%, 50% y 100% del dataset y graficar error vs. cantidad de datos etiquetados, con la PINN y el baseline analítico como líneas horizontales (no dependen del corpus). El punto de cruce es un resultado de ciencia de datos puro, del tipo que cualquier jurado de MCD entiende y valora inmediatamente.

### 3. Frontera de identificabilidad

Barrido fino de tamaño de fuga × SNR → mapa de calor de error por método. Esto convierte el fallo feo de `q=0.005 + ruido=50000` en un capítulo: *"caracterizamos la región del espacio de operación donde cada familia de métodos es aplicable"*.

Además ya tenés `scan_loss_landscape.py` y `figs/snr_heatmap.png`, así que la infraestructura está.

> **Transversal a los tres:** correr 3-5 semillas por celda y reportar media ± desvío. Es lo más barato que podés hacer para que el trabajo pase de "un experimento" a "un diseño experimental", y es probablemente lo primero que te va a pedir un tutor de DS.

---

## Cómo pitchearlo a un tutor

Texto que podés mandar prácticamente tal cual:

> Estoy trabajando en un estudio comparativo sobre **cuánto conocimiento del dominio conviene incorporar a un modelo de aprendizaje** cuando los datos son ruidosos y no hay etiquetas disponibles.
>
> Comparo cuatro enfoques sobre el mismo problema de estimación de parámetros: (1) un modelo analítico cerrado, (2) una red con el conocimiento del dominio incorporado en la arquitectura, (3) una red con ese conocimiento incorporado como penalización en la función de pérdida, y (4) una LSTM puramente data-driven. El diseño es factorial sobre nivel de ruido, intensidad de señal y cantidad de sensores disponibles, con repeticiones por semilla.
>
> El caso de aplicación es localización de fugas en ductos, y cuento con un simulador validado que me da **ground truth exacto**, lo que me permite medir sesgo y varianza de cada estimador directamente — algo que rara vez se puede hacer con datos reales.
>
> La parte de simulación física ya está terminada y validada; queda en un apéndice. Lo que necesito discutir con un director es el **diseño experimental, la comparación de modelos y el análisis de identificabilidad**, no la física.

Ese último párrafo es el que desbloquea al tutor: le estás diciendo explícitamente que no necesita saber de fluidos para dirigirte.

**Perfil de tutor a buscar:** alguien de machine learning, aprendizaje profundo, metodología experimental o inferencia estadística. No necesitás a nadie que sepa PINNs.

---

## Sobre tu preocupación de "solo vimos una materia de redes neuronales"

Juega a tu favor con este reencuadre, por tres razones:

1. **La profundidad se corre de la arquitectura al método.** Ya no tenés que defender una arquitectura exótica; tenés que defender un diseño experimental. Eso sí lo viste en la maestría.
2. **La wave-injection se explica en tres líneas y sin física.** Es residual learning: un modelo base (la solución analítica) hace la predicción gruesa y la red aprende solo el error que queda. Es la misma lógica que boosting o que una conexión residual. Cualquier docente de DS lo entiende de inmediato.
3. **Las PINNs dejan de ser el tema y pasan a ser un tratamiento.** No estás escribiendo "una tesis sobre PINNs" — estás escribiendo una tesis sobre comparación de modelos, en la que dos de los cuatro brazos resultan ser PINNs. Eso baja muchísimo la barrera de entrada para el director.

---

## Qué cambia en el `to_do_list.md`

La prioridad se reordena. El punto que antes era "🔴 investigar el fallo catastrófico" deja de ser una crisis y pasa a ser el **experimento 3** (frontera de identificabilidad). Lo que sube a prioridad máxima es:

1. Sacar `pinn_model.py` de `archive/` — es el brazo "física blanda", no código muerto.
2. Correr los cuatro brazos juntos, mismas semillas, misma sesión (experimento 1).
3. Curva de eficiencia de datos del LSTM (experimento 2).
4. Multi-seed en todo.
5. Recién después, la ablation y la corrección del bias de 150 m.

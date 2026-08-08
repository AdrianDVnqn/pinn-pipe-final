# Guía de estudio personal — Tesis MCD

> Documento de uso interno. No es un entregable ni un plan de tareas (eso está en [`to_do_list.md`](./to_do_list.md)). El objetivo acá es **que entiendas de verdad el tema**, en el orden correcto, con cada concepto anclado a una línea concreta de tu propio código.
>
> Está escrito asumiendo que viste una materia de redes neuronales y nada de PINNs ni de problemas inversos. Ninguna de las siete ideas de abajo requiere matemática que no hayas visto en la maestría.

---

## Cómo usar este documento

Las siete ideas están ordenadas por dependencia: cada una se apoya en la anterior. **No saltees**. La tentación va a ser ir directo a la idea 4 (PINNs) porque suena a lo más técnico, pero si no entendés antes las ideas 1 a 3 vas a poder implementar una PINN sin poder defender por qué la usaste — que es exactamente lo que te van a preguntar.

Cada idea tiene tres partes: **qué es**, **dónde está en tu código**, y **qué tenés que poder responder** antes de pasar a la siguiente. Usá esa tercera parte como autoevaluación: si no podés responderla en voz alta sin mirar, todavía no la entendiste.

Al final hay un plan de lectura por semanas y una lista de preguntas de defensa.

---

## Mapa de las siete ideas

```
                    ┌─────────────────────────────┐
                    │ 1. SESGO INDUCTIVO          │  ← la espina dorsal de la tesis
                    │    (cuánto prior meter)     │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
┌─────────────────────────────┐         ┌─────────────────────────────┐
│ 2. PROBLEMA INVERSO         │         │ 7. DISEÑO EXPERIMENTAL      │
│    (de datos a parámetros)  │         │    (cómo se mide todo esto) │
└──────────────┬──────────────┘         └─────────────────────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 3. IDENTIFICABILIDAD        │  ← acá vive tu "fallo" de 1985 m
│    (¿los datos alcanzan?)   │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ 4. QUÉ ES UNA PINN          │
│    (física en el loss)      │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐         ┌─────────────────────────────┐
│ 5. TU WAVE-INJECTION        │◄────────│ 6. LA FÍSICA MÍNIMA         │
│    (física en la arquitect.)│         │    (lo justo para defender) │
└─────────────────────────────┘         └─────────────────────────────┘
```

---

## Idea 1 — Sesgo inductivo

### Qué es

Un modelo entrenado con datos finitos tiene infinitas formas de extrapolar más allá de esos datos. **El sesgo inductivo es el conjunto de supuestos que hacen que el modelo elija una de esas formas y no otra.** Sin ningún supuesto, generalizar es imposible: cualquier función que pase por tus datos es igual de válida.

Ejemplos que ya conocés sin haberlos llamado así:

| Modelo | Su sesgo inductivo |
|---|---|
| Regresión lineal | "La relación es una recta" |
| CNN | "Lo que importa es local y se repite en toda la imagen" (localidad + invarianza traslacional) |
| LSTM | "Hay dependencia temporal secuencial" |
| Tu wave-PINN | "El campo de presión es una onda de D'Alembert propagándose desde un punto, más una corrección suave" |

La clave: **cuanto más fuerte el sesgo, menos datos necesitás — pero peor te va si el sesgo está equivocado.** Eso es literalmente el trade-off sesgo-varianza, aplicado no a la complejidad del modelo sino al conocimiento del dominio que le inyectás.

Y acá está la conexión con tu tesis: tus cuatro métodos son **cuatro puntos en un eje continuo de fuerza del sesgo inductivo**.

```
  SESGO FUERTE                                              SESGO DÉBIL
  Pocos datos                                               Muchos datos
  Frágil si el modelo físico está mal                       Frágil si hay poco dato
       │                                                          │
       ▼                                                          ▼
  ┌─────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │ NPW /   │    │ Wave-PINN    │    │ PINN vanilla │    │ LSTM         │
  │ balance │    │ física en la │    │ física en el │    │ sin física   │
  │ de masa │    │ ARQUITECTURA │    │ LOSS         │    │              │
  └─────────┘    └──────────────┘    └──────────────┘    └──────────────┘
  Fórmula        Restricción         Restricción          Todo aprendido
  cerrada        DURA                BLANDA               de los datos
```

La diferencia entre los dos del medio es sutil pero central en tu tesis, así que asegurate de tenerla clarísima:

- **Física en el loss (PINN vanilla):** la red *podría* violar la física, pero la penalizás si lo hace. Es una sugerencia fuerte.
- **Física en la arquitectura (wave-PINN):** la red *no puede* violar la parte inyectada, porque está hardcodeada en la forma funcional de la salida. Es una imposición.

### Dónde está en tu código

- Sesgo nulo → `archive/legacy_code/baseline_lstm.py`, clase `LeakLSTM`
- Sesgo blando → `archive/legacy_code/pinn_model.py`, función `compute_loss()` con el término `L_fisica`
- Sesgo duro → `wave_pinn.py`, la descomposición `P_total = P_ss + P_mlp_res + P_sing`
- Sesgo total → `baseline_mass_balance.py`, la fórmula `x = (L + a·Δt)/2`

### Qué tenés que poder responder

1. ¿Por qué un modelo sin ningún sesgo inductivo no puede generalizar?
2. En tu problema, ¿cuál de los cuatro métodos esperás que gane si tenés 3 sensores y muchísimo ruido? ¿Y si tuvieras 50 sensores limpios y 10.000 fugas históricas etiquetadas? *(Si podés responder esto, ya tenés la conclusión de tu tesis.)*
3. ¿Cuál es el riesgo de meter un sesgo demasiado fuerte?

---

## Idea 2 — Problema directo vs. problema inverso

### Qué es

- **Problema directo (forward):** conozco los parámetros → calculo qué observaría. *Si la fuga está en 6000 m y pierde 0.015 m³/s, ¿qué mide el sensor?* Eso es tu simulador MOC.
- **Problema inverso:** observo mediciones → infiero los parámetros que las produjeron. *El sensor midió esto, ¿dónde está la fuga?* Eso es tu PINN.

En ciencia de datos esto es familiar aunque no se llame así: el problema inverso es **estimación de parámetros**. Tu `x_leak` y `q_leak` son parámetros latentes que no observás directamente y que querés inferir a partir de observaciones ruidosas.

Lo importante: **los problemas inversos son estructuralmente más difíciles que los directos**, y no por una cuestión de esfuerzo computacional. Un problema bien planteado (Hadamard) requiere tres cosas: que exista solución, que sea única, y que dependa de forma continua de los datos. Los problemas inversos suelen violar la segunda y la tercera.

Violar la tercera es lo que te está pasando: **una perturbación chica en los datos produce un cambio enorme en el parámetro estimado**. Eso es exactamente tu colapso a 1985 m cuando subís el ruido a 50 kPa con fuga chica. No es un bug de implementación, es la naturaleza del problema.

Esta es la razón profunda por la que meter física ayuda: el prior **regulariza** el problema inverso, restringiendo el espacio de soluciones posibles y devolviéndole estabilidad.

> **Conexión clave con la Idea 1:** ahora podés decir, con vocabulario de DS, por qué tu tesis tiene sentido: *el conocimiento físico funciona como regularizador de un problema inverso mal condicionado, y la pregunta es cuánta regularización conviene y dónde inyectarla.* Esa frase sola justifica todo el trabajo.

### Dónde está en tu código

- Directo: `simulator.py` / `moc_simulator.py` / `generate_dataset.py`
- Inverso: `wave_pinn.py`, donde `x_leak` y `q_leak` son `nn.Parameter` optimizables junto con los pesos

### Qué tenés que poder responder

1. ¿Por qué el problema inverso es más difícil que el directo, más allá del costo computacional?
2. ¿Cuál de las tres condiciones de Hadamard viola tu problema, y qué evidencia empírica tenés de eso en tus propios resultados?
3. ¿En qué sentido la física actúa como regularizador?

---

## Idea 3 — Identificabilidad

### Qué es

Un parámetro es **identificable** si valores distintos del parámetro producen observaciones distinguibles. Si dos valores muy diferentes de `x_leak` producen esencialmente la misma señal en tus sensores, ningún método —por sofisticado que sea— va a poder distinguirlos. **No es un problema del modelo, es un problema de los datos.**

Se distingue entre:

- **Identificabilidad estructural:** en un mundo ideal sin ruido y con observación continua, ¿es único el parámetro? Esto se responde con matemática, antes de mirar un solo dato.
- **Identificabilidad práctica:** con *este* nivel de ruido, *esta* cantidad de sensores y *esta* duración de registro, ¿puedo distinguirlo? Esto se responde empíricamente.

Tu problema es estructuralmente identificable (por eso funciona sin ruido) pero **prácticamente no identificable** en la esquina del espacio de operación donde la fuga es chica y el ruido alto. Ese es exactamente tu Experimento 3.

### 🔍 Mirá esto ahora: ya tenés la evidencia y no la estás usando

Abrí `results/loss_landscape.csv`. Es un barrido de la pérdida en función de `x_leak`:

| x_leak | loss |
|---|---|
| 1000 | 0.002894 |
| 2000 | 0.002804 |
| 3000 | 0.011310 |
| 3500 | 0.086576 |
| 4000 | 0.002889 |
| 5000 | 0.002824 |
| 6000 | 0.002876 |
| 7000 | 0.002903 |
| 8000 | 0.003063 |
| 9000 | 0.002875 |

**La pérdida es prácticamente plana** (~0.0028-0.0031) en todo el dominio, salvo un pico raro en 3000-3500. No hay un mínimo pronunciado en ningún lado. Y la columna `q_leak_pred` vale ~0.01503 en *todas* las posiciones.

Eso es la firma visual de la no identificabilidad: **el paisaje de pérdida no distingue dónde está la fuga.** Si la superficie es plana, el optimizador no tiene hacia dónde bajar, y termina donde lo dejó la inicialización. Toda tu saga de "mínimos locales" documentada en `todo.md` y `avances.md` probablemente sea, en el fondo, esto.

Dos advertencias antes de que lo uses:

1. Ese CSV es del 12 de junio, o sea del modelo viejo (`pinn_model.py`), no del wave-PINN actual. **Regenerálo con el modelo actual** antes de sacar conclusiones.
2. Averiguá qué escenario se usó y por qué hay un pico en 3000-3500 — puede ser un artefacto de la posición de un sensor, y si es así es un hallazgo en sí mismo.

Si al regenerarlo el paisaje sigue plano, ese gráfico es **la figura más importante de tu tesis**: explica en una imagen por qué el problema es difícil, por qué hacen falta priors, y por qué tus métodos fallan donde fallan.

### Dónde está en tu código

- `scan_loss_landscape.py` — la herramienta, ya escrita
- `results/loss_landscape.csv` — la evidencia, desactualizada
- `figs/snr_heatmap.png` — infraestructura para el mapa de calor del Experimento 3

### Qué tenés que poder responder

1. Diferencia entre identificabilidad estructural y práctica, con un ejemplo de cada una en tu problema.
2. Si el paisaje de pérdida es plano, ¿tiene sentido culpar al optimizador? ¿Por qué no?
3. ¿Qué tres cosas podrías cambiar para volver identificable un caso que no lo es? *(Pista: más sensores, menos ruido, más prior. Notá que las tres son ejes de tu diseño experimental.)*

---

## Idea 4 — Qué es una PINN, mínimo viable

### Qué es

Una PINN es, estructuralmente, algo mucho más simple de lo que suena:

**Una red densa que toma coordenadas `(x, t)` y devuelve el campo físico `(P, Q)` en ese punto.** Eso es todo. No procesa series temporales, no tiene convoluciones. Input de dimensión 2, output de dimensión 2. Tu red es `input(2) → [64]×5 → output(2)` con `tanh`.

Lo único no trivial es **cómo se entrena**. La pérdida tiene dos clases de términos:

```
L = L_datos  +  λ · L_física
     │              │
     │              └── ¿la red satisface la EDP en puntos donde NO tengo datos?
     └───────────────── ¿la red reproduce lo que midieron los sensores?
```

El truco central, y lo único realmente ingenioso del método, es cómo se calcula `L_física`: como la red es una función diferenciable de `(x,t)`, **podés obtener `∂P/∂t` y `∂P/∂x` exactamente con autograd**, sin malla y sin diferencias finitas. Entonces evaluás el residuo de la EDP en cualquier punto del dominio que se te ocurra (los *collocation points*, que sorteás al azar) y penalizás que no dé cero.

Eso es lo que hace que la física entre al modelo: no le enseñás la ecuación con ejemplos, la **imponés** vía autodiff.

Para el **problema inverso** hay un paso más, y es el que te importa: declarás `x_leak` y `q_leak` como parámetros entrenables más, y el optimizador los ajusta junto con los pesos de la red. Los parámetros físicos y los pesos bajan por el mismo gradiente.

### Por qué las PINNs son difíciles de entrenar

Esto tenés que entenderlo bien, porque justifica todo tu curriculum de tres fases:

1. **Los términos del loss compiten.** `L_datos` y `L_física` pueden tener gradientes de órdenes de magnitud completamente distintos. Tu propio log lo muestra descarnadamente: `Grad L_data (9.964e+01) >> Grad L_pde (1.166e-03)` — cinco órdenes de diferencia. El término físico simplemente no se ve.
2. **Optimizar parámetros físicos junto con los pesos genera co-adaptación.** La red puede "tapar" un `q_leak` equivocado deformando el campo. Vos ya diagnosticaste esto textualmente en `todo.md`: *"la MLP y los parámetros compensan sus errores mutuamente"*.
3. **Las discontinuidades son veneno.** Una red con `tanh` es infinitamente suave; un frente de onda no lo es. Aproximar un salto con funciones suaves produce oscilaciones (fenómeno de Gibbs).

Los tres problemas están documentados en la literatura ([18], [19], [20] de tu marco teórico) y **los tres los resolviste de forma independiente antes de leerlos**. Cuando escribas la tesis, conectá explícitamente cada decisión tuya con el paper que la respalda: eso convierte "lo probé y anduvo" en "apliqué una técnica fundamentada".

### 🛠 Ejercicio práctico recomendado (medio día, alto retorno)

Antes de tocar tu código, implementá desde cero una PINN de 30 líneas para la ecuación de calor 1D:

```
∂u/∂t = α ∂²u/∂x²,  u(x,0) = sin(πx),  u(0,t) = u(1,t) = 0
```

Tiene solución analítica (`u = sin(πx)·e^(-απ²t)`), así que podés verificar exactamente. Hacelo sin copiar código: escribí vos el `torch.autograd.grad` anidado para sacar la segunda derivada. Cuando eso te funcione, vas a entender tu propio `wave_pinn.py` de otra manera — y vas a poder explicarlo en una defensa sin titubear.

### Dónde está en tu código

- `wave_pinn.py` líneas ~201-217: `r_cont` y `r_mom`, los residuos de las dos EDPs
- `wave_pinn.py` líneas ~104-110: `x_leak` y `q_leak` como parámetros aprendibles
- `wave_pinn.py` `train()` línea ~274: el curriculum de tres fases
- `archive/legacy_code/pinn_model.py` `compute_loss()`: la versión con física blanda

### Qué tenés que poder responder

1. ¿Por qué una PINN no necesita malla?
2. ¿Qué son los collocation points y por qué se resortean en cada época con Adam pero se congelan para L-BFGS? *(Está en tu `avances.md` — releelo, la respuesta que escribiste es correcta.)*
3. ¿Por qué el desbalance de gradientes entre `L_datos` y `L_física` rompe el entrenamiento?

---

## Idea 5 — Tu wave-injection, en idioma de machine learning

### Qué es

Tu arquitectura descompone el campo así:

```
P_total(x,t)  =  P_ss(x)  +  P_sing(x,t)  +  P_mlp_res(x,t)
                    │            │               │
                    │            │               └── lo aprende la red
                    │            └── solución analítica de la onda (D'Alembert)
                    └── estado estacionario con fricción, analítico
```

**Traducción a DS: esto es residual learning.** Tenés un *base learner* analítico que hace la predicción gruesa, y una red que aprende solamente el residuo, o sea el error que el modelo base no captura. Es la misma lógica que boosting (un modelo débil corrige lo que dejó el anterior) o que una skip connection en una ResNet.

Explicado así no necesitás mencionar D'Alembert ni golpe de ariete para que un docente de DS entienda tu aporte en veinte segundos. **Usá esta formulación en la primera reunión con tu tutor.**

Por qué ayuda: la red ya no tiene que aprender la discontinuidad (que es donde sufría, por Gibbs), solo correcciones suaves — que es exactamente para lo que las redes con `tanh` son buenas. Le sacaste de encima la parte del problema para la que era mala.

### ⚠️ La hipótesis que puede unificar toda tu tesis

Acá va algo que creo que es el insight central que todavía no estás explotando, y que vale la pena que verifiques.

**El bias sistemático de ~150 m puede no ser un bug: puede ser el precio del sesgo inductivo.**

El razonamiento: al inyectar `P_sing` estás imponiendo la forma analítica exacta de la onda. Pero tus datos vienen del MOC, que tiene dispersión numérica — la onda simulada **no es idénticamente** la onda analítica. Si el prior no coincide perfectamente con el proceso generador, el estimador queda sesgado. Y un sesgo por especificación incorrecta del prior es, por construcción, **sistemático y aproximadamente constante**, no aleatorio.

Mirá la evidencia que ya tenés en `benchmark_v2_results.csv`: el error va de 146 a 175 m en las 36 corridas, con ruido variando de 0 a 50.000 Pa. **Un error que no cambia cuando el ruido cambia tres órdenes de magnitud no es error de estimación — es sesgo del modelo.**

Cómo testearlo, y es barato porque el Experimento 1 ya te da los datos:

| Predicción | Si la hipótesis es cierta |
|---|---|
| PINN vanilla (física blanda, sin `P_sing`) | **No** debería tener el bias de 150 m, pero sí mucha más varianza |
| Wave-PINN | Bias constante alto, varianza baja |

Si eso se confirma, tenés una demostración empírica limpísima del trade-off sesgo-varianza **en el eje del conocimiento del dominio**, medida en tu propio experimento. Eso ya no es una tesis de ingeniería con redes neuronales: es una tesis de ciencia de datos que usa un caso de ingeniería. Es exactamente el reencuadre que necesitás, y saldría de tus datos, no de una narrativa impuesta.

### Qué tenés que poder responder

1. Explicá la wave-injection en 30 segundos, sin decir "D'Alembert" ni "golpe de ariete".
2. ¿Qué pasa si la solución analítica que inyectás está equivocada?
3. ¿Cómo distinguirías empíricamente un sesgo del modelo de un error de optimización?

---

## Idea 6 — La física mínima que necesitás dominar

No necesitás ser ingeniero hidráulico. Necesitás poder defender **estas cinco cosas** y nada más. Todo lo demás va al apéndice.

**1. Qué describen las dos ecuaciones.** Son conservación de masa y conservación de cantidad de movimiento para flujo 1D en una tubería:

$$\frac{\partial P}{\partial t} + \frac{\rho a^2}{A}\frac{\partial Q}{\partial x} = 0 \qquad \frac{\partial Q}{\partial t} + \frac{A}{\rho}\frac{\partial P}{\partial x} + \frac{fQ|Q|}{2DA} = 0$$

En castellano: si entra más caudal del que sale, sube la presión (primera); el fluido se acelera por gradiente de presión y se frena por fricción (segunda). El término `fQ|Q|` es fricción de Darcy-Weisbach — es el único término no lineal, y el `|Q|` está para que la fricción siempre se oponga al movimiento sin importar la dirección.

**2. Qué es la velocidad de onda `a = 1200 m/s`.** Es la velocidad a la que viaja una perturbación de presión por el fluido. Mucho más rápida que el fluido en sí (que va a ~1 m/s). Por eso la detección es posible: la fuga se "avisa" casi instantáneamente a los extremos.

**3. Por qué una fuga genera una onda de presión negativa.** Al abrirse la fuga, se pierde masa localmente, cae la presión en ese punto, y esa caída se propaga en ambas direcciones a velocidad `a`. Midiendo *cuándo* llega la caída a cada extremo, triangulás la posición: esa es la fórmula de tu baseline NPW, `x = (L + a·Δt)/2`.

**4. Qué es el MOC en una línea.** Las dos EDPs acopladas se convierten en ecuaciones diferenciales *ordinarias* si te movés a lo largo de las curvas `dx/dt = ±a` (las características). Con `dt = dx/a` (CFL = 1 exacto) la integración es exacta sobre esas líneas, lo que hace al método muy preciso para propagación de ondas.

**5. Qué estás despreciando** — esto es lo que más te van a preguntar, porque define el alcance:

| Simplificación | Consecuencia |
|---|---|
| Fricción cuasi-estacionaria (Darcy-Weisbach) | Subestima la atenuación de transitorios rápidos ([6] trata esto) |
| Flujo 1D | Ignora efectos radiales; válido porque L/D es enorme |
| Fluido isotermo, monofásico | Sin efectos térmicos ni gas disuelto |
| Tubería recta, sin accesorios | Sin reflexiones por codos, válvulas o cambios de sección |
| Velocidad de onda constante | `a` en realidad depende de la presión y del contenido de gas |

Tenerlas listas y admitirlas sin que te las saquen es señal de dominio, no de debilidad.

### Qué tenés que poder responder

1. ¿Por qué una fuga se detecta en los extremos casi instantáneamente si el crudo se mueve a 1 m/s?
2. ¿Qué gana el MOC frente a diferencias finitas para este problema?
3. Nombrá tres supuestos de tu modelo físico y qué pasaría si no se cumplieran.

---

## Idea 7 — Diseño experimental y estadística

Esta es la parte que un jurado de MCD va a mirar con más atención, y es hoy la más floja del trabajo.

**Diseño factorial.** Estás cruzando posición × tamaño × ruido × sensores × método. Cada combinación es una celda. El punto del diseño factorial es poder atribuir la variación del resultado a cada factor y detectar interacciones (por ejemplo: *el efecto del ruido depende del tamaño de la fuga* — que es justo lo que te está pasando).

**Por qué las semillas son innegociables.** Hoy tenés una corrida por celda. Con una sola muestra no podés distinguir "el método A es mejor que el B" de "esta vez salió mejor". Con 3-5 semillas reportás media ± desvío y podés afirmar algo. Es lo más barato que podés hacer para que el trabajo pase de anecdótico a experimental.

**Descomposición sesgo-varianza del estimador.** Con múltiples semillas por celda podés separar:

- **Sesgo:** ¿la media de las predicciones da lejos del valor real? *(tus 150 m)*
- **Varianza:** ¿cuánto se dispersan las predicciones entre semillas?

Y ahí es donde cierra todo: **esa descomposición aplicada a los cuatro brazos es, literalmente, la tesis.** Es la Idea 1 vuelta un número.

**Elección de métricas.** Reportá mediana y percentiles además de la media. Tus fallos catastróficos (errores de 4000-6000 m) destrozan cualquier promedio, y una media de 800 m no describe ni el caso típico ni el caso malo. Mostrar la distribución completa es más honesto y más informativo.

**Cuidado con la comparación desigual.** Ya lo mencioné pero repito porque es fácil de olvidar al escribir: el baseline de balance de masa tiene caudalímetros en ambos extremos, así que `q_leak = Q_in − Q_out` le sale casi gratis. Las PINNs infieren `q_leak` solo desde presión. **No es la misma tarea.** Decilo explícitamente en la tabla de resultados o parecerá que ocultás algo.

### Qué tenés que poder responder

1. ¿Por qué una sola corrida por celda no permite comparar dos métodos?
2. ¿Cómo separarías sesgo de varianza en tus resultados?
3. ¿Por qué la media es mala métrica cuando hay fallos catastróficos?

---

## Plan de lectura

Tiempos estimados para lectura con notas, no lectura en diagonal.

### Semana 1 — El eje conceptual nuevo *(prioridad absoluta)*

Es la literatura que le habla a tu tutor de DS y que hoy no está en tu marco teórico.

| Lectura | Tiempo | Qué extraer |
|---|---|---|
| **von Rueden et al., "Informed Machine Learning"** (arXiv:1903.12394 / IEEE TKDE) | 3-4 h | La taxonomía de *dónde* se inyecta el conocimiento. Tus cuatro brazos son cuatro casillas de esa taxonomía — identificá exactamente cuáles y citalas. **Es el paper que estructura tu tesis.** |
| **Karpatne et al., "Theory-Guided Data Science"** (IEEE TKDE 29(10), 2017) | 2 h | El paradigma híbrido teoría+datos y su vocabulario. Útil para la introducción. |
| **Willard et al., "Integrating Physics-Based Modeling with ML: A Survey"** (arXiv:2003.04919) | 2 h, salteando | Taxonomía complementaria. Leé la intro y la tabla de clasificación; el resto es catálogo. |

### Semana 2 — PINNs, lo esencial

| Lectura | Tiempo | Qué extraer |
|---|---|---|
| **[13] Raissi et al. 2019** (ya lo tenés en PDF) | 4 h | Secciones 3-4. El framework y **especialmente** la formulación del problema *inverso*, que es la tuya. |
| **[15] Karniadakis et al. 2021** (Nature Rev. Physics) | 2 h | Panorama general. Es el puente entre "informed ML" y PINNs. |
| **Ejercicio: PINN de calor 1D desde cero** | 4 h | Vale más que las tres lecturas juntas para *entender*. No lo saltees. |

### Semana 3 — Por qué las PINNs fallan *(justifica tus decisiones de diseño)*

| Lectura | Tiempo | Qué extraer |
|---|---|---|
| **[18] Wang, Teng & Perdikaris 2021** | 3 h | Patologías del flujo de gradientes. Es la justificación teórica de tu curriculum de 3 fases. |
| **[19] Wang, Yu & Perdikaris 2022** (NTK) | 3 h | Por qué fallan. Denso; quedate con las conclusiones. |
| **[20] Krishnapriyan et al. 2021** | 2 h | Modos de falla. Alimenta directo tu capítulo de identificabilidad. |

### Semana 4 — Posicionamiento y baselines

| Lectura | Tiempo | Qué extraer |
|---|---|---|
| **[23] Tariq et al. 2023 (SPE)** | 3 h | Tu competencia directa. ¿Qué hacen distinto? ¿Asumen `q_leak` conocido? |
| **[7] Shamloo & Haghighi 2009**, **[8] Liggett & Chen 1994** | 3 h | El mismo problema inverso con optimización clásica. Tu punto de comparación histórico. |
| **[22] Cai et al. 2021** (review PINNs+fluidos) | 2 h | Estado del arte del nicho. |

### Después (baja prioridad)

Las refs [1][2][3] (surveys de detección de fugas) y [9][10][11][12] (baselines) se leen al final, cuando escribas introducción y justificación de baselines. No aportan a tu comprensión del núcleo.

---

## Preguntas de defensa — autoevaluación

Marcá honestamente cuáles podés responder hoy, en voz alta, sin mirar nada. Las que queden vacías son tu plan de estudio.

### Sobre el encuadre
- [ ] ¿Por qué esto es una tesis de ciencia de datos y no de ingeniería hidráulica?
- [ ] ¿Cuál es tu pregunta de investigación en una sola oración?
- [ ] ¿Qué aprende alguien de tu tesis que **no** trabaje con oleoductos?

### Sobre el método
- [ ] ¿Por qué una PINN y no una red convencional con más datos?
- [ ] ¿Qué diferencia hay entre poner la física en el loss y ponerla en la arquitectura?
- [ ] Explicá tu wave-injection sin usar vocabulario de física.
- [ ] ¿Por qué entrenás en tres fases y no todo junto?

### Sobre los resultados *(las incómodas — preparalas especialmente)*
- [ ] Tu baseline de 1996 le gana a tu red neuronal con ruido bajo. ¿Por qué tiene sentido tu trabajo igual?
- [ ] Tenés un error sistemático de 150 m. ¿Por qué no lo corregís restándolo?
- [ ] ¿Por qué el modelo predice ~1985 m sin importar dónde esté la fuga, en algunos casos?
- [ ] ¿Cómo sabés que tus resultados no son producto de una semilla afortunada?
- [ ] Tu baseline tiene caudalímetros y tu PINN no. ¿Es justa la comparación?

### Sobre las limitaciones
- [ ] Validaste solo con datos de tu propio simulador. ¿Qué vale entonces tu conclusión?
- [ ] ¿Qué pasaría con una tubería con codos, o con flujo bifásico?
- [ ] ¿En qué condiciones **no** recomendarías tu método?

> Las de la sección "resultados" son las que deciden tu nota. Un jurado que ve que las anticipaste y las tenés respondidas concluye que dominás el tema. Uno que ve que te sorprenden concluye lo contrario. **Escribí las respuestas antes de la defensa, literalmente, en un documento aparte.**

---

## Glosario de traducción

Para cuando tengas que hablar con alguien de un lado o del otro.

| Física / CFD | Ciencia de datos |
|---|---|
| Simulador MOC | Proceso generador de datos con ground truth |
| Problema inverso de la EDP | Estimación de parámetros latentes |
| `x_leak`, `q_leak` | Parámetros a estimar; se analiza sesgo y varianza del estimador |
| Inyección de solución analítica | Sesgo inductivo estructural / residual learning |
| Residuo de la EDP en el loss | Regularización basada en conocimiento del dominio |
| Collocation points | Puntos de evaluación de la restricción física |
| Ruido del sensor (Pa) | Degradación controlada de SNR |
| Cantidad de sensores | Presupuesto de adquisición / dimensionalidad de la observación |
| Fuga no detectable | Parámetro prácticamente no identificable |
| Curriculum de 3 fases | Estrategia de optimización multi-objetivo con gradientes en conflicto |
| Fenómeno de Gibbs | Sesgo de suavidad del aproximador frente a discontinuidades |
| Estado estacionario | Modelo base sobre el que se aprende el residuo |

---

## Si solo tuvieras una semana

En orden: leé von Rueden (te da el esqueleto conceptual y el vocabulario), hacé el ejercicio de la PINN de calor 1D (te da la intuición mecánica), y regenerá `loss_landscape.csv` con el modelo actual (te dice si tu mejor figura existe o no). Con esas tres cosas podés sentarte con un tutor y sostener una conversación de una hora sin quedar expuesto.

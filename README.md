# Simulador MOC de Oleoducto

El objetivo de este proyecto es desarrollar y evaluar una **Physics-Informed Neural Network (PINN)** capaz de detectar, localizar y cuantificar fugas en un oleoducto utilizando únicamente mediciones transitorias de presión y caudal, resolviendo el problema inverso de las ecuaciones de Navier-Stokes unidimensionales.

## Configuración de sensores

Basada en instrumentación industrial real para oleoductos:

| Tipo | Posiciones | Cantidad | Justificación |
|---|---|---|---|
| Transmisor de presión | 1000, 5000, 9000 m | 2–3 | Económico, instalable en campo |
| Caudalímetro | 0, 10000 m | 2 (fijo) | Caro, solo en transferencia de custodia |

Los sensores de presión varían entre 2 y 3 en el experimento factorial. Los caudalímetros en extremos son siempre 2 y no forman parte del eje experimental — son una constante del sistema.

## Idea conceptual

El modelo representa un ducto de longitud `L = 10000 m`, con un fluido compresible equivalente al petróleo, y resuelve la evolución temporal de:

- `P(x,t)`: presión
- `Q(x,t)`: caudal volumétrico

### 1. Estado estacionario

Antes de simular la dinámica transitoria, el sistema se inicializa con la solución estacionaria analítica:

- el caudal es uniforme en todo el ducto,
- la presión cae linealmente por fricción Darcy-Weisbach.

Eso sirve como condición inicial física y evita transitorios artificiales al comenzar en `t = 0`.

### 2. Simulación transitoria con MOC

El núcleo del simulador usa las ecuaciones de compatibilidad del MOC:

- `C+ : P + B Q = constante`
- `C- : P - B Q = constante`

con fricción incluida en el término característico. En cada paso temporal se actualizan:

- nodos interiores,
- condición de contorno upstream con presión fija,
- condición de contorno downstream con caudal fijo,
- nodo de fuga cuando se activa.

### 3. Fuga

La fuga se activa en un tiempo configurable `t_leak` y en una posición `x_leak`. Cuando se activa, el nodo divide el caudal en dos ramas:

- caudal que llega desde la izquierda,
- caudal que sigue hacia la derecha,
- caudal perdido por la fuga.

Esto produce una perturbación que viaja a velocidad aproximadamente igual a la celeridad `a`.

## Archivos principales

- `simulator.py`: script principal con el estado estacionario, el MOC, la extracción de sensores y los 4 tests.

## Qué valida el script

El archivo principal corre automáticamente cuatro chequeos:

1. **Estado estacionario estable**: sin fuga, el perfil no debería moverse.
2. **Velocidad de onda correcta**: la perturbación debe llegar a los extremos en los tiempos esperados.
3. **Balance de masa**: sin fuga se conserva el caudal; con fuga aparece la diferencia esperada.
4. **Señal vs ruido**: la fuga debe seguir siendo visible con ruido agregado.

## Requisitos

- Python 3.10 o superior
- `numpy`
- `matplotlib`

## Modo de uso

### Ejecutar todo el simulador

```bash
python simulator.py
```

Eso ejecuta los 4 tests y abre los gráficos.

### Usarlo como módulo

```python
import simulator as sim

result = sim.run_moc(Q_leak=0.015, x_leak=6000, t_leak=50)
sensores = sim.get_sensor_data(result, [0, 5000, 10000])
```

### Variables útiles

- `Q_leak`: caudal de fuga
- `x_leak`: posición de la fuga
- `t_leak`: instante de activación
- `noise_std`: ruido gaussiano en presión

## Salidas de `run_moc`

El diccionario devuelto contiene:

- `t`: vector de tiempos
- `x`: vector espacial
- `P`: presión `P(x,t)`
- `Q`: caudal `Q(x,t)`
- `P_ss`, `Q_ss`: estado estacionario
- `dP`, `dQ`: desviaciones respecto del estacionario

## Notas técnicas

- El loop temporal del MOC usa solo NumPy.
- La interpolación de sensores también se hace sin SciPy.
- El script arranca desde el estacionario para que la dinámica observada provenga solo de la fuga.

## Siguiente paso natural

Cuando este simulador esté listo, el siguiente paso es usarlo para generar datasets sintéticos con múltiples escenarios de fuga y después entrenar la PINN.

---

## Estructura del proyecto

```text
├── moc_simulator.py      # Simulador MOC: run_moc() y get_sensor_data()
├── generate_dataset.py   # Generación del dataset sintético
├── dataset.h5            # Dataset generado (13 escenarios)
├── figs/                 # Figuras de análisis exploratorio
│   ├── overview_leak_sizes.png
│   ├── overview_leak_positions.png
│   ├── snr_heatmap.png
│   └── wave_arrival_times.png
└── README.md

## Modelo PINN

Arquitectura:
- Red fully-connected: input(2) → [64]×5 → output(2)
- Input: (x, t) normalizados a [0,1]
- Output: P(x,t) y Q(x,t) en unidades físicas
- Activación: tanh
- Parámetros entrenables adicionales: x_leak, q_leak

Función de pérdida:
- L_datos:     MSE entre P predicha y mediciones de sensores
- L_física:    residuo de las EDPs de flujo con término de fuga
- L_contorno:  condiciones de borde (P_in, Q_out)
- L_inicial:   condición inicial (estado estacionario)

Checkpoints:
- Los puntos de control del entrenamiento se guardan en el directorio `checkpoints/`.

```

## Dataset

- **13 escenarios** generados con el simulador MOC
- **12 escenarios con fuga**: grid factorial de 4 posiciones × 3 tamaños
- **1 escenario sin fuga**: baseline para detección binaria
- Datos guardados sin ruido — el ruido se agrega en entrenamiento
- Sensores en `x = {1000, 5000, 9000} m`

| Variable | Valores |
|---|---|
| Posiciones de fuga | 2000, 4000, 6000, 8000 m |
| Tamaños de fuga | 0.005, 0.015, 0.030 m³/s |
| Tiempo de inicio | 50 s en todos los casos |
| Duración simulación | 200 s |

## Próximos pasos

- [ ] Implementar PINN para caso base (sin ruido, 3 sensores)
- [ ] Implementar baselines (balance de masa, gradiente de presión, LSTM)
- [ ] Experimento factorial: ruido × cantidad de sensores

## Archivos nuevos

- `config.py`: parámetros globales del experimento y rutas
- `data_utils.py`: carga del dataset y aplicación de ruido en ejecución
- `verify_data_utils.py`: verificación de carga, ruido y subconjuntos

## Experimento factorial

El aporte diferencial de la tesis es el análisis sistemático de robustez bajo dos ejes:

| Variable | Valores |
|---|---|
| Nivel de ruido | trivial (500 Pa), fácil (2000 Pa), moderado (8000 Pa), difícil (25000 Pa), muy difícil (50000 Pa) |
| Cantidad de sensores | 2, 3 |

Esto genera 5 × 2 = 10 condiciones experimentales por escenario.
El ruido se aplica en tiempo de ejecución, no se almacena en el dataset, y usa semilla fija (`seed=42`) para garantizar reproducibilidad.

## Baselines

### Baseline 1: SCADA Clásico (Mass Balance + NPW)

Representa el estado del arte actual en la industria sin uso de Machine Learning.
* **Detección**: Balance de masa tradicional midiendo la diferencia entre el caudalímetro de entrada y el de salida. Si $\Delta Q$ supera un umbral de $3\sigma$, se declara alarma de fuga.
* **Localización**: Método de la Onda de Presión Negativa (NPW). Cuando ocurre la ruptura, se propaga una onda de descompresión a la velocidad del sonido en ambas direcciones. Midiendo el $T_{arrival}$ en los sensores de presión, se triangula la posición usando $x = \frac{L + a \cdot \Delta t}{2}$.

Resultados guardados en: `results/baseline_mass_balance.csv`

```bash
python baseline_mass_balance.py
```

### Baseline 2: Gradiente de Presión

Método basado en régimen cuasi-estacionario.
Trabaja en dominio espacial (no temporal como NPW).

Principio: una fuga crea un quiebre en el perfil lineal
de presión. Upstream el gradiente es mayor (más flujo),
downstream menor (menos flujo).

* **Detección**: cambio relativo entre gradientes de segmentos adyacentes > 15% → fuga detectada
* **Localización**: intersección de las dos líneas de gradiente (upstream desde x=0, downstream desde x=L) estimadas a partir de los caudales medidos en los extremos.

Latencia de detección: ~120s (necesita régimen estacionario) vs. ~55s del NPW. Trade-off: más robusto a transitorios.

Resultados guardados en: `results/baseline_pressure_gradient.csv`

```bash
python baseline_pressure_gradient.py
```

### Baseline 3: LSTM Puro (sin física)

Red neuronal recurrente entrenada sobre series temporales
de sensores. No incorpora ecuaciones de flujo.

Arquitectura:
  - LSTM bidireccional: 2 capas, hidden_size=128
  - Input: 5 canales (3 dP + 2 dQ) × 2401 timesteps
  - Output: x_leak_norm, q_leak_norm, has_leak (prob)

Dataset de entrenamiento:
  - 3440 samples generados con MOC + augmentación de ruido
  - Test: los 12 escenarios estándar de dataset.h5

Ventaja sobre PINN: inferencia en ~5ms (vs ~14 min de PINN)
Desventaja sobre PINN: requiere dataset de entrenamiento supervisado; no aprovecha conocimiento físico del sistema.

Resultados guardados en: `results/baseline_lstm.csv`

## Estructura del proyecto

```text
├── config.py                 # Parámetros globales del experimento
├── data_utils.py             # Carga de escenarios y aplicación de ruido
├── verify_data_utils.py      # Verificación de utilidades de datos
├── moc_simulator.py          # Simulador MOC: run_moc() y get_sensor_data()
├── generate_dataset.py       # Generación del dataset sintético base
├── generate_lstm_dataset.py  # Generación del dataset extendido para LSTM
├── pinn_model.py             # Red PINN para problema inverso
├── baseline_mass_balance.py  # Baseline 1: Balance de Masa + NPW
├── baseline_pressure_gradient.py # Baseline 2: Gradiente de Presión
├── baseline_lstm.py          # Baseline 3: LSTM Supervisado
├── dataset.h5                # Dataset base generado (13 escenarios)
├── lstm_dataset.h5           # Dataset extendido para LSTM
├── results/                  # Resultados de evaluación (CSV)
│   ├── baseline_mass_balance.csv
│   ├── baseline_pressure_gradient.csv
│   └── baseline_lstm.csv
├── figs/                     # Figuras de análisis
│   ├── overview_leak_sizes.png
│   ├── overview_leak_positions.png
│   ├── snr_heatmap.png
│   ├── wave_arrival_times.png
│   ├── mb_detection_example.png
│   ├── mb_error_vs_noise.png
│   └── mb_detection_rate.png
└── README.md
```
## Experimento factorial comparativo

### Ejecución

`ash
# Paso 1: correr experimento completo (overnight)
python run_factorial_experiment.py

# Paso 2: generar figuras de la tesis
python generate_thesis_figures.py

# Para saltar corridas ya completadas (resume):
python run_factorial_experiment.py --skip-pinn
`

### Resultados

`	ext
results/
├── baseline_mass_balance.csv
├── baseline_pressure_gradient.csv
├── baseline_lstm.csv
├── pinn_factorial.csv
├── master_results.csv         ← todos los métodos combinados
└── aggregate_metrics.csv      ← métricas por método × ruido × sensores

figs/
├── thesis_main_comparison.png  ← figura principal de la tesis
├── thesis_error_heatmap.png
├── thesis_by_leak_size.png
├── thesis_sensor_impact.png
└── thesis_summary_table.png   ← para presentación de defensa
`

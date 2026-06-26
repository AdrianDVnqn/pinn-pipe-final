# Detección de Fugas en Oleoductos con Wave-Injection PINNs

Proyecto final de la Maestría en Ciencia de Datos. El objetivo es desarrollar y evaluar una **Physics-Informed Neural Network (PINN)** capaz de detectar, localizar y cuantificar fugas en un oleoducto, resolviendo el problema inverso de las ecuaciones de flujo transitorio 1D utilizando únicamente mediciones de sensores de presión y caudal.

## Idea conceptual

El modelo representa un ducto de longitud `L = 10000 m` con un fluido compresible (crudo), y resuelve la evolución temporal de:

- `P(x,t)`: presión
- `Q(x,t)`: caudal volumétrico

Se genera una fuga transitoria a partir de `t = 50s` y la PINN infiere simultáneamente la **posición** (`x_leak`) y el **caudal** (`q_leak`) de la fuga a partir de datos ruidosos de sensores.

---

## Arquitectura: WaveLeakPINN

La arquitectura clásica de PINNs sufre al modelar las discontinuidades abruptas generadas por una fuga transitoria (fenómeno de Gibbs). Para solucionar esto, implementamos una **Inyección de Solución Analítica**:

$$P_{total}(x,t) = P_{ss}(x) + P_{mlp\_res}(x,t) + P_{sing}(x,t)$$

1. **Estado Estacionario** ($P_{ss}$): Gradiente de presión lineal inicial con fricción Darcy-Weisbach, sin fuga.
2. **Solución Singular** ($P_{sing}$): Solución analítica exacta de la ecuación de onda (tipo D'Alembert) que representa un frente de onda propagándose desde $x_{leak}$ a velocidad $a$. Tanto $x_{leak}$ como $q_{leak}$ son **parámetros físicos aprendibles**.
3. **Residual de Red** ($P_{mlp\_res}$): Red fully-connected `input(2) → [64]×5 → output(2)` con activación `tanh`. Solo aprende lo que falta: reflexiones en bordes y curvaturas por fricción no lineal.

### Entrenamiento: Curriculum Learning de 3 Fases

| Fase | Epochs | Objetivo | Mecanismo |
|------|--------|----------|-----------|
| **1. Alineación** | 0–40% | Inferir `q_leak` y acercar `x_leak` al vecindario correcto | Solo Data Loss. Onda suave ($k$: 0.5→50) para gradientes no nulos. Parámetros físicos libres, MLP congelado. |
| **2. Residual** | 40–70% | MLP modela dinámica PDE residual | Parámetros físicos congelados. Se introduce gradualmente el peso PDE Loss. |
| **3. Ajuste fino** | 70–100% | Relajación conjunta del sistema | Todo libre con LR reducido para parámetros físicos. |

---

## Simulador MOC

El núcleo del generador de datos usa el **Método de las Características (MOC)** para resolver las ecuaciones de flujo transitorio:

- `C+`: $P + BQ = \text{const}$ a lo largo de $dx/dt = +a$
- `C-`: $P - BQ = \text{const}$ a lo largo de $dx/dt = -a$

donde $B = \rho a / A$ es la impedancia característica.

El simulador inicializa desde el estado estacionario analítico y activa la fuga en un nodo configurable, dividiendo el caudal en rama izquierda, derecha y pérdida.

### Configuración de sensores

Basada en instrumentación industrial real:

| Tipo | Posiciones | Cantidad | Justificación |
|------|------------|----------|---------------|
| Transmisor de presión | 1000, 5000, 9000 m | 2–3 | Económico, instalable en campo |
| Caudalímetro | 0, 10000 m | 2 (fijo) | Caro, solo en transferencia de custodia |

---

## Benchmark

El script `benchmark_wave_pinn.py` itera sobre un grid factorial de escenarios:

| Variable | Valores |
|----------|---------|
| Posiciones de fuga | 2000, 4000, 6000, 8000 m |
| Tamaños de fuga | 0.005, 0.015, 0.030 m³/s |
| Ruido en sensores | 0, 8000, 50000 Pa |

Resultados guardados en `results/benchmark_results.csv`.

```bash
python benchmark_wave_pinn.py
```

### Sesgo sistemático (~150m)

Se observó un sesgo consistente de ~150m hacia aguas arriba (~1.5% sobre 10 km). Análisis de diagnóstico determinó que es un **desfasaje temporal** (150m / 1200 m/s = 0.125s) causado por la interacción entre el suavizado Heaviside y la dispersión no-lineal del MOC. A efectos prácticos de ingeniería, este error es aceptable.

---

## Estructura del proyecto

```text
├── wave_pinn.py              # Modelo PINN principal (Wave-Injection)
├── benchmark_wave_pinn.py    # Benchmark factorial automatizado
├── benchmark_cpu_gpu.py      # Benchmark de rendimiento CPU vs GPU
├── simulator.py              # Simulador MOC de flujo transitorio
├── config.py                 # Parámetros globales del experimento
├── data_utils.py             # Carga del dataset y aplicación de ruido
├── generate_dataset.py       # Generación del dataset sintético base
├── moc_simulator.py          # Wrapper del simulador MOC
├── xpinn_model.py            # Modelo XPINN (experimental)
├── baseline_mass_balance.py  # Baseline: Balance de Masa + NPW
├── generate_thesis_figures.py # Generación de figuras para la tesis
│
├── app/
│   └── app.py                # Dashboard Streamlit de demostración
│
├── results/                  # Resultados de benchmarks (CSV)
├── figs/                     # Figuras generadas
├── checkpoints/              # Checkpoints de entrenamiento
├── archive/                  # Código antiguo y experimentos descartados
│
├── README.md
├── avances.md                # Notas de avance del proyecto
├── todo.md                   # Pendientes
└── requirements.txt
```

## Requisitos

- Python 3.10+
- `numpy`, `matplotlib`, `torch`, `pandas`

```bash
pip install -r requirements.txt
```

## Uso rápido

### Correr el modelo PINN

```bash
python wave_pinn.py --x_leak 6000 --q_leak 0.030 --noise 8000 --epochs 4000
```

### Correr el benchmark completo

```bash
python benchmark_wave_pinn.py
```

### Lanzar el dashboard interactivo

```bash
cd app
streamlit run app.py
```

### Correr el simulador MOC standalone

```bash
python simulator.py
```

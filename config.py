"""Configuración central del experimento de detección de fugas."""

# ── Parámetros del ducto ──────────────────────────────────────
PIPE_LENGTH = 10_000
PIPE_DIAMETER = 0.5
FRICTION_FACTOR = 0.02
FLUID_DENSITY = 850.0
WAVE_SPEED = 1_200.0

# ── Condiciones de contorno ───────────────────────────────────
P_INLET = 5_000_000.0
Q_OUTLET = 0.3

# ── Discretización ────────────────────────────────────────────
N_NODES = 101
T_TOTAL = 200.0
T_LEAK_START = 50.0

# ── Instrumentación ───────────────────────────────────────────
# Configuración basada en instrumentación industrial real:
# Los transmisores de presión son económicos y se instalan
# en múltiples puntos a lo largo del ducto.
# Los caudalímetros son costosos y solo existen en los
# extremos del sistema (puntos de transferencia de custodia).
# Esta distinción es estándar en oleoductos de transporte.

# Sensores de presión (transmisores) — baratos, puntos intermedios
PRESSURE_SENSOR_POSITIONS = [1_000, 5_000, 9_000]

# Caudalímetros (flow meters) — caros, solo en extremos
FLOW_METER_POSITIONS = [0, 10_000]

# Subconjuntos para experimento factorial
# Solo varía la cantidad de sensores de PRESIÓN.
# Los caudalímetros en extremos son siempre 2 (fijos).
PRESSURE_SENSOR_SUBSETS = {
    2: [1_000, 9_000],
    3: [1_000, 5_000, 9_000],
}
N_PRESSURE_SENSOR_LEVELS = [2, 3]

# ── Escenarios del dataset ────────────────────────────────────
X_LEAK_VALUES = [2_000, 4_000, 6_000, 8_000]
Q_LEAK_VALUES = [0.005, 0.015, 0.030]
Q_LEAK_LABELS = {0.005: 'small', 0.015: 'medium', 0.030: 'large'}

# ── Experimento factorial: niveles de ruido ───────────────────
NOISE_LEVELS = {
    'trivial': 500,
    'facil': 2_000,
    'moderado': 8_000,
    'dificil': 25_000,
    'muy_dificil': 50_000,
}

# ── Reproducibilidad ──────────────────────────────────────────
RANDOM_SEED = 42

# ── Rutas ─────────────────────────────────────────────────────
DATASET_PATH = 'dataset.h5'
FIGS_DIR = 'figs'
RESULTS_DIR = 'results'
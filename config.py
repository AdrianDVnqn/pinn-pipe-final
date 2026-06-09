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

# ── Sensores ──────────────────────────────────────────────────
SENSOR_POSITIONS = [1_000, 5_000, 9_000]

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

# ── Experimento factorial: cantidad de sensores ───────────────
N_SENSORS_LEVELS = [2, 3]
SENSOR_SUBSETS = {
    2: [1_000, 9_000],
    3: [1_000, 5_000, 9_000],
}

# ── Reproducibilidad ──────────────────────────────────────────
RANDOM_SEED = 42

# ── Rutas ─────────────────────────────────────────────────────
DATASET_PATH = 'dataset.h5'
FIGS_DIR = 'figs'
RESULTS_DIR = 'results'
# config.py

SENSOR_POSITIONS = [1000, 5000, 9000]  # metros

# Niveles de ruido del experimento factorial
NOISE_LEVELS = {
    "trivial":    500,
    "facil":    2_000,
    "moderado": 8_000,
    "dificil":  25_000,
    "muy_dificil": 50_000,
}

# Cantidades de sensores a evaluar (Paso E)
N_SENSORS_LEVELS = [2, 3]

# Semilla para reproducibilidad
RANDOM_SEED = 42
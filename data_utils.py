import json

import h5py
import numpy as np
import pandas as pd

import config as cfg


def _dataset_path():
    return cfg.DATASET_PATH


def _scenario_name(scenario_id: int) -> str:
    return f'scenario_{int(scenario_id):02d}'


def _pipe_area() -> float:
    return np.pi * cfg.PIPE_DIAMETER**2 / 4.0


def _read_string(dataset):
    raw = dataset[()]
    if isinstance(raw, bytes):
        return raw.decode('utf-8')
    return raw


def _reconstruct_pressure_profile(x: np.ndarray) -> np.ndarray:
    area = _pipe_area()
    return cfg.P_INLET - cfg.FRICTION_FACTOR * cfg.FLUID_DENSITY * x * (cfg.Q_OUTLET * abs(cfg.Q_OUTLET)) / (2.0 * cfg.PIPE_DIAMETER * area**2)


def _resample_sensor_block(source_positions, source_values, target_positions):
    source_positions = np.asarray(source_positions, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)
    source_values = np.asarray(source_values)
    resampled = np.empty((len(target_positions), source_values.shape[1]), dtype=source_values.dtype)
    for idx in range(source_values.shape[1]):
        resampled[:, idx] = np.interp(target_positions, source_positions, source_values[:, idx])
    return resampled


def load_scenario(scenario_id: int) -> dict:
    """Carga un escenario del dataset limpio y reconstruye el perfil estacionario."""
    scenario_group_name = _scenario_name(scenario_id)

    with h5py.File(_dataset_path(), 'r') as h5:
        metadata = h5['metadata']
        scenario_group = h5['scenarios'][scenario_group_name]
        labels = json.loads(_read_string(scenario_group['labels']))
        x = metadata['x'][()]

        return {
            'id': int(scenario_id),
            'has_leak': bool(labels.get('has_leak', False)),
            'x_leak': labels.get('x_leak'),
            'q_leak': float(labels.get('q_leak', 0.0)),
            'leak_size': labels.get('leak_size', 'none'),
            't': metadata['t'][()],
            'x': x,
            'x_sensors': metadata['x_sensors'][()],
            'P_sensors': scenario_group['P_sensors'][()],
            'Q_sensors': scenario_group['Q_sensors'][()],
            'dP_sensors': scenario_group['dP_sensors'][()],
            'P_full': scenario_group['P_full'][()],
            'P_ss': _reconstruct_pressure_profile(x),
        }


def apply_noise(data: np.ndarray, noise_std: float, seed: int = cfg.RANDOM_SEED) -> np.ndarray:
    """Agrega ruido gaussiano reproducible al array recibido."""
    data = np.asarray(data)
    if noise_std <= 0.0:
        return data.copy()
    rng = np.random.default_rng(seed)
    return data + rng.normal(0.0, noise_std, size=data.shape)


def get_training_data(scenario_id: int, noise_level: str, n_sensors: int = 3) -> dict:
    """Carga un escenario, selecciona sensores y aplica ruido sólo en ejecución.

    When n_sensors > len(x_sensors), pressure is interpolated from P_full
    (the full spatial field) instead of the sparse P_sensors array.
    """
    if noise_level not in cfg.NOISE_LEVELS:
        valid_levels = ', '.join(cfg.NOISE_LEVELS)
        raise ValueError(f'noise_level inválido: {noise_level}. Valores válidos: {valid_levels}')
    if n_sensors not in cfg.SENSOR_SUBSETS:
        valid_counts = ', '.join(str(value) for value in cfg.SENSOR_SUBSETS)
        raise ValueError(f'n_sensors inválido: {n_sensors}. Valores válidos: {valid_counts}')

    scenario = load_scenario(scenario_id)
    target_positions = np.asarray(cfg.SENSOR_SUBSETS[n_sensors], dtype=float)
    base_positions = np.asarray(scenario['x_sensors'], dtype=float)

    # Use P_full (full spatial grid) when requesting more positions than
    # available in P_sensors.  This allows arbitrary sensor layouts without
    # regenerating the dataset.
    if n_sensors > len(base_positions):
        x_full = np.asarray(scenario['x'], dtype=float)        # (101,)
        P_full = np.asarray(scenario['P_full'], dtype=float)    # (101, Nt)
        P_subset = _resample_sensor_block(x_full, P_full, target_positions)
    else:
        P_subset = _resample_sensor_block(base_positions, scenario['P_sensors'], target_positions)

    Q_subset = _resample_sensor_block(base_positions, scenario['Q_sensors'], target_positions)
    P_ss_subset = np.interp(target_positions, scenario['x'], scenario['P_ss'])

    noise_std = float(cfg.NOISE_LEVELS[noise_level])
    P_noisy = apply_noise(P_subset, noise_std, seed=cfg.RANDOM_SEED)
    Q_noisy = apply_noise(Q_subset, noise_std, seed=cfg.RANDOM_SEED + 1)
    dP_noisy = P_noisy - P_ss_subset[:, None]

    training_data = dict(scenario)
    training_data.update(
        {
            'noise_std': noise_std,
            'noise_level': noise_level,
            'n_sensors': n_sensors,
            'x_sensors_used': target_positions,
            'P_noisy': P_noisy,
            'Q_noisy': Q_noisy,
            'dP_noisy': dP_noisy,
        }
    )
    return training_data


def list_scenarios() -> pd.DataFrame:
    """Devuelve un resumen tabular de los 13 escenarios del dataset."""
    rows = []
    with h5py.File(_dataset_path(), 'r') as h5:
        for scenario_name in sorted(h5['scenarios'].keys()):
            scenario_id = int(scenario_name.split('_')[1])
            labels = json.loads(_read_string(h5['scenarios'][scenario_name]['labels']))
            rows.append(
                {
                    'scenario_id': scenario_id,
                    'has_leak': bool(labels.get('has_leak', False)),
                    'x_leak': labels.get('x_leak'),
                    'q_leak': labels.get('q_leak'),
                    'leak_size': labels.get('leak_size'),
                }
            )
    return pd.DataFrame(rows, columns=['scenario_id', 'has_leak', 'x_leak', 'q_leak', 'leak_size'])

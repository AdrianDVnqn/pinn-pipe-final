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
    """Carga un escenario del dataset separando sensores de presión y caudalímetros.

    Returns dict with:
        - Pressure sensors (multiple intermediate points)
        - Flow meters (only at extremes: x=0, x=L)
        - Full fields for interpolation
    """
    scenario_group_name = _scenario_name(scenario_id)

    with h5py.File(_dataset_path(), 'r') as h5:
        metadata = h5['metadata']
        scenario_group = h5['scenarios'][scenario_group_name]
        labels = json.loads(_read_string(scenario_group['labels']))
        x = metadata['x'][()]               # full spatial grid (101,)
        t = metadata['t'][()]               # time vector

        P_full = scenario_group['P_full'][()]    # (101, Nt)
        # Q_full might not exist as a separate dataset; derive from Q_sensors
        # or from the full grid if available
        if 'Q_full' in scenario_group:
            Q_full = scenario_group['Q_full'][()]
        else:
            # Q_sensors is at the original 3 sensor positions
            # For flow meters at extremes, interpolate from Q_sensors
            # or use P_full-based approach
            Q_full = None

        x_sensors_orig = metadata['x_sensors'][()]
        P_sensors_orig = scenario_group['P_sensors'][()]
        Q_sensors_orig = scenario_group['Q_sensors'][()]

        # ── Pressure sensors: interpolate from P_full ──
        x_pressure = np.asarray(cfg.PRESSURE_SENSOR_POSITIONS, dtype=float)
        P_sensors = _resample_sensor_block(x, P_full, x_pressure)

        # ── Flow meters: extract Q at extremes (x=0, x=L) ──
        x_flow = np.asarray(cfg.FLOW_METER_POSITIONS, dtype=float)
        if Q_full is not None:
            Q_flow_meters = _resample_sensor_block(x, Q_full, x_flow)
        else:
            # Interpolate from original Q_sensors
            Q_flow_meters = _resample_sensor_block(x_sensors_orig, Q_sensors_orig, x_flow)

        # ── Steady-state profiles for delta computation ──
        P_ss = _reconstruct_pressure_profile(x)
        P_ss_at_pressure = np.interp(x_pressure, x, P_ss)
        # Q steady state is uniform = Q_OUTLET everywhere
        Q_ss = cfg.Q_OUTLET

        # ── Delta (deviation from steady state) ──
        dP_sensors = P_sensors - P_ss_at_pressure[:, None]
        dQ_flow_meters = Q_flow_meters - Q_ss

        return {
            'id': int(scenario_id),
            'has_leak': bool(labels.get('has_leak', False)),
            'x_leak': labels.get('x_leak'),
            'q_leak': float(labels.get('q_leak', 0.0)),
            'leak_size': labels.get('leak_size', 'none'),
            't': t,
            'x': x,
            # Pressure sensors — multiple intermediate points
            'x_pressure_sensors': x_pressure,
            'P_sensors': P_sensors,
            'dP_sensors': dP_sensors,
            # Flow meters — only at extremes
            'x_flow_meters': x_flow,
            'Q_flow_meters': Q_flow_meters,
            'dQ_flow_meters': dQ_flow_meters,
            # Full fields for PINN interpolation
            'P_full': P_full,
            'P_ss': P_ss,
        }


def apply_noise(data: np.ndarray, noise_std: float, seed: int = cfg.RANDOM_SEED) -> np.ndarray:
    """Agrega ruido gaussiano reproducible al array recibido."""
    data = np.asarray(data)
    if noise_std <= 0.0:
        return data.copy()
    rng = np.random.default_rng(seed)
    return data + rng.normal(0.0, noise_std, size=data.shape)


def get_training_data(scenario_id: int, noise_level: str, n_pressure_sensors: int = 3) -> dict:
    """Carga un escenario, selecciona sensores de presión y aplica ruido.

    Parameters
    ----------
    scenario_id : int
    noise_level : str
        Key de NOISE_LEVELS en config.py
    n_pressure_sensors : int
        Cantidad de sensores de presión intermedios (2 o 3).
        Los caudalímetros en extremos son siempre 2 (fijos).
    """
    if noise_level not in cfg.NOISE_LEVELS:
        valid_levels = ', '.join(cfg.NOISE_LEVELS)
        raise ValueError(f'noise_level inválido: {noise_level}. Valores válidos: {valid_levels}')
    if n_pressure_sensors not in cfg.PRESSURE_SENSOR_SUBSETS:
        valid_counts = ', '.join(str(value) for value in cfg.PRESSURE_SENSOR_SUBSETS)
        raise ValueError(f'n_pressure_sensors inválido: {n_pressure_sensors}. Valores válidos: {valid_counts}')

    scenario = load_scenario(scenario_id)

    # ── Pressure sensors: select subset ──
    target_p_positions = np.asarray(cfg.PRESSURE_SENSOR_SUBSETS[n_pressure_sensors], dtype=float)
    base_p_positions = scenario['x_pressure_sensors']

    # Resample P from the base pressure sensor positions (or P_full if needed)
    x_full = np.asarray(scenario['x'], dtype=float)
    P_full = np.asarray(scenario['P_full'], dtype=float)
    P_subset = _resample_sensor_block(x_full, P_full, target_p_positions)

    P_ss_subset = np.interp(target_p_positions, scenario['x'], scenario['P_ss'])

    # ── Flow meters: always 2 at extremes (no subset selection) ──
    Q_flow = scenario['Q_flow_meters']           # (2, Nt)
    x_flow = scenario['x_flow_meters']            # [0, 10000]

    # ── Apply noise ──
    noise_std = float(cfg.NOISE_LEVELS[noise_level])
    P_noisy = apply_noise(P_subset, noise_std, seed=cfg.RANDOM_SEED)

    # Flow meter noise: scaled proportionally to Q magnitude
    # (flow meters have their own noise characteristics, not Pa-scale)
    q_noise_std = noise_std * cfg.Q_OUTLET / cfg.P_INLET
    Q_noisy = apply_noise(Q_flow, q_noise_std, seed=cfg.RANDOM_SEED + 1)

    # ── Deltas ──
    dP_noisy = P_noisy - P_ss_subset[:, None]
    dQ_noisy = Q_noisy - cfg.Q_OUTLET

    training_data = dict(scenario)
    training_data.update(
        {
            'noise_std': noise_std,
            'q_noise_std': q_noise_std,
            'noise_level': noise_level,
            'n_pressure_sensors': n_pressure_sensors,
            # Pressure sensors (variable count)
            'x_pressure_sensors_used': target_p_positions,
            'P_noisy': P_noisy,
            'dP_noisy': dP_noisy,
            # Flow meters (always 2 at extremes)
            'x_flow_meters': x_flow,
            'Q_noisy': Q_noisy,
            'dQ_noisy': dQ_noisy,
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

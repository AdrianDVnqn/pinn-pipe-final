import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as cfg
import moc_simulator as sim


X_LEAK_VALUES = cfg.X_LEAK_VALUES
Q_LEAK_VALUES = cfg.Q_LEAK_VALUES
LEAK_SIZE_LABELS = cfg.Q_LEAK_LABELS
SENSOR_POSITIONS = cfg.SENSOR_SUBSETS[3]
T_LEAK = cfg.T_LEAK_START
REFERENCE_NOISE_STD = cfg.NOISE_LEVELS['trivial']


def build_scenario_catalog():
    scenarios = [
        {
            'name': 'scenario_00',
            'has_leak': False,
            'x_leak': None,
            'q_leak': 0.0,
            'leak_size': 'none',
        }
    ]

    scenario_idx = 1
    for x_leak in X_LEAK_VALUES:
        for q_leak in Q_LEAK_VALUES:
            scenarios.append(
                {
                    'name': f'scenario_{scenario_idx:02d}',
                    'has_leak': True,
                    'x_leak': x_leak,
                    'q_leak': q_leak,
                    'leak_size': LEAK_SIZE_LABELS[q_leak],
                }
            )
            scenario_idx += 1
    return scenarios


def build_metadata_payload():
    params = {
        'L': cfg.PIPE_LENGTH,
        'D': cfg.PIPE_DIAMETER,
        'rho': cfg.FLUID_DENSITY,
        'a': cfg.WAVE_SPEED,
        'f': cfg.FRICTION_FACTOR,
        'P_in': cfg.P_INLET,
        'Q_out': cfg.Q_OUTLET,
        'Nx': cfg.N_NODES,
        'T': cfg.T_TOTAL,
        'sensor_positions': SENSOR_POSITIONS,
        'x_leak_values': X_LEAK_VALUES,
        'q_leak_values': Q_LEAK_VALUES,
        't_leak': T_LEAK,
        'noise_levels': cfg.NOISE_LEVELS,
        'sensor_subsets': cfg.SENSOR_SUBSETS,
        'random_seed': cfg.RANDOM_SEED,
    }
    return params


def _write_scalar_string(group, name, value):
    dtype = h5py.string_dtype(encoding='utf-8')
    group.create_dataset(name, data=value, dtype=dtype)


def _write_array(group, name, array, compression='gzip', compression_opts=4):
    data = np.asarray(array)
    group.create_dataset(
        name,
        data=data,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=True,
    )


def _read_string(dataset):
    raw = dataset[()]
    if isinstance(raw, bytes):
        return raw.decode('utf-8')
    return raw


def _interp_1d(x_grid, values, x_query):
    return float(np.interp(x_query, x_grid, values))


def generate_dataset(h5_path='dataset.h5', fig_dir='figs', scenario_subset=None):
    scenarios = build_scenario_catalog()
    if scenario_subset is not None:
        scenarios = [scenarios[i] for i in scenario_subset]

    h5_path = Path(h5_path)
    fig_dir = Path(fig_dir) if fig_dir is not None else None
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'w') as h5:
        metadata = h5.create_group('metadata')
        x_ref, t_ref = sim.build_grid()
        _write_array(metadata, 't', t_ref.astype(np.float64))
        _write_array(metadata, 'x', x_ref.astype(np.float64))
        _write_array(metadata, 'x_sensors', np.asarray(SENSOR_POSITIONS, dtype=np.float64))
        _write_scalar_string(metadata, 'params', json.dumps(build_metadata_payload(), ensure_ascii=False))

        scenarios_group = h5.create_group('scenarios')
        cached_records = []

        total = len(scenarios)
        for idx, scenario in enumerate(scenarios, start=1):
            if scenario['has_leak']:
                print(
                    f"Corriendo escenario {idx}/{total}: "
                    f"x_leak={scenario['x_leak']}m, q_leak={scenario['q_leak']:.3f} m³/s..."
                )
            else:
                print(f"Corriendo escenario {idx}/{total}: sin fuga...")

            if scenario['has_leak']:
                moc_result = sim.run_moc(Q_leak=scenario['q_leak'], x_leak=scenario['x_leak'], t_leak=T_LEAK)
            else:
                moc_result = sim.run_moc(Q_leak=0.0, x_leak=cfg.X_LEAK_VALUES[0], t_leak=T_LEAK)

            sensor_data = sim.get_sensor_data(moc_result, SENSOR_POSITIONS, noise_std=0.0)
            dQ_sensors = []
            for sensor_idx, xpos in enumerate(SENSOR_POSITIONS):
                q_ss_sensor = _interp_1d(moc_result['x'], moc_result['Q_ss'], xpos)
                dQ_sensors.append(sensor_data['Q_sensors'][sensor_idx] - q_ss_sensor)
            dQ_sensors = np.asarray(dQ_sensors, dtype=np.float32)

            scenario_group = scenarios_group.create_group(scenario['name'])

            _write_array(scenario_group, 'P_sensors', sensor_data['P_sensors'].astype(np.float32))
            _write_array(scenario_group, 'Q_sensors', sensor_data['Q_sensors'].astype(np.float32))
            _write_array(scenario_group, 'dP_sensors', sensor_data['dP_sensors'].astype(np.float32))
            _write_array(scenario_group, 'dQ_sensors', dQ_sensors)
            _write_array(scenario_group, 'P_full', moc_result['P'].astype(np.float32))
            _write_scalar_string(scenario_group, 'labels', json.dumps(scenario, ensure_ascii=False))

            cached_records.append(
                {
                    'name': scenario['name'],
                    'x_leak': scenario['x_leak'],
                    'q_leak': scenario['q_leak'],
                    'leak_size': scenario['leak_size'],
                    'has_leak': scenario['has_leak'],
                    't': moc_result['t'].copy(),
                    'dP_sensors': sensor_data['dP_sensors'].copy(),
                }
            )

    if fig_dir is not None:
        generate_exploratory_figures(h5_path, fig_dir)

    return h5_path, cached_records


def load_scenario(h5_path, scenario_id):
    if isinstance(scenario_id, int):
        scenario_name = f'scenario_{scenario_id:02d}'
    else:
        scenario_name = str(scenario_id)
        if not scenario_name.startswith('scenario_'):
            scenario_name = f'scenario_{int(scenario_name):02d}'

    with h5py.File(h5_path, 'r') as h5:
        metadata = h5['metadata']
        scenario_group = h5['scenarios'][scenario_name]
        labels = json.loads(_read_string(scenario_group['labels']))

        return {
            'scenario_id': scenario_name,
            't': metadata['t'][()],
            'x': metadata['x'][()],
            'x_sensors': metadata['x_sensors'][()],
            'params': json.loads(_read_string(metadata['params'])),
            'P_sensors': scenario_group['P_sensors'][()],
            'Q_sensors': scenario_group['Q_sensors'][()],
            'dP_sensors': scenario_group['dP_sensors'][()],
            'dQ_sensors': scenario_group['dQ_sensors'][()],
            'P_full': scenario_group['P_full'][()],
            'labels': labels,
        }


def _load_labels(scenario_group):
    return json.loads(_read_string(scenario_group['labels']))


def _select_scenario(records, *, x_leak=None, q_leak=None, has_leak=None):
    for record in records:
        if x_leak is not None and record['x_leak'] != x_leak:
            continue
        if q_leak is not None and not np.isclose(record['q_leak'], q_leak):
            continue
        if has_leak is not None and record['has_leak'] != has_leak:
            continue
        return record
    raise ValueError('Scenario not found for the requested filters.')


def generate_exploratory_figures(h5_path, fig_dir):
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as h5:
        t = h5['metadata']['t'][()]
        x_sensors = h5['metadata']['x_sensors'][()]
        scenario_names = sorted(h5['scenarios'].keys())
        records = []
        for name in scenario_names:
            grp = h5['scenarios'][name]
            labels = _load_labels(grp)
            records.append(
                {
                    'name': name,
                    't': t,
                    'x_leak': labels.get('x_leak'),
                    'q_leak': labels.get('q_leak'),
                    'leak_size': labels.get('leak_size'),
                    'has_leak': labels.get('has_leak'),
                    'dP_sensors': grp['dP_sensors'][()],
                }
            )

    _plot_leak_sizes(records, t, x_sensors, fig_dir / 'overview_leak_sizes.png')
    _plot_leak_positions(records, t, x_sensors, fig_dir / 'overview_leak_positions.png')
    _plot_snr_heatmap(records, fig_dir / 'snr_heatmap.png')
    _plot_wave_arrival_times(records, t, x_sensors, fig_dir / 'wave_arrival_times.png')


def _plot_leak_sizes(records, t, x_sensors, output_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    q_order = [0.005, 0.015, 0.030]

    for ax, q_leak in zip(axes, q_order):
        record = _select_scenario(records, x_leak=6000, q_leak=q_leak, has_leak=True)
        for sensor_idx, xpos in enumerate(x_sensors):
            ax.plot(t, record['dP_sensors'][sensor_idx], color=colors[sensor_idx], label=f'Sensor x={int(xpos)} m')
        ax.set_title(f"Leak size = {record['leak_size']} | q_leak = {q_leak:.3f} m³/s | x_leak = 6000 m")
        ax.set_ylabel('dP [Pa]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('t [s]')
    fig.suptitle('Variación de la señal según tamaño de fuga')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_leak_positions(records, t, x_sensors, output_path):
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    x_order = [2000, 4000, 6000, 8000]
    q_medium = 0.015

    for ax, x_leak in zip(axes, x_order):
        record = _select_scenario(records, x_leak=x_leak, q_leak=q_medium, has_leak=True)
        for sensor_idx, xpos in enumerate(x_sensors):
            ax.plot(t, record['dP_sensors'][sensor_idx], color=colors[sensor_idx], label=f'Sensor x={int(xpos)} m')
        ax.set_title(f"x_leak = {x_leak} m | q_leak = {q_medium:.3f} m³/s ({record['leak_size']})")
        ax.set_ylabel('dP [Pa]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('t [s]')
    fig.suptitle('Variación de la señal según posición de fuga')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_snr_heatmap(records, output_path):
    q_order = [0.005, 0.015, 0.030]
    x_order = [2000, 4000, 6000, 8000]
    snr_db = np.zeros((len(q_order), len(x_order)), dtype=float)

    for i, q_leak in enumerate(q_order):
        for j, x_leak in enumerate(x_order):
            record = _select_scenario(records, x_leak=x_leak, q_leak=q_leak, has_leak=True)
            max_signal = np.max(np.abs(record['dP_sensors']))
            snr_db[i, j] = 20.0 * np.log10(max_signal / REFERENCE_NOISE_STD)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    im = ax.imshow(snr_db, origin='lower', aspect='auto', cmap='viridis')
    ax.set_xticks(np.arange(len(x_order)))
    ax.set_xticklabels([str(x) for x in x_order])
    ax.set_yticks(np.arange(len(q_order)))
    ax.set_yticklabels([f'{q:.3f}' for q in q_order])
    ax.set_xlabel('x_leak [m]')
    ax.set_ylabel('q_leak [m³/s]')
    ax.set_title('SNR (dB) por combinación de fuga')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('SNR [dB]')

    for i in range(len(q_order)):
        for j in range(len(x_order)):
            ax.text(j, i, f'{snr_db[i, j]:.1f}', ha='center', va='center', color='white' if snr_db[i, j] < np.max(snr_db) * 0.5 else 'black')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_wave_arrival_times(records, t, x_sensors, output_path):
    x_order = [2000, 4000, 6000, 8000]
    q_medium = 0.015
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for ax, x_leak in zip(axes, x_order):
        record = _select_scenario(records, x_leak=x_leak, q_leak=q_medium, has_leak=True)
        for sensor_idx, xpos in enumerate(x_sensors):
            arrival_time = T_LEAK + abs(float(xpos) - float(x_leak)) / sim.a
            ax.plot(t, record['dP_sensors'][sensor_idx], color=colors[sensor_idx], label=f'Sensor x={int(xpos)} m')
            ax.axvline(arrival_time, color=colors[sensor_idx], linestyle='--', alpha=0.8)
        ax.set_title(f'Arrivals teóricos vs MOC | x_leak = {x_leak} m | q_leak = {q_medium:.3f} m³/s')
        ax.set_ylabel('dP [Pa]')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('t [s]')
    fig.suptitle('Tiempos de llegada de onda por posición de fuga')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic leak dataset for PINN training.')
    parser.add_argument('--h5', default=cfg.DATASET_PATH, help='Output HDF5 file path.')
    parser.add_argument('--fig-dir', default=cfg.FIGS_DIR, help='Directory where exploratory figures are saved.')
    parser.add_argument('--subset', nargs='*', type=int, default=None, help='Optional scenario indices to generate for a quick smoke test.')
    return parser.parse_args()


def main():
    args = parse_args()
    generate_dataset(h5_path=args.h5, fig_dir=args.fig_dir, scenario_subset=args.subset)


if __name__ == '__main__':
    main()

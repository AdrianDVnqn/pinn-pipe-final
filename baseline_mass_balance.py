"""
Baseline 1 — Balance de Masa + Onda de Presión Negativa (NPW)
==============================================================

Método clásico de detección de fugas usado en sistemas SCADA industriales.
No usa ML ni optimización — solo física y aritmética.

Componentes:
  1. DETECCIÓN:      balance de masa entre sensores upstream/downstream
  2. LOCALIZACIÓN:   método de onda de presión negativa (NPW)
  3. CUANTIFICACIÓN: estimación de q_leak por diferencia de caudales
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from data_utils import get_training_data, list_scenarios


# ═══════════════════════════════════════════════════════════════
# CLASE 1 — DETECCIÓN POR BALANCE DE MASA
# ═══════════════════════════════════════════════════════════════

class MassBalanceDetector:
    """Detecta fugas comparando caudal upstream vs downstream.

    Principio: en estado estacionario sin fuga, Q_upstream ≈ Q_downstream.
    Cuando hay fuga, Q_upstream - Q_downstream > 0 de forma sostenida.
    """

    def __init__(self, window_size=20, n_sigma=3.0):
        self.window_size = window_size
        self.n_sigma = n_sigma
        self._baseline_mean = None
        self._baseline_std = None

    def _moving_average(self, signal):
        """Suavizado con ventana deslizante (media móvil centrada)."""
        kernel = np.ones(self.window_size) / self.window_size
        # mode='same' mantiene la longitud; los bordes usan padding parcial
        smoothed = np.convolve(signal, kernel, mode='same')
        return smoothed

    def fit_baseline(self, t, Q_sensors, t_end_baseline=10.0):
        """Calcula mean y std de la línea base usando datos
        anteriores a t_end_baseline segundos.

        Parameters
        ----------
        t : np.ndarray, shape (Nt,)
            Vector de tiempos.
        Q_sensors : np.ndarray, shape (n_sensors, Nt)
            Caudal en cada sensor a lo largo del tiempo.
        t_end_baseline : float
            Segundos iniciales usados como referencia pre-fuga.
        """
        delta_Q = Q_sensors[0, :] - Q_sensors[-1, :]
        delta_Q_smooth = self._moving_average(delta_Q)

        mask_baseline = t < t_end_baseline
        if mask_baseline.sum() < 2:
            # Fallback: usar los primeros 10 pasos
            mask_baseline = np.zeros(len(t), dtype=bool)
            mask_baseline[:10] = True

        self._baseline_mean = np.mean(delta_Q_smooth[mask_baseline])
        self._baseline_std = np.std(delta_Q_smooth[mask_baseline])
        # Evitar std = 0 en datos perfectos (sin ruido)
        if self._baseline_std < 1e-12:
            self._baseline_std = 1e-12

    def detect(self, t, Q_sensors):
        """Ejecuta la detección de fuga por balance de masa.

        Parameters
        ----------
        t : np.ndarray, shape (Nt,)
        Q_sensors : np.ndarray, shape (n_sensors, Nt)

        Returns
        -------
        dict con claves:
            leak_detected, t_detection, q_leak_est,
            delta_Q, delta_Q_smooth, threshold
        """
        if self._baseline_mean is None:
            self.fit_baseline(t, Q_sensors)

        delta_Q = Q_sensors[0, :] - Q_sensors[-1, :]
        delta_Q_smooth = self._moving_average(delta_Q)

        threshold = self._baseline_mean + self.n_sigma * self._baseline_std

        # Buscar primer cruce del umbral
        exceedances = np.where(delta_Q_smooth > threshold)[0]

        if len(exceedances) == 0:
            return {
                'leak_detected': False,
                't_detection': None,
                'q_leak_est': 0.0,
                'delta_Q': delta_Q,
                'delta_Q_smooth': delta_Q_smooth,
                'threshold': threshold,
            }

        idx_detection = exceedances[0]
        t_detection = t[idx_detection]

        # Estimar q_leak: promedio de delta_Q en régimen post-detección
        # (al menos 20s después de la detección para estabilizarse)
        t_stable = t_detection + 20.0
        mask_stable = t > t_stable
        if mask_stable.sum() > 0:
            q_leak_est = float(np.mean(delta_Q_smooth[mask_stable]))
        else:
            # Si no hay datos 20s después, usar lo que quede
            q_leak_est = float(np.mean(delta_Q_smooth[idx_detection:]))

        return {
            'leak_detected': True,
            't_detection': float(t_detection),
            'q_leak_est': q_leak_est,
            'delta_Q': delta_Q,
            'delta_Q_smooth': delta_Q_smooth,
            'threshold': threshold,
        }


# ═══════════════════════════════════════════════════════════════
# CLASE 2 — LOCALIZACIÓN POR ONDA DE PRESIÓN NEGATIVA (NPW)
# ═══════════════════════════════════════════════════════════════

class NPWLocalizer:
    """Localiza fugas por diferencia de tiempos de llegada de la
    onda de presión negativa entre pares de sensores.

    Principio:
      Una fuga genera una onda que viaja a velocidad `a`.
      Con dos sensores que bracketean la fuga:
        x_leak = (a * (t1 - t2) + x_s1 + x_s2) / 2
    """

    def __init__(self, wave_speed=None, k_threshold=2.0, window_size=10):
        self.wave_speed = wave_speed if wave_speed is not None else cfg.WAVE_SPEED
        self.k_threshold = k_threshold
        self.window_size = window_size

    def _moving_average(self, signal):
        """Suavizado con ventana deslizante."""
        kernel = np.ones(self.window_size) / self.window_size
        return np.convolve(signal, kernel, mode='same')

    def find_arrival_times(self, t, dP_sensors, x_sensors, t_detection=None):
        """Encuentra el tiempo de llegada de la onda negativa en cada sensor.

        Parameters
        ----------
        t : np.ndarray, shape (Nt,)
        dP_sensors : np.ndarray, shape (n_sensors, Nt)
            Desviación de presión respecto de la baseline (P - P_baseline).
        x_sensors : np.ndarray, shape (n_sensors,)
        t_detection : float or None
            Si se provee, solo busca arrivals después de este tiempo.

        Returns
        -------
        list of (float or None)
            Tiempo de llegada por sensor. None si no se detecta.
        """
        n_sensors = dP_sensors.shape[0]
        arrival_times = []

        # Zona de baseline: antes de la fuga (t < 10s o antes de t_detection)
        t_baseline_end = 10.0
        mask_baseline = t < t_baseline_end

        for i in range(n_sensors):
            dP_i = dP_sensors[i, :]
            dP_smooth = self._moving_average(dP_i)

            # Estadísticas de baseline para este sensor
            if mask_baseline.sum() > 1:
                baseline_std = np.std(dP_smooth[mask_baseline])
            else:
                baseline_std = np.std(dP_smooth[:10])
            if baseline_std < 1e-12:
                baseline_std = 1e-12

            # Umbral de arrival: caída negativa significativa
            arrival_threshold = -self.k_threshold * baseline_std

            # Buscar primer cruce por debajo del umbral
            # Solo buscar después de t_detection si está disponible
            if t_detection is not None:
                search_mask = t >= t_detection
            else:
                search_mask = t >= t_baseline_end

            candidates = np.where(
                (dP_smooth < arrival_threshold) & search_mask
            )[0]

            if len(candidates) > 0:
                arrival_times.append(float(t[candidates[0]]))
            else:
                arrival_times.append(None)

        return arrival_times

    def localize(self, t, dP_sensors, x_sensors, t_detection=None):
        """Estima x_leak usando diferencia de tiempos de llegada.

        Lógica:
          1. Encontrar tiempos de llegada por sensor
          2. Para cada par de sensores adyacentes que bracketean
             la fuga: calcular x_leak
          3. Promediar las estimaciones de todos los pares
          4. Si no hay pares válidos → usar el sensor con menor
             arrival time como estimación gruesa

        Returns
        -------
        dict con claves:
            x_leak_est, arrival_times, n_pairs_used, estimates
        """
        x_sensors = np.asarray(x_sensors, dtype=float)
        arrival_times = self.find_arrival_times(
            t, dP_sensors, x_sensors, t_detection
        )

        a = self.wave_speed
        estimates = []

        # Para cada par de sensores adyacentes
        for j in range(len(x_sensors) - 1):
            t1 = arrival_times[j]      # sensor upstream
            t2 = arrival_times[j + 1]  # sensor downstream

            if t1 is None or t2 is None:
                continue

            x_s1 = x_sensors[j]
            x_s2 = x_sensors[j + 1]

            x_est = (a * (t1 - t2) + x_s1 + x_s2) / 2.0

            # Solo aceptar si la estimación cae entre ambos sensores
            # (es decir, el par efectivamente bracketea la fuga)
            margin = 0.1 * (x_s2 - x_s1)  # 10% de margen
            if (x_s1 - margin) <= x_est <= (x_s2 + margin):
                estimates.append(float(x_est))

        if len(estimates) > 0:
            x_leak_est = float(np.mean(estimates))
        else:
            # Fallback: el sensor con arrival más temprano está más cerca
            valid = [
                (arrival_times[i], x_sensors[i])
                for i in range(len(arrival_times))
                if arrival_times[i] is not None
            ]
            if valid:
                valid.sort(key=lambda pair: pair[0])
                x_leak_est = float(valid[0][1])
            else:
                x_leak_est = float(np.mean(x_sensors))

        return {
            'x_leak_est': x_leak_est,
            'arrival_times': arrival_times,
            'n_pairs_used': len(estimates),
            'estimates': estimates,
        }


# ═══════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL DE INFERENCIA
# ═══════════════════════════════════════════════════════════════

def run_mass_balance(scenario_id, noise_level='trivial', n_pressure_sensors=3,
                     window_size=20, n_sigma=3.0, k_threshold=2.0):
    """Corre el baseline completo (detección + localización) para un escenario.

    Instrumentación:
      - DETECCIÓN: usa Q_flow_meters (caudalímetros en x=0 y x=L, siempre 2).
        delta_Q(t) = Q_noisy[0,t] - Q_noisy[1,t] (entrada - salida).
      - LOCALIZACIÓN: usa dP_noisy de sensores de presión intermedios.
        La cantidad de sensores de presión varía (2 o 3).

    Retorna dict con formato comparable al de la PINN.
    """
    data = get_training_data(scenario_id, noise_level, n_pressure_sensors)

    t = data['t']                                       # (Nt,)
    x_p_sensors = data['x_pressure_sensors_used']       # (n_p_sensors,)
    dP_noisy = data['dP_noisy']                         # (n_p_sensors, Nt)
    Q_noisy = data['Q_noisy']                           # (2, Nt) — flow meters

    x_leak_true = data['x_leak']
    q_leak_true = data['q_leak']
    has_leak = data['has_leak']

    # ── 1. Detección por balance de masa (caudalímetros en extremos) ──
    detector = MassBalanceDetector(window_size=window_size, n_sigma=n_sigma)
    detector.fit_baseline(t, Q_noisy, t_end_baseline=10.0)
    det_result = detector.detect(t, Q_noisy)

    leak_detected = det_result['leak_detected']
    t_detection = det_result['t_detection']
    q_leak_pred = det_result['q_leak_est']

    # ── 2. Localización por NPW (sensores de presión intermedios) ──
    x_leak_pred = None
    arrival_times = []
    n_pairs_used = 0
    estimates_npw = []

    if leak_detected:
        localizer = NPWLocalizer(
            wave_speed=cfg.WAVE_SPEED,
            k_threshold=k_threshold,
            window_size=10
        )
        loc_result = localizer.localize(
            t, dP_noisy, x_p_sensors, t_detection
        )
        x_leak_pred = loc_result['x_leak_est']
        arrival_times = loc_result['arrival_times']
        n_pairs_used = loc_result['n_pairs_used']
        estimates_npw = loc_result['estimates']

    # ── 3. Calcular errores ──
    x_leak_error_km = None
    q_leak_error_pct = None

    if has_leak and leak_detected and x_leak_pred is not None:
        x_leak_error_km = abs(x_leak_pred - x_leak_true) / 1000.0

    if has_leak and leak_detected and q_leak_true > 0:
        q_leak_error_pct = abs(q_leak_pred - q_leak_true) / q_leak_true * 100.0

    return {
        # Resultados de detección
        'leak_detected': leak_detected,
        't_detection': t_detection,
        # Resultados de localización
        'x_leak_pred': x_leak_pred,
        'x_leak_true': x_leak_true if has_leak else None,
        'x_leak_error_km': x_leak_error_km,
        # Resultados de cuantificación
        'q_leak_pred': q_leak_pred,
        'q_leak_true': q_leak_true,
        'q_leak_error_pct': q_leak_error_pct,
        # Metadata
        'method': 'mass_balance_npw',
        'scenario_id': scenario_id,
        'noise_level': noise_level,
        'n_pressure_sensors': n_pressure_sensors,
        'has_leak': has_leak,
        # Diagnóstico
        'delta_Q': det_result['delta_Q'],
        'delta_Q_smooth': det_result['delta_Q_smooth'],
        'arrival_times': arrival_times,
        'threshold': det_result['threshold'],
        'n_pairs_used': n_pairs_used,
        'estimates_npw': estimates_npw,
        # Datos para gráficos
        't': t,
        'x_p_sensors': x_p_sensors,
        'dP_noisy': dP_noisy,
    }


# ═══════════════════════════════════════════════════════════════
# EVALUACIÓN EN TODO EL DATASET
# ═══════════════════════════════════════════════════════════════

def evaluate_mass_balance(noise_levels=None, n_p_sensors_list=None):
    """Corre el baseline en todos los escenarios × ruido × sensores.

    Returns
    -------
    pd.DataFrame con una fila por combinación.
    """
    if noise_levels is None:
        noise_levels = list(cfg.NOISE_LEVELS.keys())
    if n_p_sensors_list is None:
        n_p_sensors_list = cfg.N_PRESSURE_SENSOR_LEVELS

    scenarios = list_scenarios()
    total = len(scenarios) * len(noise_levels) * len(n_p_sensors_list)
    rows = []
    counter = 0

    for n_ps in n_p_sensors_list:
        for noise in noise_levels:
            noise_std = cfg.NOISE_LEVELS[noise]
            for _, sc in scenarios.iterrows():
                sid = sc['scenario_id']
                counter += 1
                print(
                    f'  Corriendo mass_balance | noise={noise:<12s} '
                    f'| n_p_sensors={n_ps} '
                    f'| escenario {sid:>2d} '
                    f'({counter}/{total})'
                )

                result = run_mass_balance(
                    scenario_id=sid,
                    noise_level=noise,
                    n_pressure_sensors=n_ps
                )

                rows.append({
                    'scenario_id': sid,
                    'has_leak': sc['has_leak'],
                    'x_leak_true': sc['x_leak'],
                    'q_leak_true': sc['q_leak'],
                    'leak_size': sc['leak_size'],
                    'noise_level': noise,
                    'noise_std': noise_std,
                    'n_pressure_sensors': n_ps,
                    'leak_detected': result['leak_detected'],
                    't_detection': result['t_detection'],
                    'x_leak_pred': result['x_leak_pred'],
                    'x_leak_error_km': result['x_leak_error_km'],
                    'q_leak_pred': result['q_leak_pred'],
                    'q_leak_error_pct': result['q_leak_error_pct'],
                    'method': 'mass_balance_npw',
                })

    df = pd.DataFrame(rows)

    # Guardar resultados
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(cfg.RESULTS_DIR, 'baseline_mass_balance.csv')
    df.to_csv(csv_path, index=False)
    print(f'\n  Resultados guardados en: {csv_path}')

    return df


# ═══════════════════════════════════════════════════════════════
# VISUALIZACIÓN Y MÉTRICAS
# ═══════════════════════════════════════════════════════════════

def plot_mass_balance_results(df, save_dir=None):
    """Genera las 3 figuras de diagnóstico del baseline."""
    if save_dir is None:
        save_dir = cfg.FIGS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # ── Figura 1: Ejemplo de detección (scenario 8, noise moderado) ──
    _plot_detection_example(save_dir)

    # ── Figura 2: Error de localización vs nivel de ruido ──
    _plot_error_vs_noise(df, save_dir)

    # ── Figura 3: Tasa de detección como heatmap ──
    _plot_detection_rate_heatmap(df, save_dir)


def _plot_detection_example(save_dir):
    """Figura 1: ejemplo visual de detección para scenario 8, moderado."""
    result = run_mass_balance(scenario_id=8, noise_level='moderado', n_pressure_sensors=3)
    t = result['t']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        'Baseline: Balance de Masa — Ejemplo de Detección\n'
        '(Escenario 8: x_leak=6000m, q_leak=0.015, ruido=moderado, 3 sensores)',
        fontsize=13, fontweight='bold'
    )

    # Subplot 1: dP en cada sensor
    ax1 = axes[0]
    dP = result['dP_noisy']
    x_p_sensors = result['x_p_sensors']
    for i in range(dP.shape[0]):
        ax1.plot(t, dP[i, :], label=f'P sensor x={x_p_sensors[i]:.0f} m', alpha=0.8)
    if result['t_detection'] is not None:
        ax1.axvline(result['t_detection'], color='red', ls='--', lw=1.5,
                     label=f't_detect = {result["t_detection"]:.1f} s')
    ax1.set_ylabel('ΔP  [Pa]')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.set_title('Desviación de presión en sensores')
    ax1.grid(True, alpha=0.3)

    # Subplot 2: delta_Q suavizado + threshold
    ax2 = axes[1]
    ax2.plot(t, result['delta_Q'], color='lightblue', alpha=0.5,
             label='ΔQ raw')
    ax2.plot(t, result['delta_Q_smooth'], color='blue', lw=1.5,
             label='ΔQ suavizado')
    ax2.axhline(result['threshold'], color='orange', ls='--', lw=1.5,
                label=f'Umbral (3σ) = {result["threshold"]:.5f}')
    if result['t_detection'] is not None:
        ax2.axvline(result['t_detection'], color='red', ls='--', lw=1.5,
                     label=f't_detect = {result["t_detection"]:.1f} s')
    ax2.set_xlabel('Tiempo  [s]')
    ax2.set_ylabel('ΔQ  [m³/s]')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.set_title('Balance de masa: Q_upstream − Q_downstream')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'mb_detection_example.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figura guardada: {path}')


def _plot_error_vs_noise(df, save_dir):
    """Figura 2: error de localización vs nivel de ruido, una línea por n_sensors."""
    noise_order = ['trivial', 'facil', 'moderado', 'dificil', 'muy_dificil']

    # Solo escenarios con fuga donde se detectó correctamente
    df_ok = df[(df['has_leak']) & (df['leak_detected']) &
               (df['x_leak_error_km'].notna())].copy()

    if df_ok.empty:
        print('  [WARN] No hay detecciones correctas para graficar error vs ruido.')
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = {2: 's', 3: 'o', 5: '^', 11: 'D'}

    for ns in sorted(df_ok['n_pressure_sensors'].unique()):
        sub = df_ok[df_ok['n_pressure_sensors'] == ns]
        means = []
        valid_noises = []
        for noise in noise_order:
            vals = sub[sub['noise_level'] == noise]['x_leak_error_km']
            if len(vals) > 0:
                means.append(vals.mean())
                valid_noises.append(noise)
        if means:
            ax.plot(valid_noises, means,
                    marker=markers.get(ns, 'o'), lw=2, ms=8,
                    label=f'{ns} sensores')

    ax.set_xlabel('Nivel de Ruido', fontsize=12)
    ax.set_ylabel('Error medio de localización  [km]', fontsize=12)
    ax.set_title('Baseline Mass Balance + NPW: Error de localización vs Ruido',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'mb_error_vs_noise.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figura guardada: {path}')


def _plot_detection_rate_heatmap(df, save_dir):
    """Figura 3: tasa de detección como heatmap (filas=leak_size, cols=noise)."""
    noise_order = ['trivial', 'facil', 'moderado', 'dificil', 'muy_dificil']

    # Solo escenarios con fuga
    df_leak = df[df['has_leak']].copy()
    if df_leak.empty:
        print('  [WARN] No hay escenarios con fuga para el heatmap.')
        return

    # Pivot: filas = leak_size, columnas = noise_level
    sizes = ['small', 'medium', 'large']
    available_sizes = [s for s in sizes if s in df_leak['leak_size'].values]
    available_noises = [n for n in noise_order if n in df_leak['noise_level'].values]

    rate_matrix = np.full((len(available_sizes), len(available_noises)), np.nan)
    for i, size in enumerate(available_sizes):
        for j, noise in enumerate(available_noises):
            mask = (df_leak['leak_size'] == size) & (df_leak['noise_level'] == noise)
            sub = df_leak[mask]
            if len(sub) > 0:
                rate_matrix[i, j] = sub['leak_detected'].mean() * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(rate_matrix, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')

    ax.set_xticks(range(len(available_noises)))
    ax.set_xticklabels(available_noises, fontsize=10)
    ax.set_yticks(range(len(available_sizes)))
    ax.set_yticklabels([f'q_leak {s}' for s in available_sizes], fontsize=10)

    # Anotar valores
    for i in range(len(available_sizes)):
        for j in range(len(available_noises)):
            val = rate_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                        fontsize=12, fontweight='bold', color=color)

    ax.set_xlabel('Nivel de Ruido', fontsize=12)
    ax.set_ylabel('Tamaño de Fuga', fontsize=12)
    ax.set_title('Baseline Mass Balance: Tasa de Detección (%)',
                 fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Tasa de detección (%)')

    plt.tight_layout()
    path = os.path.join(save_dir, 'mb_detection_rate.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figura guardada: {path}')


def print_summary(df):
    """Imprime tabla resumen por nivel de ruido."""
    noise_order = ['trivial', 'facil', 'moderado', 'dificil', 'muy_dificil']

    n_ps_vals = df['n_pressure_sensors'].unique()
    n_ps_label = ', '.join(str(v) for v in sorted(n_ps_vals))

    print()
    print('══════════════════════════════════════════════════════')
    print('  RESULTADOS BASELINE: BALANCE DE MASA + NPW')
    print(f'  Sensores de presión: {n_ps_label}  |  Caudalímetros: 2 (fijos)')
    print('══════════════════════════════════════════════════════')
    print(f'  {"Noise level":<14s} {"Det.rate":>8s} {"x_err(km)":>10s} {"q_err(%)":>10s}')
    print(f'  {"─" * 14} {"─" * 8} {"─" * 10} {"─" * 10}')

    for noise in noise_order:
        sub = df[df['noise_level'] == noise]
        if sub.empty:
            continue

        # Solo escenarios con fuga para detection rate
        sub_leak = sub[sub['has_leak']]
        if len(sub_leak) > 0:
            det_rate = sub_leak['leak_detected'].mean() * 100.0
        else:
            det_rate = float('nan')

        # Errores solo sobre detecciones correctas de escenarios con fuga
        sub_detected = sub_leak[sub_leak['leak_detected']]
        if len(sub_detected) > 0 and sub_detected['x_leak_error_km'].notna().any():
            x_err = sub_detected['x_leak_error_km'].mean()
            x_err_str = f'{x_err:.3f} km'
        else:
            x_err_str = '  N/A'

        if len(sub_detected) > 0 and sub_detected['q_leak_error_pct'].notna().any():
            q_err = sub_detected['q_leak_error_pct'].mean()
            q_err_str = f'{q_err:.1f}%'
        else:
            q_err_str = '  N/A'

        print(f'  {noise:<14s} {det_rate:>7.0f}% {x_err_str:>10s} {q_err_str:>10s}')

    # Falsos positivos (escenarios sin fuga que detectan fuga)
    sub_noleak = df[~df['has_leak']]
    if len(sub_noleak) > 0:
        fp_rate = sub_noleak['leak_detected'].mean() * 100.0
        print(f'\n  Falsos positivos (sin fuga): {fp_rate:.0f}% '
              f'({sub_noleak["leak_detected"].sum()}/{len(sub_noleak)})')

    print('══════════════════════════════════════════════════════')
    print()


# ═══════════════════════════════════════════════════════════════
# BLOQUE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 1. Verificación rápida en un caso conocido
    print('─── Verificación caso base ───')
    result = run_mass_balance(
        scenario_id=8,      # x_leak=6000m, q_leak=0.015
        noise_level='trivial',
        n_pressure_sensors=3
    )
    print(f'  Leak detected: {result["leak_detected"]}')
    if result['x_leak_pred'] is not None:
        print(f'  x_leak pred:   {result["x_leak_pred"]:.0f} m  '
              f'(real: {result["x_leak_true"]:.0f} m)')
        print(f'  x_leak error:  {result["x_leak_error_km"]:.3f} km')
    else:
        print('  x_leak pred:   N/A (no se detectó fuga)')
    print(f'  q_leak pred:   {result["q_leak_pred"]:.4f} m³/s  '
          f'(real: {result["q_leak_true"]:.4f} m³/s)')
    if result['q_leak_error_pct'] is not None:
        print(f'  q_leak error:  {result["q_leak_error_pct"]:.1f}%')
    print(f'  Arrival times: {result["arrival_times"]}')
    print(f'  Pares NPW:    {result["n_pairs_used"]}')
    print('──────────────────────────────')

    # 2. Evaluación completa
    print('\n═══ Evaluación completa del baseline ═══')
    df = evaluate_mass_balance()

    # 3. Visualizaciones
    print('\n─── Generando figuras ───')
    plot_mass_balance_results(df)

    # 4. Resumen
    for n_ps in sorted(df['n_pressure_sensors'].unique()):
        print_summary(df[df['n_pressure_sensors'] == n_ps])

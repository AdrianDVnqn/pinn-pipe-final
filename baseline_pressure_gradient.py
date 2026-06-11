import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
from data_utils import get_training_data, list_scenarios

# ═══════════════════════════════════════════════════════════════
# PARTE 1 — EXTRACCIÓN DEL PERFIL ESPACIAL
# ═══════════════════════════════════════════════════════════════

def extract_steady_state_profile(t, P_noisy, dP_noisy, t_steady_start=120.0, t_steady_end=200.0):
    '''
    Promedia las lecturas de presión en la ventana
    [t_steady_start, t_steady_end] para obtener el
    perfil espacial en régimen cuasi-estacionario.
    
    Retorna:
      P_steady:    np.ndarray [n_sensors]  presión promedio
      P_std:       np.ndarray [n_sensors]  std en la ventana
      dP_steady:   np.ndarray [n_sensors]  desviación promedio
    '''
    mask = (t >= t_steady_start) & (t <= t_steady_end)
    P_steady  = np.mean(P_noisy[:, mask],  axis=1)
    P_std     = np.std(P_noisy[:, mask],   axis=1)
    dP_steady = np.mean(dP_noisy[:, mask], axis=1)
    return P_steady, P_std, dP_steady


# ═══════════════════════════════════════════════════════════════
# PARTE 2 — DETECCIÓN POR GRADIENTE
# ═══════════════════════════════════════════════════════════════

class PressureGradientDetector:
    def __init__(self,
                 f=cfg.FRICTION_FACTOR,
                 rho=cfg.FLUID_DENSITY,
                 D=cfg.PIPE_DIAMETER,
                 gradient_change_threshold=0.15):
        self.f = f
        self.rho = rho
        self.D = D
        self.threshold = gradient_change_threshold

    def compute_gradients(self, x_sensors, P_steady):
        gradients = np.diff(P_steady) / np.diff(x_sensors)
        x_midpoints = x_sensors[:-1] + np.diff(x_sensors) / 2.0
        return gradients, x_midpoints

    def compute_expected_gradient(self, Q_flow_meters):
        A = np.pi * self.D**2 / 4.0
        G_upstream   = -self.f * self.rho * Q_flow_meters[0]**2 / (2 * self.D * A**2)
        G_downstream = -self.f * self.rho * Q_flow_meters[1]**2 / (2 * self.D * A**2)
        return G_upstream, G_downstream

    def detect(self, x_sensors, P_steady, Q_steady_flow_meters):
        gradients, x_midpoints = self.compute_gradients(x_sensors, P_steady)
        
        G_upstream_obs = gradients[0]
        G_downstream_obs = gradients[-1]
        
        G_upstream_exp, G_downstream_exp = self.compute_expected_gradient(Q_steady_flow_meters)
        G_expected = G_upstream_exp

        if len(gradients) == 2:
            anomaly_score = abs(gradients[0] - gradients[1]) / (abs(np.mean(gradients)) + 1e-8)
        else:
            anomaly_score = np.std(gradients) / (abs(np.mean(gradients)) + 1e-8)

        leak_detected = anomaly_score > self.threshold

        return {
            "leak_detected":   bool(leak_detected),
            "anomaly_score":   float(anomaly_score),
            "gradients":       gradients,
            "x_midpoints":     x_midpoints,
            "G_expected":      float(G_expected),
            "G_upstream_obs":  float(G_upstream_obs),
            "G_downstream_obs":float(G_downstream_obs),
        }


# ═══════════════════════════════════════════════════════════════
# PARTE 3 — LOCALIZACIÓN POR INTERSECCIÓN DE GRADIENTES
# ═══════════════════════════════════════════════════════════════

class GradientLocalizer:
    def __init__(self, L=cfg.PIPE_LENGTH, a=cfg.WAVE_SPEED,
                 f=cfg.FRICTION_FACTOR, rho=cfg.FLUID_DENSITY,
                 D=cfg.PIPE_DIAMETER):
        self.L = L
        self.a = a
        self.f = f
        self.rho = rho
        self.D = D

    def estimate_gradients_from_flow(self, Q_in, Q_out):
        A = np.pi * self.D**2 / 4.0
        G1 = -self.f * self.rho * Q_in**2 / (2 * self.D * A**2)
        G2 = -self.f * self.rho * Q_out**2 / (2 * self.D * A**2)
        return G1, G2

    def localize_intersection(self, x_sensors, P_steady, Q_steady_flow_meters):
        G1, G2 = self.estimate_gradients_from_flow(Q_steady_flow_meters[0], Q_steady_flow_meters[1])
        
        # Extrapolar desde el primer sensor para obtener P en x=0
        P_in_est = P_steady[0] - G1 * x_sensors[0]
        # Extrapolar desde el último sensor para obtener P en x=L
        P_out_est = P_steady[-1] - G2 * (x_sensors[-1] - self.L)
        
        def P_line_upstream(x):
            return P_in_est + G1 * x
            
        def P_line_downstream(x):
            return P_out_est + G2 * (x - self.L)

        if abs(G1 - G2) < 1e-8:
            x_leak_est = self.L / 2.0
        else:
            x_leak_est = (P_out_est - P_in_est - G2 * self.L) / (G1 - G2)

        method_used = "intersection"
        if not (0 <= x_leak_est <= self.L):
            gradients = np.diff(P_steady) / np.diff(x_sensors)
            x_leak_est = self.localize_kink(x_sensors, P_steady, gradients)
            method_used = "fallback"

        return {
            "x_leak_est":      float(x_leak_est),
            "method_used":     method_used,
            "G1_est":          float(G1),
            "G2_est":          float(G2),
            "P_line_upstream": P_line_upstream,
            "P_line_downstream": P_line_downstream
        }

    def localize_kink(self, x_sensors, P_steady, gradients):
        diffs = np.abs(np.diff(gradients))
        if len(diffs) > 0:
            kink_idx = np.argmax(diffs) + 1
            return x_sensors[kink_idx]
        else:
            return x_sensors[0]


# ═══════════════════════════════════════════════════════════════
# PARTE 4 — CUANTIFICACIÓN
# ═══════════════════════════════════════════════════════════════

def estimate_q_leak_from_flow(t, Q_noisy_flow_meters, t_steady_start=120.0):
    mask = t >= t_steady_start
    if not np.any(mask):
        mask = t >= 0
    Q_in_mean = np.mean(Q_noisy_flow_meters[0, mask])
    Q_out_mean = np.mean(Q_noisy_flow_meters[1, mask])
    q_leak_est = Q_in_mean - Q_out_mean
    return q_leak_est, Q_in_mean, Q_out_mean


# ═══════════════════════════════════════════════════════════════
# PARTE 5 — FUNCIÓN PRINCIPAL DE INFERENCIA
# ═══════════════════════════════════════════════════════════════

def run_pressure_gradient(scenario_id,
                           noise_level="trivial",
                           n_pressure_sensors=3,
                           t_steady_start=120.0,
                           gradient_change_threshold=0.15,
                           localization_method="intersection"):
    
    data = get_training_data(scenario_id, noise_level, n_pressure_sensors)
    
    t = data['t']
    x_p_sensors = data['x_pressure_sensors_used']
    P_noisy = data['P_noisy']
    dP_noisy = data['dP_noisy']
    Q_noisy = data['Q_noisy']
    
    x_leak_true = data['x_leak']
    q_leak_true = data['q_leak']
    has_leak = data['has_leak']

    P_steady, P_std, dP_steady = extract_steady_state_profile(
        t, P_noisy, dP_noisy, t_steady_start=t_steady_start
    )
    
    mask_steady = t >= t_steady_start
    Q_steady_flow_meters = np.mean(Q_noisy[:, mask_steady], axis=1)
    
    detector = PressureGradientDetector(gradient_change_threshold=gradient_change_threshold)
    det_result = detector.detect(x_p_sensors, P_steady, Q_steady_flow_meters)
    
    leak_detected = det_result["leak_detected"]
    
    t_detection = t_steady_start if leak_detected else None
    x_leak_pred = None
    x_leak_error_km = None
    
    localizer = GradientLocalizer()
    loc_result = localizer.localize_intersection(
        x_p_sensors, P_steady, Q_steady_flow_meters
    )
    
    if leak_detected:
        x_leak_pred = loc_result["x_leak_est"]
        if has_leak and x_leak_pred is not None:
            x_leak_error_km = abs(x_leak_pred - x_leak_true) / 1000.0
            
    q_leak_pred, Q_in_mean, Q_out_mean = estimate_q_leak_from_flow(
        t, Q_noisy, t_steady_start=t_steady_start
    )
    q_leak_error_pct = None
    
    if has_leak and leak_detected and q_leak_true > 0:
        q_leak_error_pct = abs(q_leak_pred - q_leak_true) / q_leak_true * 100.0
        
    return {
        "leak_detected":      leak_detected,
        "t_detection":        t_detection,
        "x_leak_pred":        x_leak_pred,
        "x_leak_true":        x_leak_true if has_leak else None,
        "x_leak_error_km":    x_leak_error_km,
        "q_leak_pred":        q_leak_pred,
        "q_leak_true":        q_leak_true,
        "q_leak_error_pct":   q_leak_error_pct,
        "method":             "pressure_gradient",
        "scenario_id":        scenario_id,
        "noise_level":        noise_level,
        "n_pressure_sensors": n_pressure_sensors,
        "has_leak":           has_leak,
        
        # Para diagnóstico y plots
        "gradients":          det_result["gradients"],
        "anomaly_score":      det_result["anomaly_score"],
        "G_expected":         det_result["G_expected"],
        "G_upstream_obs":     det_result["G_upstream_obs"],
        "G_downstream_obs":   det_result["G_downstream_obs"],
        "G1_est":             loc_result["G1_est"],
        "G2_est":             loc_result["G2_est"],
        "P_steady":           P_steady,
        "P_std":              P_std,
        "x_p_sensors":        x_p_sensors,
        "loc_result":         loc_result,
        "t":                  t,
        "P_noisy":            P_noisy,
        "Q_noisy":            Q_noisy,
        "dP_noisy":           dP_noisy,
        "x_midpoints":        det_result["x_midpoints"]
    }


# ═══════════════════════════════════════════════════════════════
# PARTE 6 — EVALUACIÓN EN TODO EL DATASET
# ═══════════════════════════════════════════════════════════════

def evaluate_pressure_gradient(noise_levels=None, n_p_sensors_list=None):
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
                    f'  Corriendo PG baseline | noise={noise:<12s} '
                    f'| n_p_sensors={n_ps} '
                    f'| escenario {sid:>2d} '
                    f'({counter}/{total})'
                )

                result = run_pressure_gradient(
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
                    'method': 'pressure_gradient',
                })

    df = pd.DataFrame(rows)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(cfg.RESULTS_DIR, 'baseline_pressure_gradient.csv')
    df.to_csv(csv_path, index=False)
    print(f'\n  Resultados guardados en: {csv_path}')

    return df


# ═══════════════════════════════════════════════════════════════
# PARTE 7 — VISUALIZACIÓN Y MÉTRICAS
# ═══════════════════════════════════════════════════════════════

def plot_pressure_gradient_results(df, save_dir=None):
    if save_dir is None:
        save_dir = cfg.FIGS_DIR
    os.makedirs(save_dir, exist_ok=True)

    _plot_profile_example(save_dir)
    _plot_error_vs_noise(df, save_dir)
    _plot_detection_rate_heatmap(df, save_dir)


def _plot_profile_example(save_dir):
    result = run_pressure_gradient(scenario_id=8, noise_level="moderado", n_pressure_sensors=3)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(
        'Baseline 2: Gradiente de Presión — Ejemplo Visual\n'
        '(Escenario 8: x_leak=6000m, ruido=moderado, 3 sensores de presión)',
        fontsize=13, fontweight='bold'
    )
    
    # a) Perfil espacial de presión
    ax1 = axes[0]
    x_p = result['x_p_sensors']
    P_st = result['P_steady']
    P_std = result['P_std']
    
    ax1.errorbar(x_p, P_st, yerr=P_std, fmt='o', color='k', capsize=5, label='Presión media medida')
    
    if result['leak_detected'] and result['x_leak_pred'] is not None:
        xx1 = np.linspace(0, result['x_leak_pred'], 100)
        xx2 = np.linspace(result['x_leak_pred'], float(cfg.PIPE_LENGTH), 100)
        ax1.plot(xx1, result['loc_result']['P_line_upstream'](xx1), 'b-', label='Gradiente upstream estimado')
        ax1.plot(xx2, result['loc_result']['P_line_downstream'](xx2), 'r-', label='Gradiente downstream estimado')
        
        ax1.axvline(result['x_leak_pred'], color='red', linestyle='--', label=f'Fuga estimada ({result["x_leak_pred"]:.0f}m)')
        ax1.axvline(result['x_leak_true'], color='green', linestyle=':', label=f'Fuga real ({result["x_leak_true"]:.0f}m)')
        
    ax1.set_xlabel('Posición x [m]')
    ax1.set_ylabel('Presión [Pa]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # b) Gradientes por segmento
    ax2 = axes[1]
    x_mid = result['x_midpoints']
    grads = result['gradients']
    
    ax2.bar(x_mid, grads, width=2000, alpha=0.6, label='Gradiente observado')
    ax2.axhline(result['G_expected'], color='gray', linestyle='--', label='Gradiente sin fuga (esperado)')
    ax2.set_xlabel('Posición de segmento [m]')
    ax2.set_ylabel('Gradiente [Pa/m]')
    ax2.set_title(f'Anomaly score: {result["anomaly_score"]:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # c) Evolución temporal de dP
    ax3 = axes[2]
    t = result['t']
    dP = result['dP_noisy']
    for i in range(dP.shape[0]):
        ax3.plot(t, dP[i, :], label=f'P sensor @ {x_p[i]:.0f}m', alpha=0.7)
    
    ax3.axvline(120.0, color='purple', linestyle='--', label='t_steady_start')
    ax3.set_xlabel('Tiempo [s]')
    ax3.set_ylabel('ΔP [Pa]')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'pg_profile_example.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figura guardada: {path}')

def _plot_error_vs_noise(df, save_dir):
    noise_order = ['trivial', 'facil', 'moderado', 'dificil', 'muy_dificil']
    df_ok = df[(df['has_leak']) & (df['leak_detected']) & (df['x_leak_error_km'].notna())].copy()

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
            ax.plot(valid_noises, means, marker=markers.get(ns, 'o'), lw=2, ms=8, label=f'{ns} sensores presión')

    ax.set_xlabel('Nivel de Ruido', fontsize=12)
    ax.set_ylabel('Error medio de localización  [km]', fontsize=12)
    ax.set_title('Baseline Pressure Gradient: Error de localización vs Ruido', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'pg_error_vs_noise.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figura guardada: {path}')

def _plot_detection_rate_heatmap(df, save_dir):
    noise_order = ['trivial', 'facil', 'moderado', 'dificil', 'muy_dificil']
    df_leak = df[df['has_leak']].copy()
    
    if df_leak.empty:
        return

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

    for i in range(len(available_sizes)):
        for j in range(len(available_noises)):
            val = rate_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 50 else 'black'
                ax.text(j, i, f'{val:.0f}%', ha='center', va='center', fontsize=12, fontweight='bold', color=color)

    ax.set_xlabel('Nivel de Ruido', fontsize=12)
    ax.set_ylabel('Tamaño de Fuga', fontsize=12)
    ax.set_title('Baseline Pressure Gradient: Tasa de Detección (%)', fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Tasa de detección (%)')

    plt.tight_layout()
    path = os.path.join(save_dir, 'pg_detection_rate.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Figura guardada: {path}')

def print_summary(df):
    noise_order = ['trivial', 'facil', 'moderado', 'dificil', 'muy_dificil']
    n_ps_vals = df['n_pressure_sensors'].unique()
    n_ps_label = ', '.join(str(v) for v in sorted(n_ps_vals))

    print()
    print('══════════════════════════════════════════════════════')
    print('  RESULTADOS BASELINE: GRADIENTE DE PRESIÓN')
    print(f'  Sensores de presión: {n_ps_label}  |  Caudalímetros: 2 (fijos)')
    print('══════════════════════════════════════════════════════')
    print(f'  {"Noise level":<14s} {"Det.rate":>8s} {"x_err(km)":>10s} {"q_err(%)":>10s}')
    print(f'  {"─" * 14} {"─" * 8} {"─" * 10} {"─" * 10}')

    for noise in noise_order:
        sub = df[df['noise_level'] == noise]
        if sub.empty: continue

        sub_leak = sub[sub['has_leak']]
        if len(sub_leak) > 0:
            det_rate = sub_leak['leak_detected'].mean() * 100.0
        else:
            det_rate = float('nan')

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

    sub_noleak = df[~df['has_leak']]
    if len(sub_noleak) > 0:
        fp_rate = sub_noleak['leak_detected'].mean() * 100.0
        print(f'\n  Falsos positivos (sin fuga): {fp_rate:.0f}% ({sub_noleak["leak_detected"].sum()}/{len(sub_noleak)})')
    print('══════════════════════════════════════════════════════\n')


# ═══════════════════════════════════════════════════════════════
# PARTE 8 — BLOQUE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 1. Verificación caso base
    print("─── Verificación caso base ───")
    result = run_pressure_gradient(
        scenario_id=8,
        noise_level="trivial",
        n_pressure_sensors=3,
    )
    print(f"Leak detected:  {result['leak_detected']}")
    if result['leak_detected']:
        print(f"x_leak pred:    {result['x_leak_pred']:.0f} m (real: {result['x_leak_true']:.0f} m)")
        print(f"x_leak error:   {result['x_leak_error_km']:.3f} km")
    else:
        print("x_leak pred:    N/A")
    print(f"q_leak pred:    {result['q_leak_pred']:.4f} m³/s (real: {result['q_leak_true']:.4f} m³/s)")
    print(f"Anomaly score:  {result['anomaly_score']:.3f}")
    print("──────────────────────────────")

    # 2. Evaluación completa
    df = evaluate_pressure_gradient()

    # 3. Visualizaciones
    plot_pressure_gradient_results(df)

    # 4. Resumen
    print_summary(df[df["n_pressure_sensors"] == 3])
    print_summary(df[df["n_pressure_sensors"] == 2])

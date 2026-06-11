import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

import config as cfg
from pinn_model import train_pinn
from data_utils import get_training_data

# ──────────────────────────────────────────
# PARTE 1 — DEFINICIÓN DEL GRID EXPERIMENTAL
# ──────────────────────────────────────────

ESCENARIOS_CON_FUGA = list(range(1, 13))   # scenario_id 1 a 12
ESCENARIO_SIN_FUGA  = [0]                  # para tasa de falsos positivos

NOISE_LEVELS = ["trivial", "facil", "moderado",
                "dificil", "muy_dificil"]

N_SENSORS_LIST = [2, 3]                    # sensores de presión

# Total de corridas de PINN:
# 12 escenarios × 5 niveles × 2 configs = 120 runs (con fuga)
# + 1 escenario (0) × 5 niveles × 2 configs = 10 runs (sin fuga)
# Total = 130 runs

# ──────────────────────────────────────────
# PARTE 2 — CORRIDAS DE LA PINN
# ──────────────────────────────────────────

PINN_FACTORIAL_CONFIG = {
    "n_epochs":       5_000,
    "n_collocation":  10_000,
    "lr":             1e-3,
    "verbose":        False,
    # skip LBFGS for speed in factorial or use small amount
    "use_lbfgs":      True,
    "lbfgs_epochs":   500,
}

def run_pinn_factorial(output_path="results/pinn_factorial.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Manejo de interrupciones (resume)
    done_runs = set()
    if os.path.exists(output_path):
        df_done = pd.read_csv(output_path)
        for _, row in df_done.iterrows():
            done_runs.add((int(row['scenario_id']), str(row['noise_level']), int(row['n_sensors'])))
    else:
        # Create CSV header
        pd.DataFrame(columns=[
            'scenario_id', 'has_leak', 'x_leak_true', 'q_leak_true',
            'leak_size', 'noise_level', 'noise_std', 'n_sensors',
            'leak_detected', 'x_leak_pred', 'x_leak_error_km',
            'q_leak_pred', 'q_leak_error_pct',
            't_detection', 'training_time_s', 'method'
        ]).to_csv(output_path, index=False)

    grid = []
    for s_id in ESCENARIOS_CON_FUGA + ESCENARIO_SIN_FUGA:
        for noise in NOISE_LEVELS:
            for n_sens in N_SENSORS_LIST:
                grid.append((s_id, noise, n_sens))
    
    total_runs = len(grid)
    run_times = []
    
    for i, (s_id, noise, n_sens) in enumerate(grid, 1):
        if (s_id, noise, n_sens) in done_runs:
            print(f"PINN | noise={noise} | sensors={n_sens} | escenario {s_id} | run {i}/{total_runs} | Saltando (ya completada)")
            continue
            
        eta_str = "Calculando..."
        if run_times:
            avg_time = sum(run_times[-5:]) / len(run_times[-5:])
            eta_s = avg_time * (total_runs - i + 1)
            eta_str = f"{eta_s/3600:.1f} h"
            
        print(f"PINN | noise={noise} | sensors={n_sens} | escenario {s_id} | run {i}/{total_runs} | ETA: {eta_str}")
        
        t0 = time.time()
        
        try:
            # Extraer metadata de data_utils para guardar en CSV
            data_meta = get_training_data(s_id, noise, n_sens)
            has_leak = data_meta['has_leak']
            leak_size = data_meta['leak_size']
            x_leak_true = data_meta['x_leak'] if has_leak else 0.0
            q_leak_true = data_meta['q_leak'] if has_leak else 0.0
            noise_std = data_meta['noise_std']
            
            # Correr entrenamiento PINN
            result = train_pinn(
                scenario_id=s_id,
                noise_level=noise,
                n_pressure_sensors=n_sens,
                n_epochs=PINN_FACTORIAL_CONFIG["n_epochs"],
                n_collocation=PINN_FACTORIAL_CONFIG["n_collocation"],
                lr=PINN_FACTORIAL_CONFIG["lr"],
                verbose=PINN_FACTORIAL_CONFIG["verbose"],
                use_lbfgs=PINN_FACTORIAL_CONFIG["use_lbfgs"],
                lbfgs_epochs=PINN_FACTORIAL_CONFIG["lbfgs_epochs"],
                progress_every=1000 # reduce logs
            )
            
            x_leak_pred = result['x_leak_pred']
            q_leak_pred = result['q_leak_pred']
            training_time_s = result['training_time_s']
            
            # Error calcs
            if has_leak:
                x_err_km = abs(x_leak_pred - x_leak_true) / 1000.0
                q_err_pct = abs(q_leak_pred - q_leak_true) / max(q_leak_true, 1e-6) * 100.0
            else:
                x_err_km = np.nan
                q_err_pct = np.nan
            
            leak_detected = q_leak_pred >= 0.003
            t_detection = np.nan # PINN no tiene un "tiempo de detección" secuencial
            
            row = pd.DataFrame([{
                'scenario_id': s_id,
                'has_leak': has_leak,
                'x_leak_true': x_leak_true,
                'q_leak_true': q_leak_true,
                'leak_size': leak_size,
                'noise_level': noise,
                'noise_std': noise_std,
                'n_sensors': n_sens,
                'leak_detected': leak_detected,
                'x_leak_pred': x_leak_pred,
                'x_leak_error_km': x_err_km,
                'q_leak_pred': q_leak_pred,
                'q_leak_error_pct': q_err_pct,
                't_detection': t_detection,
                'training_time_s': training_time_s,
                'method': 'pinn'
            }])
            
            row.to_csv(output_path, mode='a', header=False, index=False)
            
            run_time = time.time() - t0
            run_times.append(run_time)
            
        except Exception as e:
            print(f"  -> Error en run {i}: {e}")
            continue

# ──────────────────────────────────────────
# PARTE 3 — COMBINAR TODOS LOS RESULTADOS
# ──────────────────────────────────────────

def build_master_dataframe():
    files = [
        "results/baseline_mass_balance.csv",
        "results/baseline_pressure_gradient.csv",
        "results/baseline_lstm.csv",
        "results/pinn_factorial.csv"
    ]
    
    dfs = []
    for f in files:
        if not os.path.exists(f):
            print(f"Error: Falta {f}. Por favor, corre el script correspondiente.")
            sys.exit(1)
        dfs.append(pd.read_csv(f))
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Agregar columna noise_std_db (normalizada respecto a 500 Pa)
    df['noise_std_db'] = 20 * np.log10(np.clip(df['noise_std'], 1e-5, None) / 500.0)
    
    df.to_csv("results/master_results.csv", index=False)
    return df

# ──────────────────────────────────────────
# PARTE 4 — MÉTRICAS AGREGADAS
# ──────────────────────────────────────────

def compute_aggregate_metrics(df):
    results = []
    methods = df['method'].unique()
    noises = df['noise_level'].unique()
    sensors = df['n_sensors'].unique()
    
    for m in methods:
        for n in noises:
            for s in sensors:
                sub = df[(df['method'] == m) & (df['noise_level'] == n) & (df['n_sensors'] == s)]
                if sub.empty:
                    continue
                
                sub_leak = sub[sub['has_leak'] == True]
                sub_noleak = sub[sub['has_leak'] == False]
                
                det_rate = 0.0
                if not sub_leak.empty:
                    det_rate = (sub_leak['leak_detected'].sum() / len(sub_leak)) * 100.0
                    
                fp_rate = 0.0
                if not sub_noleak.empty:
                    fp_rate = (sub_noleak['leak_detected'].sum() / len(sub_noleak)) * 100.0
                    
                correct_dets = sub_leak[sub_leak['leak_detected'] == True]
                
                x_mean = correct_dets['x_leak_error_km'].mean() if not correct_dets.empty else np.nan
                x_std = correct_dets['x_leak_error_km'].std() if len(correct_dets) > 1 else 0.0
                x_median = correct_dets['x_leak_error_km'].median() if not correct_dets.empty else np.nan
                q_mean_pct = correct_dets['q_leak_error_pct'].mean() if not correct_dets.empty else np.nan
                
                inf_time = sub['training_time_s'].mean() if 'training_time_s' in sub.columns else np.nan
                
                results.append({
                    'method': m,
                    'noise_level': n,
                    'n_sensors': s,
                    'detection_rate': det_rate,
                    'false_positive_rate': fp_rate,
                    'x_error_mean_km': x_mean,
                    'x_error_std_km': x_std,
                    'x_error_median_km': x_median,
                    'q_error_mean_pct': q_mean_pct,
                    'inference_time_s': inf_time
                })
                
    agg_df = pd.DataFrame(results)
    
    # Calcular degradation factor = x_error_muy_dificil / x_error_trivial
    agg_df['degradation_factor'] = np.nan
    for m in methods:
        for s in sensors:
            m_s_df = agg_df[(agg_df['method'] == m) & (agg_df['n_sensors'] == s)]
            if not m_s_df.empty:
                err_triv = m_s_df[m_s_df['noise_level'] == 'trivial']['x_error_mean_km']
                err_mdif = m_s_df[m_s_df['noise_level'] == 'muy_dificil']['x_error_mean_km']
                if not err_triv.empty and not err_mdif.empty:
                    val_triv = err_triv.values[0]
                    val_mdif = err_mdif.values[0]
                    if pd.notna(val_triv) and val_triv > 0:
                        deg_factor = val_mdif / val_triv
                        agg_df.loc[(agg_df['method'] == m) & (agg_df['n_sensors'] == s), 'degradation_factor'] = deg_factor
    
    agg_df.to_csv("results/aggregate_metrics.csv", index=False)
    return agg_df

# ──────────────────────────────────────────
# PARTE 5 — TABLA RESUMEN PARA LA TESIS
# ──────────────────────────────────────────

def print_master_summary(agg_df, n_sensors=3):
    print(f"")
    print(f"══════════════════════════════════════════════════════")
    print(f"EXPERIMENTO FACTORIAL — RESULTADOS COMPARATIVOS")
    print(f"Sensores de presión: {n_sensors}")
    print(f"══════════════════════════════════════════════════════")
    
    print("\nError de localización x_leak [km]")
    print("┌─────────────────┬─────────┬───────┬──────────┬──────────┬─────────────┐")
    print("│ Método          │ Trivial │ Fácil │ Moderado │ Difícil  │ Muy difícil │")
    print("├─────────────────┼─────────┼───────┼──────────┼──────────┼─────────────┤")
    
    noises = ["trivial", "facil", "moderado", "dificil", "muy_dificil"]
    methods = {"pinn": "PINN", "mass_balance_npw": "Balance masa", "pressure_gradient": "Grad. presión", "lstm": "LSTM"}
    
    for m_key, m_name in methods.items():
        row_str = f"│ {m_name:<15} │"
        for n in noises:
            val = agg_df[(agg_df['method'] == m_key) & (agg_df['n_sensors'] == n_sensors) & (agg_df['noise_level'] == n)]['x_error_mean_km']
            if not val.empty and pd.notna(val.values[0]):
                row_str += f" {val.values[0]:5.3f} │"
            else:
                row_str += "   N/A │"
        print(row_str)
    print("└─────────────────┴─────────┴───────┴──────────┴──────────┴─────────────┘")

    print("\nTasa de detección [%]")
    print("┌─────────────────┬─────────┬───────┬──────────┬──────────┬─────────────┐")
    print("│ Método          │ Trivial │ Fácil │ Moderado │ Difícil  │ Muy difícil │")
    print("├─────────────────┼─────────┼───────┼──────────┼──────────┼─────────────┤")
    
    for m_key, m_name in methods.items():
        row_str = f"│ {m_name:<15} │"
        for n in noises:
            val = agg_df[(agg_df['method'] == m_key) & (agg_df['n_sensors'] == n_sensors) & (agg_df['noise_level'] == n)]['detection_rate']
            if not val.empty and pd.notna(val.values[0]):
                row_str += f"  {val.values[0]:3.0f}% │"
            else:
                row_str += "   N/A │"
        print(row_str)
    print("└─────────────────┴─────────┴───────┴──────────┴──────────┴─────────────┘")
    
    print("\nFactor de degradación (x_err muy_dificil / x_err trivial):")
    for m_key, m_name in methods.items():
        val = agg_df[(agg_df['method'] == m_key) & (agg_df['n_sensors'] == n_sensors)]['degradation_factor'].iloc[0]
        if pd.notna(val):
            print(f"  {m_name:<15}: {val:4.1f}x")
        else:
            print(f"  {m_name:<15}: N/A")
            
    print(f"══════════════════════════════════════════════════════")

# ──────────────────────────────────────────
# PARTE 6 — BLOQUE PRINCIPAL
# ──────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-pinn', action='store_true',
        help='Saltar corridas de PINN (usar CSV existente)')
    parser.add_argument('--skip-baselines', action='store_true',
        help='Saltar re-corrida de baselines')
    args = parser.parse_args()

    # 1. Correr PINN (puede tomar horas)
    if not args.skip_pinn:
        print("═══ CORRIENDO PINN FACTORIAL ═══")
        print("Esto puede tomar ~8 horas. Dejar correr overnight.")
        print("El progreso se guarda automáticamente.")
        print("Para resumir si se interrumpe: correr el script de nuevo.")
        run_pinn_factorial()
    else:
        print("Saltando PINN — usando results/pinn_factorial.csv")

    # 2. Re-correr baselines si se pide
    if not args.skip_baselines:
        for csv_f in ["results/baseline_mass_balance.csv",
                    "results/baseline_pressure_gradient.csv",
                    "results/baseline_lstm.csv"]:
            if not os.path.exists(csv_f):
                print(f"Falta {csv_f}. Correr el baseline correspondiente.")
                sys.exit(1)

    # 3. Combinar resultados
    print("═══ COMBINANDO RESULTADOS ═══")
    df = build_master_dataframe()
    print(f"Master DataFrame: {len(df)} filas × {len(df.columns)} columnas")

    # 4. Métricas agregadas
    agg = compute_aggregate_metrics(df)

    # 5. Tablas resumen
    print_master_summary(agg, n_sensors=3)
    print_master_summary(agg, n_sensors=2)

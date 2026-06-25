"""
benchmark_wave_pinn.py -- Benchmark automatizado para WaveLeakPINN

Itera sobre distintas ubicaciones de fuga (x_leak), tamaños de fuga (q_leak),
y niveles de ruido en los sensores (noise_std) definidos en config.py.
Guarda los resultados en 'benchmark_results.csv'.
"""

import os
import time
import pandas as pd
import torch

from wave_pinn import generate_data, train, L_PIPE

# ── Configuraciones a iterar ─────────────────────────
X_LEAKS = [2000.0, 4000.0, 6000.0, 8000.0]
Q_LEAKS = [0.005, 0.015, 0.030]
NOISES = [0.0, 8000.0, 50000.0]  # Sin ruido, moderado, muy difícil
EPOCHS = 4000

RESULTS_FILE = "benchmark_results.csv"

def run_benchmark():
    print("=" * 60)
    print("  INICIANDO BENCHMARK WAVE-PINN (Q_LEAK LEARNABLE)")
    print("=" * 60)
    
    # Check if we resume
    if os.path.exists(RESULTS_FILE):
        df_existing = pd.read_csv(RESULTS_FILE)
        results = df_existing.to_dict('records')
        print(f"[*] Encontrado archivo de resultados previo con {len(results)} corridas.")
    else:
        results = []
        df_existing = pd.DataFrame()

    total_runs = len(X_LEAKS) * len(Q_LEAKS) * len(NOISES)
    print(f"[*] Total de configuraciones a evaluar: {total_runs}")
    
    run_idx = 0
    
    for q_true in Q_LEAKS:
        for x_true in X_LEAKS:
            for noise in NOISES:
                run_idx += 1
                
                # Check if already done
                if not df_existing.empty:
                    already_run = df_existing[
                        (df_existing['true_q_leak'] == q_true) &
                        (df_existing['true_x_leak'] == x_true) &
                        (df_existing['noise_pa'] == noise)
                    ]
                    if not already_run.empty:
                        print(f"[{run_idx}/{total_runs}] Skip x={x_true}, q={q_true}, noise={noise} (Ya procesado)")
                        continue
                
                print(f"\n[{run_idx}/{total_runs}] Evaluando -> x_leak: {x_true}m | q_leak: {q_true} m3/s | Noise: {noise} Pa")
                
                t0 = time.time()
                
                try:
                    # Generar datos
                    data = generate_data(x_true, q_true, noise_std=noise)
                    
                    # Entrenar PINN
                    model, history = train(data, q_leak_true=q_true, n_epochs=EPOCHS)
                    
                    x_pred = model.x_leak.item()
                    q_pred = model.q_leak.item()
                    
                    err_x = abs(x_pred - x_true)
                    err_q = abs(q_pred - q_true)
                    err_x_pct = err_x / L_PIPE * 100
                    err_q_pct = err_q / q_true * 100
                    
                    elapsed = time.time() - t0
                    
                    print(f"  [RESULTADO] x_pred: {x_pred:.0f}m (Err: {err_x:.0f}m) | q_pred: {q_pred:.4f} (Err: {err_q_pct:.1f}%) | {elapsed:.1f}s")
                    
                    # Guardar fila
                    res = {
                        'true_x_leak': x_true,
                        'true_q_leak': q_true,
                        'noise_pa': noise,
                        'pred_x_leak': x_pred,
                        'pred_q_leak': q_pred,
                        'err_x_m': err_x,
                        'err_x_pct': err_x_pct,
                        'err_q_abs': err_q,
                        'err_q_pct': err_q_pct,
                        'time_s': elapsed
                    }
                    results.append(res)
                    
                    # Guardar a CSV inmediatamente para no perder datos si crashea
                    df = pd.DataFrame(results)
                    df.to_csv(RESULTS_FILE, index=False)
                    
                except Exception as e:
                    print(f"  [ERROR] Falló la corrida: {e}")

    print("\n" + "=" * 60)
    print("  BENCHMARK FINALIZADO")
    print(f"  Resultados guardados en: {RESULTS_FILE}")
    print("=" * 60)

if __name__ == '__main__':
    run_benchmark()

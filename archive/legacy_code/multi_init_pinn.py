import json
import time
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import concurrent.futures
import torch

from perfect_pinn_model import train_pinn

SCENARIO_ID = 6  # Fuga verdadera en 4000m
X_TRUE = 4000.0
Q_TRUE = 0.03

INITIALIZATIONS = [2000.0, 4000.0, 6000.0, 8000.0]

def run_single_init(x_init):
    print(f"[{x_init}m] Iniciando entrenamiento de 10000 epochs...")
    try:
        train_result = train_pinn(
            scenario_id=SCENARIO_ID,
            noise_level="trivial",
            n_pressure_sensors=3,
            n_epochs=10000,
            verbose=False,
            initial_x_leak=x_init
        )
        
        model = train_result["model"]
        history = train_result["history"].to_dict('records')
        
        x_leak_pred = float(model.x_leak.item())
        q_leak_pred = float(model.q_leak.item())
        
        x_err_m = abs(x_leak_pred - X_TRUE)
        q_err_pct = abs(q_leak_pred - Q_TRUE) / Q_TRUE * 100.0
        
        final_loss = history[-1]['loss_total']
        loss_data = history[-1]['L_P'] + history[-1]['L_Q']
        
        return {
            "x_init": x_init,
            "x_leak_pred": x_leak_pred,
            "x_err_m": x_err_m,
            "q_leak_pred": q_leak_pred,
            "q_err_pct": q_err_pct,
            "final_loss": final_loss,
            "loss_data": loss_data,
            "history": history
        }
    except Exception as e:
        print(f"Error en init {x_init}: {str(e)}")
        return None

def plot_multi_init(results, save_dir="figs"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Trajectories
    ax_x = axs[0]
    for r in results:
        hist = r["history"]
        epochs = [h["epoch"] for h in hist]
        x_pred = [h["x_leak_pred"] for h in hist]
        ax_x.plot(epochs, x_pred, label=f"Init: {r['x_init']}m (Final: {r['x_leak_pred']:.0f}m)")
        
    ax_x.axhline(X_TRUE, color="red", linestyle="--", linewidth=2, label=f"x_leak True ({X_TRUE}m)")
    ax_x.set_title("Trayectorias de x_leak para múltiples inicializaciones")
    ax_x.set_xlabel("Epoch")
    ax_x.set_ylabel("x_leak [m]")
    ax_x.legend()
    
    # Right: Data Loss over time
    ax_loss = axs[1]
    for r in results:
        hist = r["history"]
        epochs = [h["epoch"] for h in hist]
        loss_data = [h["L_P"] + h["L_Q"] for h in hist]
        ax_loss.plot(epochs, loss_data, label=f"Init: {r['x_init']}m (Final Loss: {loss_data[-1]:.2e})")
        
    ax_loss.set_yscale("log")
    ax_loss.set_title("Evolución del Loss de Datos (Sensores)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("L_P + L_Q")
    ax_loss.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "multi_init_trajectories.png")
    plt.savefig(save_path, dpi=150)
    print(f"Plot guardado en: {save_path}")
    plt.close()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    print("  ═══════════════════════════════════════════════")
    print("  PRUEBA ALTERNATIVA: MULTI-INICIALIZACIÓN")
    print("  ═══════════════════════════════════════════════")
    print(f"Escenario: {SCENARIO_ID} (x_leak real = {X_TRUE}m)")
    print(f"Lanzando {len(INITIALIZATIONS)} inicializaciones en paralelo...")
    
    start_time = time.time()
    
    results = []
    # Run in parallel using ProcessPoolExecutor
    # Use max_workers=2 or 4. 4 should fit in 12.9GB VRAM if each takes ~1-2GB
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_single_init, x): x for x in INITIALIZATIONS}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
                
    elapsed = time.time() - start_time
    print(f"\nEntrenamiento paralelo finalizado en {elapsed/60:.1f} min.\n")
    
    # Sort by initial x_leak for printing
    results.sort(key=lambda r: r['x_init'])
    
    print("  RESULTADOS POR INICIALIZACIÓN:")
    print("  Init (m) | x_pred (m) | Error (m) | q_pred     | L_datos final")
    print("  ─────────┼────────────┼───────────┼────────────┼──────────────")
    for r in results:
        print(f"  {r['x_init']:<8.0f} | {r['x_leak_pred']:<10.0f} | {r['x_err_m']:<9.0f} | {r['q_leak_pred']:.6f} | {r['loss_data']:.2e}")
        
    # Find the winner (minimum data loss)
    # The PDE loss might be low if it's hiding at the boundary, so we trust DATA loss to find the true leak!
    winner = min(results, key=lambda r: r['loss_data'])
    
    print("\n  =======================================================")
    print(f"  🏆 GANADOR (Menor Loss de Datos): Inicialización en {winner['x_init']}m")
    print(f"      Predicción Final: {winner['x_leak_pred']:.0f}m")
    print(f"      Error contra real: {winner['x_err_m']:.0f}m")
    print("  =======================================================\n")
    
    plot_multi_init(results)
    
    print("Prueba completada.")

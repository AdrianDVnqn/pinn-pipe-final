import json
import time
import os
import matplotlib.pyplot as plt
import numpy as np

from xpinn_model import train_xpinn

TEST_SCENARIOS = [
    {
        "scenario_id": 6,
        "descripcion": "Fuga moderada al medio (Escenario Sencillo)",
        "x_leak_true": 6000,
        "q_leak_true": 0.015,
        "dificultad": "FACIL",
    }
]

def run_single_test(scenario, noise_level="moderado", use_npw_init=True):
    print(f"Entrenando... (10000 epochs)")
    
    # Run the XPINN
    train_result = train_xpinn(
        scenario_id=scenario["scenario_id"],
        n_epochs=10000,
        noise_level=noise_level,
        n_pressure_sensors=2,
        use_npw_init=use_npw_init
    )
    
    model = train_result["model"]
    history = train_result["history"]
    training_time = train_result["training_time_s"]
    
    # Extract final predictions
    x_leak_pred = float(model.x_leak.item())
    q_leak_pred = float(model.q_leak.item())
    
    x_leak_true = scenario["x_leak_true"]
    q_leak_true = scenario["q_leak_true"]
    
    x_err_m = abs(x_leak_pred - x_leak_true)
    x_err_km = x_err_m / 1000.0
    q_err_pct = abs(q_leak_pred - q_leak_true) / q_leak_true * 100.0
    
    final_loss = history[-1]['L_total']
    loss_converged = bool(final_loss < 1e-2)
    
    # Subsample trajectories (every 500 epochs roughly, or just all since history is every 10 epochs)
    # Actually, the user asked for every 500 epochs
    x_traj = [h['x_leak_pred'] for h in history if h['epoch'] % 500 == 0]
    q_traj = [h['q_leak_pred'] for h in history if h['epoch'] % 500 == 0]
    # Keep full history for plotting
    
    # Verdict
    if x_err_km <= 0.5 and q_err_pct <= 30.0:
        veredicto = "EXITO"
    elif x_err_km <= 1.5 and q_err_pct <= 60.0:
        veredicto = "MARGINAL"
    else:
        veredicto = "FALLO"
        
    result = {
        "scenario_id": scenario["scenario_id"],
        "x_leak_true": x_leak_true,
        "q_leak_true": q_leak_true,
        "x_leak_pred": x_leak_pred,
        "x_leak_error_m": x_err_m,
        "x_leak_error_km": x_err_km,
        "q_leak_pred": q_leak_pred,
        "q_leak_error_pct": q_err_pct,
        "final_loss": final_loss,
        "loss_converged": loss_converged,
        "veredicto": veredicto,
        "training_time_s": training_time,
        "x_leak_trajectory": x_traj,
        "q_leak_trajectory": q_traj,
        "full_history": history,  # keep for plotting, remove before saving json
        "descripcion": scenario["descripcion"],
        "dificultad": scenario["dificultad"]
    }
    
    return result

def print_scenario_result(result):
    print("  Resultado:")
    print(f"    x_leak: {result['x_leak_true']}m real -> {result['x_leak_pred']:.0f}m pred | error: {result['x_leak_error_m']:.0f} m  [{result['veredicto']}]")
    print(f"    q_leak: {result['q_leak_true']} real -> {result['q_leak_pred']:.4f} pred | error: {result['q_leak_error_pct']:.1f}%")
    print(f"    Loss final: {result['final_loss']:.2e} | Tiempo: {result['training_time_s']/60:.1f} min")
    print("  " + "─"*47)

def print_summary(results):
    print("\n  ═══════════════════════════════════════════════")
    print("  RESUMEN QUICK TEST")
    print("  ═══════════════════════════════════════════════")
    print()
    print("  Escenario | Dificultad | x_err (m) | q_err (%) | Veredicto")
    print("  ──────────┼────────────┼───────────┼───────────┼──────────")
    
    exitos = sum(1 for r in results if r["veredicto"] == "EXITO")
    
    for r in results:
        check = "✓" if r["veredicto"] == "EXITO" else ("~" if r["veredicto"] == "MARGINAL" else "❌")
        print(f"  id={r['scenario_id']:<6} | {r['dificultad']:<10} | {r['x_leak_error_m']:9.0f} | {r['q_leak_error_pct']:8.1f}% | {r['veredicto']} {check}")
        
    print(f"\n  RESULTADO GLOBAL: {exitos}/4 éxitos\n")
    print("  RECOMENDACIÓN:")
    if exitos == 4:
        print("    4/4 éxitos  -> ✅ Lanzar factorial overnight")
        reco = "lanzar_factorial"
    elif exitos == 3:
        print("    3/4 éxitos  -> ⚠️  Revisar el escenario fallido antes de continuar")
        reco = "revisar_fallido"
    else:
        print("    2/4 o menos -> ❌ Ajustar hiperparámetros antes del factorial")
        reco = "ajustar_hiperparametros"
    print("  ═══════════════════════════════════════════════\n")
    return reco, exitos

def plot_quick_test(results, save_dir="figs"):
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    
    for i, r in enumerate(results):
        ax_x = axs[i, 0]
        ax_loss = axs[i, 1]
        
        hist = r["full_history"]
        epochs = [h["epoch"] for h in hist]
        x_pred = [h["x_leak_pred"] for h in hist]
        
        l_total = [h["L_total"] for h in hist]
        l_p = [h["L_P"] for h in hist]
        l_q = [h["L_Q"] for h in hist]
        l_pde = [h["L_pde"] for h in hist]
        l_masa = [h["L_masa"] for h in hist]
        
        # Left plot: x_leak trajectory
        ax_x.plot(epochs, x_pred, label="x_leak pred", color="blue")
        ax_x.axhline(r["x_leak_true"], color="red", linestyle="-", label="x_leak_true")
        ax_x.axhline(5000, color="gray", linestyle="--", label="L/2 (5000m)")
        
        # LR decay & annealing marks
        ax_x.axvline(1000, color="orange", linestyle=":", label="LR decay 1")
        ax_x.axvline(2500, color="orange", linestyle=":", label="LR decay 2")
        ax_x.axvline(8000, color="purple", linestyle="-.", label="Annealing end")
        
        ax_x.set_title(f"Escenario {r['scenario_id']} | x_leak={r['x_leak_true']}m | q_leak={r['q_leak_true']} | {r['veredicto']}")
        ax_x.set_ylabel("x_leak [m]")
        ax_x.set_xlabel("Epoch")
        ax_x.legend()
        
        # Right plot: losses
        ax_loss.plot(epochs, l_total, label="L_total", color="black")
        ax_loss.plot(epochs, l_p, label="L_P")
        ax_loss.plot(epochs, l_q, label="L_Q")
        ax_loss.plot(epochs, l_pde, label="L_pde")
        ax_loss.plot(epochs, l_masa, label="L_masa")
        
        ax_loss.set_yscale("log")
        ax_loss.axvline(1000, color="orange", linestyle=":")
        ax_loss.axvline(2500, color="orange", linestyle=":")
        ax_loss.axvline(8000, color="purple", linestyle="-.")
        
        ax_loss.set_title("Evolución de componentes del Loss")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.legend()
        
    plt.tight_layout()
    save_path = os.path.join(save_dir, "quick_test_trajectories.png")
    plt.savefig(save_path, dpi=150)
    print(f"Plot guardado en: {save_path}")
    plt.close()

def save_results(results, reco, exitos):
    os.makedirs("results", exist_ok=True)
    
    # Clean up full_history before saving
    clean_results = []
    for r in results:
        r_copy = dict(r)
        del r_copy["full_history"]
        clean_results.append(r_copy)
        
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "n_epochs": 10000,
            "noise_level": "moderado",
            "n_pressure_sensors": 3,
            "use_npw_init": True
        },
        "scenarios": clean_results,
        "global_result": f"{exitos}/{len(clean_results)} exitos",
        "recommendation": reco
    }
    
    with open("results/quick_test_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Resultados guardados en: results/quick_test_results.json")

if __name__ == '__main__':
    print("  ═══════════════════════════════════════════════")
    print("  QUICK TEST — VERIFICACIÓN PRE-FACTORIAL")
    print("  ═══════════════════════════════════════════════")
    print("Corriendo quick test con Warm-Start (NPW) y ruido moderado...")
    print(f"Tiempo estimado: ~{len(TEST_SCENARIOS)*10} minutos\n")

    results = []
    for i, scenario in enumerate(TEST_SCENARIOS):
        print(f"[{i+1}/{len(TEST_SCENARIOS)}] Escenario {scenario['scenario_id']} | x_leak={scenario['x_leak_true']}m | q={scenario['q_leak_true']} | dificultad={scenario['dificultad']}")
        res = run_single_test(scenario, noise_level="moderado", use_npw_init=True)
        results.append(res)
        print_scenario_result(res)

    reco, exitos = print_summary(results)
    plot_quick_test(results)
    save_results(results, reco, exitos)

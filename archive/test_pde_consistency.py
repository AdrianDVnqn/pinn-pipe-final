import numpy as np
import matplotlib.pyplot as plt
import os

from config import *
import config as cfg
from simulator import run_moc

SCENARIO = {
    "scenario_id": 8,       # x_leak=6000m, q_leak=0.015
    "x_leak_true": 6000.0,  # m
    "q_leak_true": 0.015,   # m³/s
    "t_leak":      50.0,    # s
}

SIGMA_VALUES = [500.0, 1000.0, 2000.0]
T_START = 0.0

def compute_residuals_from_moc(moc_data, sigma=None):
    t_array = moc_data["t"]
    x_array = moc_data["x"]
    P_moc = moc_data["P"]
    Q_moc = moc_data["Q"]
    
    mask_t = t_array >= T_START
    t_filt = t_array[mask_t]
    P_filt = P_moc[:, mask_t]
    Q_filt = Q_moc[:, mask_t]
    
    dx = x_array[1] - x_array[0]
    dt = t_filt[1] - t_filt[0]
    
    dP_dt = (P_filt[:, 2:] - P_filt[:, :-2]) / (2 * dt)
    dQ_dx = (Q_filt[2:, :] - Q_filt[:-2, :]) / (2 * dx)
    dQ_dt = (Q_filt[:, 2:] - Q_filt[:, :-2]) / (2 * dt)
    dP_dx = (P_filt[2:, :] - P_filt[:-2, :]) / (2 * dx)
    
    x_int = x_array[1:-1]
    t_int = t_filt[1:-1]
    Q_int = Q_filt[1:-1, 1:-1]
    
    dP_dt_int = dP_dt[1:-1, :]
    dQ_dx_int = dQ_dx[:, 1:-1]
    dQ_dt_int = dQ_dt[1:-1, :]
    dP_dx_int = dP_dx[:, 1:-1]
    
    A = np.pi * cfg.PIPE_DIAMETER**2 / 4.0
    
    if sigma is not None:
        X_int, T_int = np.meshgrid(x_int, t_int, indexing='ij')
        spatial = np.exp(-0.5 * ((X_int - SCENARIO["x_leak_true"]) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
        temporal = 1.0 / (1.0 + np.exp(-(T_int - SCENARIO["t_leak"]) / 2.0))
        S = SCENARIO["q_leak_true"] * spatial * temporal
    else:
        S = 0.0
        
    r_cont = dP_dt_int + (cfg.FLUID_DENSITY * cfg.WAVE_SPEED**2 / A) * dQ_dx_int + (cfg.FLUID_DENSITY * cfg.WAVE_SPEED**2 / A) * S
    
    friction = (cfg.FRICTION_FACTOR * Q_int * np.abs(Q_int)) / (2.0 * cfg.PIPE_DIAMETER * A)
    r_mom = dQ_dt_int + (A / cfg.FLUID_DENSITY) * dP_dx_int + friction
    
    return r_cont, r_mom, x_int, t_int

def report_residuals(r_cont, r_mom, sigma, label=""):
    print(f"\n{'─'*50}")
    if sigma is not None:
        print(f"σ = {sigma:.0f} m  {label}")
    else:
        print(f"σ = None  {label}")
    print(f"{'─'*50}")

    print(f"Residuo continuidad r_cont:")
    print(f"  mean(|r_cont|) = {np.abs(r_cont).mean():.4e}")
    print(f"  max(|r_cont|)  = {np.abs(r_cont).max():.4e}")
    print(f"  std(r_cont)    = {np.std(r_cont):.4e}")

    r_cont_norm = np.abs(r_cont).mean() / cfg.P_INLET
    print(f"  mean(|r_cont|) / P_INLET = {r_cont_norm:.4e}")

    print(f"Residuo momentum r_mom:")
    print(f"  mean(|r_mom|) = {np.abs(r_mom).mean():.4e}")
    print(f"  max(|r_mom|)  = {np.abs(r_mom).max():.4e}")

    r_mom_norm = np.abs(r_mom).mean() / cfg.Q_OUTLET
    print(f"  mean(|r_mom|) / Q_OUTLET = {r_mom_norm:.4e}")

    if r_cont_norm < 1e-3 and r_mom_norm < 1e-3:
        print(f"  ✓ CONSISTENTE: residuos normalizados < 1e-3")
        return True
    elif r_cont_norm < 1e-2 and r_mom_norm < 1e-2:
        print(f"  ⚠ ACEPTABLE: residuos normalizados < 1e-2")
        print(f"    La PINN puede compensar esta imprecisión")
        return True
    else:
        print(f"  ✗ INCONSISTENTE: residuos demasiado grandes")
        print(f"    La solución MOC no satisface esta PDE")
        return False

def plot_consistency_check(moc, SIGMA_VALUES, r_cont_list, r_mom_list, r_cont_nl, r_mom_nl, x_int, t_int):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    extent = [t_int[0], t_int[-1], x_int[-1], x_int[0]]

    def plot_heatmap(ax, data, title):
        im = ax.imshow(data, aspect='auto', extent=extent, cmap='hot', interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.axhline(SCENARIO['x_leak_true'], color='red', linestyle='--')
        if t_int[0] <= SCENARIO['t_leak'] <= t_int[-1]:
            ax.axvline(SCENARIO['t_leak'], color='red', linestyle='--')
        plt.colorbar(im, ax=ax)

    plot_heatmap(axes[0,0], np.abs(r_cont_list[0]), f"|r_cont| (σ = {SIGMA_VALUES[0]}m)")
    plot_heatmap(axes[0,1], np.abs(r_mom_list[0]), f"|r_mom| (σ = {SIGMA_VALUES[0]}m)")

    plot_heatmap(axes[1,0], np.abs(r_cont_list[1]), f"|r_cont| (σ = {SIGMA_VALUES[1]}m)")
    plot_heatmap(axes[1,1], np.abs(r_mom_list[1]), f"|r_mom| (σ = {SIGMA_VALUES[1]}m)")

    plot_heatmap(axes[2,0], np.abs(r_cont_nl), "|r_cont| (Sin fuga)")
    plot_heatmap(axes[2,1], np.abs(r_mom_nl), "|r_mom| (Sin fuga)")

    plt.suptitle("Test de consistencia PDE — Solución MOC real", fontsize=16)
    plt.tight_layout()
    plt.savefig('figs/pde_consistency_check.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    print("═"*55)
    print("TEST DE CONSISTENCIA PDE")
    print("Verificando que la solución MOC satisface")
    print("la formulación de PDE usada en la PINN")
    print("═"*55)

    moc = run_moc(Q_leak=SCENARIO["q_leak_true"],
                  x_leak=SCENARIO["x_leak_true"],
                  t_leak=SCENARIO["t_leak"],
                  noise_std=0.0)

    print(f"Rango temporal MOC completo: [{moc['t'][0]:.0f}, {moc['t'][-1]:.0f}] s")
    mask_t = moc["t"] >= T_START
    print(f"Rango temporal filtrado:     [{moc['t'][mask_t][0]:.0f}, {moc['t'][mask_t][-1]:.0f}] s")
    print(f"Nodos espaciales: {len(moc['x'])}")
    print(f"Pasos temporales filtrados: {np.sum(mask_t)}")
    
    dx = moc['x'][1] - moc['x'][0]
    dt = moc['t'][mask_t][1] - moc['t'][mask_t][0]
    print(f"dx = {dx:.2f} m")
    print(f"dt = {dt:.6f} s")

    r_cont_list = []
    r_mom_list = []
    verdicts = []

    for sigma in SIGMA_VALUES:
        r_cont, r_mom, x_int, t_int = compute_residuals_from_moc(moc, sigma)
        r_cont_list.append(r_cont)
        r_mom_list.append(r_mom)
        verdicts.append(report_residuals(r_cont, r_mom, sigma))

    moc_noleak = run_moc(Q_leak=0.0, noise_std=0.0)
    r_cont_nl, r_mom_nl, _, _ = compute_residuals_from_moc(moc_noleak, sigma=None)
    
    print("\n═══ BASELINE NUMÉRICO (sin fuga) ═══")
    report_residuals(r_cont_nl, r_mom_nl, sigma=None, label="(sin término fuente)")

    if not os.path.exists('figs'):
        os.makedirs('figs')
    plot_consistency_check(moc, SIGMA_VALUES, r_cont_list, r_mom_list, r_cont_nl, r_mom_nl, x_int, t_int)

    print("\n" + "═"*55)
    print("RESUMEN: TEST DE CONSISTENCIA PDE")
    print("═"*55)
    print(f"Escenario: x_leak={SCENARIO['x_leak_true']:.0f}m, q_leak={SCENARIO['q_leak_true']:.3f} m³/s")
    print(f"Rango temporal: [{T_START:.0f}, {cfg.T_TOTAL:.0f}] s")
    print()

    for sigma, r_cont, r_mom, is_ok in zip(SIGMA_VALUES, r_cont_list, r_mom_list, verdicts):
        estado = "✓ OK" if is_ok else "✗ FALLO"
        r_cont_norm = np.abs(r_cont).mean() / cfg.P_INLET
        r_mom_norm  = np.abs(r_mom).mean() / cfg.Q_OUTLET
        print(f"  σ={sigma:5.0f}m: r_cont={r_cont_norm:.2e} r_mom={r_mom_norm:.2e}  {estado}")

    print()
    print("INTERPRETACIÓN:")
    print("  Si ✓ para algún sigma → la formulación es correcta.")
    print("     El problema es de optimización, no de física.")
    print("     → Proceder con grid search / multi-start.")
    print()
    print("  Si ✗ para todos → hay un bug en la PDE.")
    print("     El problema es de formulación.")
    print("     → Revisar signos, escalas y término fuente")
    print("       antes de cualquier otra modificación.")
    print("═"*55)

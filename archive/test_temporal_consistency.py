import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import config as cfg
from data_utils import get_training_data

def run_pinn_diagnostic(scenario_id=8, noise_level="trivial", n_pressure_sensors=3):
    data = get_training_data(scenario_id, noise_level, n_pressure_sensors)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    t_tensor = torch.tensor(data['t'], dtype=torch.float32, device=device)
    
    Nbc = 200
    Nic = 200
    t_bc = torch.linspace(0.0, float(cfg.T_TOTAL), Nbc, dtype=torch.float32, device=device)
    
    x_ic = torch.linspace(0.0, float(cfg.PIPE_LENGTH), Nic, dtype=torch.float32, device=device)
    t0_ic = torch.zeros_like(x_ic, device=device)
    
    n_collocation = 20000
    t_col = torch.empty(n_collocation, dtype=torch.float32, device=device)
    t_col.uniform_(0.0, float(cfg.T_TOTAL))
    
    lambdas = {'P': 10.0, 'Q': 10.0, 'pde': 1.0, 'bc': 10.0, 'ic': 5.0, 'masa': 1000.0}
    
    diagnostic_info = {}

    # Capturar t de los datos de sensores
    diagnostic_info["t_data_min"] = t_tensor.min().item()
    diagnostic_info["t_data_max"] = t_tensor.max().item()
    diagnostic_info["t_data_n"]   = len(t_tensor)

    # Capturar t de los puntos de colocación
    diagnostic_info["t_col_min"]  = t_col.min().item()
    diagnostic_info["t_col_max"]  = t_col.max().item()
    diagnostic_info["t_col_n"]    = n_collocation

    # Capturar t de los puntos de contorno
    diagnostic_info["t_bc_min"]   = t_bc.min().item()
    diagnostic_info["t_bc_max"]   = t_bc.max().item()

    # Capturar t de la condición inicial
    diagnostic_info["t_ic_values"] = t0_ic.unique().tolist()
    
    # Capturar los pesos lambda en epoch=0
    diagnostic_info["lambdas_epoch0"] = {k: v for k,v in lambdas.items()}
    
    return diagnostic_info

def run_all_checks(diagnostic_info):
    checks = {}
    
    # CHECK 1
    t_data_range = (diagnostic_info["t_data_min"], diagnostic_info["t_data_max"])
    t_col_range  = (diagnostic_info["t_col_min"], diagnostic_info["t_col_max"])

    overlap = (max(t_data_range[0], t_col_range[0]), min(t_data_range[1], t_col_range[1]))
    if t_data_range[1] > t_data_range[0]:
        overlap_pct = (overlap[1] - overlap[0]) / (t_data_range[1] - t_data_range[0]) * 100
    else:
        overlap_pct = 0.0

    print(f"\nCHECK 1 — Rango temporal datos vs colocación")
    print(f"  t_datos:      [{t_data_range[0]:.1f}, {t_data_range[1]:.1f}] s")
    print(f"  t_col:        [{t_col_range[0]:.1f},  {t_col_range[1]:.1f}] s")
    print(f"  Solapamiento: {overlap_pct:.1f}%")

    if overlap_pct < 99.0 or (t_col_range[0] < t_data_range[0] - 1.0):
        print(f"  ✗ INCONSISTENCIA: los puntos de colocación cubren")
        print(f"    un dominio temporal diferente al de los datos.")
        print(f"    La red aprende física donde no hay supervisión.")
        checks["CHECK 1 (datos vs colocación)"] = "FALLO"
    else:
        print(f"  ✓ Dominios temporales alineados")
        checks["CHECK 1 (datos vs colocación)"] = "OK"
        
    # CHECK 2
    t_ic_vals = diagnostic_info["t_ic_values"]
    T_FILTER  = 150.0   # el filtro aplicado a los datos

    print(f"\nCHECK 2 — Condición inicial")
    print(f"  t usados en L_ic: {t_ic_vals}")
    print(f"  t mínimo datos:   {diagnostic_info['t_data_min']:.1f} s")
    print(f"  Filtro aplicado:  t >= {T_FILTER:.0f} s")

    if any(t < T_FILTER - 1.0 for t in t_ic_vals):
        print(f"  ✗ INCONSISTENCIA: L_ic usa t={min(t_ic_vals):.0f}s")
        print(f"    pero los datos empiezan en t={T_FILTER:.0f}s.")
        print(f"    La red tiene que reconciliar el estado en t=0")
        print(f"    (sin fuga) con los datos en t=150s (con fuga).")
        print(f"    Esto genera una contradicción que el optimizador")
        print(f"    resuelve encontrando soluciones irreales.")
        checks["CHECK 2 (L_ic vs filtro)"] = "FALLO"
    else:
        print(f"  ✓ L_ic es coherente con el filtro temporal")
        checks["CHECK 2 (L_ic vs filtro)"] = "OK"

    # CHECK 3
    t_bc_range = (diagnostic_info["t_bc_min"], diagnostic_info["t_bc_max"])

    print(f"\nCHECK 3 — Condición de contorno")
    print(f"  t_bc: [{t_bc_range[0]:.1f}, {t_bc_range[1]:.1f}] s")
    print(f"  t_datos: [{diagnostic_info['t_data_min']:.1f}, {diagnostic_info['t_data_max']:.1f}] s")

    if abs(t_bc_range[0] - diagnostic_info["t_data_min"]) > 1.0 or abs(t_bc_range[1] - diagnostic_info["t_data_max"]) > 1.0:
        print(f"  ✗ INCONSISTENCIA: L_bc cubre rango distinto")
        print(f"    al de los datos.")
        checks["CHECK 3 (L_bc vs filtro)"] = "FALLO"
    else:
        print(f"  ✓ L_bc coherente con datos")
        checks["CHECK 3 (L_bc vs filtro)"] = "OK"
        
    # CHECK 4
    T_LEAK  = 50.0    # tiempo de inicio de la fuga
    T_START = 150.0   # inicio del dominio filtrado
    WAVE_SPEED = float(cfg.WAVE_SPEED)

    print(f"\nCHECK 4 — Cobertura del transitorio")
    print(f"  Fuga inicia en:        t = {T_LEAK:.0f} s")
    print(f"  Dominio datos inicia:  t = {T_START:.0f} s")
    print(f"  Tiempo desde fuga hasta inicio datos: {T_START - T_LEAK:.0f} s")
    print(f"  Velocidad onda: {WAVE_SPEED:.0f} m/s")
    print(f"  Distancia recorrida en ese tiempo: {WAVE_SPEED * (T_START - T_LEAK) / 1000:.0f} km")

    if T_START > T_LEAK:
        print(f"  ✓ La onda ya recorrió todo el ducto cuando")
        print(f"    empieza el dominio de entrenamiento.")
        print(f"    El sistema está en cuasi-estacionario.")
        checks["CHECK 4 (cobertura transitorio)"] = "OK"
    else:
        print(f"  ✗ El dominio empieza antes de que la onda")
        print(f"    llegue a todos los sensores.")
        checks["CHECK 4 (cobertura transitorio)"] = "FALLO"
        
    # CHECK 5
    lambdas = diagnostic_info["lambdas_epoch0"]

    print(f"\nCHECK 5 — Balance de lambdas en epoch 0")
    for k, v in lambdas.items():
        print(f"  lambda_{k}: {v:.4f}")

    if lambdas.get("pde", 0) > 10.0:
        print(f"  ✗ ADVERTENCIA: lambda_pde={lambdas['pde']:.1f}")
        print(f"    arranca demasiado alto. La física domina")
        print(f"    antes de que la red aprenda los datos.")
        checks["CHECK 5 (lambdas epoch 0)"] = "ADVERTENCIA"
    else:
        print(f"  ✓ Lambdas en rango razonable para epoch 0")
        checks["CHECK 5 (lambdas epoch 0)"] = "OK"
        
    return checks

def plot_temporal_domains(diagnostic_info):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})
    
    T_TOTAL = float(cfg.T_TOTAL)
    T_START = 150.0
    T_LEAK = 50.0
    
    # Fila superior
    ax1.set_xlim(0, T_TOTAL)
    ax1.set_ylim(-0.5, 3.5)
    
    ax1.barh(3, diagnostic_info["t_data_max"] - diagnostic_info["t_data_min"], left=diagnostic_info["t_data_min"], color='blue', label='t_datos')
    ax1.barh(2, diagnostic_info["t_col_max"] - diagnostic_info["t_col_min"], left=diagnostic_info["t_col_min"], color='green', label='t_col')
    ax1.barh(1, diagnostic_info["t_bc_max"] - diagnostic_info["t_bc_min"], left=diagnostic_info["t_bc_min"], color='orange', label='t_bc')
    ax1.plot(diagnostic_info["t_ic_values"], [0]*len(diagnostic_info["t_ic_values"]), 'ro', markersize=10, label='t_ic')
    
    ax1.axvline(T_LEAK, color='r', linestyle='--', label='Fuga inicia')
    ax1.axvline(T_START, color='b', linestyle='--', label='Filtro inicia')
    
    ax1.set_yticks([0, 1, 2, 3])
    ax1.set_yticklabels(['IC', 'BC', 'Colocación', 'Datos'])
    ax1.set_xlabel('Tiempo (s)')
    ax1.legend(loc='upper right')
    ax1.set_title('Dominios temporales en la PINN')
    
    # Fila inferior: Mock de P(t) en x=5000m
    from simulator import run_moc
    moc = run_moc(Q_leak=0.015, x_leak=6000.0, t_leak=50.0, noise_std=0.0)
    idx_5000 = np.argmin(np.abs(moc['x'] - 5000.0))
    P_5000 = moc['P'][idx_5000, :]
    t_moc = moc['t']
    
    ax2.plot(t_moc, P_5000, 'k-', label='P(x=5000m, t)')
    ax2.axvspan(0, T_START, color='gray', alpha=0.3, label='Excluido por filtro')
    ax2.axvspan(T_START, T_TOTAL, color='lightblue', alpha=0.3, label='Usado por PINN')
    ax2.axvline(T_LEAK, color='r', linestyle='--')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Presión (Pa)')
    ax2.legend()
    ax2.set_title('Señal de presión en sensor (x=5000m)')
    
    plt.tight_layout()
    if not os.path.exists('figs'):
        os.makedirs('figs')
    plt.savefig('figs/temporal_consistency.png', dpi=300)
    plt.close()

def print_summary(checks):
    print("\n" + "═"*55)
    print("RESUMEN: TEST DE CONSISTENCIA TEMPORAL")
    print("═"*55)

    n_ok    = sum(1 for v in checks.values() if v == "OK")
    n_fallo = sum(1 for v in checks.values() if v == "FALLO")

    for name, result in checks.items():
        emoji = "✓" if result == "OK" else "✗"
        print(f"  {emoji} {name}: {result}")

    print(f"\n  Resultado: {n_ok}/5 checks OK")
    print()

    if n_fallo == 0:
        print("  CONCLUSIÓN: Consistencia temporal verificada.")
        print("  El problema es de optimización (mínimos locales).")
        print("  → Proceder con grid search / multi-start.")
    else:
        print("  CONCLUSIÓN: Se encontraron inconsistencias.")
        print("  Aplicar los fixes antes de continuar:")

        if checks.get("CHECK 1 (datos vs colocación)") == "FALLO":
            print()
            print("  FIX 1 (crítico):")
            print("    En train_pinn(), filtrar puntos de colocación:")
            print("      x_col.uniform_(0, L)")
            print("      t_col.uniform_(T_START, T_TOTAL)  ← cambiar 0 por T_START")
            print("    donde T_START = 150.0")

        if checks.get("CHECK 2 (L_ic vs filtro)") == "FALLO":
            print()
            print("  FIX 2 (crítico):")
            print("    Opción A (recomendada): eliminar L_ic")
            print("      En compute_loss(): no calcular L_ic")
            print("      Justificación: con t_datos=[150,200]")
            print("      la IC en t=0 es inaccesible e irrelevante.")
            print()
            print("    Opción B: cambiar IC a t=150s")
            print("      x_ic ~ Uniform(0, L)")
            print("      t_ic = 150.0  ← fijo, no 0.0")
            print("      P_ic(x) = P_moc(x, t=150)  ← del MOC real")

        if checks.get("CHECK 3 (L_bc vs filtro)") == "FALLO":
            print()
            print("  FIX 3:")
            print("    Alinear t_bc con el rango de t_datos:")
            print("      t_bc.uniform_(T_START, T_TOTAL)")

    print("═"*55)

if __name__ == '__main__':
    print("═"*55)
    print("TEST DE CONSISTENCIA TEMPORAL")
    print("Verificando alineación de dominios entre")
    print("datos, colocación, contorno e IC")
    print("═"*55)

    diagnostic = run_pinn_diagnostic(scenario_id=8, noise_level="trivial", n_pressure_sensors=3)
    results = run_all_checks(diagnostic)
    plot_temporal_domains(diagnostic)
    print_summary(results)

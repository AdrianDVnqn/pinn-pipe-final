"""
diagnose_bias.py -- ¿El sesgo de ~150m es física o es el optimizador?

Corre 3 experimentos rápidos con wave_pinn:
  1. INIT PERFECTO: x_leak arranca en el valor verdadero (6000m)
     → Si se aleja ~150m: el sesgo es un atractor del paisaje de loss (física)
     → Si se queda: el problema es que la Fase 1 no converge bien

  2. MÁS EPOCHS: mismo setup pero con 8000 epochs (2x más)
     → Si el error baja: el modelo necesitaba más tiempo
     → Si no cambia: el sesgo es estable

  3. FASE 3 AGRESIVA: lr de fase 3 más alto y sin decay
     → Si el error baja: la Fase 3 estaba subentrenada
     → Si no cambia: confirma que es un atractor físico
"""

import time
import torch
import numpy as np

from wave_pinn import (
    generate_data, WaveLeakPINN, compute_loss, compute_k,
    L_PIPE, T_TOTAL, P_INLET, Q_OUTLET, F_DARCY, RHO, D_PIPE, A_PIPE,
    X_PRESSURE_SENSORS, X_FLOW_METERS, T_LEAK_START
)

X_TRUE = 6000.0
Q_TRUE = 0.030


def train_custom(data, x_init_raw, n_epochs, phase3_lr, phase3_decay, label):
    """
    Entrenamiento con parámetros configurables para diagnóstico.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = WaveLeakPINN().to(device)

    # Override x_leak initialization
    with torch.no_grad():
        model.x_leak_raw.data = torch.tensor(x_init_raw, dtype=torch.float32, device=device)

    print(f"\n  [{label}] x_leak init: {model.x_leak.item():.0f}m | q_leak init: {model.q_leak.item():.4f}")

    lambdas = {'P': 100.0, 'Q': 100.0, 'pde': 1.0, 'bc': 10.0, 'ic': 10.0}

    t_tensor = torch.tensor(data['t'], dtype=torch.float32, device=device)
    P_sensors_t = torch.tensor(data['P_sensors'], dtype=torch.float32, device=device)
    Q_sensors_t = torch.tensor(data['Q_sensors'], dtype=torch.float32, device=device)

    t_bc = torch.linspace(0, T_TOTAL, 200, dtype=torch.float32, device=device)
    x_ic = torch.linspace(0, L_PIPE, 200, dtype=torch.float32, device=device)
    t0_ic = torch.zeros_like(x_ic)
    P_ss_ic = torch.tensor(
        P_INLET - F_DARCY * RHO * x_ic.cpu().numpy() * (Q_OUTLET * abs(Q_OUTLET)) / (2.0 * D_PIPE * A_PIPE**2),
        dtype=torch.float32, device=device
    )

    data_tensors = {
        't': t_tensor, 'P_sensors': P_sensors_t, 'Q_sensors': Q_sensors_t,
        't_bc': t_bc, 'x_ic': x_ic, 't0_ic': t0_ic, 'P_ss_ic': P_ss_ic,
    }

    n_collocation = 10000
    x_col = torch.empty(n_collocation, dtype=torch.float32, device=device)
    t_col = torch.empty(n_collocation, dtype=torch.float32, device=device)

    opt_mlp = torch.optim.Adam(model.network_params(), lr=1e-3)
    opt_phys = torch.optim.Adam(model.physical_params(), lr=5e-2)

    phase1_epochs = int(0.4 * n_epochs)
    phase2_epochs = int(0.3 * n_epochs)

    t_start = time.time()
    snapshots = []  # (epoch, x_leak, q_leak)

    for epoch in range(1, n_epochs + 1):
        k_current = compute_k(epoch, phase1_epochs)

        torch.manual_seed(epoch + 42)
        with torch.no_grad():
            x_col.uniform_(0.0, L_PIPE)
            t_col.uniform_(0.0, T_TOTAL)
        x_col.requires_grad_(True)
        t_col.requires_grad_(True)

        if epoch <= phase1_epochs:
            opt_phys.zero_grad()
            Nt = t_tensor.shape[0]
            L_P_tensor = torch.tensor(0.0, device=device)
            for i, xs in enumerate(X_PRESSURE_SENSORS):
                x_t = torch.full((Nt,), xs, dtype=torch.float32, device=device)
                P_pred, _, _, _ = model(x_t, t_tensor, k_current)
                L_P_tensor += torch.mean((P_pred / P_INLET - data_tensors['P_sensors'][i] / P_INLET)**2)
            L_P_tensor /= len(X_PRESSURE_SENSORS)

            L_Q_tensor = torch.tensor(0.0, device=device)
            for i, xs in enumerate(X_FLOW_METERS):
                x_t = torch.full((Nt,), xs, dtype=torch.float32, device=device)
                _, Q_pred, _, _ = model(x_t, t_tensor, k_current)
                L_Q_tensor += torch.mean((Q_pred / Q_OUTLET - data_tensors['Q_sensors'][i] / Q_OUTLET)**2)
            L_Q_tensor /= len(X_FLOW_METERS)

            L_phase1 = lambdas['P'] * L_P_tensor + lambdas['Q'] * L_Q_tensor
            L_phase1.backward()
            opt_phys.step()

        elif epoch <= phase1_epochs + phase2_epochs:
            opt_mlp.zero_grad()
            progress = min(1.0, (epoch - phase1_epochs) / float(phase2_epochs))
            lambdas['pde'] = 1.0 + (1000.0 - 1.0) * progress
            L_total, comps = compute_loss(model, data_tensors, x_col, t_col, lambdas, k_current)
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_mlp.step()

        else:
            if epoch == phase1_epochs + phase2_epochs + 1:
                for pg in opt_phys.param_groups:
                    pg['lr'] = phase3_lr

            opt_mlp.zero_grad()
            opt_phys.zero_grad()
            L_total, comps = compute_loss(model, data_tensors, x_col, t_col, lambdas, k_current)
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_mlp.step()
            opt_phys.step()

            if phase3_decay:
                if epoch == int(0.80 * n_epochs) or epoch == int(0.90 * n_epochs):
                    for pg in opt_phys.param_groups:
                        pg['lr'] *= 0.2

        if epoch % 100 == 0 or epoch == n_epochs:
            snapshots.append((epoch, model.x_leak.item(), model.q_leak.item()))

        if epoch % 1000 == 0 or epoch == n_epochs:
            xl = model.x_leak.item()
            ql = model.q_leak.item()
            print(f"    Epoch {epoch:5d} | x_leak: {xl:7.0f}m | q_leak: {ql:.5f} | err_x: {abs(xl-X_TRUE):.0f}m")

    elapsed = time.time() - t_start
    x_pred = model.x_leak.item()
    q_pred = model.q_leak.item()
    x_err = abs(x_pred - X_TRUE)
    q_err = abs(q_pred - Q_TRUE)

    print(f"  [{label}] RESULT: x={x_pred:.0f}m (err {x_err:.0f}m) | q={q_pred:.5f} (err {q_err:.5f}) | {elapsed:.0f}s")
    return x_pred, q_pred, x_err, snapshots


def x_to_raw(x_target):
    """Convert desired x_leak to raw sigmoid-space parameter."""
    p = (x_target - 500.0) / 9000.0
    p = max(1e-5, min(1.0 - 1e-5, p))
    return float(np.log(p / (1 - p)))


if __name__ == '__main__':
    print("=" * 60)
    print("  DIAGNÓSTICO DE SESGO: ¿Física o Optimizador?")
    print("=" * 60)

    # Default raw init (-1.0 → ~2920m)
    default_raw = -1.0

    # Generate data once
    print("\n[0] Generando datos MOC...")
    data = generate_data(X_TRUE, Q_TRUE, noise_std=0.0)

    results = {}

    # ── Test 1: Init perfecto ──
    print("\n" + "=" * 60)
    print("  TEST 1: Inicialización perfecta (x_leak = 6000m)")
    print("  Si se aleja ~150m → sesgo es físico (atractor)")
    print("  Si se queda → Fase 1 no converge bien")
    print("=" * 60)
    perfect_raw = x_to_raw(X_TRUE)
    x1, q1, e1, s1 = train_custom(data, perfect_raw, n_epochs=4000,
                                    phase3_lr=2e-3, phase3_decay=True,
                                    label="INIT_PERFECTO")
    results['init_perfecto'] = e1

    # ── Test 2: Más epochs ──
    print("\n" + "=" * 60)
    print("  TEST 2: Doble de epochs (8000 vs 4000)")
    print("  Si el error baja → necesitaba más entrenamiento")
    print("  Si no cambia → sesgo es estable")
    print("=" * 60)
    x2, q2, e2, s2 = train_custom(data, default_raw, n_epochs=8000,
                                    phase3_lr=2e-3, phase3_decay=True,
                                    label="MÁS_EPOCHS")
    results['mas_epochs'] = e2

    # ── Test 3: Fase 3 agresiva ──
    print("\n" + "=" * 60)
    print("  TEST 3: Fase 3 con lr alto (1e-2) y sin decay")
    print("  Si el error baja → Fase 3 estaba subentrenada")
    print("  Si no cambia → confirma atractor")
    print("=" * 60)
    x3, q3, e3, s3 = train_custom(data, default_raw, n_epochs=4000,
                                    phase3_lr=1e-2, phase3_decay=False,
                                    label="FASE3_AGRESIVA")
    results['fase3_agresiva'] = e3

    # ── Resumen ──
    print("\n" + "=" * 60)
    print("  RESUMEN DIAGNÓSTICO")
    print("=" * 60)
    print(f"  {'Test':<20} | {'Error x_leak':>12}")
    print(f"  {'-'*20}-+-{'-'*12}")
    for name, err in results.items():
        print(f"  {name:<20} | {err:>10.0f} m")

    print()
    if results['init_perfecto'] > 100:
        print("  → CONCLUSIÓN: El sesgo de ~150m ES un atractor del paisaje de loss.")
        print("    Incluso partiendo del valor exacto, el modelo converge a ~150m de error.")
        print("    Esto es un artefacto de la inyección D'Alembert sin fricción.")
    else:
        print("  → CONCLUSIÓN: El sesgo es un problema de optimización.")
        print("    Con inicialización perfecta el error es bajo.")
        print("    Hay que mejorar la Fase 1 o la convergencia general.")
    print("=" * 60)

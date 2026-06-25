"""
simple_xleak_pinn.py -- PINN simplificada: solo busca x_leak (q_leak conocido)

Usa el simulador MOC directamente (sin dataset.h5).
Inyecta la física del escalón de caudal y quiebre de presión en la arquitectura.
Solo un parámetro escalar a optimizar: x_leak.

Uso:
    python simple_xleak_pinn.py
    python simple_xleak_pinn.py --x_leak 4000 --q_leak 0.015 --epochs 8000
"""

import os
import math
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import config as cfg
from simulator import run_moc, get_sensor_data

# ═══════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS (from config)
# ═══════════════════════════════════════════════════════════════
L_PIPE   = float(cfg.PIPE_LENGTH)      # 10000 m
D_PIPE   = float(cfg.PIPE_DIAMETER)    # 0.5 m
A_PIPE   = math.pi * D_PIPE**2 / 4.0
RHO      = float(cfg.FLUID_DENSITY)    # 850 kg/m³
A_WAVE   = float(cfg.WAVE_SPEED)       # 1200 m/s
F_DARCY  = float(cfg.FRICTION_FACTOR)  # 0.02
P_INLET  = float(cfg.P_INLET)         # 5e6 Pa
Q_OUTLET = float(cfg.Q_OUTLET)        # 0.3 m³/s
T_TOTAL  = float(cfg.T_TOTAL)         # 200 s
T_LEAK_START = float(cfg.T_LEAK_START) # 50 s

# Sensor positions
X_PRESSURE_SENSORS = [1000.0, 5000.0, 9000.0]
X_FLOW_METERS      = [0.0, 10000.0]


# ═══════════════════════════════════════════════════════════════
# DATA GENERATION (direct MOC call)
# ═══════════════════════════════════════════════════════════════

def generate_data(x_leak_true, q_leak_true, noise_std=500.0):
    """Run MOC simulator and extract sensor data."""
    print(f"  Simulando MOC: x_leak={x_leak_true}m, q_leak={q_leak_true}")
    moc = run_moc(Q_leak=q_leak_true, x_leak=x_leak_true,
                  t_leak=T_LEAK_START, noise_std=0.0)

    # Extract at pressure sensor positions
    p_data = get_sensor_data(moc, X_PRESSURE_SENSORS, noise_std=noise_std)
    # Extract at flow meter positions
    q_data = get_sensor_data(moc, X_FLOW_METERS, noise_std=noise_std * Q_OUTLET / P_INLET)

    # Steady-state for IC
    x_grid = moc['x']
    P_ss = P_INLET - F_DARCY * RHO * x_grid * (Q_OUTLET * abs(Q_OUTLET)) / (2.0 * D_PIPE * A_PIPE**2)

    return {
        't': moc['t'],
        'x': x_grid,
        'P_full': moc['P'],
        'Q_full': moc['Q'],
        'P_sensors': p_data['P_sensors'],     # [3, Nt]
        'Q_sensors': q_data['Q_sensors'],      # [2, Nt] at x=0 and x=L
        'P_ss': P_ss,
    }


# ═══════════════════════════════════════════════════════════════
# ARCHITECTURE: SimpleLeakPINN
# ═══════════════════════════════════════════════════════════════

class SimpleLeakPINN(nn.Module):
    """
    Physics-injection PINN for leak localization.

    The total field is:
        P(x,t) = P_mlp(x,t) + P_sing(x,t; x_leak)
        Q(x,t) = Q_mlp(x,t) + Q_sing(x,t; x_leak)

    Where P_sing and Q_sing encode the step/kink structure of the leak,
    and P_mlp/Q_mlp are smooth residual corrections from the MLP.

    Only x_leak is trainable. q_leak is fixed.
    """

    def __init__(self, q_leak_known, hidden_layers=5, hidden_size=64):
        super().__init__()
        self.q_leak_known = q_leak_known

        # MLP: maps (x/L, t/T) -> (P_norm, Q_norm)
        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 2))
        self.net = nn.Sequential(*layers)

        # Xavier init
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # The ONLY trainable scalar: x_leak position
        # sigmoid(0) = 0.5 -> 5000m (center of pipe)
        self.x_leak_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def x_leak(self):
        """x_leak in [500, 9500] via sigmoid."""
        return torch.sigmoid(self.x_leak_raw) * 9000.0 + 500.0

    def network_params(self):
        """All parameters except x_leak_raw."""
        excl = {id(self.x_leak_raw)}
        return [p for p in self.parameters() if id(p) not in excl]

    def forward(self, x, t, k):
        """
        Returns: P_total, Q_total, P_mlp, Q_mlp

        k: sharpness of the step function (continuation parameter)
        """
        orig_shape = x.shape

        # MLP prediction (smooth component)
        x_in = (x / L_PIPE).reshape(-1, 1)
        t_in = (t / T_TOTAL).reshape(-1, 1)
        inp = torch.cat([x_in, t_in], dim=1)
        out = self.net(inp)
        P_mlp = out[:, 0].reshape(orig_shape) * P_INLET
        Q_mlp = out[:, 1].reshape(orig_shape) * Q_OUTLET

        # Physics injection: step in Q, kink in P
        q_leak = self.q_leak_known

        # Normalized sigmoid step: 0 at x=0, 1 at x=L
        S_x = torch.sigmoid(k * (x - self.x_leak))
        S_0 = torch.sigmoid(k * (0.0 - self.x_leak))
        S_L = torch.sigmoid(k * (L_PIPE - self.x_leak))
        S_norm = (S_x - S_0) / (S_L - S_0 + 1e-12)

        # Q drops by q_leak after the leak
        Q_sing_base = -q_leak * S_norm

        # P kink: pressure gradient changes after leak due to reduced flow
        # Analytical: m_slope ≈ f * rho * q_leak * |Q_out| / (D * A²)
        # This is the change in dP/dx due to the flow deficit
        m_slope = F_DARCY * RHO * q_leak * abs(Q_OUTLET) / (D_PIPE * A_PIPE**2)

        SP_x = torch.nn.functional.softplus(x - self.x_leak, beta=k)
        SP_0 = torch.nn.functional.softplus(torch.zeros_like(x) - self.x_leak, beta=k)
        P_sing_base = m_slope * (SP_x - SP_0)

        # Temporal activation: leak starts at T_LEAK_START
        leak_active = torch.sigmoid(10.0 * (t - T_LEAK_START))

        P_sing = P_sing_base * leak_active
        Q_sing = Q_sing_base * leak_active

        P_total = P_mlp + P_sing
        Q_total = Q_mlp + Q_sing

        return P_total, Q_total, P_mlp, Q_mlp


# ═══════════════════════════════════════════════════════════════
# CONTINUATION PARAMETER k
# ═══════════════════════════════════════════════════════════════

def compute_k(epoch, n_epochs):
    """Anneal k from blurry (1/3000) to sharp (1/50)."""
    k_start = 1.0 / 3000.0
    k_end   = 1.0 / 50.0
    t1 = int(0.25 * n_epochs)
    if epoch < t1:
        alpha = epoch / float(t1)
        return k_start + (k_end - k_start) * alpha
    return k_end


def compute_sigma(epoch, n_epochs, sigma_start=3000.0, sigma_end=500.0):
    """Anneal the Gaussian source sigma (for the PDE source term)."""
    t1 = int(0.20 * n_epochs)
    t2 = int(0.60 * n_epochs)
    if epoch < t1:
        alpha = epoch / t1
        return sigma_start - (sigma_start - 1500.0) * alpha
    elif epoch < t2:
        alpha = (epoch - t1) / (t2 - t1)
        return 1500.0 - 800.0 * alpha
    else:
        alpha = (epoch - t2) / (n_epochs - t2)
        return 700.0 - 200.0 * alpha


# ═══════════════════════════════════════════════════════════════
# PDE RESIDUALS
# ═══════════════════════════════════════════════════════════════

def compute_pde_residuals(model, x_col, t_col, k):
    """
    PDE residuals for the water hammer equations:
      Continuity: dP/dt + (ρa²/A) dQ/dx = 0   (for the MLP part only)
      Momentum:   dQ/dt + (A/ρ) dP/dx + friction = 0  (on total fields)
    """
    x_col = x_col.clone().detach().requires_grad_(True)
    t_col = t_col.clone().detach().requires_grad_(True)

    P_total, Q_total, P_mlp, Q_mlp = model(x_col, t_col, k)

    ones = torch.ones_like(P_total)

    # Continuity on MLP part (the singular part satisfies it by construction)
    dP_mlp_dt = torch.autograd.grad(P_mlp, t_col, grad_outputs=ones, create_graph=True)[0]
    dQ_mlp_dx = torch.autograd.grad(Q_mlp, x_col, grad_outputs=ones, create_graph=True)[0]
    r_cont = dP_mlp_dt + (RHO * A_WAVE**2 / A_PIPE) * dQ_mlp_dx

    # Momentum on total fields (includes friction)
    dP_total_dx = torch.autograd.grad(P_total, x_col, grad_outputs=ones, create_graph=True)[0]
    dQ_total_dt = torch.autograd.grad(Q_total, t_col, grad_outputs=ones, create_graph=True)[0]
    friction = F_DARCY * Q_total * torch.abs(Q_total) / (2.0 * D_PIPE * A_PIPE)
    r_mom = dQ_total_dt + (A_PIPE / RHO) * dP_total_dx + friction

    return r_cont, r_mom


# ═══════════════════════════════════════════════════════════════
# LOSS FUNCTION (5 terms, no L_masa)
# ═══════════════════════════════════════════════════════════════

def compute_loss(model, data_tensors, x_col, t_col, lambdas, k):
    """
    L_P:    pressure sensor MSE
    L_Q:    flow meter MSE
    L_pde:  PDE residual
    L_bc:   boundary conditions
    L_ic:   initial conditions
    """
    device = x_col.device
    t = data_tensors['t']
    Nt = t.shape[0]

    # ── L_P: pressure sensors ──────────────────────────────────
    L_P = torch.tensor(0.0, device=device)
    for i, xs in enumerate(X_PRESSURE_SENSORS):
        x_t = torch.full((Nt,), xs, dtype=torch.float32, device=device)
        P_pred, _, _, _ = model(x_t, t, k)
        P_target = data_tensors['P_sensors'][i]
        L_P = L_P + torch.mean((P_pred / P_INLET - P_target / P_INLET)**2)
    L_P = L_P / len(X_PRESSURE_SENSORS)

    # ── L_Q: flow meters ───────────────────────────────────────
    L_Q = torch.tensor(0.0, device=device)
    for i, xs in enumerate(X_FLOW_METERS):
        x_t = torch.full((Nt,), xs, dtype=torch.float32, device=device)
        _, Q_pred, _, _ = model(x_t, t, k)
        Q_target = data_tensors['Q_sensors'][i]
        L_Q = L_Q + torch.mean((Q_pred / Q_OUTLET - Q_target / Q_OUTLET)**2)
    L_Q = L_Q / len(X_FLOW_METERS)

    # ── L_pde: PDE residual ────────────────────────────────────
    r_cont, r_mom = compute_pde_residuals(model, x_col, t_col, k)
    p_cont_scale = P_INLET * A_WAVE / L_PIPE
    p_mom_scale  = P_INLET / L_PIPE
    L_pde = torch.mean((r_cont / p_cont_scale)**2) + torch.mean((r_mom / p_mom_scale)**2)

    # ── L_bc: P(0,t) = P_inlet, Q(L,t) = Q_outlet ────────────
    t_bc = data_tensors['t_bc']
    P_x0, _, _, _ = model(torch.zeros_like(t_bc), t_bc, k)
    _, Q_xL, _, _ = model(torch.full_like(t_bc, L_PIPE), t_bc, k)
    L_bc = (torch.mean((P_x0 / P_INLET - 1.0)**2)
            + torch.mean((Q_xL / Q_OUTLET - 1.0)**2))

    # ── L_ic: start from steady state ─────────────────────────
    x_ic = data_tensors['x_ic']
    t0_ic = data_tensors['t0_ic']
    P_ic, Q_ic, _, _ = model(x_ic, t0_ic, k)
    P_ss_ic = data_tensors['P_ss_ic']
    L_ic = (torch.mean((P_ic / P_INLET - P_ss_ic / P_INLET)**2)
            + torch.mean((Q_ic / Q_OUTLET - 1.0)**2))

    # ── Total ──────────────────────────────────────────────────
    L_total = (lambdas['P'] * L_P
               + lambdas['Q'] * L_Q
               + lambdas['pde'] * L_pde
               + lambdas['bc'] * L_bc
               + lambdas['ic'] * L_ic)

    comps = {
        'L_total': L_total.item(),
        'L_P': L_P.item(),
        'L_Q': L_Q.item(),
        'L_pde': L_pde.item(),
        'L_bc': L_bc.item(),
        'L_ic': L_ic.item(),
    }
    return L_total, comps


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train(data, q_leak_known, n_epochs=5000, lr=1e-3, n_collocation=15000,
          use_lbfgs=True, lbfgs_epochs=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = SimpleLeakPINN(q_leak_known).to(device)
    print(f"  x_leak inicial: {model.x_leak.item():.0f}m")
    print(f"  q_leak (fijo):  {q_leak_known}")

    # Lambdas
    lambdas = {'P': 10.0, 'Q': 10.0, 'pde': 1.0, 'bc': 10.0, 'ic': 5.0}

    # Prepare data tensors
    t_tensor = torch.tensor(data['t'], dtype=torch.float32, device=device)
    P_sensors_t = torch.tensor(data['P_sensors'], dtype=torch.float32, device=device)
    Q_sensors_t = torch.tensor(data['Q_sensors'], dtype=torch.float32, device=device)

    Nbc = 200
    t_bc = torch.linspace(0, T_TOTAL, Nbc, dtype=torch.float32, device=device)

    Nic = 200
    x_ic = torch.linspace(0, L_PIPE, Nic, dtype=torch.float32, device=device)
    t0_ic = torch.zeros_like(x_ic)
    P_ss_ic = torch.tensor(
        P_INLET - F_DARCY * RHO * x_ic.cpu().numpy()
        * (Q_OUTLET * abs(Q_OUTLET)) / (2.0 * D_PIPE * A_PIPE**2),
        dtype=torch.float32, device=device
    )

    data_tensors = {
        't': t_tensor,
        'P_sensors': P_sensors_t,
        'Q_sensors': Q_sensors_t,
        't_bc': t_bc,
        'x_ic': x_ic,
        't0_ic': t0_ic,
        'P_ss_ic': P_ss_ic,
    }

    # Collocation points (pre-allocate, re-sample in-place)
    x_col = torch.empty(n_collocation, dtype=torch.float32, device=device)
    t_col = torch.empty(n_collocation, dtype=torch.float32, device=device)

    # Optimizer: separate LR for network vs x_leak
    optimizer = torch.optim.Adam([
        {'params': model.network_params(), 'lr': lr},
        {'params': [model.x_leak_raw],     'lr': 5e-3},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5)

    # Lambda annealing for PDE
    lambda_pde_start = 1.0
    lambda_pde_end   = 1000.0
    lambda_anneal_epochs = int(0.80 * n_epochs)

    history = []
    t_start = time.time()

    print(f"\n  {'Epoch':>6} | {'L_total':>9} | {'L_P':>9} | {'L_Q':>9} | "
          f"{'L_pde':>9} | {'x_leak':>8} | {'k':>8}")
    print("  " + "-" * 78)

    # ── Phase 1: Adam ────────────────────────────────────────
    for epoch in range(1, n_epochs + 1):
        # Lambda annealing
        if epoch < lambda_anneal_epochs:
            progress = epoch / lambda_anneal_epochs
            lambdas['pde'] = lambda_pde_start + (lambda_pde_end - lambda_pde_start) * progress
        else:
            lambdas['pde'] = lambda_pde_end

        k_current = compute_k(epoch, n_epochs)

        # Resample collocation points each epoch
        torch.manual_seed(epoch + 42)
        with torch.no_grad():
            x_col.uniform_(0.0, L_PIPE)
            t_col.uniform_(0.0, T_TOTAL)
        x_col.requires_grad_(True)
        t_col.requires_grad_(True)

        optimizer.zero_grad()
        loss_total, comps = compute_loss(
            model, data_tensors, x_col, t_col, lambdas, k_current)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # LR decay for x_leak at 60% and 80%
        if epoch == int(0.60 * n_epochs) or epoch == int(0.80 * n_epochs):
            for pg in optimizer.param_groups:
                if pg is optimizer.param_groups[1]:  # x_leak group
                    pg['lr'] *= 0.1
                    print(f"  [Epoch {epoch}] x_leak LR -> {pg['lr']:.2e}")

        if epoch % 10 == 0:
            history.append({
                'epoch': epoch,
                'x_leak': model.x_leak.item(),
                **comps,
            })

        if epoch == 1 or epoch % 500 == 0 or epoch == n_epochs:
            xl = model.x_leak.item()
            print(f"  {epoch:6d} | {comps['L_total']:9.3e} | {comps['L_P']:9.3e} | "
                  f"{comps['L_Q']:9.3e} | {comps['L_pde']:9.3e} | "
                  f"{xl:8.0f}m | {k_current:8.5f}")

    # ── Phase 2: L-BFGS refinement ───────────────────────────
    if use_lbfgs and lbfgs_epochs > 0:
        print(f"\n  ── L-BFGS refinement ({lbfgs_epochs} steps) ──")

        # Fixed collocation for deterministic landscape
        torch.manual_seed(42 + 999)
        with torch.no_grad():
            x_col_fixed = torch.empty(n_collocation, dtype=torch.float32, device=device)
            t_col_fixed = torch.empty(n_collocation, dtype=torch.float32, device=device)
            x_col_fixed.uniform_(0.0, L_PIPE)
            t_col_fixed.uniform_(0.0, T_TOTAL)
        x_col_fixed.requires_grad_(True)
        t_col_fixed.requires_grad_(True)

        k_final = compute_k(n_epochs, n_epochs)

        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=1,
            line_search_fn="strong_wolfe")

        prev_loss = float('inf')
        for step in range(1, lbfgs_epochs + 1):
            comps_lbfgs = {}
            def closure():
                nonlocal comps_lbfgs
                optimizer_lbfgs.zero_grad()
                loss, c = compute_loss(
                    model, data_tensors, x_col_fixed, t_col_fixed,
                    lambdas, k_final)
                loss.backward()
                comps_lbfgs = c
                return loss

            optimizer_lbfgs.step(closure)

            epoch_idx = n_epochs + step
            if step % 10 == 0:
                history.append({
                    'epoch': epoch_idx,
                    'x_leak': model.x_leak.item(),
                    **comps_lbfgs,
                })

            if step == 1 or step % 200 == 0 or step == lbfgs_epochs:
                xl = model.x_leak.item()
                print(f"  LBFGS {step:5d} | L_total: {comps_lbfgs['L_total']:.3e} | "
                      f"L_P: {comps_lbfgs['L_P']:.3e} | L_Q: {comps_lbfgs['L_Q']:.3e} | "
                      f"x_leak: {xl:.0f}m")

            # Convergence check
            curr = comps_lbfgs.get('L_total', prev_loss)
            if abs(prev_loss - curr) < 1e-12:
                print(f"  L-BFGS converged at step {step}")
                break
            prev_loss = curr

    elapsed = time.time() - t_start
    x_pred = model.x_leak.item()

    print(f"\n  {'='*50}")
    print(f"  x_leak pred: {x_pred:.0f} m")
    print(f"  Tiempo:      {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"  {'='*50}")

    return model, history


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_diagnostics(data, model, history, x_leak_true, q_leak_true,
                     save_dir='figs'):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    x_pred = model.x_leak.item()
    k_final = compute_k(100000, 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── (0,0) x_leak trajectory ─────────────────────────────
    ax = axes[0, 0]
    epochs = [h['epoch'] for h in history]
    xls = [h['x_leak'] for h in history]
    ax.plot(epochs, xls, 'b-', lw=1.5)
    ax.axhline(x_leak_true, color='r', ls='--', lw=2, label=f'True ({x_leak_true}m)')
    ax.axhline(x_pred, color='g', ls=':', lw=1.5, label=f'Pred ({x_pred:.0f}m)')
    ax.set_title(f'x_leak Convergence | Error: {abs(x_pred-x_leak_true):.0f}m')
    ax.set_xlabel('Epoch'); ax.set_ylabel('x_leak (m)')
    ax.legend()

    # ── (0,1) Loss components ───────────────────────────────
    ax = axes[0, 1]
    for key, label in [('L_total','Total'), ('L_P','Pressure'),
                        ('L_Q','Flow'), ('L_pde','PDE'),
                        ('L_bc','BC'), ('L_ic','IC')]:
        vals = [h[key] for h in history]
        ax.plot(epochs, vals, label=label)
    ax.set_yscale('log')
    ax.set_title('Loss Components')
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=8)

    # ── (0,2) Pressure field comparison ─────────────────────
    ax = axes[0, 2]
    t_arr = data['t']
    x_arr = data['x']
    P_moc = data['P_full']

    # Sample at a few time slices
    t_slices = [0.0, 60.0, 100.0, 150.0, 190.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(t_slices)))
    for t_s, col in zip(t_slices, colors):
        t_idx = np.argmin(np.abs(t_arr - t_s))
        ax.plot(x_arr, P_moc[:, t_idx] / 1e6, color=col, lw=1.5,
                label=f't={t_s:.0f}s (MOC)')
        # PINN prediction
        x_t = torch.tensor(x_arr, dtype=torch.float32, device=device)
        t_t = torch.full_like(x_t, t_arr[t_idx])
        with torch.no_grad():
            P_pinn, _, _, _ = model(x_t, t_t, k_final)
        ax.plot(x_arr, P_pinn.cpu().numpy() / 1e6, color=col, ls='--', lw=1)
    ax.axvline(x_leak_true, color='r', ls='--', alpha=0.5, label='Leak (true)')
    ax.axvline(x_pred, color='g', ls=':', alpha=0.5, label='Leak (pred)')
    ax.set_title('P(x) at time slices: MOC(─) vs PINN(--)')
    ax.set_xlabel('x (m)'); ax.set_ylabel('P (MPa)')
    ax.legend(fontsize=7)

    # ── (1,0) Pressure sensor fit ───────────────────────────
    ax = axes[1, 0]
    t_t = torch.tensor(t_arr, dtype=torch.float32, device=device)
    colors_s = ['tab:blue', 'tab:orange', 'tab:green']
    for i, xs in enumerate(X_PRESSURE_SENSORS):
        ax.plot(t_arr, data['P_sensors'][i] / 1e6,
                color=colors_s[i], lw=1.2, label=f'Data x={xs:.0f}m')
        x_t = torch.full_like(t_t, xs)
        with torch.no_grad():
            P_pred, _, _, _ = model(x_t, t_t, k_final)
        ax.plot(t_arr, P_pred.cpu().numpy() / 1e6,
                color=colors_s[i], ls='--', lw=1, alpha=0.8)
    ax.set_title('Pressure Sensors: Data(─) vs PINN(--)')
    ax.set_xlabel('t (s)'); ax.set_ylabel('P (MPa)')
    ax.legend(fontsize=8)

    # ── (1,1) Flow meter fit ────────────────────────────────
    ax = axes[1, 1]
    colors_q = ['tab:red', 'tab:purple']
    labels_q = ['Inlet (x=0)', 'Outlet (x=L)']
    for i, (xs, lbl) in enumerate(zip(X_FLOW_METERS, labels_q)):
        ax.plot(t_arr, data['Q_sensors'][i],
                color=colors_q[i], lw=1.2, label=f'Data {lbl}')
        x_t = torch.full_like(t_t, xs)
        with torch.no_grad():
            _, Q_pred, _, _ = model(x_t, t_t, k_final)
        ax.plot(t_arr, Q_pred.cpu().numpy(),
                color=colors_q[i], ls='--', lw=1, alpha=0.8)
    ax.set_title('Flow Meters: Data(─) vs PINN(--)')
    ax.set_xlabel('t (s)'); ax.set_ylabel('Q (m³/s)')
    ax.legend(fontsize=8)

    # ── (1,2) Summary text ──────────────────────────────────
    ax = axes[1, 2]
    ax.axis('off')
    err_m = abs(x_pred - x_leak_true)
    err_pct = err_m / L_PIPE * 100
    summary = (
        f"Results Summary\n"
        f"{'='*35}\n"
        f"x_leak true:  {x_leak_true:.0f} m\n"
        f"x_leak pred:  {x_pred:.0f} m\n"
        f"Error:        {err_m:.0f} m ({err_pct:.1f}%)\n"
        f"{'='*35}\n"
        f"q_leak (fixed): {q_leak_true}\n"
        f"{'='*35}\n"
        f"Final L_P:    {history[-1]['L_P']:.3e}\n"
        f"Final L_Q:    {history[-1]['L_Q']:.3e}\n"
        f"Final L_pde:  {history[-1]['L_pde']:.3e}\n"
    )
    ax.text(0.1, 0.5, summary, transform=ax.transAxes,
            fontsize=12, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle(
        f"Simple x_leak PINN | True: {x_leak_true}m | "
        f"Pred: {x_pred:.0f}m | Error: {err_m:.0f}m",
        fontsize=13, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'simple_xleak_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"  Figura guardada: {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple x_leak PINN')
    parser.add_argument('--x_leak', type=float, default=6000.0,
                        help='True leak position (default: 6000)')
    parser.add_argument('--q_leak', type=float, default=0.030,
                        help='Known leak flow rate (default: 0.030)')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Adam epochs (default: 5000)')
    parser.add_argument('--lbfgs', type=int, default=1000,
                        help='L-BFGS epochs (default: 1000)')
    parser.add_argument('--noise', type=float, default=500.0,
                        help='Noise std in Pa (default: 500)')
    parser.add_argument('--no_lbfgs', action='store_true',
                        help='Disable L-BFGS refinement')
    args = parser.parse_args()

    print("=" * 55)
    print("  Simple x_leak PINN (q_leak known)")
    print("=" * 55)

    print(f"\n[1/3] Generating data (MOC simulator)...")
    data = generate_data(args.x_leak, args.q_leak, noise_std=args.noise)
    print(f"  t: {data['t'].shape}, P_sensors: {data['P_sensors'].shape}, "
          f"Q_sensors: {data['Q_sensors'].shape}")

    print(f"\n[2/3] Training ({args.epochs} Adam + "
          f"{0 if args.no_lbfgs else args.lbfgs} L-BFGS)...")
    model, history = train(
        data, q_leak_known=args.q_leak,
        n_epochs=args.epochs,
        use_lbfgs=not args.no_lbfgs,
        lbfgs_epochs=args.lbfgs,
    )

    x_pred = model.x_leak.item()
    err = abs(x_pred - args.x_leak)
    print(f"\n  ======================================")
    print(f"  x_leak TRUE:  {args.x_leak:.0f} m")
    print(f"  x_leak PRED:  {x_pred:.0f} m")
    print(f"  ERROR:        {err:.0f} m ({err/L_PIPE*100:.1f}%)")
    print(f"  q_leak KNOWN: {args.q_leak}")
    if err < 500:
        print(f"  VERDICT:      SUCCESS (< 500m)")
    elif err < 1500:
        print(f"  VERDICT:      ~ MARGINAL")
    else:
        print(f"  VERDICT:      FAIL")
    print(f"  ======================================")

    print(f"\n[3/3] Plotting diagnostics...")
    plot_diagnostics(data, model, history, args.x_leak, args.q_leak)

    print("\nDone.")

"""
wave_pinn.py -- Wave-Injection PINN for leak localization

Cambio respecto a v1: q_leak ya NO se asume conocido.
Tanto x_leak como q_leak son parámetros aprendibles que la PINN
infiere simultáneamente a partir de los datos de sensores.
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
# PHYSICAL CONSTANTS
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
B_IMPEDANCE = RHO * A_WAVE / A_PIPE    # Characteristic impedance
X_PRESSURE_SENSORS = [1000.0, 5000.0, 9000.0]
X_FLOW_METERS      = [0.0, 10000.0]

# Rango físico razonable para q_leak [m³/s]
Q_LEAK_MIN = 0.001   # Fuga mínima detectable
Q_LEAK_MAX = 0.10    # Fuga máxima (~33% del caudal nominal)


# ═══════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_data(x_leak_true, q_leak_true, noise_std=500.0):
    print(f"  Simulando MOC: x_leak={x_leak_true}m, q_leak={q_leak_true}")
    moc = run_moc(Q_leak=q_leak_true, x_leak=x_leak_true,
                  t_leak=T_LEAK_START, noise_std=0.0)

    p_data = get_sensor_data(moc, X_PRESSURE_SENSORS, noise_std=noise_std)
    q_data = get_sensor_data(moc, X_FLOW_METERS, noise_std=noise_std * Q_OUTLET / P_INLET)

    x_grid = moc['x']
    P_ss = P_INLET - F_DARCY * RHO * x_grid * (Q_OUTLET * abs(Q_OUTLET)) / (2.0 * D_PIPE * A_PIPE**2)

    return {
        't': moc['t'],
        'x': x_grid,
        'P_full': moc['P'],
        'Q_full': moc['Q'],
        'P_sensors': p_data['P_sensors'],
        'Q_sensors': q_data['Q_sensors'],
        'P_ss': P_ss,
    }


# ═══════════════════════════════════════════════════════════════
# ARCHITECTURE: WaveLeakPINN_v2
# ═══════════════════════════════════════════════════════════════

class WaveLeakPINN(nn.Module):
    def __init__(self, hidden_layers=5, hidden_size=64):
        super().__init__()

        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 2))
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Override final layer to be exactly zero.
        # This ensures P_mlp_res starts at exactly 0.0,
        # protecting Phase 1 from massive random initialization noise!
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

        # Initialize off-center to break symmetry and avoid saddle points.
        # -1.0 maps to ~2920m (sigmoid(-1)*9000 + 500)
        self.x_leak_raw = nn.Parameter(torch.tensor(-1.0))

        # q_leak is now LEARNABLE.
        # Initialize at sigmoid(0)=0.5 → 0.5*(0.10-0.001)+0.001 ≈ 0.05 m³/s
        # This is deliberately above most real values so the wave is visible
        # from the start, providing strong gradients for x_leak.
        self.q_leak_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def x_leak(self):
        return torch.sigmoid(self.x_leak_raw) * 9000.0 + 500.0

    @property
    def q_leak(self):
        return torch.sigmoid(self.q_leak_raw) * (Q_LEAK_MAX - Q_LEAK_MIN) + Q_LEAK_MIN

    def physical_params(self):
        """Returns list of learnable physical parameters (x_leak, q_leak)."""
        return [self.x_leak_raw, self.q_leak_raw]

    def network_params(self):
        phys_ids = {id(self.x_leak_raw), id(self.q_leak_raw)}
        return [p for p in self.parameters() if id(p) not in phys_ids]

    def forward(self, x, t, k):
        orig_shape = x.shape

        # MLP prediction (now representing just the residual from steady state)
        x_in = (x / L_PIPE).reshape(-1, 1)
        t_in = (t / T_TOTAL).reshape(-1, 1)
        inp = torch.cat([x_in, t_in], dim=1)
        out = self.net(inp)
        
        # Scale residuals (10% of total scale is plenty for friction changes/reflections)
        P_mlp_res = out[:, 0].reshape(orig_shape) * (P_INLET * 0.1)
        Q_mlp_res = out[:, 1].reshape(orig_shape) * (Q_OUTLET * 0.1)

        # ── STEADY STATE BASELINE ──
        P_ss = P_INLET - F_DARCY * RHO * x * (Q_OUTLET * abs(Q_OUTLET)) / (2.0 * D_PIPE * A_PIPE**2)
        Q_ss = Q_OUTLET

        # Smoothed absolute value to prevent undefined derivatives
        # and allow exact cancellation in the momentum equation.
        abs_x = torch.sqrt((x - self.x_leak)**2 + (5.0)**2)
        q_leak = self.q_leak  # ← NOW LEARNABLE
        
        # Propagation delay from leak to x
        delay = abs_x / A_WAVE
        
        # Wave front argument: (t - t_leak - delay)
        z = t - T_LEAK_START - delay
        
        # Smoothed Heaviside step propagating outward at speed A_WAVE
        H_wave = torch.sigmoid(k * z)

        # Pressure drops by (B/2)*q_leak behind the wave front
        P_sing = - (B_IMPEDANCE / 2.0) * q_leak * H_wave

        # For the momentum equation (dQ/dt + A/rho dP/dx = 0) to be satisfied 
        # EXACTLY by the analytical part, we MUST have S(x) = A_WAVE * d(delay)/dx
        sgn_x = (x - self.x_leak) / abs_x
        Q_sing = - (q_leak / 2.0) * sgn_x * H_wave

        # Total fields
        P_total = P_ss + P_mlp_res + P_sing
        Q_total = Q_ss + Q_mlp_res + Q_sing

        # P_mlp and Q_mlp now contain the baseline so the PDE loss works transparently
        P_mlp = P_ss + P_mlp_res
        Q_mlp = Q_ss + Q_mlp_res

        return P_total, Q_total, P_mlp_res, Q_mlp_res


# ═══════════════════════════════════════════════════════════════
# PDE RESIDUALS & LOSS
# ═══════════════════════════════════════════════════════════════

def compute_k(epoch, phase1_epochs):
    # The arrival time requires a sharp wave eventually.
    # But to avoid vanishing gradients and local minima, we MUST start 
    # with a very smooth wave (k=0.5 covers the whole pipe delay).
    k_start = 0.5
    k_end   = 50.0
    t1 = phase1_epochs * 0.8  # Reach max sharpness before Phase 1 ends
    if epoch < t1:
        alpha = epoch / float(t1)
        return k_start + (k_end - k_start) * alpha
    return k_end

def compute_pde_residuals(model, x_col, t_col, k):
    x_col = x_col.clone().detach().requires_grad_(True)
    t_col = t_col.clone().detach().requires_grad_(True)

    P_total, Q_total, P_mlp_res, Q_mlp_res = model(x_col, t_col, k)
    ones = torch.ones_like(P_total)

    # ── CONTINUITY ──
    # The analytical P_sing and Q_sing exactly satisfy the source term (delta function).
    # P_ss and Q_ss satisfy the homogeneous continuity equation (0).
    # Therefore, the residual MLP part must satisfy the homogeneous equation!
    # Evaluating it only on MLP avoids the massive delta function penalty at x_leak.
    dP_mlp_dt = torch.autograd.grad(P_mlp_res, t_col, grad_outputs=ones, create_graph=True)[0]
    dQ_mlp_dx = torch.autograd.grad(Q_mlp_res, x_col, grad_outputs=ones, create_graph=True)[0]
    r_cont = dP_mlp_dt + (RHO * A_WAVE**2 / A_PIPE) * dQ_mlp_dx

    # ── MOMENTUM ──
    # Momentum has a non-linear friction term, so we MUST evaluate it on the TOTAL fields.
    # Fortunately, P_sing and Q_sing satisfy the linear momentum equation exactly
    # and have NO delta functions (only bounded kinks), so this won't explode.
    dP_total_dx = torch.autograd.grad(P_total, x_col, grad_outputs=ones, create_graph=True)[0]
    dQ_total_dt = torch.autograd.grad(Q_total, t_col, grad_outputs=ones, create_graph=True)[0]
    friction = F_DARCY * Q_total * torch.abs(Q_total) / (2.0 * D_PIPE * A_PIPE)
    r_mom = dQ_total_dt + (A_PIPE / RHO) * dP_total_dx + friction

    return r_cont, r_mom


def compute_loss(model, data_tensors, x_col, t_col, lambdas, k):
    device = x_col.device
    t = data_tensors['t']
    Nt = t.shape[0]

    L_P = torch.tensor(0.0, device=device)
    for i, xs in enumerate(X_PRESSURE_SENSORS):
        x_t = torch.full((Nt,), xs, dtype=torch.float32, device=device)
        P_pred, _, _, _ = model(x_t, t, k)
        P_target = data_tensors['P_sensors'][i]
        L_P = L_P + torch.mean((P_pred / P_INLET - P_target / P_INLET)**2)
    L_P = L_P / len(X_PRESSURE_SENSORS)

    L_Q = torch.tensor(0.0, device=device)
    for i, xs in enumerate(X_FLOW_METERS):
        x_t = torch.full((Nt,), xs, dtype=torch.float32, device=device)
        _, Q_pred, _, _ = model(x_t, t, k)
        Q_target = data_tensors['Q_sensors'][i]
        L_Q = L_Q + torch.mean((Q_pred / Q_OUTLET - Q_target / Q_OUTLET)**2)
    L_Q = L_Q / len(X_FLOW_METERS)

    # PDE
    r_cont, r_mom = compute_pde_residuals(model, x_col, t_col, k)
    p_cont_scale = P_INLET * A_WAVE / L_PIPE
    p_mom_scale  = P_INLET / L_PIPE
    L_pde = torch.mean((r_cont / p_cont_scale)**2) + torch.mean((r_mom / p_mom_scale)**2)

    # BC
    t_bc = data_tensors['t_bc']
    P_x0, _, _, _ = model(torch.zeros_like(t_bc), t_bc, k)
    _, Q_xL, _, _ = model(torch.full_like(t_bc, L_PIPE), t_bc, k)
    L_bc = (torch.mean((P_x0 / P_INLET - 1.0)**2) + torch.mean((Q_xL / Q_OUTLET - 1.0)**2))

    # IC
    x_ic = data_tensors['x_ic']
    t0_ic = data_tensors['t0_ic']
    P_ic, Q_ic, _, _ = model(x_ic, t0_ic, k)
    P_ss_ic = data_tensors['P_ss_ic']
    L_ic = (torch.mean((P_ic / P_INLET - P_ss_ic / P_INLET)**2) + torch.mean((Q_ic / Q_OUTLET - 1.0)**2))

    L_total = (lambdas['P'] * L_P + lambdas['Q'] * L_Q + 
               lambdas['pde'] * L_pde + lambdas['bc'] * L_bc + 
               lambdas['ic'] * L_ic)

    comps = {'L_total': L_total.item(), 'L_P': L_P.item(), 'L_Q': L_Q.item(), 'L_pde': L_pde.item()}
    return L_total, comps


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train(data, q_leak_true, n_epochs=5000, lr=1e-3, n_collocation=10000):
    """
    q_leak_true is only used for logging/diagnostics, NOT passed to the model.
    The model infers q_leak from data.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = WaveLeakPINN().to(device)
    print(f"  x_leak inicial: {model.x_leak.item():.0f}m")
    print(f"  q_leak inicial: {model.q_leak.item():.4f} m³/s (true: {q_leak_true:.4f})")

    lambdas = {'P': 100.0, 'Q': 100.0, 'pde': 1.0, 'bc': 10.0, 'ic': 10.0}

    # Data tensors
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

    x_col = torch.empty(n_collocation, dtype=torch.float32, device=device)
    t_col = torch.empty(n_collocation, dtype=torch.float32, device=device)

    # Optimizers
    opt_mlp = torch.optim.Adam(model.network_params(), lr=lr)
    # Joint optimizer for both physical parameters
    opt_phys = torch.optim.Adam(model.physical_params(), lr=5e-2)

    history = []
    t_start = time.time()

    print(f"\n  {'Epoch':>6} | {'Phase':>6} | {'L_total':>9} | {'L_P':>9} | {'L_pde':>9} | {'x_leak':>8} | {'q_leak':>8} | {'k':>8}")
    print("  " + "-" * 90)

    # Dynamic phases based on total epochs
    phase1_epochs = int(0.4 * n_epochs)
    phase2_epochs = int(0.3 * n_epochs)

    for epoch in range(1, n_epochs + 1):
        k_current = compute_k(epoch, phase1_epochs)

        torch.manual_seed(epoch + 42)
        with torch.no_grad():
            x_col.uniform_(0.0, L_PIPE)
            t_col.uniform_(0.0, T_TOTAL)
        x_col.requires_grad_(True)
        t_col.requires_grad_(True)

        if epoch <= phase1_epochs:
            phase = "1(phys)"
            # Optimize BOTH x_leak and q_leak using Data Loss only.
            # q_leak has linear gradients (amplitude scaling), so it converges
            # very quickly and doesn't interfere with x_leak (position/timing).
            opt_phys.zero_grad()
            
            # Compute data loss with graph for physical params
            L_P_tensor = torch.tensor(0.0, device=device)
            Nt = t_tensor.shape[0]
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

            # Log components from what we already computed (no PDE in Phase 1)
            comps = {
                'L_total': L_phase1.item(),
                'L_P': L_P_tensor.item(),
                'L_Q': L_Q_tensor.item(),
                'L_pde': 0.0,
            }
            
        elif epoch <= phase1_epochs + phase2_epochs:
            phase = "2(mlp)"
            
            # Phase 2: Freeze physical params, train MLP only.
            opt_mlp.zero_grad()
            
            # Anneal PDE weight in Phase 2
            progress = min(1.0, (epoch - phase1_epochs) / float(phase2_epochs))
            lambdas['pde'] = 1.0 + (1000.0 - 1.0) * progress

            L_total, comps = compute_loss(model, data_tensors, x_col, t_col, lambdas, k_current)
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_mlp.step()
            
        else:
            phase = "3(all)"
            
            # Phase 3: Joint fine-tuning of everything.
            if epoch == phase1_epochs + phase2_epochs + 1:
                for pg in opt_phys.param_groups:
                    pg['lr'] = 2e-3  # Moderate LR for fine-tuning
                    
            opt_mlp.zero_grad()
            opt_phys.zero_grad()
            
            L_total, comps = compute_loss(model, data_tensors, x_col, t_col, lambdas, k_current)
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_mlp.step()
            opt_phys.step()

            # Decay LR for physical params
            if epoch == int(0.80 * n_epochs) or epoch == int(0.90 * n_epochs):
                for pg in opt_phys.param_groups:
                    pg['lr'] *= 0.2

        if epoch % 10 == 0:
            history.append({
                'epoch': epoch,
                'x_leak': model.x_leak.item(),
                'q_leak': model.q_leak.item(),
                **comps
            })

        if epoch == 1 or epoch % 500 == 0 or epoch == phase1_epochs or epoch == n_epochs:
            xl = model.x_leak.item()
            ql = model.q_leak.item()
            print(f"  {epoch:6d} | {phase:>7} | {comps['L_total']:9.3e} | {comps['L_P']:9.3e} | {comps['L_pde']:9.3e} | {xl:7.0f}m | {ql:8.5f} | {k_current:8.5f}")

    elapsed = time.time() - t_start
    x_pred = model.x_leak.item()
    q_pred = model.q_leak.item()

    print(f"\n  {'='*55}")
    print(f"  x_leak pred: {x_pred:.0f} m")
    print(f"  q_leak pred: {q_pred:.5f} m³/s (true: {q_leak_true:.5f})")
    print(f"  Tiempo:      {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"  {'='*55}")

    return model, history


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_diagnostics(data, model, history, x_leak_true, q_leak_true, save_dir='figs'):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    # Use max k for the final plot
    k_final = compute_k(100000, 10000)

    x_pred = model.x_leak.item()
    q_pred = model.q_leak.item()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── x_leak convergence ──
    ax = axes[0, 0]
    epochs = [h['epoch'] for h in history]
    xls = [h['x_leak'] for h in history]
    ax.plot(epochs, xls, 'b-', lw=1.5)
    ax.axhline(x_leak_true, color='r', ls='--', lw=2, label=f'True ({x_leak_true}m)')
    ax.axhline(x_pred, color='g', ls=':', lw=1.5, label=f'Pred ({x_pred:.0f}m)')
    ax.set_title(f'x_leak Convergence | Error: {abs(x_pred-x_leak_true):.0f}m')
    ax.set_xlabel('Epoch'); ax.set_ylabel('x_leak (m)')
    ax.legend()

    # ── q_leak convergence ──
    ax = axes[0, 1]
    qls = [h['q_leak'] for h in history]
    ax.plot(epochs, qls, 'b-', lw=1.5)
    ax.axhline(q_leak_true, color='r', ls='--', lw=2, label=f'True ({q_leak_true:.4f})')
    ax.axhline(q_pred, color='g', ls=':', lw=1.5, label=f'Pred ({q_pred:.4f})')
    q_err_pct = abs(q_pred - q_leak_true) / q_leak_true * 100
    ax.set_title(f'q_leak Convergence | Error: {q_err_pct:.1f}%')
    ax.set_xlabel('Epoch'); ax.set_ylabel('q_leak (m³/s)')
    ax.legend()

    # ── Loss components ──
    ax = axes[0, 2]
    for key, label in [('L_total','Total'), ('L_P','Pressure'), ('L_Q','Flow'), ('L_pde','PDE')]:
        vals = [h[key] for h in history]
        ax.plot(epochs, vals, label=label)
    ax.set_yscale('log')
    ax.set_title('Loss Components')
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=8)

    # ── Pressure sensors ──
    ax = axes[1, 0]
    t_arr = data['t']
    t_t = torch.tensor(t_arr, dtype=torch.float32, device=device)
    colors_s = ['tab:blue', 'tab:orange', 'tab:green']
    for i, xs in enumerate(X_PRESSURE_SENSORS):
        ax.plot(t_arr, data['P_sensors'][i] / 1e6, color=colors_s[i], lw=1.2, label=f'Data x={xs:.0f}m')
        x_t = torch.full_like(t_t, xs)
        with torch.no_grad():
            P_pred, _, _, _ = model(x_t, t_t, k_final)
        ax.plot(t_arr, P_pred.cpu().numpy() / 1e6, color=colors_s[i], ls='--', lw=1, alpha=0.8)
    ax.set_title('Pressure Sensors: Data(-) vs PINN(--)')
    ax.set_xlabel('t (s)'); ax.set_ylabel('P (MPa)')
    ax.legend(fontsize=8)

    # ── Flow meters ──
    ax = axes[1, 1]
    colors_q = ['tab:red', 'tab:purple']
    labels_q = ['Inlet (x=0)', 'Outlet (x=L)']
    for i, (xs, lbl) in enumerate(zip(X_FLOW_METERS, labels_q)):
        ax.plot(t_arr, data['Q_sensors'][i], color=colors_q[i], lw=1.2, label=f'Data {lbl}')
        x_t = torch.full_like(t_t, xs)
        with torch.no_grad():
            _, Q_pred, _, _ = model(x_t, t_t, k_final)
        ax.plot(t_arr, Q_pred.cpu().numpy(), color=colors_q[i], ls='--', lw=1, alpha=0.8)
    ax.set_title('Flow Meters: Data(-) vs PINN(--)')
    ax.set_xlabel('t (s)'); ax.set_ylabel('Q (m3/s)')
    ax.legend(fontsize=8)

    # ── Summary box ──
    ax = axes[1, 2]
    ax.axis('off')
    err_m = abs(x_pred - x_leak_true)
    err_pct = err_m / L_PIPE * 100
    summary = (
        f"Results Summary (v2)\n"
        f"{'='*35}\n"
        f"x_leak true:  {x_leak_true:.0f} m\n"
        f"x_leak pred:  {x_pred:.0f} m\n"
        f"x error:      {err_m:.0f} m ({err_pct:.1f}%)\n"
        f"{'='*35}\n"
        f"q_leak true:  {q_leak_true:.5f} m³/s\n"
        f"q_leak pred:  {q_pred:.5f} m³/s\n"
        f"q error:      {abs(q_pred-q_leak_true):.5f} ({q_err_pct:.1f}%)\n"
        f"{'='*35}\n"
        f"Final L_P:    {history[-1]['L_P']:.3e}\n"
        f"Final L_Q:    {history[-1]['L_Q']:.3e}\n"
        f"Final L_pde:  {history[-1]['L_pde']:.3e}\n"
    )
    ax.text(0.1, 0.5, summary, transform=ax.transAxes, fontsize=11, verticalalignment='center', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle(f"Wave-PINN v2 | x: {x_leak_true}→{x_pred:.0f}m (err {err_m:.0f}m) | q: {q_leak_true:.4f}→{q_pred:.4f}", fontsize=13, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'wave_pinn_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"  Figura guardada: {save_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wave-Injection PINN v2 (q_leak learnable)')
    parser.add_argument('--x_leak', type=float, default=6000.0, help='True leak position (default: 6000)')
    parser.add_argument('--q_leak', type=float, default=0.030, help='True leak flow rate for data generation (default: 0.030)')
    parser.add_argument('--epochs', type=int, default=5000, help='Adam epochs (default: 5000)')
    parser.add_argument('--noise', type=float, default=500.0, help='Noise std in Pa (default: 500)')
    args = parser.parse_args()

    print("=" * 55)
    print("  Wave-Injection PINN v2 (q_leak LEARNABLE)")
    print("=" * 55)

    print(f"\n[1/3] Generating data (MOC simulator)...")
    data = generate_data(args.x_leak, args.q_leak, noise_std=args.noise)

    print(f"\n[2/3] Training ({args.epochs} epochs)...")
    model, history = train(data, q_leak_true=args.q_leak, n_epochs=args.epochs)

    x_pred = model.x_leak.item()
    q_pred = model.q_leak.item()
    x_err = abs(x_pred - args.x_leak)
    q_err = abs(q_pred - args.q_leak)
    print(f"\n  ======================================")
    print(f"  x_leak TRUE:  {args.x_leak:.0f} m")
    print(f"  x_leak PRED:  {x_pred:.0f} m")
    print(f"  x ERROR:      {x_err:.0f} m ({x_err/L_PIPE*100:.1f}%)")
    print(f"  --------------------------------------")
    print(f"  q_leak TRUE:  {args.q_leak:.5f} m³/s")
    print(f"  q_leak PRED:  {q_pred:.5f} m³/s")
    print(f"  q ERROR:      {q_err:.5f} ({q_err/args.q_leak*100:.1f}%)")
    print(f"  --------------------------------------")
    if x_err < 500:
        print(f"  x VERDICT:    SUCCESS (< 500m)")
    elif x_err < 1500:
        print(f"  x VERDICT:    ~ MARGINAL")
    else:
        print(f"  x VERDICT:    FAIL")
    print(f"  ======================================")

    print(f"\n[3/3] Plotting diagnostics...")
    plot_diagnostics(data, model, history, args.x_leak, args.q_leak)

    print("\nDone.")

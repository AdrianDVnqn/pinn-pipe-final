"""
wave_pinn_v3.py -- Wave-Injection PINN for leak localization (v3)

Cambios respecto a v2:
  - Agrega amortiguamiento por fricción a la onda analítica.
    En lugar de P_sing = -(B/2)*q*H (onda sin pérdida), usa:
      P_sing = -(B/2)*q * exp(-alpha * dist) * H
    donde alpha es un coeficiente de atenuación APRENDIBLE.
  - Esto modela la pérdida de amplitud de la onda transitoria
    por fricción de Darcy-Weisbach a medida que se propaga.
  - Hipótesis: el sesgo sistemático de ~150m en v1/v2 se debe
    a que la onda analítica sin fricción no coincide con la onda
    MOC con fricción, y el optimizador compensa desplazando x_leak.
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
Q_LEAK_MIN = 0.001
Q_LEAK_MAX = 0.10

# Rango para el coeficiente de atenuación alpha [1/m]
# Estimación teórica: alpha ~ f*v/(2*D*a) ≈ 0.02*1.53/(2*0.5*1200) ≈ 2.5e-5
# Rango amplio para dejar que el modelo encuentre el óptimo
ALPHA_MAX = 5e-4  # Demasiado alto causaría atenuación total en pocos km


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
# ARCHITECTURE: WaveLeakPINN_v3 (with friction damping)
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
        
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

        # x_leak: off-center init to break symmetry
        self.x_leak_raw = nn.Parameter(torch.tensor(-1.0))

        # q_leak: learnable amplitude
        self.q_leak_raw = nn.Parameter(torch.tensor(0.0))

        # alpha: friction damping coefficient [1/m]
        # Iniciar en ~0.0 para que la Fase 1 se comporte como v2 (sin fricción)
        # y ubique x_leak por timing. En Fase 3 se descongela para ajustar el sesgo.
        self.alpha_raw = nn.Parameter(torch.tensor(-10.0))

    @property
    def x_leak(self):
        return torch.sigmoid(self.x_leak_raw) * 9000.0 + 500.0

    @property
    def q_leak(self):
        return torch.sigmoid(self.q_leak_raw) * (Q_LEAK_MAX - Q_LEAK_MIN) + Q_LEAK_MIN

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw) * ALPHA_MAX

    def physical_params(self):
        """Returns list of learnable physical parameters."""
        return [self.x_leak_raw, self.q_leak_raw, self.alpha_raw]

    def network_params(self):
        phys_ids = {id(self.x_leak_raw), id(self.q_leak_raw), id(self.alpha_raw)}
        return [p for p in self.parameters() if id(p) not in phys_ids]

    def forward(self, x, t, k):
        orig_shape = x.shape

        # MLP residual
        x_in = (x / L_PIPE).reshape(-1, 1)
        t_in = (t / T_TOTAL).reshape(-1, 1)
        inp = torch.cat([x_in, t_in], dim=1)
        out = self.net(inp)
        
        P_mlp_res = out[:, 0].reshape(orig_shape) * (P_INLET * 0.1)
        Q_mlp_res = out[:, 1].reshape(orig_shape) * (Q_OUTLET * 0.1)

        # ── STEADY STATE BASELINE ──
        P_ss = P_INLET - F_DARCY * RHO * x * (Q_OUTLET * abs(Q_OUTLET)) / (2.0 * D_PIPE * A_PIPE**2)
        Q_ss = Q_OUTLET

        # Distance from leak (smoothed)
        abs_x = torch.sqrt((x - self.x_leak)**2 + (5.0)**2)
        q_leak = self.q_leak
        alpha = self.alpha
        
        # Propagation delay
        delay = abs_x / A_WAVE
        
        # Wave front
        z = t - T_LEAK_START - delay
        H_wave = torch.sigmoid(k * z)

        # ── FRICTION DAMPING ──
        # The wave attenuates exponentially as it propagates away from the leak.
        # This models the energy loss due to Darcy-Weisbach friction along the path.
        damping = torch.exp(-alpha * abs_x)

        # Damped analytical wave injection
        P_sing = - (B_IMPEDANCE / 2.0) * q_leak * damping * H_wave
        
        sgn_x = (x - self.x_leak) / abs_x
        Q_sing = - (q_leak / 2.0) * sgn_x * damping * H_wave

        # Total fields
        P_total = P_ss + P_mlp_res + P_sing
        Q_total = Q_ss + Q_mlp_res + Q_sing

        return P_total, Q_total, P_mlp_res, Q_mlp_res


# ═══════════════════════════════════════════════════════════════
# PDE RESIDUALS & LOSS
# ═══════════════════════════════════════════════════════════════

def compute_k(epoch, phase1_epochs):
    k_start = 0.5
    k_end   = 50.0
    t1 = phase1_epochs * 0.8
    if epoch < t1:
        a = epoch / float(t1)
        return k_start + (k_end - k_start) * a
    return k_end

def compute_pde_residuals(model, x_col, t_col, k):
    x_col = x_col.clone().detach().requires_grad_(True)
    t_col = t_col.clone().detach().requires_grad_(True)

    P_total, Q_total, P_mlp_res, Q_mlp_res = model(x_col, t_col, k)
    ones = torch.ones_like(P_total)

    # CONTINUITY (on MLP residual only — analytical part satisfies source term)
    dP_mlp_dt = torch.autograd.grad(P_mlp_res, t_col, grad_outputs=ones, create_graph=True)[0]
    dQ_mlp_dx = torch.autograd.grad(Q_mlp_res, x_col, grad_outputs=ones, create_graph=True)[0]
    r_cont = dP_mlp_dt + (RHO * A_WAVE**2 / A_PIPE) * dQ_mlp_dx

    # MOMENTUM (on total fields — includes non-linear friction)
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
    The model infers x_leak, q_leak, AND alpha from data.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    model = WaveLeakPINN().to(device)
    print(f"  x_leak inicial: {model.x_leak.item():.0f}m")
    print(f"  q_leak inicial: {model.q_leak.item():.4f} m³/s (true: {q_leak_true:.4f})")
    print(f"  alpha  inicial: {model.alpha.item():.2e} 1/m")

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

    # Freeze alpha initially to prevent parametric degeneracy in Phase 1 & 2
    model.alpha_raw.requires_grad_(False)

    # Optimizers
    opt_mlp = torch.optim.Adam(model.network_params(), lr=lr)
    opt_phys = torch.optim.Adam(model.physical_params(), lr=5e-2)

    history = []
    t_start = time.time()

    hdr = f"  {'Epoch':>6} | {'Phase':>7} | {'L_total':>9} | {'L_P':>9} | {'L_pde':>9} | {'x_leak':>7} | {'q_leak':>8} | {'alpha':>9} | {'k':>6}"
    print(f"\n{hdr}")
    print("  " + "-" * len(hdr))

    # Dynamic phases
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
            opt_phys.zero_grad()
            
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

            comps = {
                'L_total': L_phase1.item(),
                'L_P': L_P_tensor.item(),
                'L_Q': L_Q_tensor.item(),
                'L_pde': 0.0,
            }
            
        elif epoch <= phase1_epochs + phase2_epochs:
            phase = "2(mlp)"
            opt_mlp.zero_grad()
            
            progress = min(1.0, (epoch - phase1_epochs) / float(phase2_epochs))
            lambdas['pde'] = 1.0 + (1000.0 - 1.0) * progress

            L_total, comps = compute_loss(model, data_tensors, x_col, t_col, lambdas, k_current)
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_mlp.step()
            
        else:
            phase = "3(all)"
            
            if epoch == phase1_epochs + phase2_epochs + 1:
                model.alpha_raw.requires_grad_(True)
                for pg in opt_phys.param_groups:
                    pg['lr'] = 2e-3

            opt_mlp.zero_grad()
            opt_phys.zero_grad()
            
            L_total, comps = compute_loss(model, data_tensors, x_col, t_col, lambdas, k_current)
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt_mlp.step()
            opt_phys.step()

            if epoch == int(0.80 * n_epochs) or epoch == int(0.90 * n_epochs):
                for pg in opt_phys.param_groups:
                    pg['lr'] *= 0.2

        if epoch % 10 == 0:
            history.append({
                'epoch': epoch,
                'x_leak': model.x_leak.item(),
                'q_leak': model.q_leak.item(),
                'alpha': model.alpha.item(),
                **comps
            })

        if epoch == 1 or epoch % 500 == 0 or epoch == phase1_epochs or epoch == n_epochs:
            xl = model.x_leak.item()
            ql = model.q_leak.item()
            al = model.alpha.item()
            print(f"  {epoch:6d} | {phase:>7} | {comps['L_total']:9.3e} | {comps['L_P']:9.3e} | {comps['L_pde']:9.3e} | {xl:6.0f}m | {ql:8.5f} | {al:9.2e} | {k_current:6.1f}")

    elapsed = time.time() - t_start
    x_pred = model.x_leak.item()
    q_pred = model.q_leak.item()
    a_pred = model.alpha.item()

    # Compute damping at max sensor distance for context
    max_dist = max(abs(xs - x_pred) for xs in X_PRESSURE_SENSORS + X_FLOW_METERS)
    atten_pct = (1.0 - math.exp(-a_pred * max_dist)) * 100

    print(f"\n  {'='*60}")
    print(f"  x_leak pred:  {x_pred:.0f} m")
    print(f"  q_leak pred:  {q_pred:.5f} m³/s (true: {q_leak_true:.5f})")
    print(f"  alpha pred:   {a_pred:.2e} 1/m")
    print(f"  Max damping:  {atten_pct:.1f}% at {max_dist:.0f}m from leak")
    print(f"  Tiempo:       {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"  {'='*60}")

    return model, history


# ═══════════════════════════════════════════════════════════════
# DIAGNOSTIC PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_diagnostics(data, model, history, x_leak_true, q_leak_true, save_dir='figs'):
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    k_final = compute_k(100000, 10000)

    x_pred = model.x_leak.item()
    q_pred = model.q_leak.item()
    a_pred = model.alpha.item()

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    # ── x_leak convergence ──
    ax = axes[0, 0]
    epochs = [h['epoch'] for h in history]
    xls = [h['x_leak'] for h in history]
    ax.plot(epochs, xls, 'b-', lw=1.5)
    ax.axhline(x_leak_true, color='r', ls='--', lw=2, label=f'True ({x_leak_true}m)')
    ax.axhline(x_pred, color='g', ls=':', lw=1.5, label=f'Pred ({x_pred:.0f}m)')
    err_m = abs(x_pred - x_leak_true)
    ax.set_title(f'x_leak | Error: {err_m:.0f}m ({err_m/L_PIPE*100:.2f}%)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('x_leak (m)')
    ax.legend(fontsize=8)

    # ── q_leak convergence ──
    ax = axes[0, 1]
    qls = [h['q_leak'] for h in history]
    ax.plot(epochs, qls, 'b-', lw=1.5)
    ax.axhline(q_leak_true, color='r', ls='--', lw=2, label=f'True ({q_leak_true:.4f})')
    ax.axhline(q_pred, color='g', ls=':', lw=1.5, label=f'Pred ({q_pred:.4f})')
    q_err_pct = abs(q_pred - q_leak_true) / q_leak_true * 100
    ax.set_title(f'q_leak | Error: {q_err_pct:.1f}%')
    ax.set_xlabel('Epoch'); ax.set_ylabel('q_leak (m³/s)')
    ax.legend(fontsize=8)

    # ── alpha convergence ──
    ax = axes[0, 2]
    als = [h['alpha'] for h in history]
    ax.plot(epochs, als, 'b-', lw=1.5)
    ax.axhline(a_pred, color='g', ls=':', lw=1.5, label=f'Final ({a_pred:.2e})')
    ax.set_title(f'α (damping) = {a_pred:.2e} 1/m')
    ax.set_xlabel('Epoch'); ax.set_ylabel('α (1/m)')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    ax.legend(fontsize=8)

    # ── Loss components ──
    ax = axes[0, 3]
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
    ax.set_title('Pressure: Data(-) vs PINN(--)')
    ax.set_xlabel('t (s)'); ax.set_ylabel('P (MPa)')
    ax.legend(fontsize=7)

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
    ax.set_title('Flow: Data(-) vs PINN(--)')
    ax.set_xlabel('t (s)'); ax.set_ylabel('Q (m³/s)')
    ax.legend(fontsize=8)

    # ── Damping profile ──
    ax = axes[1, 2]
    x_plot = np.linspace(0, L_PIPE, 500)
    dist_from_leak = np.abs(x_plot - x_pred)
    damping_profile = np.exp(-a_pred * dist_from_leak)
    ax.plot(x_plot, damping_profile * 100, 'b-', lw=2)
    ax.axvline(x_pred, color='g', ls=':', alpha=0.7, label=f'Leak ({x_pred:.0f}m)')
    ax.axvline(x_leak_true, color='r', ls='--', alpha=0.7, label=f'True ({x_leak_true:.0f}m)')
    for xs in X_PRESSURE_SENSORS:
        d = abs(xs - x_pred)
        atten = math.exp(-a_pred * d) * 100
        ax.plot(xs, atten, 'ko', ms=6)
        ax.annotate(f'{atten:.0f}%', (xs, atten), textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_title('Wave amplitude vs distance')
    ax.set_xlabel('x (m)'); ax.set_ylabel('Amplitude (%)')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)

    # ── Summary box ──
    ax = axes[1, 3]
    ax.axis('off')
    err_pct = err_m / L_PIPE * 100
    summary = (
        f"Results Summary (v3)\n"
        f"{'='*35}\n"
        f"x_leak true:  {x_leak_true:.0f} m\n"
        f"x_leak pred:  {x_pred:.0f} m\n"
        f"x error:      {err_m:.0f} m ({err_pct:.2f}%)\n"
        f"{'='*35}\n"
        f"q_leak true:  {q_leak_true:.5f} m³/s\n"
        f"q_leak pred:  {q_pred:.5f} m³/s\n"
        f"q error:      {abs(q_pred-q_leak_true):.5f} ({q_err_pct:.1f}%)\n"
        f"{'='*35}\n"
        f"alpha:        {a_pred:.2e} 1/m\n"
        f"{'='*35}\n"
        f"Final L_P:    {history[-1]['L_P']:.3e}\n"
        f"Final L_pde:  {history[-1]['L_pde']:.3e}\n"
    )
    ax.text(0.05, 0.5, summary, transform=ax.transAxes, fontsize=10, verticalalignment='center', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle(f"Wave-PINN v3 (damped) | x: {x_leak_true}→{x_pred:.0f}m (err {err_m:.0f}m) | q: {q_leak_true:.4f}→{q_pred:.4f} | α={a_pred:.2e}", fontsize=12, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'wave_pinn_v3_results.png')
    plt.savefig(save_path, dpi=150)
    print(f"  Figura guardada: {save_path}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wave-Injection PINN v3 (damped)')
    parser.add_argument('--x_leak', type=float, default=6000.0, help='True leak position (default: 6000)')
    parser.add_argument('--q_leak', type=float, default=0.030, help='True leak flow rate (default: 0.030)')
    parser.add_argument('--epochs', type=int, default=5000, help='Adam epochs (default: 5000)')
    parser.add_argument('--noise', type=float, default=500.0, help='Noise std in Pa (default: 500)')
    args = parser.parse_args()

    print("=" * 60)
    print("  Wave-Injection PINN v3 (FRICTION-DAMPED WAVE)")
    print("=" * 60)

    print(f"\n[1/3] Generating data (MOC simulator)...")
    data = generate_data(args.x_leak, args.q_leak, noise_std=args.noise)

    print(f"\n[2/3] Training ({args.epochs} epochs)...")
    model, history = train(data, q_leak_true=args.q_leak, n_epochs=args.epochs)

    x_pred = model.x_leak.item()
    q_pred = model.q_leak.item()
    a_pred = model.alpha.item()
    x_err = abs(x_pred - args.x_leak)
    q_err = abs(q_pred - args.q_leak)
    print(f"\n  ==========================================")
    print(f"  x_leak TRUE:  {args.x_leak:.0f} m")
    print(f"  x_leak PRED:  {x_pred:.0f} m")
    print(f"  x ERROR:      {x_err:.0f} m ({x_err/L_PIPE*100:.2f}%)")
    print(f"  ------------------------------------------")
    print(f"  q_leak TRUE:  {args.q_leak:.5f} m³/s")
    print(f"  q_leak PRED:  {q_pred:.5f} m³/s")
    print(f"  q ERROR:      {q_err:.5f} ({q_err/args.q_leak*100:.1f}%)")
    print(f"  ------------------------------------------")
    print(f"  alpha:        {a_pred:.2e} 1/m")
    print(f"  ------------------------------------------")
    if x_err < 500:
        print(f"  x VERDICT:    SUCCESS (< 500m)")
    elif x_err < 1500:
        print(f"  x VERDICT:    ~ MARGINAL")
    else:
        print(f"  x VERDICT:    FAIL")
    print(f"  ==========================================")

    print(f"\n[3/3] Plotting diagnostics...")
    plot_diagnostics(data, model, history, args.x_leak, args.q_leak)

    print("\nDone.")

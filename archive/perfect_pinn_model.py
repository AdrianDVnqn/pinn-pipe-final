import os
import math
import time
import logging
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import config as cfg
from data_utils import get_training_data


def setup_logger(log_dir: str = 'logs') -> logging.Logger:
    """Create a logger that writes to both console and a timestamped file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')

    logger = logging.getLogger('pinn')
    logger.setLevel(logging.DEBUG)
    # avoid duplicate handlers on re-import
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f'Log file: {os.path.abspath(log_file)}')
    return logger


log = setup_logger()


class PerfectLeakPINN(nn.Module):
    def __init__(self, hidden_layers: int = 5, hidden_size: int = 64, activation: str = "tanh"):
        super().__init__()
        self.L = float(cfg.PIPE_LENGTH)
        self.T_total = float(cfg.T_TOTAL)
        self.P_in = float(cfg.P_INLET)
        self.Q_out = float(cfg.Q_OUTLET)

        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(negative_slope=0.01)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(self.act)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(self.act)
        layers.append(nn.Linear(hidden_size, 2))

        self.net = nn.Sequential(*layers)

        # trainable leak params (raw, unconstrained)
        # initialize x_leak ~ L/2 (center of domain)
        self.x_leak_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5 → 5000 m
        self.q_leak_raw = nn.Parameter(torch.tensor(0.0))
        # m_slope represents the change in pressure gradient (Pa/m) due to the leak
        self.m_slope = nn.Parameter(torch.tensor(0.0))

        # initialize weights with Xavier uniform
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def x_leak(self) -> float:
        # sigmoid -> [0,1] then scale to [500,9500]
        return torch.sigmoid(self.x_leak_raw) * 9000.0 + 500.0

    @property
    def q_leak(self) -> float:
        # sigmoid -> [0,1] then scale to [0.001, 0.050]
        # symmetric gradient around raw=0, no vanishing gradient trap
        return torch.sigmoid(self.q_leak_raw) * 0.049 + 0.001

    def network_params(self):
        # Return all parameters except the trainable leak scalars.
        exclude = {id(self.x_leak_raw), id(self.q_leak_raw), id(self.m_slope)}
        return [p for p in self.parameters() if id(p) not in exclude]

    def forward(self, x: torch.Tensor, t: torch.Tensor, k: float) -> torch.Tensor:
        """
        x, t: tensors of same shape [...]. Returns P, Q in physical units.
        """
        orig_shape = x.shape
        x_in = (x / self.L).reshape(-1, 1)
        t_in = (t / self.T_total).reshape(-1, 1)
        
        inp = torch.cat([x_in, t_in], dim=1)
        out = self.net(inp)
        P_norm = out[:, 0]
        Q_norm = out[:, 1]
        P_mlp = P_norm.reshape(orig_shape) * self.P_in
        Q_mlp = Q_norm.reshape(orig_shape) * self.Q_out
        
        # Perfect Physics Injection with Continuation Method (k)
        import torch.nn.functional as F
        
        # Q_sing uses sigmoid to create a smooth C0 step.
        # We normalize it so Q_sing(0) == 0 and Q_sing(L) == -q_leak exactly, regardless of k
        L_pipe = 10000.0 # Assuming fixed pipe length for simplicity, but can use cfg.PIPE_LENGTH
        S_x = torch.sigmoid(k * (x - self.x_leak))
        S_0 = torch.sigmoid(k * (0.0 - self.x_leak))
        S_L = torch.sigmoid(k * (L_pipe - self.x_leak))
        S_norm = (S_x - S_0) / (S_L - S_0)
        
        Q_sing_base = -self.q_leak * S_norm
        
        SP_x = torch.nn.functional.softplus(x - self.x_leak, beta=k)
        SP_0 = torch.nn.functional.softplus(torch.zeros_like(x) - self.x_leak, beta=k)
        
        # Scale m_slope by characteristic pressure gradient (P_in / L) so Adam can train it efficiently
        m_slope_scaled = self.m_slope * (self.P_in / self.L)
        P_sing_base = m_slope_scaled * (SP_x - SP_0)
        
        # Temporal activation for the leak (starts at t=50s)
        leak_active = torch.sigmoid(10.0 * (t - 50.0))
        
        P_sing = P_sing_base * leak_active
        Q_sing = Q_sing_base * leak_active
        
        P_total = P_mlp + P_sing
        Q_total = Q_mlp + Q_sing

        return P_total, Q_total, P_mlp, Q_mlp


def compute_sigma(epoch, n_epochs, sigma_start=3000.0, sigma_end=500.0):
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




def compute_k(epoch, n_epochs):
    # Continuation Method: Start very blurry (k_start) to provide gradients to all sensors,
    # then progressively sharpen (k_end) to localize the leak perfectly.
    k_start = 1.0 / 3000.0
    k_end = 1.0 / 50.0
    t1 = int(0.25 * n_epochs)  # Anneal over the first 25% of epochs
    if epoch < t1:
        alpha = epoch / float(t1)
        return k_start + (k_end - k_start) * alpha
    else:
        return k_end

def compute_pde_residuals(model: PerfectLeakPINN, x_col: torch.Tensor, t_col: torch.Tensor, k: float):
    device = x_col.device
    x_col = x_col.clone().detach().requires_grad_(True)
    t_col = t_col.clone().detach().requires_grad_(True)

    P_total, Q_total, P_mlp, Q_mlp = model(x_col, t_col, k)

    ones = torch.ones_like(P_total, device=device)

    # Continuity equation is satisfied exactly by the smooth parts
    dP_mlp_dt = torch.autograd.grad(P_mlp, t_col, grad_outputs=ones, create_graph=True)[0]
    dQ_mlp_dx = torch.autograd.grad(Q_mlp, x_col, grad_outputs=ones, create_graph=True)[0]

    rho = float(cfg.FLUID_DENSITY)
    a = float(cfg.WAVE_SPEED)
    A = math.pi * cfg.PIPE_DIAMETER ** 2 / 4.0

    r_cont = dP_mlp_dt + (rho * a * a / A) * dQ_mlp_dx

    # Momentum equation is evaluated on the TOTAL fields because m_slope cancels the step in friction
    dP_total_dx = torch.autograd.grad(P_total, x_col, grad_outputs=ones, create_graph=True)[0]
    dQ_total_dt = torch.autograd.grad(Q_total, t_col, grad_outputs=ones, create_graph=True)[0]

    f = float(cfg.FRICTION_FACTOR)
    D = float(cfg.PIPE_DIAMETER)
    friction = f * Q_total * torch.abs(Q_total) / (2.0 * D * A)

    r_mom = dQ_total_dt + (A / rho) * dP_total_dx + friction

    return r_cont, r_mom


def compute_loss(model: PerfectLeakPINN, data_dict: Dict, x_col: torch.Tensor, t_col: torch.Tensor, lambdas: Dict, k: float):
    device = x_col.device
    mse = nn.MSELoss()

    t = data_dict['t_tensor']
    P_noisy_tensor = data_dict['P_noisy_tensor']
    Q_noisy_tensor = data_dict['Q_noisy_tensor']
    x_pressure_sensors = data_dict['x_pressure_sensors_tensor']
    x_flow_meters = data_dict['x_flow_meters_tensor']
    P_inlet = data_dict['P_INLET_t']
    Q_outlet = data_dict['Q_OUTLET_t']

    Nt = t.shape[0]

    L_P = torch.tensor(0.0, device=device)
    for i in range(x_pressure_sensors.shape[0]):
        x_val = x_pressure_sensors[i].item()
        x_tensor = torch.full((Nt,), x_val, dtype=torch.float32, device=device)
        P_pred, _, _, _ = model(x_tensor, t, k)
        P_target = P_noisy_tensor[i]
        L_P = L_P + mse(P_pred / P_inlet, P_target / P_inlet)
    L_P = L_P / float(x_pressure_sensors.shape[0])

    L_Q = torch.tensor(0.0, device=device)
    for i in range(x_flow_meters.shape[0]):
        x_val = x_flow_meters[i].item()
        x_tensor = torch.full((Nt,), x_val, dtype=torch.float32, device=device)
        _, Q_pred, _, _ = model(x_tensor, t, k)
        Q_target = Q_noisy_tensor[i]
        L_Q = L_Q + mse(Q_pred / Q_outlet, Q_target / Q_outlet)
    L_Q = L_Q / float(x_flow_meters.shape[0])

    r_cont, r_mom = compute_pde_residuals(model, x_col, t_col, k)
    p_cont_scale = float(cfg.P_INLET) * float(cfg.WAVE_SPEED) / float(cfg.PIPE_LENGTH)
    p_mom_scale = float(cfg.P_INLET) / float(cfg.PIPE_LENGTH)
    L_fisica = torch.mean((r_cont / p_cont_scale) ** 2) + torch.mean((r_mom / p_mom_scale) ** 2)

    P_x0, _, _, _ = model(data_dict['x0_bc'], data_dict['t_bc'], k)
    _, Q_xL, _, _ = model(data_dict['xL_bc'], data_dict['t_bc'], k)
    L_contorno = mse(P_x0 / P_inlet, torch.ones_like(P_x0)) + mse(Q_xL / Q_outlet, torch.ones_like(Q_xL))

    P_ic, Q_ic, _, _ = model(data_dict['x_ic'], data_dict['t0_ic'], k)
    L_ic = mse(P_ic / P_inlet, data_dict['P_ss_x'] / P_inlet) + mse(Q_ic / Q_outlet, torch.ones_like(Q_ic))

    idx_late = slice(int(Nt * 0.8), Nt)
    Q_in_late = Q_noisy_tensor[0, idx_late]
    Q_out_late = Q_noisy_tensor[1, idx_late]
    q_leak_data_est = torch.mean(Q_in_late - Q_out_late)
    
    L_masa = mse(model.q_leak / Q_outlet, q_leak_data_est / Q_outlet)

    L_total = (
        lambdas.get('P', 10.0) * L_P
        + lambdas.get('Q', 20.0) * L_Q
        + lambdas.get('pde', 1.0) * L_fisica
        + lambdas.get('bc', 5.0) * L_contorno
        + lambdas.get('ic', 5.0) * L_ic
        + lambdas.get('masa', 10.0) * L_masa
    )

    components = {
        'L_total': float(L_total.detach().cpu().item()),
        'L_P': float(L_P.detach().cpu().item()),
        'L_Q': float(L_Q.detach().cpu().item()),
        'L_fisica': float(L_fisica.detach().cpu().item()),
        'L_contorno': float(L_contorno.detach().cpu().item()),
        'L_ic': float(L_ic.detach().cpu().item()),
        'L_masa': float(L_masa.detach().cpu().item()),
    }
    tensors = {
        'L_P': L_P,
        'L_Q': L_Q,
        'L_fisica': L_fisica,
        'L_bc': L_contorno,
        'L_ic': L_ic,
        'L_masa': L_masa,
    }
    return L_total, components, tensors


def train_pinn(scenario_id: int, noise_level: str = 'trivial', n_pressure_sensors: int = 3, lambdas: Dict = None,
               n_epochs: int = 10000, lr: float = 1e-3, n_collocation: int = None, use_lbfgs: bool = True,
               lbfgs_epochs: int = 2000, initial_x_leak: float = 5000.0, activation: str = "tanh", verbose: bool = True):
    if lambdas is None:
        lambdas = {"P": 10.0, "Q": 10.0, "pde": 1000.0, "bc": 10.0, "masa": 1000.0}

    torch.set_float32_matmul_precision('high')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data = get_training_data(scenario_id, noise_level, n_pressure_sensors)

    if n_collocation is None:
        n_collocation = 20_000 if device.type == 'cuda' else 8_000

    log.info(f"Device: {device}")

    actual_adam_epochs = n_epochs - lbfgs_epochs if use_lbfgs else n_epochs
    actual_lbfgs_epochs = lbfgs_epochs if use_lbfgs else 0

    model = PerfectLeakPINN(hidden_layers=5, hidden_size=64, activation=activation).to(device)
    
    p = (initial_x_leak - 500.0) / 9000.0
    p = max(1e-5, min(1.0 - 1e-5, p))
    logit = math.log(p / (1 - p))
    model.x_leak_raw.data = torch.tensor(logit, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam([
        {'params': model.network_params(), 'lr': lr},
        {'params': [model.x_leak_raw], 'lr': 5e-3},
        {'params': [model.q_leak_raw], 'lr': 5e-3},
        {'params': [model.m_slope], 'lr': 5e-3},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, actual_adam_epochs), eta_min=1e-5)

    history = []

    T_START = 0.0
    mask_t = data['t'] >= T_START
    t_filt = data['t'][mask_t]
    P_noisy_filt = data['P_noisy'][:, mask_t]
    Q_noisy_filt = data['Q_noisy'][:, mask_t]

    t_tensor = torch.tensor(t_filt, dtype=torch.float32, device=device)
    P_noisy_tensor = torch.tensor(P_noisy_filt, dtype=torch.float32, device=device)
    Q_noisy_tensor = torch.tensor(Q_noisy_filt, dtype=torch.float32, device=device)
    x_pressure_sensors_tensor = torch.tensor(data['x_pressure_sensors_used'], dtype=torch.float32, device=device)
    x_flow_meters_tensor = torch.tensor(data['x_flow_meters'], dtype=torch.float32, device=device)
    P_INLET_t = torch.tensor(float(cfg.P_INLET), dtype=torch.float32, device=device)
    Q_OUTLET_t = torch.tensor(float(cfg.Q_OUTLET), dtype=torch.float32, device=device)
    RHO_t = torch.tensor(float(cfg.FLUID_DENSITY), dtype=torch.float32, device=device)

    Nbc = 200
    t_bc = torch.linspace(T_START, float(cfg.T_TOTAL), Nbc, dtype=torch.float32, device=device)
    x0_bc = torch.zeros_like(t_bc, device=device)
    xL_bc = torch.full_like(t_bc, float(cfg.PIPE_LENGTH), device=device)
    x_ic = torch.linspace(0.0, float(cfg.PIPE_LENGTH), 200, dtype=torch.float32, device=device)
    t0_ic = torch.zeros_like(x_ic, device=device)
    A = math.pi * cfg.PIPE_DIAMETER ** 2 / 4.0
    P_ss_x = (float(cfg.P_INLET) - float(cfg.FRICTION_FACTOR) * float(cfg.FLUID_DENSITY) * x_ic * (float(cfg.Q_OUTLET) * abs(float(cfg.Q_OUTLET))) / (2.0 * float(cfg.PIPE_DIAMETER) * A ** 2))
    P_ss_x = P_ss_x.clone().detach().to(dtype=torch.float32, device=device)

    device_data = {
        'scenario_id': scenario_id,
        'noise_level': noise_level,
        'n_pressure_sensors': n_pressure_sensors,
        't_tensor': t_tensor,
        'P_noisy_tensor': P_noisy_tensor,
        'Q_noisy_tensor': Q_noisy_tensor,
        'x_pressure_sensors_tensor': x_pressure_sensors_tensor,
        'x_flow_meters_tensor': x_flow_meters_tensor,
        'P_INLET_t': P_INLET_t,
        'Q_OUTLET_t': Q_OUTLET_t,
        'RHO_t': RHO_t,
        't_bc': t_bc,
        'x0_bc': x0_bc,
        'xL_bc': xL_bc,
        'x_ic': x_ic,
        't0_ic': t0_ic,
        'P_ss_x': P_ss_x,
    }

    try:
        from simulator import run_moc
        moc_test = run_moc(Q_leak=0.015, x_leak=6000.0, t_leak=50.0, noise_std=0.0)
        dt_test = moc_test["t"][1] - moc_test["t"][0]
        dx_test = moc_test["x"][1] - moc_test["x"][0]
        P_test = moc_test["P"]
        dP_dt_typical = np.abs(np.diff(P_test, axis=1) / dt_test).mean()
        dP_dx_typical = np.abs(np.diff(P_test, axis=0) / dx_test).mean()
        p_cont_scale_val = float(cfg.P_INLET) * float(cfg.WAVE_SPEED) / float(cfg.PIPE_LENGTH)
        p_mom_scale_val = float(cfg.P_INLET) / float(cfg.PIPE_LENGTH)
        if verbose:
            log.info("─── Verificación de escalas ───")
            log.info(f"  |∂P/∂t| típico: {dP_dt_typical:.2e} Pa/s")
            log.info(f"  p_cont_scale:   {p_cont_scale_val:.2e} Pa/s")
            ratio_cont = dP_dt_typical / p_cont_scale_val
            log.info(f"  ratio cont:     {ratio_cont:.3f}  ← debe ser O(1)")
            log.info(f"  |∂P/∂x| típico: {dP_dx_typical:.2e} Pa/m")
            log.info(f"  p_mom_scale:    {p_mom_scale_val:.2e} Pa/m")
            ratio_mom = dP_dx_typical / p_mom_scale_val
            log.info(f"  ratio mom:      {ratio_mom:.3f}  ← debe ser O(1)")
            if ratio_cont < 0.01 or ratio_cont > 100 or ratio_mom < 0.01 or ratio_mom > 100:
                log.warning("  ✗ La escala está mal: ajustar p_cont_scale o p_mom_scale")
            else:
                log.info("  ✓ Escalas correctas: L_fisica tendrá peso real")
            log.info("─────────────────────────────────────")
    except Exception as e:
        log.warning(f"No se pudo correr la verificación dimensional: {str(e)}")

    x_col = torch.empty(n_collocation, dtype=torch.float32, device=device)
    t_col = torch.empty(n_collocation, dtype=torch.float32, device=device)
    t_start_total = time.time()
    t_epoch_start = time.time()

    lambda_pde_start = 1.0
    lambda_pde_end   = 1000.0
    lambda_pde_anneal_epochs = int(0.80 * actual_adam_epochs)

    for epoch in range(1, actual_adam_epochs + 1):
        model.current_epoch = epoch
        if epoch < lambda_pde_anneal_epochs:
            progress = epoch / lambda_pde_anneal_epochs
            lambdas['pde'] = lambda_pde_start + (lambda_pde_end - lambda_pde_start) * progress
        else:
            lambdas['pde'] = lambda_pde_end
            
        k_current = compute_k(epoch, actual_adam_epochs)
        
        torch.manual_seed(epoch + int(cfg.RANDOM_SEED))
        with torch.no_grad():
            x_col.uniform_(0.0, float(cfg.PIPE_LENGTH))
            t_col.uniform_(0.0, float(cfg.T_TOTAL))
        x_col.requires_grad_()
        t_col.requires_grad_()

        optimizer.zero_grad()
        loss_total, comps, comps_tensors = compute_loss(model, {**device_data}, x_col, t_col, lambdas, k_current)
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        epoch_fase3 = int(0.60 * actual_adam_epochs)
        epoch_decay2 = int(0.80 * actual_adam_epochs)

        if epoch == epoch_fase3:
            for pg in optimizer.param_groups:
                if 'name' in pg and pg['name'] == 'x_leak' or pg is optimizer.param_groups[1]:
                    pg['lr'] *= 0.1
                    log.info(f"  [Epoch {epoch}] LR decay x_leak: lr → {pg['lr']:.2e}")

        if epoch == epoch_decay2:
            for pg in optimizer.param_groups:
                if 'name' in pg and pg['name'] == 'x_leak' or pg is optimizer.param_groups[1]:
                    pg['lr'] *= 0.1
                    log.info(f"  [Epoch {epoch}] LR decay x_leak: lr → {pg['lr']:.2e}")

        if epoch % 10 == 0:
            history.append({
                'epoch': epoch,
                'loss_total': comps['L_total'],
                'L_P': comps['L_P'],
                'L_Q': comps['L_Q'],
                'L_fisica': comps['L_fisica'],
                'L_contorno': comps['L_contorno'],
                'L_ic': comps['L_ic'],
                'L_masa': comps['L_masa'],
                'x_leak_pred': float(model.x_leak.detach().cpu().numpy()),
                'q_leak_pred': float(model.q_leak.detach().cpu().numpy()),
            })

        if epoch == 1000 and verbose:
            log.info("─── Balance de loss en epoch 1000 ───")
            log.info(f"  L_P    × λ_P    = {comps['L_P'] * lambdas['P']:.3e}")
            log.info(f"  L_Q    × λ_Q    = {comps['L_Q'] * lambdas['Q']:.3e}")
            log.info(f"  L_fis  × λ_pde  = {comps['L_fisica'] * lambdas['pde']:.3e}")
            log.info(f"  L_masa × λ_masa = {comps['L_masa'] * lambdas['masa']:.3e}")
            log.info("  → Si L_fis << L_P: subir λ_pde o bajar λ_P")
            log.info("  → Si L_fis >> L_P: bajar λ_pde")
            log.info("─────────────────────────────────────")

        if verbose and (epoch == 1 or epoch % 500 == 0 or epoch == actual_adam_epochs):
            elapsed = time.time() - t_start_total
            log.info(f"Epoch {epoch:5d} | Loss: {loss_total:.3e} | x_leak: {model.x_leak.item():.0f}m")
            t_epoch_start = time.time()

    total_epochs_run = actual_adam_epochs
    if use_lbfgs and actual_lbfgs_epochs > 0:
        log.info(f"─── Fase L-BFGS: Refinamiento ({actual_lbfgs_epochs} iteraciones máx) ───")
        
        torch.manual_seed(42 + int(cfg.RANDOM_SEED))
        with torch.no_grad():
            x_col_fixed = torch.empty(n_collocation, dtype=torch.float32, device=device)
            t_col_fixed = torch.empty(n_collocation, dtype=torch.float32, device=device)
            x_col_fixed.uniform_(0.0, float(cfg.PIPE_LENGTH))
            t_col_fixed.uniform_(0.0, float(cfg.T_TOTAL))
        x_col_fixed.requires_grad_()
        t_col_fixed.requires_grad_()

        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=1,
            line_search_fn="strong_wolfe"
        )
        prev_loss = float('inf')
        
        for step in range(1, actual_lbfgs_epochs + 1):
            epoch_idx = actual_adam_epochs + step
            total_epochs_run = epoch_idx
            model.current_epoch = epoch_idx
            
            comps_lbfgs = None
            loss_val = None
            
            def closure():
                nonlocal comps_lbfgs, loss_val
                optimizer_lbfgs.zero_grad()
                k_end = compute_k(epoch_idx, actual_adam_epochs)
                loss_total, comps, comps_tensors = compute_loss(model, device_data, x_col_fixed, t_col_fixed, lambdas, k_end)
                loss_total.backward()
                comps_lbfgs = comps
                loss_val = loss_total.item()
                return loss_total
            
            optimizer_lbfgs.step(closure)
            
            # Retrieve values after the step
            l_val = loss_val if loss_val is not None else prev_loss
            c_lbfgs = comps_lbfgs
            
            # Log progress
            if c_lbfgs is not None:
                # Add to history
                if epoch_idx % 10 == 0:
                    history.append({
                        'epoch': epoch_idx,
                        'loss_total': c_lbfgs['L_total'],
                        'L_P': c_lbfgs['L_P'],
                        'L_Q': c_lbfgs['L_Q'],
                        'L_fisica': c_lbfgs['L_fisica'],
                        'L_contorno': c_lbfgs['L_contorno'],
                        'L_ic': c_lbfgs['L_ic'],
                        'L_masa': c_lbfgs['L_masa'],
                        'x_leak_pred': float(model.x_leak.detach().cpu().numpy()),
                        'q_leak_pred': float(model.q_leak.detach().cpu().numpy()),
                    })
                
                if verbose and (step == 1 or step % progress_every == 0 or step == actual_lbfgs_epochs):
                    elapsed = time.time() - t_start_total
                    recent_epochs = progress_every if step >= progress_every else step
                    epoch_ms = (time.time() - t_epoch_start) / float(recent_epochs) * 1000.0
                    vram_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
                    log.info(
                        f"Epoch {epoch_idx:5d} (LBFGS) | "
                        f"L_total: {c_lbfgs['L_total']:.3e} | "
                        f"L_P: {c_lbfgs['L_P']:.3e} | "
                        f"L_Q: {c_lbfgs['L_Q']:.3e} | "
                        f"L_fis: {c_lbfgs['L_fisica']:.3e} | "
                        f"L_bc: {c_lbfgs['L_contorno']:.3e} | "
                        f"L_ic: {c_lbfgs['L_ic']:.3e} | "
                        f"L_m: {c_lbfgs['L_masa']:.3e} | "
                        f"x_leak: {model.x_leak.item():.0f}m | "
                        f"q_leak: {model.q_leak.item():.5f} | "
                        f"λ_pde: {lambdas['pde']:.1f} | "
                        f"{epoch_ms:.1f} ms/epoch | Elapsed: {elapsed/60:.1f} min"
                    )
                    t_epoch_start = time.time()
            
            # Convergence check
            if abs(prev_loss - l_val) < 1e-12:
                log.info(f"L-BFGS converged at step {step} (change in loss < 1e-12). Stopping.")
                break
                
            prev_loss = l_val
            
            # periodic checkpoint during L-BFGS
            if epoch_idx % 2000 == 0:
                ckpt = {
                    'epoch': epoch_idx,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer_lbfgs.state_dict(),
                    'x_leak_pred': model.x_leak.item(),
                    'q_leak_pred': model.q_leak.item(),
                    'loss_total': l_val,
                    'history': history,
                }
                torch.save(ckpt, os.path.join('checkpoints', f'pinn_epoch_{epoch_idx}.pt'))

    df = pd.DataFrame(history)

    x_pred = float(model.x_leak.detach().cpu().numpy())
    q_pred = float(model.q_leak.detach().cpu().numpy())

    total_time = time.time() - t_start_total
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    result = {
        'model': model,
        'history': df,
        'scenario_id': scenario_id,
        'noise_level': noise_level,
        'n_pressure_sensors': n_pressure_sensors,
        'x_leak_pred': x_pred,
        'q_leak_pred': q_pred,
        'x_leak_true': data.get('x_leak'),
        'q_leak_true': data.get('q_leak'),
        'x_leak_error_km': abs(x_pred - data['x_leak']) / 1000.0 if data.get('x_leak') is not None else np.nan,
        'q_leak_error_pct': abs(q_pred - data['q_leak']) / data['q_leak'] * 100.0 if data.get('q_leak') is not None and data['q_leak'] > 0 else np.nan,
        'peak_vram_gb': peak_vram,
        'training_time_s': total_time,
        'ms_per_epoch': total_time / float(total_epochs_run) * 1000.0 if total_epochs_run > 0 else 0.0,
    }
    return result


def plot_training_diagnostics(train_result: Dict, save_dir: str = 'figs'):
    os.makedirs(save_dir, exist_ok=True)
    df = train_result['history']
    model = train_result['model']

    # Loss curves
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs = axs.flatten()
    if not df.empty:
        axs[0].plot(df['epoch'], df['loss_total'])
        axs[0].set_yscale('log')
        axs[0].set_title('Total loss')

        axs[1].plot(df['epoch'], df['L_P'], label='L_P')
        axs[1].plot(df['epoch'], df['L_Q'], label='L_Q')
        axs[1].plot(df['epoch'], df['L_fisica'], label='physics')
        axs[1].plot(df['epoch'], df['L_contorno'], label='bc')
        axs[1].plot(df['epoch'], df['L_inicial'], label='ic')
        axs[1].set_yscale('log')
        axs[1].legend()
        axs[1].set_title('Loss components')

        axs[2].plot(df['epoch'], df['x_leak_pred'])
        axs[2].axhline(train_result['x_leak_true'], color='k', linestyle='--')
        axs[2].set_title('x_leak vs epoch')

        axs[3].plot(df['epoch'], df['q_leak_pred'])
        axs[3].axhline(train_result['q_leak_true'], color='k', linestyle='--')
        axs[3].set_title('q_leak vs epoch')

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'pinn_loss_curves.png'))
    plt.close(fig)

    # Pressure field comparison
    data = get_training_data(
        train_result.get('scenario_id', 7),
        train_result.get('noise_level', 'trivial'),
        train_result.get('n_pressure_sensors', 3),
    )
    x = data['x']
    t = data['t']
    P_moc = data['P_full']

    Xg, Tg = np.meshgrid(x, t, indexing='xy')
    device = next(model.parameters()).device
    x_flat = torch.tensor(Xg.flatten(), dtype=torch.float32, device=device)
    t_flat = torch.tensor(Tg.flatten(), dtype=torch.float32, device=device)
    k_final = compute_k(100000, 1) # get max k
    with torch.no_grad():
        P_pred_flat, _, _, _ = model(x_flat, t_flat, k_final)
    P_pred = P_pred_flat.detach().cpu().numpy().reshape(len(t), len(x)).T

    log.info(f"P_moc shape:  {P_moc.shape}")
    log.info(f"P_pred shape: {P_pred.shape}")
    log.info(f"Diferencia shape: {np.abs(P_moc - P_pred).shape}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    vmin = np.min(P_moc)
    vmax = np.max(P_moc)
    axs[0].imshow(P_moc, aspect='auto', origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axs[0].set_title('MOC')
    axs[1].imshow(P_pred, aspect='auto', origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axs[1].set_title('PINN')
    axs[2].imshow(np.abs(P_moc - P_pred), aspect='auto', origin='lower', cmap='viridis')
    axs[2].set_title('Abs diff')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'pinn_pressure_field.png'))
    plt.close(fig)

    # Pressure sensor fit
    x_p_sensors = data['x_pressure_sensors_used']
    P_noisy = data['P_noisy']
    fig, axs = plt.subplots(len(x_p_sensors), 1, figsize=(8, 3 * len(x_p_sensors)))
    if len(x_p_sensors) == 1:
        axs = [axs]
    for i, xs in enumerate(x_p_sensors):
        t = data['t']
        t_t = torch.tensor(t, dtype=torch.float32)
        x_t = torch.full_like(t_t, float(xs))
        P_pred, _, _, _ = model(x_t.to(device), t_t.to(device), k_final)
        axs[i].plot(t, P_pred.detach().cpu().numpy(), label='PINN')
        axs[i].scatter(t, P_noisy[i, :], s=6, color='k', alpha=0.6, label='data')
        axs[i].set_title(f'P sensor @ {int(xs)} m')
        axs[i].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'pinn_sensor_fit.png'))
    plt.close(fig)

    # Print summary
    log.info('════════════════════════════════')
    log.info('RESULTADO CASO BASE')
    log.info(f"  x_leak real:  {train_result['x_leak_true']} m")
    log.info(f"  x_leak pred:  {train_result['x_leak_pred']:.0f} m")
    log.info(f"  Error:        {train_result['x_leak_error_km']:.3f} km")
    log.info(f"  q_leak real:  {train_result['q_leak_true']:.4f} m³/s")
    log.info(f"  q_leak pred:  {train_result['q_leak_pred']:.4f} m³/s")
    log.info(f"  Error:        {train_result['q_leak_error_pct']:.2f} %")
    log.info('════════════════════════════════')


def resume_training(checkpoint_path: str, n_epochs_extra: int = 5_000, **train_kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = PerfectLeakPINN().to(device)
    model.load_state_dict(ckpt['model_state'])

    # prepare optimizer and state
    # recreate train call to continue
    result = train_pinn(n_epochs=n_epochs_extra, **train_kwargs)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train Leak PINN')
    parser.add_argument('--scenario', type=int, default=8, help='Scenario ID (default: 8)')
    parser.add_argument('--noise', type=str, default='trivial', help='Noise level (default: trivial)')
    parser.add_argument('--n_pressure_sensors', type=int, default=3, help='Number of pressure sensors: 2,3 (default: 3)')
    parser.add_argument('--n_epochs', type=int, default=20000, help='Training epochs (default: 20000)')
    parser.add_argument('--skip_benchmark', action='store_true', help='Skip the 100-epoch benchmark')
    parser.add_argument('--no_lbfgs', action='store_true', help='Disable L-BFGS refinement')
    parser.add_argument('--lbfgs_epochs', type=int, default=2000, help='L-BFGS refinement epochs (default: 2000)')
    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)

    if not args.skip_benchmark:
        log.info('─── Benchmark rápido (100 epochs) ───')
        t0 = time.time()
        try:
            _ = train_pinn(n_epochs=100, n_pressure_sensors=args.n_pressure_sensors, verbose=True, progress_every=20, use_lbfgs=False)
            t_100 = time.time() - t0
            log.info(f'100 epochs: {t_100:.2f}s')
            log.info(f'Estimado para {args.n_epochs:,} epochs: {t_100 * args.n_epochs / 100:.1f} s (~{t_100 * args.n_epochs / 100 / 60:.1f} min)')
        except torch.cuda.OutOfMemoryError:
            log.error('OOM en benchmark. Saltando benchmark.')

    # Full training with OOM handling
    try:
        log.info(f'─── Entrenamiento completo ({args.n_epochs:,} epochs, {args.n_pressure_sensors} sensores P + 2 caudalímetros) ───')
        result = train_pinn(
            scenario_id=args.scenario,
            noise_level=args.noise,
            n_pressure_sensors=args.n_pressure_sensors,
            n_epochs=args.n_epochs,
            n_collocation=None,
            verbose=True,
            progress_every=500,
            use_lbfgs=not args.no_lbfgs,
            lbfgs_epochs=args.lbfgs_epochs,
        )
    except torch.cuda.OutOfMemoryError:
        log.error('OOM en GPU. Reduciendo n_collocation a 10000 y reintentando...')
        torch.cuda.empty_cache()
        result = train_pinn(
            scenario_id=args.scenario,
            noise_level=args.noise,
            n_pressure_sensors=args.n_pressure_sensors,
            n_epochs=args.n_epochs,
            n_collocation=10000,
            verbose=True,
            progress_every=500,
            use_lbfgs=not args.no_lbfgs,
            lbfgs_epochs=args.lbfgs_epochs,
        )

    plot_training_diagnostics(result)
    ckpt_name = f'pinn_s{args.scenario}_nP{args.n_pressure_sensors}_{args.noise}.pt'
    torch.save(result['model'].state_dict(), os.path.join('checkpoints', ckpt_name))
    log.info(f'Training complete. Model saved to checkpoints/{ckpt_name}')
    log.info(f'Log saved. Check logs/ directory for full history.')


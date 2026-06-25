import torch
import torch.nn as nn
import numpy as np
import math
import time
import logging
from typing import Dict, Tuple
from tqdm import tqdm
from data_utils import get_training_data

log = logging.getLogger("XPINN")
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')

class MLP(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=32):
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
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        return self.net(x)

class LeakXPINN(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=32, initial_x_leak=5000.0, initial_q_leak=0.015):
        super().__init__()
        self.L = 10000.0
        self.T_total = 100.0
        self.P_in = 1.05e7
        self.Q_out = 0.2
        
        self.net_up = MLP(hidden_layers, hidden_size)
        self.net_down = MLP(hidden_layers, hidden_size)
        
        # Init x_leak mapping
        p = (initial_x_leak - 500.0) / 9000.0
        p = max(1e-5, min(1.0 - 1e-5, p))
        logit = math.log(p / (1 - p))
        self.x_leak_raw = nn.Parameter(torch.tensor(logit, dtype=torch.float32))
        
        # Init q_leak mapping (rango: 0.001 a 0.050)
        pq = (initial_q_leak - 0.001) / 0.049
        pq = max(1e-5, min(1.0 - 1e-5, pq))
        logit_q = math.log(pq / (1 - pq))
        self.q_leak_raw = nn.Parameter(torch.tensor(logit_q, dtype=torch.float32))

    @property
    def x_leak(self) -> torch.Tensor:
        p = torch.sigmoid(self.x_leak_raw)
        return p * 9000.0 + 500.0

    @property
    def q_leak(self) -> torch.Tensor:
        p = torch.sigmoid(self.q_leak_raw)
        return p * 0.049 + 0.001

    def network_params(self):
        exclude = {id(self.x_leak_raw), id(self.q_leak_raw)}
        return [p for p in self.parameters() if id(p) not in exclude]
        
    def forward_up(self, xi: torch.Tensor, t: torch.Tensor):
        t_in = (t / self.T_total).reshape(-1, 1)
        xi_in = xi.reshape(-1, 1)
        out = self.net_up(torch.cat([xi_in, t_in], dim=1))
        P = out[:, 0].reshape(xi.shape) * self.P_in
        Q = out[:, 1].reshape(xi.shape) * self.Q_out
        return P, Q

    def forward_down(self, xi: torch.Tensor, t: torch.Tensor):
        t_in = (t / self.T_total).reshape(-1, 1)
        xi_in = xi.reshape(-1, 1)
        out = self.net_down(torch.cat([xi_in, t_in], dim=1))
        P = out[:, 0].reshape(xi.shape) * self.P_in
        Q = out[:, 1].reshape(xi.shape) * self.Q_out
        return P, Q

    def predict(self, x: torch.Tensor, t: torch.Tensor, detach_x_L=False):
        x_L = self.x_leak
        if detach_x_L:
            x_L = x_L.detach()
        mask_up = (x < x_L)
        mask_down = ~mask_up
        
        P_pred = torch.zeros_like(x)
        Q_pred = torch.zeros_like(x)
        
        if mask_up.any():
            x_up = x[mask_up]
            t_up = t[mask_up]
            xi_up = x_up / x_L
            P_up, Q_up = self.forward_up(xi_up, t_up)
            P_pred[mask_up] = P_up
            Q_pred[mask_up] = Q_up
            
        if mask_down.any():
            x_down = x[mask_down]
            t_down = t[mask_down]
            xi_down = (x_down - x_L) / (self.L - x_L)
            P_down, Q_down = self.forward_down(xi_down, t_down)
            P_pred[mask_down] = P_down
            Q_pred[mask_down] = Q_down
            
        return P_pred, Q_pred

def compute_pde_residuals_xpinn(model: LeakXPINN, xi_col: torch.Tensor, t_col: torch.Tensor):
    xi_col.requires_grad_(True)
    t_col.requires_grad_(True)
    
    rho = 800.0
    a = 1000.0
    D = 0.5
    A = math.pi * (D / 2)**2
    f = 0.015
    
    x_L = model.x_leak
    L_total = model.L
    
    # ==== Upstream ====
    P1, Q1 = model.forward_up(xi_col, t_col)
    dP1_dt = torch.autograd.grad(P1, t_col, grad_outputs=torch.ones_like(P1), create_graph=True)[0]
    dQ1_dt = torch.autograd.grad(Q1, t_col, grad_outputs=torch.ones_like(Q1), create_graph=True)[0]
    dP1_dxi = torch.autograd.grad(P1, xi_col, grad_outputs=torch.ones_like(P1), create_graph=True)[0]
    dQ1_dxi = torch.autograd.grad(Q1, xi_col, grad_outputs=torch.ones_like(Q1), create_graph=True)[0]
    
    x_L_det = model.x_leak.detach()
    
    dP1_dx = dP1_dxi / x_L_det
    dQ1_dx = dQ1_dxi / x_L_det
    
    fric1 = f * Q1 * torch.abs(Q1) / (2.0 * D * A)
    r_cont1 = dP1_dt + (rho * a**2 / A) * dQ1_dx
    r_mom1 = dQ1_dt + (A / rho) * dP1_dx + fric1
    
    # ==== Downstream ====
    P2, Q2 = model.forward_down(xi_col, t_col)
    dP2_dt = torch.autograd.grad(P2, t_col, grad_outputs=torch.ones_like(P2), create_graph=True)[0]
    dQ2_dt = torch.autograd.grad(Q2, t_col, grad_outputs=torch.ones_like(Q2), create_graph=True)[0]
    dP2_dxi = torch.autograd.grad(P2, xi_col, grad_outputs=torch.ones_like(P2), create_graph=True)[0]
    dQ2_dxi = torch.autograd.grad(Q2, xi_col, grad_outputs=torch.ones_like(Q2), create_graph=True)[0]
    
    dP2_dx = dP2_dxi / (L_total - x_L_det)
    dQ2_dx = dQ2_dxi / (L_total - x_L_det)
    
    fric2 = f * Q2 * torch.abs(Q2) / (2.0 * D * A)
    r_cont2 = dP2_dt + (rho * a**2 / A) * dQ2_dx
    r_mom2 = dQ2_dt + (A / rho) * dP2_dx + fric2
    
    return r_cont1, r_mom1, r_cont2, r_mom2

def compute_loss_xpinn(model: LeakXPINN, data_dict: Dict, xi_col: torch.Tensor, t_col: torch.Tensor, lambdas: Dict):
    mse = nn.MSELoss()
    
    # 1. Sensores (Datos)
    t = data_dict['t_tensor']
    P_noisy = data_dict['P_noisy_tensor']
    Q_noisy = data_dict['Q_noisy_tensor']
    x_sensors = data_dict['x_pressure_sensors_tensor']
    
    t_exp = t.expand(-1, x_sensors.size(0)).reshape(-1, 1)
    x_exp = x_sensors.expand(t.size(0), -1).reshape(-1, 1)
    P_pred_flat, _ = model.predict(x_exp, t_exp, detach_x_L=False)
    P_pred = P_pred_flat.reshape(t.size(0), x_sensors.size(0)).T
    L_P = mse(P_pred / model.P_in, P_noisy / model.P_in)
    
    x_in_flat = torch.zeros_like(t).reshape(-1, 1)
    x_out_flat = torch.full_like(t, model.L).reshape(-1, 1)
    t_flat = t.reshape(-1, 1)
    
    _, Q_pred_0 = model.predict(x_in_flat, t_flat, detach_x_L=True)
    _, Q_pred_L = model.predict(x_out_flat, t_flat, detach_x_L=True)
    
    Q_noisy_0 = Q_noisy[0, :].reshape(-1, 1)
    Q_noisy_L = Q_noisy[1, :].reshape(-1, 1)
    
    L_Q = mse(Q_pred_0 / model.Q_out, Q_noisy_0 / model.Q_out) + \
          mse(Q_pred_L / model.Q_out, Q_noisy_L / model.Q_out)
    
    # 2. Inicial / Contorno (Limites del caño)
    # IC: t=0, x in [0, L]
    x_ic = (torch.rand_like(t) * model.L).reshape(-1, 1)
    t_zero = torch.zeros_like(x_ic).reshape(-1, 1)
    P_ic, Q_ic = model.predict(x_ic, t_zero, detach_x_L=True)
    
    # Condición inicial nominal
    rho, a, A = 800.0, 1000.0, math.pi*(0.5/2)**2
    f, L, D = 0.015, 10000.0, 0.5
    Q0 = 0.15
    fric_term = f * Q0**2 / (2 * D * A)
    P0_nominal = 1.05e7 - (A / rho) * fric_term * x_ic
    Q0_nominal = torch.full_like(x_ic, Q0)
    
    L_ic = mse(P_ic.view_as(P0_nominal) / model.P_in, P0_nominal / model.P_in) + \
           mse(Q_ic.view_as(Q0_nominal) / model.Q_out, Q0_nominal / model.Q_out)
    
    # BC: t>0, x=0 y x=L (Presiones dadas por diseño o bomba)
    P_bc_0, _ = model.predict(x_in_flat, t_flat, detach_x_L=True)
    P_bc_L, _ = model.predict(x_out_flat, t_flat, detach_x_L=True)
    # Suponemos que los sensores extremos P(0) y P(L) son las BC
    P_true_0 = P_noisy[0, :].reshape(-1, 1)
    P_true_L = P_noisy[-1, :].reshape(-1, 1)
    L_bc = mse(P_bc_0 / model.P_in, P_true_0 / model.P_in) + \
           mse(P_bc_L / model.P_in, P_true_L / model.P_in)
    
    # 3. Interfaz (XPINN)
    # En la interfaz, xi_up = 1.0, xi_down = 0.0
    # Evaluamos en los mismos tiempos t_col
    t_int = t_col.clone().detach()
    xi_up = torch.ones_like(t_int)
    xi_down = torch.zeros_like(t_int)
    
    P1_int, Q1_int = model.forward_up(xi_up, t_int)
    P2_int, Q2_int = model.forward_down(xi_down, t_int)
    
    # Balance temporal de masa en la interfaz
    t_leak = 10.0
    tau = 2.0
    temporal = torch.sigmoid((t_int - t_leak) / tau)
    q_l_t = model.q_leak.detach() * temporal # Q leak scale is trained via L_masa
    
    L_int_P = mse(P1_int / model.P_in, P2_int / model.P_in)
    L_int_Q = mse(Q1_int / model.Q_out, (Q2_int + q_l_t) / model.Q_out)
    L_int = L_int_P + L_int_Q
    
    # 4. PDE
    rho_t = 800.0 / model.T_total
    r_c1, r_m1, r_c2, r_m2 = compute_pde_residuals_xpinn(model, xi_col, t_col)
    
    L_pde_up = torch.mean((r_c1 / model.P_in)**2 + (r_m1 / (model.Q_out * rho_t))**2)
    L_pde_down = torch.mean((r_c2 / model.P_in)**2 + (r_m2 / (model.Q_out * rho_t))**2)
    L_pde = L_pde_up + L_pde_down
    
    # 5. Balance de Masa Global
    Nt = Q_noisy.shape[1]
    idx_late = slice(int(Nt * 0.8), Nt)
    Q_in_late = Q_noisy[0, idx_late]
    Q_out_late = Q_noisy[1, idx_late]
    q_leak_data_est = torch.mean(Q_in_late - Q_out_late)
    
    L_masa = mse(model.q_leak / model.Q_out, q_leak_data_est / model.Q_out)

    loss = (lambdas['P'] * L_P + 
            lambdas['Q'] * L_Q + 
            lambdas['bc'] * L_bc + 
            lambdas['ic'] * L_ic + 
            lambdas['pde'] * L_pde + 
            10.0 * L_int + # Interface weight
            lambdas['masa'] * L_masa)
            
    comps = {
        'L_total': loss.item(), 'L_P': L_P.item(), 'L_Q': L_Q.item(),
        'L_bc': L_bc.item(), 'L_ic': L_ic.item(), 'L_pde': L_pde.item(),
        'L_int': L_int.item(), 'L_masa': L_masa.item()
    }
    
    return loss, comps

def train_xpinn(scenario_id=1, n_epochs=10000, n_collocation=20000, lr=1e-3, lambdas=None, 
                noise_level="trivial", n_pressure_sensors=3, use_npw_init=False,
                initial_x_leak=5000.0, initial_q_leak=0.015):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if lambdas is None:
        lambdas = {"P": 10.0, "Q": 20.0, "pde": 1.0, "bc": 5.0, "ic": 5.0, "masa": 20.0}
        
    if use_npw_init:
        try:
            from baseline_mass_balance import run_mass_balance
            res = run_mass_balance(scenario_id, noise_level=noise_level, n_pressure_sensors=n_pressure_sensors)
            if res['leak_detected']:
                initial_x_leak = res['x_leak_pred']
                initial_q_leak = res['q_leak_pred']
                log.info(f"Warm-Start activado: NPW init -> x_leak={initial_x_leak:.1f}m, q_leak={initial_q_leak:.4f}m3/s")
            else:
                log.warning("Warm-Start: No se detectó fuga en el baseline. Usando valores por defecto.")
        except ImportError:
            log.warning("No se pudo importar baseline_mass_balance para el Warm-Start.")
            
    data = get_training_data(scenario_id, noise_level, n_pressure_sensors)
    data_dict = {
        't_tensor': torch.tensor(data['t'], dtype=torch.float32, device=device).unsqueeze(1),
        'P_noisy_tensor': torch.tensor(data['P_noisy'], dtype=torch.float32, device=device),
        'Q_noisy_tensor': torch.tensor(data['Q_noisy'], dtype=torch.float32, device=device),
        'x_pressure_sensors_tensor': torch.tensor(data['x_pressure_sensors_used'], dtype=torch.float32, device=device)
    }
    
    # xi collocation instead of x collocation
    xi_col = torch.rand((n_collocation, 1), device=device)
    t_col = torch.rand((n_collocation, 1), device=device) * 100.0
    
    model = LeakXPINN(initial_x_leak=initial_x_leak, initial_q_leak=initial_q_leak).to(device)
    
    optimizer = torch.optim.Adam([
        {'params': model.network_params(), 'lr': lr},
        {'params': [model.x_leak_raw], 'lr': 5e-3},
        {'params': [model.q_leak_raw], 'lr': 5e-3},
    ])
    
    history = []
    
    # Return format matching original test scripts
    start_time = time.time()
    pbar = tqdm(range(n_epochs), desc=f"XPINN Esc {scenario_id}")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        # dynamic lambdas
        if epoch < 1000:
            lambdas['pde'] = 1.0
        elif epoch < 5000:
            lambdas['pde'] = 1.0 + (1000.0 - 1.0) * (epoch - 1000) / 4000.0
            
        loss, comps = compute_loss_xpinn(model, data_dict, xi_col, t_col, lambdas)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            pbar.set_postfix({
                'x_leak': f"{model.x_leak.item():.0f}m",
                'L_pde': f"{comps['L_pde']:.2e}",
                'L_int': f"{comps['L_int']:.2e}",
                'L_P': f"{comps['L_P']:.2e}"
            })
            h = comps.copy()
            h['epoch'] = epoch
            h['x_leak_pred'] = model.x_leak.item()
            h['q_leak_pred'] = model.q_leak.item()
            history.append(h)
            
    training_time_s = time.time() - start_time
    # convert history to list of dicts natively
    return {"model": model, "history": history, "training_time_s": training_time_s}

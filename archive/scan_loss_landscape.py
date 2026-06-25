import os
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sys

from perfect_pinn_model import PerfectLeakPINN, get_training_data, compute_loss
import config as cfg

def setup_logger():
    logger = logging.getLogger('landscape')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
    return logger

log = setup_logger()

def scan_loss_landscape(scenario_id=8, n_epochs_warmup=3000):
    noise_level = "trivial"
    n_pressure_sensors = 3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get the shared data once
    data = get_training_data(scenario_id, noise_level, n_pressure_sensors)
    
    x_leak_values = np.arange(1000, 9500, 500)
    
    # Default lambdas
    lambdas = {"P": 10.0, "Q": 10.0, "pde": 1000.0, "bc": 10.0, "masa": 1000.0}
    
    T_START = 0.0
    mask_t = data['t'] >= T_START
    t_filt = data['t'][mask_t]
    P_noisy_filt = data['P_noisy'][:, mask_t]
    Q_noisy_filt = data['Q_noisy'][:, mask_t]

    data['t_tensor'] = torch.tensor(t_filt, dtype=torch.float32, device=device)
    data['P_noisy_tensor'] = torch.tensor(P_noisy_filt, dtype=torch.float32, device=device)
    data['Q_noisy_tensor'] = torch.tensor(Q_noisy_filt, dtype=torch.float32, device=device)
    data['x_pressure_sensors_tensor'] = torch.tensor(data['x_pressure_sensors_used'], dtype=torch.float32, device=device)
    data['x_flow_meters_tensor'] = torch.tensor(data['x_flow_meters'], dtype=torch.float32, device=device)
    data['P_INLET_t'] = torch.tensor(float(cfg.P_INLET), dtype=torch.float32, device=device)
    data['Q_OUTLET_t'] = torch.tensor(float(cfg.Q_OUTLET), dtype=torch.float32, device=device)
    
    Nbc = 200
    t_bc = torch.linspace(T_START, float(cfg.T_TOTAL), Nbc, dtype=torch.float32, device=device)
    data['x0_bc'] = torch.zeros_like(t_bc, device=device)
    data['xL_bc'] = torch.full_like(t_bc, float(cfg.PIPE_LENGTH), device=device)
    data['t_bc'] = t_bc
    
    x_ic = torch.linspace(0.0, float(cfg.PIPE_LENGTH), 200, dtype=torch.float32, device=device)
    t0_ic = torch.zeros_like(x_ic, device=device)
    A = math.pi * cfg.PIPE_DIAMETER ** 2 / 4.0
    P_ss_x = (float(cfg.P_INLET) - float(cfg.FRICTION_FACTOR) * float(cfg.FLUID_DENSITY) * x_ic * (float(cfg.Q_OUTLET) * abs(float(cfg.Q_OUTLET))) / (2.0 * float(cfg.PIPE_DIAMETER) * A ** 2))
    P_ss_x = P_ss_x.clone().detach().to(dtype=torch.float32, device=device)
    data['x_ic'] = x_ic
    data['t0_ic'] = t0_ic
    data['P_ss_x'] = P_ss_x
    
    n_collocation = 20_000 if device.type == 'cuda' else 8_000
    torch.manual_seed(42 + int(cfg.RANDOM_SEED))
    x_col = torch.empty(n_collocation, dtype=torch.float32, device=device).uniform_(0.0, float(cfg.PIPE_LENGTH))
    t_col = torch.empty(n_collocation, dtype=torch.float32, device=device).uniform_(0.0, float(cfg.T_TOTAL))
    x_col.requires_grad_()
    t_col.requires_grad_()
    
    results = []
    
    for x_lk in x_leak_values:
        log.info(f"--- Scanning x_leak = {x_lk}m ---")
        model = PerfectLeakPINN(
            activation="tanh"
        ).to(device)
        
        p = (x_lk - 500.0) / 9000.0
        p = max(1e-5, min(1.0 - 1e-5, p))
        logit = np.log(p / (1 - p))
        model.x_leak_raw.data = torch.tensor(logit, dtype=torch.float32, device=device)
        
        # Freeze x_leak, let q_leak and MLP optimize
        model.x_leak_raw.requires_grad_(False)
        model.q_leak_raw.requires_grad_(True)
        
        optimizer = torch.optim.Adam([
            {'params': model.network_params()},
            {'params': [model.q_leak_raw], 'lr': 1e-3, 'name': 'q_leak'},
            {'params': [model.m_slope], 'lr': 1e-3, 'name': 'm_slope'}
        ], lr=1e-3)
        
        for epoch in range(1, n_epochs_warmup + 1):
            optimizer.zero_grad()
            k_val = 100.0 # Just some k value, could be ramped
            L_total, comps, _ = compute_loss(model, data, x_col, t_col, lambdas, k=k_val)
            L_total.backward()
            optimizer.step()
            
            if epoch % 500 == 0:
                log.info(f"  Epoch {epoch:4d} | Loss: {comps['L_total']:.3e} (P: {comps['L_P']:.3e}, Q: {comps['L_Q']:.3e}, pde: {comps['L_fisica']:.3e})")
                
        # Final loss evaluation
        L_total, comps, _ = compute_loss(model, data, x_col, t_col, lambdas, k=k_val)
        final_loss = comps['L_total']
        results.append({
            "x_leak": float(x_lk),
            "loss": final_loss,
            "L_P": comps['L_P'],
            "L_Q": comps['L_Q'],
            "L_fisica": comps['L_fisica'],
            "q_leak_pred": float(model.q_leak.item())
        })
        log.info(f">>> x_leak={x_lk:.0f}m -> final_loss={final_loss:.4e} | q_leak={model.q_leak.item():.4f}")
        log.info("-" * 40)
            
    df = pd.DataFrame(results)
    df.to_csv("loss_landscape.csv", index=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df['x_leak'], df['loss'], marker='o', linewidth=2)
    plt.axvline(x=6000, color='r', linestyle='--', label='True x_leak (6000m)')
    plt.yscale('log')
    plt.xlabel("x_leak (m)")
    plt.ylabel("Total Loss after Warmup")
    plt.title("Loss Landscape vs x_leak (Scenario 8, true=6000m)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("loss_landscape.png")
    log.info("Saved loss_landscape.png")
    
if __name__ == '__main__':
    scan_loss_landscape()

import os
import math
import time
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import config as cfg
from data_utils import get_training_data


class LeakPINN(nn.Module):
    def __init__(self, hidden_layers: int = 5, hidden_size: int = 64):
        super().__init__()
        self.L = float(cfg.PIPE_LENGTH)
        self.T_total = float(cfg.T_TOTAL)
        self.P_in = float(cfg.P_INLET)
        self.Q_out = float(cfg.Q_OUTLET)

        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 2))

        self.net = nn.Sequential(*layers)

        # trainable leak params (raw, unconstrained)
        # initialize so that x_leak ~ L/2 and q_leak ~ 0.008
        self.x_leak_raw = nn.Parameter(torch.tensor(0.0))
        q_target = 0.008
        # inverse softplus: raw = log(exp(y)-1)
        q_raw_init = float(math.log(math.exp((q_target - 0.001) / 0.01) - 1.0))
        self.q_leak_raw = nn.Parameter(torch.tensor(q_raw_init))

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
        # softplus to ensure positive, then scale
        return nn.functional.softplus(self.q_leak_raw) * 0.01 + 0.001

    def network_params(self):
        # Return all parameters except the trainable leak scalars.
        exclude = {id(self.x_leak_raw), id(self.q_leak_raw)}
        return [p for p in self.parameters() if id(p) not in exclude]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
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
        P = P_norm.reshape(orig_shape) * self.P_in
        Q = Q_norm.reshape(orig_shape) * self.Q_out
        return P, Q


def leak_source(x: torch.Tensor, t: torch.Tensor, x_leak: torch.Tensor, q_leak: torch.Tensor):
    sigma = 200.0
    tau = 2.0
    t_leak = float(cfg.T_LEAK_START)

    spatial = torch.exp(-0.5 * ((x - x_leak) / sigma) ** 2) / (sigma * (2 * math.pi) ** 0.5)
    temporal = torch.sigmoid((t - t_leak) / tau)
    return q_leak * spatial * temporal


def compute_pde_residuals(model: LeakPINN, x_col: torch.Tensor, t_col: torch.Tensor):
    device = x_col.device
    x_col = x_col.clone().detach().requires_grad_(True)
    t_col = t_col.clone().detach().requires_grad_(True)

    P, Q = model(x_col, t_col)

    ones = torch.ones_like(P, device=device)

    dP_dt = torch.autograd.grad(P, t_col, grad_outputs=ones, create_graph=True)[0]
    dP_dx = torch.autograd.grad(P, x_col, grad_outputs=ones, create_graph=True)[0]
    dQ_dt = torch.autograd.grad(Q, t_col, grad_outputs=ones, create_graph=True)[0]
    dQ_dx = torch.autograd.grad(Q, x_col, grad_outputs=ones, create_graph=True)[0]

    rho = float(cfg.FLUID_DENSITY)
    a = float(cfg.WAVE_SPEED)
    A = math.pi * cfg.PIPE_DIAMETER ** 2 / 4.0

    S = leak_source(x_col, t_col, model.x_leak, model.q_leak)

    r_cont = dP_dt + (rho * a * a / A) * dQ_dx + (rho * a * a / A) * S

    # friction term: f * Q * |Q| / (2 * D * A)
    f = float(cfg.FRICTION_FACTOR)
    D = float(cfg.PIPE_DIAMETER)
    friction = f * Q * torch.abs(Q) / (2.0 * D * A)

    r_mom = dQ_dt + (A / rho) * dP_dx + friction

    return r_cont, r_mom


def compute_loss(model: LeakPINN, data_dict: Dict, x_col: torch.Tensor, t_col: torch.Tensor, lambdas: Dict):
    # Now accept pre-moved tensors inside data_dict to avoid allocations.
    device = x_col.device
    mse = nn.MSELoss()

    # data tensors moved to device in train_pinn
    t = data_dict['t_tensor']
    P_noisy_tensor = data_dict['P_noisy_tensor']
    x_sensors = data_dict['x_sensors_tensor']

    # Data loss (only pressure sensors)
    L_datos = torch.tensor(0.0, device=device)
    Nt = t.shape[0]
    for i in range(x_sensors.shape[0]):
        x_val = x_sensors[i].item()
        x_tensor = torch.full((Nt,), x_val, dtype=torch.float32, device=device)
        P_pred, _ = model(x_tensor, t)
        P_target = P_noisy_tensor[i]
        L_datos = L_datos + mse(P_pred, P_target)
    L_datos = L_datos / float(x_sensors.shape[0])

    # PDE residual loss
    r_cont, r_mom = compute_pde_residuals(model, x_col, t_col)
    r_cont_norm = r_cont / data_dict['P_INLET_t']
    r_mom_norm = r_mom / (data_dict['Q_OUTLET_t'] * data_dict['RHO_t'])
    L_fisica = torch.mean(r_cont_norm ** 2) + torch.mean(r_mom_norm ** 2)

    # Boundary conditions (use preallocated tensors)
    P_x0, _ = model(data_dict['x0_bc'], data_dict['t_bc'])
    _, Q_xL = model(data_dict['xL_bc'], data_dict['t_bc'])
    L_contorno = mse(P_x0, data_dict['P_INLET_t'].expand_as(P_x0)) + mse(Q_xL, data_dict['Q_OUTLET_t'].expand_as(Q_xL))

    # Initial conditions
    P_pred_ic, Q_pred_ic = model(data_dict['x_ic'], data_dict['t0_ic'])
    P_ss_x = data_dict['P_ss_x']
    L_inicial = mse(P_pred_ic, P_ss_x) + mse(Q_pred_ic, data_dict['Q_OUTLET_t'].expand_as(Q_pred_ic))

    L_total = (
        lambdas['data'] * L_datos
        + lambdas['pde'] * L_fisica
        + lambdas['bc'] * L_contorno
        + lambdas['ic'] * L_inicial
    )

    components = {
        'L_total': float(L_total.detach().cpu().item()),
        'L_datos': float(L_datos.detach().cpu().item()),
        'L_fisica': float(L_fisica.detach().cpu().item()),
        'L_contorno': float(L_contorno.detach().cpu().item()),
        'L_inicial': float(L_inicial.detach().cpu().item()),
    }
    return L_total, components


def train_pinn(scenario_id=7,
               noise_level="trivial",
               n_sensors=3,
               n_epochs=10_000,
               n_collocation=None,
               lr=1e-3,
               lambdas=None,
               verbose=True,
               progress_every=500):
    if lambdas is None:
        lambdas = {"data": 10.0, "pde": 1.0, "bc": 5.0, "ic": 5.0}

    torch.set_float32_matmul_precision('high')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Clean CUDA cache and reset stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    data = get_training_data(scenario_id, noise_level, n_sensors)

    # VRAM-based default for collocation
    if n_collocation is None:
        if device.type == 'cuda':
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb >= 10:
                n_collocation = 20_000
            elif vram_gb >= 6:
                n_collocation = 12_000
            else:
                n_collocation = 8_000
        else:
            n_collocation = 8_000

    print(f"PyTorch version: {torch.__version__}")
    print(f"torch.compile available: {hasattr(torch, 'compile')}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        except Exception:
            pass

    if verbose:
        print(f"Training start: {n_epochs} epochs, {n_collocation:,} collocation points, reporting every {progress_every} epochs")

    model = LeakPINN().to(device)

    # param groups: network vs leak params
    optimizer = torch.optim.Adam([
        {'params': model.network_params(), 'lr': lr},
        {'params': [model.x_leak_raw], 'lr': 5e-3},
        {'params': [model.q_leak_raw], 'lr': 5e-3},
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    history = []

    # Move static data once to device
    t_tensor = torch.tensor(data['t'], dtype=torch.float32, device=device)
    P_noisy_tensor = torch.tensor(data['P_noisy'], dtype=torch.float32, device=device)
    x_sensors_tensor = torch.tensor(data['x_sensors_used'], dtype=torch.float32, device=device)
    P_INLET_t = torch.tensor(float(cfg.P_INLET), dtype=torch.float32, device=device)
    Q_OUTLET_t = torch.tensor(float(cfg.Q_OUTLET), dtype=torch.float32, device=device)
    RHO_t = torch.tensor(float(cfg.FLUID_DENSITY), dtype=torch.float32, device=device)

    # Precompute BC and IC tensors
    Nbc = 200
    Nic = 200
    t_bc = torch.linspace(0.0, float(cfg.T_TOTAL), Nbc, dtype=torch.float32, device=device)
    x0_bc = torch.zeros_like(t_bc, device=device)
    xL_bc = torch.full_like(t_bc, float(cfg.PIPE_LENGTH), device=device)

    x_ic = torch.linspace(0.0, float(cfg.PIPE_LENGTH), Nic, dtype=torch.float32, device=device)
    t0_ic = torch.zeros_like(x_ic, device=device)

    # steady state analytical on device
    A = math.pi * cfg.PIPE_DIAMETER ** 2 / 4.0
    P_ss_x = (float(cfg.P_INLET) - float(cfg.FRICTION_FACTOR) * float(cfg.FLUID_DENSITY) * x_ic * (float(cfg.Q_OUTLET) * abs(float(cfg.Q_OUTLET))) / (2.0 * float(cfg.PIPE_DIAMETER) * A ** 2))
    P_ss_x = P_ss_x.clone().detach().to(dtype=torch.float32, device=device)

    # Prepare data_dict with device tensors for compute_loss
    device_data = {
        'scenario_id': scenario_id,
        'noise_level': noise_level,
        'n_sensors': n_sensors,
        't_tensor': t_tensor,
        'P_noisy_tensor': P_noisy_tensor,
        'x_sensors_tensor': x_sensors_tensor,
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

    # Preallocate collocation tensors on device and reuse (in-place randomize)
    x_col = torch.empty(n_collocation, dtype=torch.float32, device=device)
    t_col = torch.empty(n_collocation, dtype=torch.float32, device=device)

    t_start_total = time.time()
    t_epoch_start = time.time()

    for epoch in range(1, n_epochs + 1):
        if torch.cuda.is_available() and hasattr(torch, 'compiler') and hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
            torch.compiler.cudagraph_mark_step_begin()

        # sample collocation points in-place
        torch.manual_seed(epoch + int(cfg.RANDOM_SEED))
        with torch.no_grad():
            x_col.uniform_(0.0, float(cfg.PIPE_LENGTH))
            t_col.uniform_(0.0, float(cfg.T_TOTAL))
        x_col.requires_grad_()
        t_col.requires_grad_()

        optimizer.zero_grad()
        loss_total, comps = compute_loss(model, {**device_data}, x_col, t_col, lambdas)
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            history.append({
                'epoch': epoch,
                'loss_total': comps['L_total'],
                'L_datos': comps['L_datos'],
                'L_fisica': comps['L_fisica'],
                'L_contorno': comps['L_contorno'],
                'L_inicial': comps['L_inicial'],
                'x_leak_pred': float(model.x_leak.detach().cpu().numpy()),
                'q_leak_pred': float(model.q_leak.detach().cpu().numpy()),
            })

        if epoch == 1000 and verbose:
            print("─── Balance de loss en epoch 1000 ───")
            print(f"  L_datos    × λ_data = {comps['L_datos'] * lambdas['data']:.3e}")
            print(f"  L_fisica   × λ_pde  = {comps['L_fisica'] * lambdas['pde']:.3e}")
            print(f"  L_contorno × λ_bc   = {comps['L_contorno'] * lambdas['bc']:.3e}")
            print(f"  L_inicial  × λ_ic   = {comps['L_inicial'] * lambdas['ic']:.3e}")
            print("  → Los 4 valores deberían ser del mismo orden de magnitud")
            print("────────────────────────────────────")

        if verbose and (epoch == 1 or epoch % progress_every == 0 or epoch == n_epochs):
            elapsed = time.time() - t_start_total
            recent_epochs = progress_every if epoch >= progress_every else epoch
            epoch_ms = (time.time() - t_epoch_start) / float(recent_epochs) * 1000.0
            vram_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            print(
                f"Epoch {epoch:5d} | "
                f"L_total: {comps['L_total']:.3e} | "
                f"L_dat: {comps['L_datos']:.3e} | "
                f"L_fis: {comps['L_fisica']:.3e} | "
                f"L_bc: {comps['L_contorno']:.3e} | "
                f"L_ic: {comps['L_inicial']:.3e} | "
                f"x_leak: {model.x_leak.item():.0f}m | "
                f"q_leak: {model.q_leak.item():.5f} | "
                f"{epoch_ms:.1f} ms/epoch | VRAM: {vram_used:.2f}GB | Elapsed: {elapsed/60:.1f} min"
            )
            t_epoch_start = time.time()

        # periodic checkpoint
        if epoch % 2000 == 0 and epoch > 0:
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'x_leak_pred': model.x_leak.item(),
                'q_leak_pred': model.q_leak.item(),
                'loss_total': loss_total.item(),
                'history': history,
            }
            torch.save(ckpt, os.path.join('checkpoints', f'pinn_epoch_{epoch}.pt'))


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
        'n_sensors': n_sensors,
        'x_leak_pred': x_pred,
        'q_leak_pred': q_pred,
        'x_leak_true': data['x_leak'],
        'q_leak_true': data['q_leak'],
        'x_leak_error_km': abs(x_pred - data['x_leak']) / 1000.0,
        'q_leak_error_pct': abs(q_pred - data['q_leak']) / (data['q_leak']) * 100.0,
        'peak_vram_gb': peak_vram,
        'training_time_s': total_time,
        'ms_per_epoch': total_time / float(n_epochs) * 1000.0,
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

        axs[1].plot(df['epoch'], df['L_datos'], label='data')
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
        train_result.get('n_sensors', 3),
    )
    x = data['x']
    t = data['t']
    P_moc = data['P_full']

    Xg, Tg = np.meshgrid(x, t, indexing='xy')
    device = next(model.parameters()).device
    x_flat = torch.tensor(Xg.flatten(), dtype=torch.float32, device=device)
    t_flat = torch.tensor(Tg.flatten(), dtype=torch.float32, device=device)
    with torch.no_grad():
        P_pred_flat, _ = model(x_flat, t_flat)
    P_pred = P_pred_flat.detach().cpu().numpy().reshape(len(t), len(x)).T

    print(f"P_moc shape:  {P_moc.shape}")
    print(f"P_pred shape: {P_pred.shape}")
    print(f"Diferencia shape: {np.abs(P_moc - P_pred).shape}")

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

    # Sensor fit
    x_sensors = data['x_sensors_used']
    P_noisy = data['P_noisy']
    fig, axs = plt.subplots(len(x_sensors), 1, figsize=(8, 3 * len(x_sensors)))
    if len(x_sensors) == 1:
        axs = [axs]
    for i, xs in enumerate(x_sensors):
        t = data['t']
        t_t = torch.tensor(t, dtype=torch.float32)
        x_t = torch.full_like(t_t, float(xs))
        P_pred, _ = model(x_t.to(device), t_t.to(device))
        axs[i].plot(t, P_pred.detach().cpu().numpy(), label='PINN')
        axs[i].scatter(t, P_noisy[i, :], s=6, color='k', alpha=0.6, label='data')
        axs[i].set_title(f'Sensor @ {int(xs)} m')
        axs[i].legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'pinn_sensor_fit.png'))
    plt.close(fig)

    # Print summary
    print('════════════════════════════════')
    print('RESULTADO CASO BASE')
    print(f"  x_leak real:  {train_result['x_leak_true']} m")
    print(f"  x_leak pred:  {train_result['x_leak_pred']:.0f} m")
    print(f"  Error:        {train_result['x_leak_error_km']:.3f} km")
    print(f"  q_leak real:  {train_result['q_leak_true']:.4f} m³/s")
    print(f"  q_leak pred:  {train_result['q_leak_pred']:.4f} m³/s")
    print(f"  Error:        {train_result['q_leak_error_pct']:.2f} %")
    print('════════════════════════════════')


def resume_training(checkpoint_path: str, n_epochs_extra: int = 5_000, **train_kwargs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = LeakPINN().to(device)
    model.load_state_dict(ckpt['model_state'])

    # prepare optimizer and state
    # recreate train call to continue
    result = train_pinn(n_epochs=n_epochs_extra, **train_kwargs)
    return result


if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)

    # quick benchmark
    print('─── Benchmark rápido (100 epochs) ───')
    t0 = time.time()
    try:
        _ = train_pinn(n_epochs=100, verbose=True, progress_every=20)
        t_100 = time.time() - t0
        print(f'100 epochs: {t_100:.2f}s')
        print(f'Estimado para 20.000 epochs: {t_100 * 200:.1f} s (~{t_100 * 200 / 60:.1f} min)')
    except torch.cuda.OutOfMemoryError:
        print('OOM en benchmark. Saltando benchmark.')

    # Full training with OOM handling
    try:
        print('─── Entrenamiento completo (20.000 epochs) ───')
        result = train_pinn(
            scenario_id=8,
            noise_level='trivial',
            n_sensors=3,
            n_epochs=20000,
            n_collocation=None,
            verbose=True,
            progress_every=500,
        )
    except torch.cuda.OutOfMemoryError:
        print('OOM en GPU. Reduciendo n_collocation a 10000 y reintentando...')
        torch.cuda.empty_cache()
        result = train_pinn(
            scenario_id=8,
            noise_level='trivial',
            n_sensors=3,
            n_epochs=20000,
            n_collocation=10000,
            verbose=True,
            progress_every=500,
        )

    plot_training_diagnostics(result)
    torch.save(result['model'].state_dict(), os.path.join('checkpoints', 'pinn_base.pt'))

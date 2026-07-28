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

def build_data_tensors(data, device, T_START, T_END):
    """
    Construye todos los tensores de entrenamiento a partir de los datos crudos.
    T_START y T_END controlan el rango temporal usado.
    """
    mask_t = (data['t'] >= T_START) & (data['t'] <= T_END)
    t_filt       = data['t'][mask_t]
    P_noisy_filt = data['P_noisy'][:, mask_t]
    Q_noisy_filt = data['Q_noisy'][:, mask_t]

    log.info(f"  Rango temporal datos:  [{t_filt[0]:.1f}, {t_filt[-1]:.1f}] s  "
             f"({len(t_filt)} pasos)")

    data['t_tensor']               = torch.tensor(t_filt,         dtype=torch.float32, device=device)
    data['P_noisy_tensor']         = torch.tensor(P_noisy_filt,   dtype=torch.float32, device=device)
    data['Q_noisy_tensor']         = torch.tensor(Q_noisy_filt,   dtype=torch.float32, device=device)
    data['x_pressure_sensors_tensor'] = torch.tensor(
        data['x_pressure_sensors_used'], dtype=torch.float32, device=device)
    data['x_flow_meters_tensor']   = torch.tensor(
        data['x_flow_meters'],           dtype=torch.float32, device=device)
    data['P_INLET_t']              = torch.tensor(float(cfg.P_INLET),  dtype=torch.float32, device=device)
    data['Q_OUTLET_t']             = torch.tensor(float(cfg.Q_OUTLET), dtype=torch.float32, device=device)

    # ── Condición de contorno ────────────────────────────────────────────────
    # Misma ventana temporal que los datos para evitar inconsistencias
    Nbc   = 200
    t_bc  = torch.linspace(T_START, T_END, Nbc, dtype=torch.float32, device=device)
    data['x0_bc'] = torch.zeros_like(t_bc)
    data['xL_bc'] = torch.full_like(t_bc, float(cfg.PIPE_LENGTH))
    data['t_bc']  = t_bc

    # ── Condición inicial en t = 0 (estado estacionario sin fuga) ───────────
    # Usamos t=0 siempre: el estado estacionario real antes de la fuga.
    # Si T_START > 0 la IC queda fuera del dominio de datos → la eliminamos.
    # Si T_START == 0 la IC ancla correctamente el inicio.
    A      = math.pi * cfg.PIPE_DIAMETER**2 / 4.0
    x_ic   = torch.linspace(0.0, float(cfg.PIPE_LENGTH), 200,
                             dtype=torch.float32, device=device)
    t0_ic  = torch.zeros_like(x_ic)
    P_ss_x = (float(cfg.P_INLET)
              - float(cfg.FRICTION_FACTOR) * float(cfg.FLUID_DENSITY)
              * x_ic * (float(cfg.Q_OUTLET) * abs(float(cfg.Q_OUTLET)))
              / (2.0 * float(cfg.PIPE_DIAMETER) * A**2))
    P_ss_x = P_ss_x.clone().detach().to(dtype=torch.float32, device=device)

    data['x_ic']   = x_ic
    data['t0_ic']  = t0_ic
    data['P_ss_x'] = P_ss_x
    data['use_ic'] = (T_START == 0.0)   # flag: deshabilitar IC si no incluimos t=0

    return data


def make_collocation(n_collocation, T_START, T_END, device):
    """
    Puntos de colocación DENTRO del mismo dominio temporal que los datos.
    Clave: t_col debe tener el mismo rango que t_datos.
    """
    torch.manual_seed(42 + int(cfg.RANDOM_SEED))
    x_col = torch.empty(n_collocation, dtype=torch.float32, device=device)\
                  .uniform_(0.0, float(cfg.PIPE_LENGTH))
    t_col = torch.empty(n_collocation, dtype=torch.float32, device=device)\
                  .uniform_(T_START, T_END)          # ← alineado con datos
    x_col.requires_grad_(True)
    t_col.requires_grad_(True)
    log.info(f"  Puntos colocación:     [{t_col.min().item():.1f}, "
             f"{t_col.max().item():.1f}] s  (n={n_collocation})")
    return x_col, t_col


def scan_loss_landscape(scenario_id   = 8,
                        n_epochs_warmup = 3000,
                        T_START       = 50.0,    # ← inicio del transitorio
                        T_END         = None,    # None = cfg.T_TOTAL
                        x_step        = 500):
    """
    Barre x_leak en todo el ducto manteniendo x_leak congelado.
    Entrena solo los pesos de la MLP (y q_leak) para cada posición.
    Registra el loss final → permite ver si hay un mínimo claro en x_leak_true.

    Parámetros
    ----------
    T_START : float
        Inicio del dominio temporal.
        50.0  → incluye el transitorio completo desde la fuga (RECOMENDADO)
        150.0 → solo régimen cuasi-estacionario (descartado: paisaje plano)
        0.0   → todo el rango, incluyendo antes de la fuga
    """
    if T_END is None:
        T_END = float(cfg.T_TOTAL)

    noise_level          = "trivial"
    n_pressure_sensors   = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")
    log.info(f"Dominio temporal: T_START={T_START}s  T_END={T_END}s")
    log.info(f"Epochs warmup por posición: {n_epochs_warmup}")

    # Datos crudos y tensores
    data = get_training_data(scenario_id, noise_level, n_pressure_sensors)
    data = build_data_tensors(data, device, T_START, T_END)

    # Puntos de colocación (fijos para todo el barrido para comparabilidad)
    n_col   = 20_000 if device.type == 'cuda' else 8_000
    x_col, t_col = make_collocation(n_col, T_START, T_END, device)

    # Lambdas
    lambdas = {
        "P":    10.0,
        "Q":    10.0,
        "pde":  1000.0,
        "bc":   10.0,
        "masa": 1000.0,
    }

    x_leak_values = np.arange(1000, 9500, x_step)
    results = []
    n_total = len(x_leak_values)

    log.info(f"\nIniciando barrido: {n_total} posiciones × {n_epochs_warmup} epochs")
    log.info("="*60)

    for i, x_lk in enumerate(x_leak_values):
        log.info(f"\n[{i+1}/{n_total}] x_leak_fixed = {x_lk:.0f} m")

        # Instanciar modelo fresco
        model = PerfectLeakPINN(activation="tanh").to(device)

        # Fijar x_leak en x_lk
        p     = np.clip((x_lk - 500.0) / 9000.0, 1e-5, 1.0 - 1e-5)
        logit = np.log(p / (1.0 - p))
        model.x_leak_raw.data = torch.tensor(logit, dtype=torch.float32, device=device)
        model.x_leak_raw.requires_grad_(False)   # CONGELADO
        model.q_leak_raw.requires_grad_(True)    # q_leak libre

        # Optimizador: solo MLP + q_leak (x_leak congelado)
        optimizer = torch.optim.Adam([
            {'params': model.network_params(),  'lr': 1e-3},
            {'params': [model.q_leak_raw],      'lr': 1e-3, 'name': 'q_leak'},
            {'params': [model.m_slope],         'lr': 1e-3, 'name': 'm_slope'},
        ])

        # Entrenamiento
        for epoch in range(1, n_epochs_warmup + 1):
            optimizer.zero_grad()
            L_total, comps, _ = compute_loss(
                model, data, x_col, t_col, lambdas, k=100.0
            )
            L_total.backward()
            optimizer.step()

            if epoch % 1000 == 0:
                log.info(f"  epoch {epoch:4d} | "
                         f"L_total={comps['L_total']:.3e} | "
                         f"L_P={comps['L_P']:.3e} | "
                         f"L_Q={comps['L_Q']:.3e} | "
                         f"L_pde={comps['L_fisica']:.3e} | "
                         f"q_pred={model.q_leak.item():.4f}")

        # Loss final con x_leak congelado
        with torch.no_grad():
            L_final, comps_f, _ = compute_loss(
                model, data, x_col, t_col, lambdas, k=100.0
            )

        # Guardar solo el loss de DATOS (L_P + L_Q) por separado
        # para ver si el mínimo de datos coincide con x_leak_true
        L_datos = comps_f['L_P'] + comps_f['L_Q']

        results.append({
            "x_leak_fixed":  float(x_lk),
            "loss_total":    float(comps_f['L_total']),
            "loss_datos":    float(L_datos),      # ← el más informativo
            "loss_P":        float(comps_f['L_P']),
            "loss_Q":        float(comps_f['L_Q']),
            "loss_pde":      float(comps_f['L_fisica']),
            "q_leak_pred":   float(model.q_leak.item()),
        })
        log.info(f"  >>> FINAL | L_total={comps_f['L_total']:.4e} | "
                 f"L_datos={L_datos:.4e} | q_pred={model.q_leak.item():.4f}")
        log.info("-"*60)

    # ── Guardar resultados ───────────────────────────────────────────────────
    df = pd.DataFrame(results)
    out_csv = f"loss_landscape_Tstart{int(T_START)}.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"\nGuardado: {out_csv}")

    # ── Graficar ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    x_true = 6000.0

    # Plot 1: Loss total
    ax = axes[0]
    ax.plot(df['x_leak_fixed'], df['loss_total'],
            marker='o', linewidth=2, label='Loss total')
    ax.axvline(x=x_true, color='r', linestyle='--', label=f'x_leak real ({x_true:.0f}m)')
    idx_min = df['loss_total'].idxmin()
    ax.axvline(x=df.loc[idx_min, 'x_leak_fixed'], color='g',
               linestyle=':', label=f"Mínimo en {df.loc[idx_min,'x_leak_fixed']:.0f}m")
    ax.set_yscale('log')
    ax.set_xlabel("x_leak fijo (m)")
    ax.set_ylabel("Loss total (log)")
    ax.set_title(f"Loss landscape — Loss total | T_START={T_START}s")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)

    # Plot 2: Loss de datos solamente (L_P + L_Q)
    # Este es el más informativo: muestra si los DATOS
    # tienen un mínimo claro en x_leak_true
    ax2 = axes[1]
    ax2.plot(df['x_leak_fixed'], df['loss_datos'],
             marker='s', color='darkorange', linewidth=2, label='L_datos (L_P + L_Q)')
    ax2.axvline(x=x_true, color='r', linestyle='--', label=f'x_leak real ({x_true:.0f}m)')
    idx_min2 = df['loss_datos'].idxmin()
    ax2.axvline(x=df.loc[idx_min2, 'x_leak_fixed'], color='g',
                linestyle=':', label=f"Mínimo en {df.loc[idx_min2,'x_leak_fixed']:.0f}m")
    ax2.set_yscale('log')
    ax2.set_xlabel("x_leak fijo (m)")
    ax2.set_ylabel("L_datos = L_P + L_Q (log)")
    ax2.set_title("Loss de DATOS solamente — ¿hay mínimo en x_leak_true?")
    ax2.legend(); ax2.grid(True, which="both", alpha=0.3)

    plt.suptitle(
        f"Loss Landscape | Escenario {scenario_id} | "
        f"T=[{T_START}, {T_END}]s | x_true={x_true:.0f}m",
        fontsize=13
    )
    plt.tight_layout()
    out_png = f"loss_landscape_Tstart{int(T_START)}.png"
    plt.savefig(out_png, dpi=150)
    log.info(f"Guardado: {out_png}")

    # ── Resumen en consola ────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("RESUMEN DEL BARRIDO")
    log.info("="*60)
    log.info(f"  Dominio temporal usado: [{T_START}, {T_END}] s")
    log.info(f"  x_leak verdadero:       {x_true:.0f} m")
    log.info(f"  Mínimo de L_total  en:  {df.loc[idx_min,  'x_leak_fixed']:.0f} m")
    log.info(f"  Mínimo de L_datos  en:  {df.loc[idx_min2, 'x_leak_fixed']:.0f} m")
    log.info("")

    dist_total = abs(df.loc[idx_min,  'x_leak_fixed'] - x_true)
    dist_datos = abs(df.loc[idx_min2, 'x_leak_fixed'] - x_true)

    if dist_datos <= x_step:
        log.info("  ✓ CASO A: L_datos tiene mínimo cerca de x_leak_true.")
        log.info("    El landscape tiene la señal correcta.")
        log.info("    → El staged training va a funcionar.")
        log.info("    → Proceder con train_pinn() inicializado en el mínimo.")
    else:
        log.info("  ✗ CASO B: L_datos NO tiene mínimo en x_leak_true.")
        log.info(f"    Error del mínimo de datos: {dist_datos:.0f} m")
        log.info("    El problema sigue siendo no identificable con esta config.")
        log.info("    Opciones:")
        log.info("    → Ampliar T_START (menos filtrado de transitorio)")
        log.info("    → Agregar más sensores de presión")
        log.info("    → Revisar si L_pde está activa (ver balance en epoch 1000)")
    log.info("="*60)

    return df


if __name__ == '__main__':
    # ── Correr con datos del TRANSITORIO (T_START=50s) ───────────────────────
    # Comparar con el barrido anterior donde T_START=150 (solo estacionario)
    # Si el mínimo de L_datos aparece cerca de 6000m → staged training funciona
    # Si sigue plano → el problema es de identificabilidad con 3 sensores
    log.info("Barrido 1: con transitorio completo (T_START=50s)")
    df_transitorio = scan_loss_landscape(
        scenario_id    = 8,
        n_epochs_warmup = 3000,
        T_START        = 50.0,    # ← desde el inicio de la fuga
        T_END          = None,    # hasta T_TOTAL=200s
        x_step         = 500,
    )
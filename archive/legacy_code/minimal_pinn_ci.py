"""
minimal_pinn_ci.py  —  CI-PINN para localización de fuga

Implementación de CI-PINN (Kim & Son, Mathematics 2025, 13, 1057)
aplicado al problema de localización de fuga en ducto 1D.

CAMBIO RESPECTO A minimal_pinn_leak.py:
  La pérdida PDE se pondera con pesos causales w(x_i, t_i) que priorizan
  el aprendizaje cerca de los sensores, las fronteras, y las condiciones
  iniciales, propagando la supervisión gradualmente hacia las zonas sin datos.

  w(x_i, t_i) = w_spatial(x_i) * w_temporal(t_i)

  Componente espacial:
    w_spatial = max(w_boundary(x_i), w_sensor(x_i))
    - w_boundary: decae desde x=0 y x=L hacia el interior
    - w_sensor:   decae desde cada posición de sensor hacia afuera

  Componente temporal:
    w_temporal(t_i) = exp(-epsilon_t * cumsum_residual(t < t_i))
    Propaga causalidad desde t=0 hacia adelante, forzando a la red
    a aprender primero el estado quiescente (pre-fuga).

  L_pde_causal = mean(w_i * r²_i)

FIXES vs. versión anterior:
  1. Bug: variable global 'L' (pipe length) ya no se sombrea
  2. Bins vectorizados con torch.bucketize (10x más rápido en GPU)
  3. Interpolación pura en torch (sin CPU round-trip)
  4. Último bin espacial incluye el borde x=L
  5. Causalidad temporal añadida (no solo espacial)
  6. Re-muestreo periódico de puntos de colocación
  7. Warmup: x_leak se congela los primeros N_WARMUP epochs
  8. Gradient clamp separado para x_leak_raw
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════
# PARÁMETROS
# ═══════════════════════════════════════════════════════
torch.manual_seed(42)
np.random.seed(42)

PIPE_LENGTH = 1000.0       # renamed from 'L' to avoid shadowing
WAVE_SPEED  = 500.0        # renamed from 'a' for clarity
T_SIM       = 4.0
T_LEAK      = 1.0
X_LEAK      = 650.0
Q_LEAK      = 1.0
SIGMA_SRC   = 20.0

SENSOR_POS = [50.0, 300.0, 700.0, 950.0]

NX = 201
NT = 2001

N_HIDDEN  = 5
N_NEURONS = 64
N_COL     = 10000
N_BC      = 300
N_IC      = 300
N_EPOCHS  = 15000
LR_NET    = 1e-3
LR_XLEAK  = 1e-2

LAM_DAT = 10.0
LAM_PDE =  1.0
LAM_BC  = 10.0
LAM_IC  = 10.0

# ── Hiperparámetros CI-PINN ──────────────────────────
EPSILON_DEFAULT   = 0.1    # decaimiento espacial
EPSILON_T_DEFAULT = 0.05   # decaimiento temporal
N_BINS_SPACE      = 100    # bins espaciales
N_BINS_TIME       = 50     # bins temporales

# ── Warmup: x_leak congelado durante esta fase ───────
N_WARMUP = 1000

# ── Re-muestreo de puntos de colocación ──────────────
RESAMPLE_EVERY = 1000


# ═══════════════════════════════════════════════════════
# SIMULADOR FD
# ═══════════════════════════════════════════════════════
def simulate_fd():
    x  = np.linspace(0, PIPE_LENGTH, NX)
    t  = np.linspace(0, T_SIM, NT)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    cfl_r = (WAVE_SPEED * dt / dx) ** 2
    assert cfl_r <= 1.0, f"CFL inestable: r={cfl_r:.3f}"
    print(f"  FD: dx={dx:.2f}m  dt={dt:.5f}s  CFL={cfl_r**0.5:.3f}")

    P    = np.zeros((NX, NT))
    S_sp = np.exp(-0.5 * ((x - X_LEAK) / SIGMA_SRC) ** 2)
    S_sp /= SIGMA_SRC * np.sqrt(2 * np.pi)

    for n in range(1, NT - 1):
        S_t = 1.0 / (1.0 + np.exp(-(t[n] - T_LEAK) / 0.1))
        S_n = Q_LEAK * S_sp * S_t
        P[1:-1, n+1] = (2*P[1:-1,n] - P[1:-1,n-1]
                        + cfl_r*(P[2:,n] - 2*P[1:-1,n] + P[:-2,n])
                        + dt**2 * S_n[1:-1])
        P[0, n+1] = P[-1, n+1] = 0.0

    s_idx = [np.argmin(np.abs(x - xs)) for xs in SENSOR_POS]
    return {"x": x, "t": t, "P": P, "P_sensors": P[s_idx, :]}


# ═══════════════════════════════════════════════════════
# ARQUITECTURA
# ═══════════════════════════════════════════════════════
class WavePINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(2, N_NEURONS), nn.Tanh()]
        for _ in range(N_HIDDEN - 1):
            layers += [nn.Linear(N_NEURONS, N_NEURONS), nn.Tanh()]
        layers.append(nn.Linear(N_NEURONS, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        p_init = np.clip((PIPE_LENGTH/2 - 50.0)/900.0, 1e-5, 1-1e-5)
        self.x_leak_raw = nn.Parameter(
            torch.tensor(np.log(p_init/(1-p_init)), dtype=torch.float32))

    @property
    def x_leak(self):
        return 50.0 + 900.0 * torch.sigmoid(self.x_leak_raw)

    def forward(self, x, t):
        xt = torch.stack([x/PIPE_LENGTH, t/T_SIM], dim=-1)
        return self.net(xt).squeeze(-1)

    def network_params(self):
        excl = {id(self.x_leak_raw)}
        return [p for p in self.parameters() if id(p) not in excl]


# ═══════════════════════════════════════════════════════
# SOURCE TERM
# ═══════════════════════════════════════════════════════
def source_torch(x, t, x_leak, q=Q_LEAK, sig=SIGMA_SRC,
                  t0=T_LEAK, tau=0.1):
    sp = torch.exp(-0.5 * ((x - x_leak) / sig)**2)
    sp = sp / (sig * (2*torch.pi)**0.5)
    tm = torch.sigmoid((t - t0) / tau)
    return q * sp * tm


# ═══════════════════════════════════════════════════════
# CI-PINN: PESOS CAUSALES (espacial + temporal)
# ═══════════════════════════════════════════════════════

def compute_causal_weights(r2_col, x_col, t_col, epsilon, epsilon_t,
                            n_bins_x=N_BINS_SPACE, n_bins_t=N_BINS_TIME):
    """
    Calcula pesos causales espacio-temporales para CI-PINN.

    Componente espacial (Kim & Son 2025, Sección 3.3):
      - w_boundary: propaga desde x=0 y x=L hacia adentro
      - w_sensor:   propaga desde cada sensor hacia afuera
      - w_spatial = max(w_boundary, w_sensor)

    Componente temporal (extensión para ecuación de onda):
      - w_temporal: propaga desde t=0 hacia adelante
      - Fuerza aprender primero el estado quiescente pre-fuga

    Peso final: w = w_spatial * w_temporal

    VECTORIZADO: usa torch.bucketize en vez de loops Python.
    SIN CPU ROUND-TRIP: interpolación pura en torch.

    r2_col   : [N_col] squared residuals (DETACHED)
    x_col    : [N_col] x positions (DETACHED)
    t_col    : [N_col] t positions (DETACHED)
    epsilon  : spatial causality parameter
    epsilon_t: temporal causality parameter
    """
    device = r2_col.device
    nx = n_bins_x
    nt = n_bins_t

    # ─── 1. Bin residuals spatially (vectorized) ─────────────
    x_edges   = torch.linspace(0, PIPE_LENGTH, nx + 1, device=device)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])

    # Clamp to valid range so boundary point x=L falls in last bin
    x_clamped = x_col.clamp(x_edges[0], PIPE_LENGTH - 1e-6)
    bin_idx_x = torch.bucketize(x_clamped, x_edges[1:])  # [N_col], in [0, nx-1]
    bin_idx_x = bin_idx_x.clamp(0, nx - 1)

    # Compute mean r² per spatial bin using scatter
    r2_sum_x   = torch.zeros(nx, device=device)
    count_x    = torch.zeros(nx, device=device)
    r2_sum_x.scatter_add_(0, bin_idx_x, r2_col)
    count_x.scatter_add_(0, bin_idx_x, torch.ones_like(r2_col))
    r2_bins_x = r2_sum_x / (count_x + 1e-8)

    # Normalize for numerical stability
    r2_norm_x = r2_bins_x / (r2_bins_x.mean() + 1e-8)

    # ─── 2. Spatial weights from boundaries ──────────────────
    cum_L = torch.zeros(nx, device=device)
    if nx > 1:
        cum_L[1:] = torch.cumsum(r2_norm_x[:-1], dim=0)

    cum_R = torch.zeros(nx, device=device)
    if nx > 1:
        cum_R[:-1] = torch.cumsum(r2_norm_x[1:].flip(0), dim=0).flip(0)

    w_bnd = torch.maximum(
        torch.exp(-epsilon * cum_L),
        torch.exp(-epsilon * cum_R)
    )

    # ─── 3. Spatial weights from sensors (vectorized) ────────
    sensor_t = torch.tensor(SENSOR_POS, dtype=torch.float32, device=device)
    # Find closest bin index for each sensor
    # sensor_bins: [n_sensors]
    sensor_bins = torch.argmin(
        torch.abs(x_centers.unsqueeze(0) - sensor_t.unsqueeze(1)), dim=1
    )

    w_sen = torch.zeros(nx, device=device)
    for s_idx in range(len(SENSOR_POS)):
        xs_bin = sensor_bins[s_idx].item()

        # Cumulative from sensor rightward
        cum_sr = torch.zeros(nx, device=device)
        if xs_bin + 1 < nx:
            cum_sr[xs_bin+1:] = torch.cumsum(r2_norm_x[xs_bin:nx-1], dim=0)

        # Cumulative from sensor leftward
        cum_sl = torch.zeros(nx, device=device)
        if xs_bin > 0:
            # cum_sl[j] = sum of r2_norm from j+1 to xs_bin-1
            rev_slice = r2_norm_x[1:xs_bin+1].flip(0)
            cum_sl[:xs_bin] = torch.cumsum(rev_slice, dim=0).flip(0)

        w_this = torch.maximum(
            torch.exp(-epsilon * cum_sr),
            torch.exp(-epsilon * cum_sl)
        )
        w_this[xs_bin] = 1.0   # sensor bin: maximum weight
        w_sen = torch.maximum(w_sen, w_this)

    # ─── 4. Combined spatial weight on grid ──────────────────
    w_spatial_grid = torch.maximum(w_bnd, w_sen)   # [nx]

    # ─── 5. Interpolate spatial weights to collocation points ─
    #    Pure torch: linear interpolation, no CPU round-trip
    interp_idx = torch.searchsorted(
        x_centers, x_clamped.clamp(x_centers[0], x_centers[-1])
    ) - 1
    interp_idx = interp_idx.clamp(0, nx - 2)
    frac = (x_clamped - x_centers[interp_idx]) / (
        x_centers[interp_idx + 1] - x_centers[interp_idx] + 1e-12
    )
    frac = frac.clamp(0.0, 1.0)
    w_spatial = (w_spatial_grid[interp_idx] * (1.0 - frac)
                 + w_spatial_grid[interp_idx + 1] * frac)

    # ─── 6. Temporal causality weights ───────────────────────
    #    Propagate from t=0 forward: force learning quiescent state first
    t_edges   = torch.linspace(0, T_SIM, nt + 1, device=device)
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

    t_clamped = t_col.clamp(t_edges[0], T_SIM - 1e-6)
    bin_idx_t = torch.bucketize(t_clamped, t_edges[1:]).clamp(0, nt - 1)

    r2_sum_t = torch.zeros(nt, device=device)
    count_t  = torch.zeros(nt, device=device)
    r2_sum_t.scatter_add_(0, bin_idx_t, r2_col)
    count_t.scatter_add_(0, bin_idx_t, torch.ones_like(r2_col))
    r2_bins_t = r2_sum_t / (count_t + 1e-8)

    r2_norm_t = r2_bins_t / (r2_bins_t.mean() + 1e-8)

    # Cumulative from t=0 forward
    cum_t = torch.zeros(nt, device=device)
    if nt > 1:
        cum_t[1:] = torch.cumsum(r2_norm_t[:-1], dim=0)
    w_temporal_grid = torch.exp(-epsilon_t * cum_t)  # [nt]

    # Interpolate temporal weights to collocation points
    interp_idx_t = torch.searchsorted(
        t_centers, t_clamped.clamp(t_centers[0], t_centers[-1])
    ) - 1
    interp_idx_t = interp_idx_t.clamp(0, nt - 2)
    frac_t = (t_clamped - t_centers[interp_idx_t]) / (
        t_centers[interp_idx_t + 1] - t_centers[interp_idx_t] + 1e-12
    )
    frac_t = frac_t.clamp(0.0, 1.0)
    w_temporal = (w_temporal_grid[interp_idx_t] * (1.0 - frac_t)
                  + w_temporal_grid[interp_idx_t + 1] * frac_t)

    # ─── 7. Combined weight ──────────────────────────────────
    w = w_spatial * w_temporal

    return w.detach()


# ═══════════════════════════════════════════════════════
# COLLOCATION SAMPLER (reusable)
# ═══════════════════════════════════════════════════════

def sample_collocation(n_col, device):
    """Sample fresh collocation points."""
    x_col = torch.empty(n_col, device=device).uniform_(0, PIPE_LENGTH)
    t_col = torch.empty(n_col, device=device).uniform_(0, T_SIM)
    return x_col, t_col


# ═══════════════════════════════════════════════════════
# FUNCIÓN DE PÉRDIDA CI-PINN
# ═══════════════════════════════════════════════════════

def compute_loss_ci(model, t_data, P_data, x_col, t_col,
                     t_bc, x_ic, P_scale, epsilon, epsilon_t):
    """
    Loss con pesos causales CI-PINN (espacial + temporal).

    L_dat : MSE en sensores (idéntica a vanilla PINN)
    L_pde : residuo de onda ponderado por pesos espacio-temporales
    L_bc  : P=0 en fronteras
    L_ic  : P(x,0)=0 y ∂P/∂t(x,0)=0
    """
    x_leak = model.x_leak

    # ── L_dat ──────────────────────────────────────────────────
    loss_dat = torch.tensor(0.0, device=t_data.device)
    for i, xs in enumerate(SENSOR_POS):
        xs_t   = torch.full_like(t_data, xs)
        P_pred = model(xs_t, t_data)
        loss_dat = loss_dat + torch.mean((P_pred - P_data[i])**2)
    loss_dat = loss_dat / len(SENSOR_POS) / P_scale**2

    # ── L_pde con pesos causales ────────────────────────────────
    xc = x_col.clone().requires_grad_(True)
    tc = t_col.clone().requires_grad_(True)

    Pc     = model(xc, tc)
    dPdt   = torch.autograd.grad(Pc.sum(), tc, create_graph=True)[0]
    d2Pdt2 = torch.autograd.grad(dPdt.sum(), tc, create_graph=True)[0]
    dPdx   = torch.autograd.grad(Pc.sum(), xc, create_graph=True)[0]
    d2Pdx2 = torch.autograd.grad(dPdx.sum(), xc, create_graph=True)[0]

    S = source_torch(xc, tc, x_leak)
    pde_scale = WAVE_SPEED**2 * P_scale / PIPE_LENGTH**2
    residual  = (d2Pdt2 - WAVE_SPEED**2 * d2Pdx2 - S) / pde_scale
    r2        = residual**2   # [N_col], gradient graph ACTIVE

    # Weights from DETACHED residuals (no gradient through weights)
    w = compute_causal_weights(
        r2.detach(), xc.detach(), tc.detach(), epsilon, epsilon_t
    )

    # Weighted loss: gradient flows through r2, not w
    loss_pde = torch.mean(w * r2)

    # ── L_bc ────────────────────────────────────────────────────
    P_left  = model(torch.zeros_like(t_bc), t_bc)
    P_right = model(torch.full_like(t_bc, PIPE_LENGTH), t_bc)
    loss_bc = (torch.mean(P_left**2) + torch.mean(P_right**2)) / (2 * P_scale**2)

    # ── L_ic ────────────────────────────────────────────────────
    t0   = torch.zeros_like(x_ic)
    P_i0 = model(x_ic, t0)
    loss_ic_P = torch.mean(P_i0**2) / P_scale**2

    t0g   = torch.zeros_like(x_ic, requires_grad=True)
    P_i0g = model(x_ic.detach(), t0g)
    dPdt0 = torch.autograd.grad(P_i0g.sum(), t0g, create_graph=True)[0]
    loss_ic_Pt = torch.mean(dPdt0**2) / (P_scale / T_SIM)**2
    loss_ic = loss_ic_P + loss_ic_Pt

    # ── Total ────────────────────────────────────────────────────
    loss_total = (LAM_DAT * loss_dat + LAM_PDE * loss_pde
                  + LAM_BC * loss_bc + LAM_IC * loss_ic)

    comps = {
        "total":  loss_total.item(),
        "dat":    loss_dat.item(),
        "pde":    loss_pde.item(),
        "bc":     loss_bc.item(),
        "ic":     loss_ic.item(),
        "w_mean": w.mean().item(),
        "w_min":  w.min().item(),
    }
    return loss_total, comps


# ═══════════════════════════════════════════════════════
# ENTRENAMIENTO
# ═══════════════════════════════════════════════════════

def train(sim, epsilon=EPSILON_DEFAULT, epsilon_t=EPSILON_T_DEFAULT):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}   |   epsilon={epsilon}  epsilon_t={epsilon_t}")

    P_sens  = sim["P_sensors"]
    P_scale = float(np.abs(P_sens).max()) + 1e-8
    print(f"P_scale = {P_scale:.4e}")

    t_data = torch.tensor(sim["t"],  dtype=torch.float32, device=device)
    P_data = torch.tensor(P_sens,    dtype=torch.float32, device=device)

    torch.manual_seed(0)
    x_col, t_col = sample_collocation(N_COL, device)
    t_bc  = torch.linspace(0, T_SIM, N_BC, device=device)
    x_ic  = torch.linspace(0, PIPE_LENGTH, N_IC, device=device)

    model = WavePINN().to(device)
    print(f"x_leak inicial: {model.x_leak.item():.1f} m  (real: {X_LEAK:.1f} m)")
    print(f"Warmup: x_leak congelado durante {N_WARMUP} epochs")
    print(f"Collocation resample cada {RESAMPLE_EVERY} epochs")

    # Separate optimizers for network and x_leak
    opt_net = torch.optim.Adam(model.network_params(), lr=LR_NET)
    opt_xleak = torch.optim.Adam([model.x_leak_raw], lr=LR_XLEAK)

    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_net, T_max=N_EPOCHS, eta_min=1e-5)
    # x_leak scheduler starts counting after warmup, but we create it
    # for N_EPOCHS and just don't step during warmup
    scheduler_xleak = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_xleak, T_max=N_EPOCHS - N_WARMUP, eta_min=1e-4)

    history = []

    print(f"\n{'Epoch':>6} | {'L_total':>9} | {'L_dat':>9} | {'L_pde':>9} | "
          f"{'w_mean':>7} | {'x_leak':>8} | {'error':>7}")
    print("─" * 80)

    for epoch in range(1, N_EPOCHS + 1):
        # ── Resample collocation points periodically ─────────
        if epoch > 1 and epoch % RESAMPLE_EVERY == 0:
            x_col, t_col = sample_collocation(N_COL, device)

        # ── Warmup: freeze x_leak for first N_WARMUP epochs ──
        xleak_active = (epoch > N_WARMUP)

        opt_net.zero_grad()
        if xleak_active:
            opt_xleak.zero_grad()

        loss_total, comps = compute_loss_ci(
            model, t_data, P_data, x_col, t_col,
            t_bc, x_ic, P_scale, epsilon, epsilon_t
        )
        loss_total.backward()

        # Gradient clipping: separate for network and x_leak
        torch.nn.utils.clip_grad_norm_(model.network_params(), max_norm=1.0)
        if xleak_active:
            # Tighter clamp for x_leak to prevent wild jumps
            torch.nn.utils.clip_grad_norm_([model.x_leak_raw], max_norm=0.5)

        opt_net.step()
        scheduler_net.step()

        if xleak_active:
            opt_xleak.step()
            scheduler_xleak.step()

        if epoch % 500 == 0 or epoch == 1:
            xl  = model.x_leak.item()
            err = abs(xl - X_LEAK)
            phase = "warmup" if not xleak_active else "full  "
            print(f"{epoch:6d} | {comps['total']:9.3e} | {comps['dat']:9.3e} | "
                  f"{comps['pde']:9.3e} | {comps['w_mean']:7.4f} | "
                  f"{xl:8.1f} | {err:7.1f}m  [{phase}]")
            history.append({"epoch": epoch, "x_leak": xl, "error_m": err, **comps})

    print("─" * 80)
    x_pred = model.x_leak.item()
    print(f"\n{'='*50}")
    print(f"  x_leak real:  {X_LEAK:.1f} m")
    print(f"  x_leak pred:  {x_pred:.1f} m")
    print(f"  Error final:  {abs(x_pred - X_LEAK):.1f} m")
    print(f"{'='*50}")

    return model, history


# ═══════════════════════════════════════════════════════
# VISUALIZACIÓN
# ═══════════════════════════════════════════════════════

def plot(sim, model, history, epsilon, epsilon_t, save="pinn_ci_results.png"):
    device = next(model.parameters()).device
    x_arr  = sim["x"]; t_arr = sim["t"]; P_fd = sim["P"]
    x_pred = model.x_leak.item()

    with torch.no_grad():
        Xg, Tg = np.meshgrid(x_arr, t_arr, indexing="ij")
        xf = torch.tensor(Xg.ravel(), dtype=torch.float32, device=device)
        tf = torch.tensor(Tg.ravel(), dtype=torch.float32, device=device)
        P_pred = model(xf, tf).cpu().numpy().reshape(len(x_arr), len(t_arr))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    vmax = np.abs(P_fd).max()
    kw   = dict(aspect="auto", origin="lower",
                extent=[0, PIPE_LENGTH, 0, T_SIM],
                cmap="RdBu_r", vmin=-vmax, vmax=vmax)

    im0 = axes[0,0].imshow(P_fd.T, **kw)
    axes[0,0].axvline(X_LEAK, color="yellow", lw=2, label=f"real ({X_LEAK}m)")
    for xs in SENSOR_POS:
        axes[0,0].axvline(xs, color="lime", alpha=0.5, lw=1)
    axes[0,0].set_title("FD — Ground truth"); axes[0,0].legend(fontsize=8)
    plt.colorbar(im0, ax=axes[0,0])

    im1 = axes[0,1].imshow(P_pred.T, **kw)
    axes[0,1].axvline(X_LEAK, color="yellow", lw=2, label=f"real ({X_LEAK}m)")
    axes[0,1].axvline(x_pred, color="red",    lw=2, ls="--",
                      label=f"pred ({x_pred:.1f}m)")
    axes[0,1].set_title("CI-PINN — P(x,t)"); axes[0,1].legend(fontsize=8)
    plt.colorbar(im1, ax=axes[0,1])

    im2 = axes[0,2].imshow(np.abs(P_pred-P_fd).T, aspect="auto",
                            origin="lower",
                            extent=[0, PIPE_LENGTH, 0, T_SIM], cmap="hot")
    axes[0,2].set_title("|CI-PINN - FD|")
    plt.colorbar(im2, ax=axes[0,2])

    t_t  = torch.tensor(t_arr, dtype=torch.float32, device=device)
    cols = ["tab:blue","tab:orange","tab:green","tab:red"]
    for i, xs in enumerate(SENSOR_POS):
        xs_t = torch.full_like(t_t, xs)
        with torch.no_grad():
            Ps = model(xs_t, t_t).cpu().numpy()
        axes[1,0].plot(t_arr, sim["P_sensors"][i],
                       color=cols[i], lw=1.5, label=f"FD x={xs:.0f}m")
        axes[1,0].plot(t_arr, Ps, color=cols[i], ls="--", lw=1, alpha=0.8)
    axes[1,0].axvline(T_LEAK, color="k", ls=":", label="t_leak")
    axes[1,0].set_title("Sensores: FD(─) vs CI-PINN(--)"); axes[1,0].legend(fontsize=7)

    eps_list = [h["epoch"]   for h in history]
    xls      = [h["x_leak"]  for h in history]
    wmeans   = [h["w_mean"]  for h in history]

    ax1 = axes[1,1]
    ax1.plot(eps_list, xls, "b-o", ms=4, label="x_leak")
    ax1.axhline(X_LEAK, color="r", ls="--", label=f"real ({X_LEAK}m)")
    ax1.axhline(x_pred, color="g", ls=":",  label=f"pred ({x_pred:.1f}m)")
    ax1.axvline(N_WARMUP, color="gray", ls=":", alpha=0.5, label="warmup end")
    ax1.set_title("Convergencia de x_leak"); ax1.legend(fontsize=8)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("x_leak (m)")

    ax2 = ax1.twinx()
    ax2.plot(eps_list, wmeans, "orange", ls="-.", lw=1.5, label="w_mean")
    ax2.set_ylabel("Peso causal medio", color="orange")
    ax2.legend(loc="lower right", fontsize=8)

    for key, label in [("total","total"),("dat","datos"),
                        ("pde","PDE (weighted)"),("bc","BC"),("ic","IC")]:
        axes[1,2].plot(eps_list, [h[key] for h in history], label=label)
    axes[1,2].set_yscale("log")
    axes[1,2].set_title("Loss vs epoch"); axes[1,2].legend(fontsize=8)

    plt.suptitle(
        f"CI-PINN (ε_x={epsilon}, ε_t={epsilon_t}) | x_leak real={X_LEAK}m | "
        f"pred={x_pred:.1f}m | error={abs(x_pred-X_LEAK):.1f}m", fontsize=12)
    plt.tight_layout()
    plt.savefig(save, dpi=150)
    print(f"Figura guardada: {save}")


# ═══════════════════════════════════════════════════════
# MAIN — prueba rápida con comparación de epsilon
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("CI-PINN — Causal Inverse PINN (v2, fixed)")
    print("Kim & Son, Mathematics 2025, 13, 1057")
    print("=" * 55)

    print("\n[1/4] Simulando FD...")
    sim = simulate_fd()
    print(f"  Rango P: [{sim['P'].min():.3e}, {sim['P'].max():.3e}]")

    EPSILON   = 0.1
    EPSILON_T = 0.05

    print(f"\n[2/4] Entrenando CI-PINN (epsilon={EPSILON}, "
          f"epsilon_t={EPSILON_T}, {N_EPOCHS} epochs)...")
    model, history = train(sim, epsilon=EPSILON, epsilon_t=EPSILON_T)

    print("\n[3/4] Generando figuras...")
    plot(sim, model, history, EPSILON, EPSILON_T)

    # ── Comparación rápida de epsilons (500 epochs cada uno) ──
    print("\n[4/4] Comparativa rápida de epsilon (500 epochs c/u)...")
    print(f"\n{'epsilon':>10} | {'eps_t':>7} | {'x_leak_pred':>12} | "
          f"{'error':>8} | {'w_mean_final':>13} | {'L_pde':>9}")
    print("─" * 75)

    for eps in [1.0, 0.1, 0.01, 0.001]:
        eps_t = eps * 0.5   # temporal decay is gentler
        torch.manual_seed(42)
        device = next(model.parameters()).device
        m_test = WavePINN().to(device)

        t_data = torch.tensor(sim["t"], dtype=torch.float32, device=device)
        P_data = torch.tensor(sim["P_sensors"], dtype=torch.float32, device=device)
        P_scale = float(np.abs(sim["P_sensors"]).max()) + 1e-8

        torch.manual_seed(0)
        x_col, t_col = sample_collocation(N_COL, device)
        t_bc  = torch.linspace(0, T_SIM, N_BC, device=device)
        x_ic  = torch.linspace(0, PIPE_LENGTH, N_IC, device=device)

        opt = torch.optim.Adam([
            {"params": m_test.network_params(), "lr": LR_NET},
            {"params": [m_test.x_leak_raw],     "lr": LR_XLEAK},
        ])

        for ep in range(1, 501):
            opt.zero_grad()
            loss, comps = compute_loss_ci(
                m_test, t_data, P_data, x_col, t_col,
                t_bc, x_ic, P_scale, eps, eps_t
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m_test.parameters(), 1.0)
            opt.step()

        xl  = m_test.x_leak.item()
        err = abs(xl - X_LEAK)
        print(f"{eps:>10.3f} | {eps_t:>7.3f} | {xl:>12.1f} | {err:>8.1f}m | "
              f"{comps['w_mean']:>13.4f} | {comps['pde']:>9.3e}")

    print("\nInterpretación del w_mean:")
    print("  w_mean ≈ 1.0  → pesos casi uniformes (epsilon muy bajo)")
    print("  w_mean ≈ 0.5  → propagación parcial desde sensores")
    print("  w_mean ≈ 0.1  → pesos muy concentrados en sensores (epsilon alto)")
    print("  El mejor epsilon es el que muestra x_leak moviéndose hacia 650m")

"""
minimal_pinn_v2.py  —  Coordinate Descent

Mismo problema que v1 pero con entrenamiento en dos fases:

  Fase 1: Red aprende P(x,t) desde los datos de sensores.
          x_leak congelado. Sin L_pde para evitar sesgo del x_leak incorrecto.

  Fase 2: Red congelada. Solo x_leak se mueve usando L_pde.
          El residuo r(x,t) = ∂²P_net/∂t² - a²∂²P_net/∂x² es fijo.
          x_leak encuentra dónde S(x_leak) mejor explica ese residuo.
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

L         = 1000.0
a         = 500.0
T_SIM     = 4.0
T_LEAK    = 1.0
X_LEAK    = 650.0    # ← a inferir
Q_LEAK    = 1.0
SIGMA_SRC = 20.0

SENSOR_POS = [50.0, 300.0, 700.0, 950.0]

NX = 201
NT = 2001

N_HIDDEN  = 5
N_NEURONS = 64

N_COL    = 10000
N_BC     = 300
N_IC     = 300

N_PHASE1 = 10000    # epochs solo datos
N_PHASE2 = 8000     # epochs solo física (x_leak)

LR_NET   = 1e-3
LR_XLEAK = 1e-2     # lr más alto: x_leak tiene que recorrer distancia

LAM_DAT = 10.0
LAM_BC  = 10.0
LAM_IC  = 10.0


# ═══════════════════════════════════════════════════════
# SIMULADOR FD  (idéntico a v1)
# ═══════════════════════════════════════════════════════
def simulate_fd():
    x  = np.linspace(0, L, NX)
    t  = np.linspace(0, T_SIM, NT)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    r  = (a * dt / dx) ** 2
    assert r <= 1.0, f"CFL inestable: r={r:.3f}"
    print(f"  FD: dx={dx:.2f}m  dt={dt:.5f}s  CFL={r**0.5:.3f}")

    P = np.zeros((NX, NT))
    S_sp = np.exp(-0.5 * ((x - X_LEAK) / SIGMA_SRC) ** 2)
    S_sp /= SIGMA_SRC * np.sqrt(2 * np.pi)

    for n in range(1, NT - 1):
        S_t = 1.0 / (1.0 + np.exp(-(t[n] - T_LEAK) / 0.1))
        S_n = Q_LEAK * S_sp * S_t
        P[1:-1, n+1] = (2*P[1:-1,n] - P[1:-1,n-1]
                        + r*(P[2:,n] - 2*P[1:-1,n] + P[:-2,n])
                        + dt**2 * S_n[1:-1])
        P[0, n+1] = P[-1, n+1] = 0.0

    s_idx = [np.argmin(np.abs(x - xs)) for xs in SENSOR_POS]
    return {"x": x, "t": t, "P": P, "P_sensors": P[s_idx, :],
            "sensor_idx": s_idx}


# ═══════════════════════════════════════════════════════
# ARQUITECTURA  (idéntica a v1)
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

        p_init = np.clip((L/2 - 50.0) / 900.0, 1e-5, 1-1e-5)
        self.x_leak_raw = nn.Parameter(
            torch.tensor(np.log(p_init/(1-p_init)), dtype=torch.float32))

    @property
    def x_leak(self):
        return 50.0 + 900.0 * torch.sigmoid(self.x_leak_raw)

    def forward(self, x, t):
        xt = torch.stack([x/L, t/T_SIM], dim=-1)
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
# LOSSES SEPARADOS
# ═══════════════════════════════════════════════════════
def loss_data(model, t_data, P_data, P_scale):
    """MSE en los 4 sensores (normalizado)."""
    L = torch.tensor(0.0, device=t_data.device)
    for i, xs in enumerate(SENSOR_POS):
        xs_t   = torch.full_like(t_data, xs)
        P_pred = model(xs_t, t_data)
        L = L + torch.mean((P_pred - P_data[i])**2)
    return L / len(SENSOR_POS) / P_scale**2


def loss_bc(model, t_bc, P_scale):
    """P=0 en x=0 y x=L."""
    Pl = model(torch.zeros_like(t_bc), t_bc)
    Pr = model(torch.full_like(t_bc, L), t_bc)
    return (torch.mean(Pl**2) + torch.mean(Pr**2)) / (2 * P_scale**2)


def loss_ic(model, x_ic, P_scale):
    """P(x,0)=0 y ∂P/∂t(x,0)=0."""
    t0 = torch.zeros_like(x_ic)
    Pi = model(x_ic, t0)
    L_P = torch.mean(Pi**2) / P_scale**2

    t0g  = torch.zeros_like(x_ic, requires_grad=True)
    Pi2  = model(x_ic.detach(), t0g)
    dPdt = torch.autograd.grad(Pi2.sum(), t0g, create_graph=True)[0]
    L_Pt = torch.mean(dPdt**2) / (P_scale/T_SIM)**2
    return L_P + L_Pt


def loss_pde(model, x_col, t_col, P_scale):
    """
    Residuo de la ecuación de onda:
      r = ∂²P/∂t² - a² ∂²P/∂x² - S(x,t; x_leak)
    La red y x_leak se evalúan tal como están en el modelo.
    """
    xc = x_col.clone().requires_grad_(True)
    tc = t_col.clone().requires_grad_(True)

    Pc     = model(xc, tc)
    dPdt   = torch.autograd.grad(Pc.sum(), tc, create_graph=True)[0]
    d2Pdt2 = torch.autograd.grad(dPdt.sum(), tc, create_graph=True)[0]
    dPdx   = torch.autograd.grad(Pc.sum(), xc, create_graph=True)[0]
    d2Pdx2 = torch.autograd.grad(dPdx.sum(), xc, create_graph=True)[0]

    S   = source_torch(xc, tc, model.x_leak)
    pde_scale = a**2 * P_scale / L**2

    residual = (d2Pdt2 - a**2 * d2Pdx2 - S) / pde_scale
    return torch.mean(residual**2)


def loss_pde_source_free(model, x_col, t_col, P_scale):
    """
    Residuo de la ecuación de onda SIN fuente.
    Al forzar esto, la red aprenderá la propagación pero fallará
    precisamente en la ubicación real de la fuente.
    """
    xc = x_col.clone().requires_grad_(True)
    tc = t_col.clone().requires_grad_(True)

    Pc     = model(xc, tc)
    dPdt   = torch.autograd.grad(Pc.sum(), tc, create_graph=True)[0]
    d2Pdt2 = torch.autograd.grad(dPdt.sum(), tc, create_graph=True)[0]
    dPdx   = torch.autograd.grad(Pc.sum(), xc, create_graph=True)[0]
    d2Pdx2 = torch.autograd.grad(dPdx.sum(), xc, create_graph=True)[0]

    pde_scale = a**2 * P_scale / L**2

    residual = (d2Pdt2 - a**2 * d2Pdx2) / pde_scale
    return torch.mean(residual**2)


# ═══════════════════════════════════════════════════════
# ENTRENAMIENTO EN DOS FASES
# ═══════════════════════════════════════════════════════
def train(sim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    P_sens  = sim["P_sensors"]
    P_scale = float(np.abs(P_sens).max()) + 1e-8
    print(f"P_scale = {P_scale:.4e}")

    t_data = torch.tensor(sim["t"],    dtype=torch.float32, device=device)
    P_data = torch.tensor(P_sens,      dtype=torch.float32, device=device)

    torch.manual_seed(0)
    x_col  = torch.empty(N_COL, device=device).uniform_(0, L)
    t_col  = torch.empty(N_COL, device=device).uniform_(0, T_SIM)
    t_bc   = torch.linspace(0, T_SIM, N_BC, device=device)
    x_ic   = torch.linspace(0, L, N_IC, device=device)

    model = WavePINN().to(device)
    print(f"x_leak inicial: {model.x_leak.item():.1f} m  (real: {X_LEAK:.1f} m)")

    history = []

    # ═══════════════════════════════════════════════════
    # FASE 1: entrenar la red con datos/BC/IC
    #         x_leak congelado → sin sesgo de posición incorrecta
    # ═══════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"FASE 1 ({N_PHASE1} epochs) — datos + BC + IC, x_leak congelado")
    print(f"{'─'*60}")

    model.x_leak_raw.requires_grad_(False)   # ← congelar x_leak
    opt1 = torch.optim.Adam(model.network_params(), lr=LR_NET)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=N_PHASE1, eta_min=1e-5)

    for epoch in range(1, N_PHASE1 + 1):
        opt1.zero_grad()
        Ld  = loss_data(model, t_data, P_data, P_scale)
        Lb  = loss_bc(model, t_bc, P_scale)
        Li  = loss_ic(model, x_ic, P_scale)
        Lp_sf = loss_pde_source_free(model, x_col, t_col, P_scale)
        loss_total = LAM_DAT*Ld + LAM_BC*Lb + LAM_IC*Li + 1.0 * Lp_sf
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt1.step(); sch1.step()

        if epoch % 1000 == 0:
            xl = model.x_leak.item()
            print(f"  ep {epoch:5d} | L={loss_total.item():.3e} | "
                  f"dat={Ld.item():.3e} bc={Lb.item():.3e} ic={Li.item():.3e} | "
                  f"x_leak={xl:.1f}m (congelado)")
            history.append({"phase":1, "epoch":epoch,
                             "x_leak":xl, "L_dat":Ld.item(),
                             "L_pde":0.0, "L_bc":Lb.item()})

    # ═══════════════════════════════════════════════════
    # Diagnóstico intermedio: ¿qué dice el residuo ahora?
    # ═══════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("Diagnóstico post-Fase 1: residuo de la PDE con la red fija")
    with torch.no_grad():
        # Escanear el residuo vs x_leak_test
        # Si el residuo tiene mínimo cerca de 650m → Fase 2 funcionará
        x_scan = np.arange(100, 950, 50)
        res_vals = []
        for xl_test in x_scan:
            model.x_leak_raw.data = torch.tensor(
                np.log(np.clip((xl_test-50)/900,1e-5,1-1e-5)),
                dtype=torch.float32, device=device)
            with torch.enable_grad():
                Lp = loss_pde(model, x_col, t_col, P_scale)
            res_vals.append(Lp.item())
        idx_min = np.argmin(res_vals)
        print(f"  Mínimo de L_pde en x_scan: {x_scan[idx_min]:.0f} m "
              f"(real: {X_LEAK:.0f} m)")
        print(f"  L_pde min={res_vals[idx_min]:.4e}  "
              f"max={max(res_vals):.4e}")

    # Inicializar x_leak en el mínimo encontrado por el escaneo para Fase 2
    best_x = x_scan[idx_min]
    p_init = np.clip((best_x - 50.0) / 900.0, 1e-5, 1-1e-5)
    model.x_leak_raw.data = torch.tensor(
        np.log(p_init/(1-p_init)), dtype=torch.float32, device=device)

    # ═══════════════════════════════════════════════════
    # FASE 2: congelar la red, mover solo x_leak con L_pde
    # ═══════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"FASE 2 ({N_PHASE2} epochs) — solo L_pde, red congelada")
    print(f"{'─'*60}")

    # Congelar todos los pesos de la red
    for p in model.network_params():
        p.requires_grad_(False)
    model.x_leak_raw.requires_grad_(True)   # ← solo x_leak libre

    opt2 = torch.optim.Adam([model.x_leak_raw], lr=LR_XLEAK)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=N_PHASE2, eta_min=1e-5)

    for epoch in range(1, N_PHASE2 + 1):
        opt2.zero_grad()
        Lp = loss_pde(model, x_col, t_col, P_scale)
        Lp.backward()
        opt2.step(); sch2.step()

        if epoch % 500 == 0:
            xl  = model.x_leak.item()
            err = abs(xl - X_LEAK)
            print(f"  ep {epoch:5d} | L_pde={Lp.item():.4e} | "
                  f"x_leak={xl:.1f}m | error={err:.1f}m")
            history.append({"phase":2, "epoch":N_PHASE1+epoch,
                             "x_leak":xl, "L_dat":0.0,
                             "L_pde":Lp.item(), "L_bc":0.0})

    x_pred = model.x_leak.item()
    print(f"\n{'='*50}")
    print(f"  x_leak real:  {X_LEAK:.1f} m")
    print(f"  x_leak pred:  {x_pred:.1f} m")
    print(f"  Error:        {abs(x_pred-X_LEAK):.1f} m")
    print(f"{'='*50}")

    return model, history


# ═══════════════════════════════════════════════════════
# VISUALIZACIÓN
# ═══════════════════════════════════════════════════════
def plot(sim, model, history):
    device = next(model.parameters()).device
    x_arr  = sim["x"];  t_arr = sim["t"];  P_fd = sim["P"]
    x_pred = model.x_leak.item()

    with torch.no_grad():
        Xg, Tg = np.meshgrid(x_arr, t_arr, indexing="ij")
        xf = torch.tensor(Xg.ravel(), dtype=torch.float32, device=device)
        tf = torch.tensor(Tg.ravel(), dtype=torch.float32, device=device)
        P_pred = model(xf, tf).cpu().numpy().reshape(len(x_arr), len(t_arr))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    vmax = np.abs(P_fd).max()
    kw   = dict(aspect="auto", origin="lower",
                extent=[0,L,0,T_SIM], cmap="RdBu_r",
                vmin=-vmax, vmax=vmax)

    im0 = axes[0,0].imshow(P_fd.T, **kw)
    axes[0,0].axvline(X_LEAK, color="yellow", lw=2, label=f"real ({X_LEAK}m)")
    for xs in SENSOR_POS:
        axes[0,0].axvline(xs, color="lime", alpha=0.5, lw=1)
    axes[0,0].set_title("FD — Ground truth"); axes[0,0].legend(fontsize=8)
    plt.colorbar(im0, ax=axes[0,0])

    im1 = axes[0,1].imshow(P_pred.T, **kw)
    axes[0,1].axvline(X_LEAK,  color="yellow", lw=2, label=f"real ({X_LEAK}m)")
    axes[0,1].axvline(x_pred,  color="red",    lw=2, ls="--",
                      label=f"pred ({x_pred:.1f}m)")
    axes[0,1].set_title("PINN — P(x,t) predicho"); axes[0,1].legend(fontsize=8)
    plt.colorbar(im1, ax=axes[0,1])

    im2 = axes[0,2].imshow(np.abs(P_pred-P_fd).T, aspect="auto",
                            origin="lower", extent=[0,L,0,T_SIM], cmap="hot")
    axes[0,2].set_title("|P_PINN - P_FD|")
    plt.colorbar(im2, ax=axes[0,2])

    t_t = torch.tensor(t_arr, dtype=torch.float32, device=device)
    cols = ["tab:blue","tab:orange","tab:green","tab:red"]
    for i, xs in enumerate(SENSOR_POS):
        xs_t = torch.full_like(t_t, xs)
        with torch.no_grad():
            Ps = model(xs_t, t_t).cpu().numpy()
        axes[1,0].plot(t_arr, sim["P_sensors"][i],
                       color=cols[i], lw=1.5, label=f"FD x={xs:.0f}m")
        axes[1,0].plot(t_arr, Ps, color=cols[i], ls="--", lw=1, alpha=0.8)
    axes[1,0].axvline(T_LEAK, color="k", ls=":", label="t_leak")
    axes[1,0].set_title("Sensores: FD(─) vs PINN(--)"); axes[1,0].legend(fontsize=7)

    eps = [h["epoch"]  for h in history]
    xls = [h["x_leak"] for h in history]
    ph1 = [h for h in history if h["phase"]==1]
    ph2 = [h for h in history if h["phase"]==2]
    axes[1,1].plot([h["epoch"] for h in ph1],
                   [h["x_leak"] for h in ph1], "b-o", ms=4, label="Fase 1")
    axes[1,1].plot([h["epoch"] for h in ph2],
                   [h["x_leak"] for h in ph2], "r-o", ms=4, label="Fase 2")
    axes[1,1].axhline(X_LEAK, color="k", ls="--", label=f"real ({X_LEAK}m)")
    axes[1,1].axhline(x_pred, color="g", ls=":",  label=f"pred ({x_pred:.1f}m)")
    axes[1,1].set_title("Trayectoria x_leak"); axes[1,1].legend(fontsize=8)
    axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel("x_leak (m)")

    ldat = [h["L_dat"] for h in ph1]
    lpde = [h["L_pde"] for h in ph2]
    ax   = axes[1,2]
    ax.semilogy([h["epoch"] for h in ph1], ldat, "b-", label="L_dat (Fase 1)")
    ax2  = ax.twinx()
    ax2.semilogy([h["epoch"] for h in ph2], lpde, "r-", label="L_pde (Fase 2)")
    ax.set_xlabel("Epoch"); ax.set_title("Loss por fase")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")

    plt.suptitle(
        f"PINN v2 (Coord. Descent) | x_leak real={X_LEAK}m | "
        f"pred={x_pred:.1f}m | error={abs(x_pred-X_LEAK):.1f}m", fontsize=12)
    plt.tight_layout()
    plt.savefig("pinn_wave_v2.png", dpi=150)
    print("Guardado: pinn_wave_v2.png")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("PINN v2 — Coordinate Descent")
    print("=" * 55)

    print("\n[1/3] Simulando...")
    sim = simulate_fd()
    print(f"  Rango P: [{sim['P'].min():.3e}, {sim['P'].max():.3e}]")

    print("\n[2/3] Entrenando...")
    model, history = train(sim)

    print("\n[3/3] Figuras...")
    plot(sim, model, history)
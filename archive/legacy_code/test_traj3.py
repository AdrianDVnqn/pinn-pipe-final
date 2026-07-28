import torch
import minimal

def train_test(sim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P_scale = float(abs(sim["P_sensors"]).max()) + 1e-8
    t_data = torch.tensor(sim["t"], dtype=torch.float32, device=device)
    P_data = torch.tensor(sim["P_sensors"], dtype=torch.float32, device=device)
    x_col = torch.empty(minimal.N_COL, device=device).uniform_(0, minimal.L)
    t_col = torch.empty(minimal.N_COL, device=device).uniform_(0, minimal.T_SIM)
    t_bc = torch.linspace(0, minimal.T_SIM, minimal.N_BC, device=device)
    x_ic = torch.linspace(0, minimal.L, minimal.N_IC, device=device)

    model = minimal.WavePINN().to(device)
    model.x_leak_raw.requires_grad_(False)
    
    opt1 = torch.optim.Adam(model.network_params(), lr=minimal.LR_NET)
    for epoch in range(1, 3001):
        opt1.zero_grad()
        Ld = minimal.loss_data(model, t_data, P_data, P_scale)
        Lb = minimal.loss_bc(model, t_bc, P_scale)
        Li = minimal.loss_ic(model, x_ic, P_scale)
        
        # Source-free PDE with L1 loss
        x_col.requires_grad_(True)
        t_col.requires_grad_(True)
        Pc = model(x_col, t_col) / P_scale
        dPdt = torch.autograd.grad(Pc.sum(), t_col, create_graph=True)[0]
        d2Pdt2 = torch.autograd.grad(dPdt.sum(), t_col, create_graph=True)[0]
        dPdx = torch.autograd.grad(Pc.sum(), x_col, create_graph=True)[0]
        d2Pdx2 = torch.autograd.grad(dPdx.sum(), x_col, create_graph=True)[0]
        residual = d2Pdt2 - (minimal.a**2) * d2Pdx2
        Lp = torch.mean(torch.abs(residual)) # L1 loss! Allows sparsity!
        
        loss = 10.0*Ld + 10.0*Lb + 10.0*Li + Lp
        loss.backward()
        opt1.step()

        if epoch % 1000 == 0:
            print(f"P1 ep {epoch:4d} | Lp={Lp.item():.2e} | Ld={Ld.item():.2e}")

    # Now evaluate residual over a grid to find the source!
    model.eval()
    x_grid = torch.linspace(0, minimal.L, 1000, device=device, requires_grad=True)
    t_grid = torch.full((1000,), 0.2, device=device, requires_grad=True) # t=0.2 is during source emission
    Pc = model(x_grid, t_grid) / P_scale
    dPdt = torch.autograd.grad(Pc.sum(), t_grid, create_graph=True)[0]
    d2Pdt2 = torch.autograd.grad(dPdt.sum(), t_grid, create_graph=True)[0]
    dPdx = torch.autograd.grad(Pc.sum(), x_grid, create_graph=True)[0]
    d2Pdx2 = torch.autograd.grad(dPdx.sum(), x_grid, create_graph=True)[0]
    residual = d2Pdt2 - (minimal.a**2) * d2Pdx2
    
    res_vals = torch.abs(residual).detach().cpu().numpy()
    x_vals = x_grid.detach().cpu().numpy()
    max_idx = res_vals.argmax()
    print(f"Found source at x = {x_vals[max_idx]:.1f}m (true is 650m)")

sim = minimal.simulate_fd()
train_test(sim)

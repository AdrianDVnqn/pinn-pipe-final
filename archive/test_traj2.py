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
    
    # Phase 1: Freeze x_leak at 500m
    model.x_leak_raw.requires_grad_(False)
    opt1 = torch.optim.Adam(model.network_params(), lr=minimal.LR_NET)
    for epoch in range(1, 2001):
        opt1.zero_grad()
        Ld = minimal.loss_data(model, t_data, P_data, P_scale)
        Lb = minimal.loss_bc(model, t_bc, P_scale)
        Li = minimal.loss_ic(model, x_ic, P_scale)
        Lp = minimal.loss_pde(model, x_col, t_col, P_scale, sig=100.0)
        loss = 10.0*Ld + 10.0*Lb + 10.0*Li + Lp
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt1.step()

    # Phase 2: Joint training
    model.x_leak_raw.requires_grad_(True)
    opt2 = torch.optim.Adam([
        {'params': model.network_params(), 'lr': minimal.LR_NET/10},
        {'params': [model.x_leak_raw], 'lr': 2.0}
    ])
    for epoch in range(1, 3001):
        opt2.zero_grad()
        Ld = minimal.loss_data(model, t_data, P_data, P_scale)
        Lb = minimal.loss_bc(model, t_bc, P_scale)
        Li = minimal.loss_ic(model, x_ic, P_scale)
        Lp = minimal.loss_pde(model, x_col, t_col, P_scale, sig=100.0)
        loss = 10.0*Ld + 10.0*Lb + 10.0*Li + Lp
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt2.step()
        if epoch % 500 == 0:
            print(f"ep {epoch:4d} | x_leak={model.x_leak.item():.1f} | Lp={Lp.item():.2e} | Ld={Ld.item():.2e}")

sim = minimal.simulate_fd()
train_test(sim)

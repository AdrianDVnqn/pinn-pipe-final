import os
import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')
import time
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config as cfg

# ═══════════════════════════════════════════════════════════════
# PARTE 1 — ARQUITECTURA DEL MODELO
# ═══════════════════════════════════════════════════════════════

class LeakLSTM(nn.Module):
    def __init__(self,
                 n_channels=5,
                 hidden_size=128,
                 n_layers=2,
                 dropout=0.2):
        '''
        Red LSTM para inferencia de parámetros de fuga
        a partir de series temporales de sensores.
        '''
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_channels,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x: [batch, Nt, n_channels]
        Retorna:
          leak_params: [batch, 2]  (x_leak_norm, q_leak_norm)
          has_leak:    [batch, 1]  probabilidad de fuga
        '''
        lstm_out, (h_n, _) = self.lstm(x)

        # Último estado para LSTM bidireccional
        last_forward  = h_n[-2, :, :]
        last_backward = h_n[-1, :, :]
        combined = torch.cat([last_forward, last_backward], dim=1)

        leak_params = self.regression_head(combined)
        has_leak    = self.classification_head(combined)
        return leak_params, has_leak

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    input_size = state_dict['lstm.weight_ih_l0'].shape[1]
    hidden_size = state_dict['lstm.weight_ih_l0'].shape[0] // 4
    n_layers = 1
    while f'lstm.weight_ih_l{n_layers}' in state_dict:
        n_layers += 1
    model = LeakLSTM(n_channels=input_size, hidden_size=hidden_size, n_layers=n_layers)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def diagnose_collapse(model_path="checkpoints/lstm_best.pt",
                      h5_path="lstm_dataset.h5"):
    '''
    Verifica si el modelo colapsó a la media.
    '''
    model = load_model(model_path)
    input_size = model.lstm.input_size
    n_pressure_sensors = 3 if input_size == 5 else 2
    test_loader = get_dataloaders(h5_path, n_pressure_sensors=n_pressure_sensors)[1]

    all_x_preds = []
    all_x_trues = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            leak_params, _ = model(X.to(device))
            all_x_preds.append(leak_params[:, 0].cpu().numpy())
            all_x_trues.append(y[:, 0].cpu().numpy())

    x_preds = np.concatenate(all_x_preds) * cfg.PIPE_LENGTH
    x_trues = np.concatenate(all_x_trues) * cfg.PIPE_LENGTH

    print("─── Diagnóstico de colapso ───")
    print(f"Predicciones únicas de x_leak: {np.unique(x_preds.round(-2))}")
    print(f"Std de predicciones:           {np.std(x_preds):.1f} m")
    print(f"Media de predicciones:         {np.mean(x_preds):.1f} m")
    print(f"Media del training set:        {cfg.PIPE_LENGTH/2:.1f} m  (esperado si colapsa)")
    if np.std(x_preds) < 500:
        print("✗ CONFIRMADO: el modelo colapsó a la media")
    else:
        print("✓ El modelo muestra variación en las predicciones")
    print("─────────────────────────────")

# ═══════════════════════════════════════════════════════════════
# PARTE 2 — DATASET Y DATALOADER
# ═══════════════════════════════════════════════════════════════

class LeakDataset(Dataset):
    def __init__(self, h5_path, split="train", noise_level=None, n_pressure_sensors=3):
        '''
        Carga X e y desde lstm_dataset.h5
        '''
        with h5py.File(h5_path, 'r') as f:
            grp = f[split]
            self.X_all = grp['X'][:]
            self.y_all = grp['y'][:]
            self.meta = json.loads(grp['metadata'][()])

        self.indices = []
        for i, m in enumerate(self.meta):
            if noise_level is not None:
                if 'noise_name' in m:
                    if m['noise_name'] != noise_level:
                        continue
                else:
                    target_std = cfg.NOISE_LEVELS[noise_level]
                    if abs(m['noise_std'] - target_std) > 1.0:
                        continue
            self.indices.append(i)

        self.X = self.X_all[self.indices]
        self.y = self.y_all[self.indices]

        # Si hay 2 sensores, descartar el del medio (canal 1 de los 3 de dP)
        if n_pressure_sensors == 2:
            # Canales: 0 (dP0), 1 (dP1), 2 (dP2), 3 (dQ0), 4 (dQ1)
            # Conservamos: 0, 2, 3, 4
            self.X = self.X[:, :, [0, 2, 3, 4]]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(h5_path, batch_size=64, noise_level=None, n_pressure_sensors=3):
    '''
    Retorna train_loader y val_loader (split 85/15) y test_loader.
    '''
    train_dataset = LeakDataset(h5_path, split="train", n_pressure_sensors=n_pressure_sensors)
    
    N = len(train_dataset)
    indices = np.random.RandomState(42).permutation(N)
    val_size = int(0.15 * N)
    
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    test_dataset = LeakDataset(h5_path, split="test", noise_level=noise_level, n_pressure_sensors=n_pressure_sensors)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# ═══════════════════════════════════════════════════════════════
# PARTE 3 — FUNCIÓN DE PÉRDIDA COMBINADA
# ═══════════════════════════════════════════════════════════════

def compute_lstm_loss(leak_params_pred, has_leak_pred,
                       leak_params_true, has_leak_true,
                       lambda_reg=10.0, lambda_cls=0.3):
    bce = nn.BCELoss()

    has_leak_true = has_leak_true.view(-1, 1)
    L_cls = bce(has_leak_pred, has_leak_true)

    mask = (has_leak_true > 0.5).view(-1)
    if mask.sum() > 0:
        pred = leak_params_pred[mask]
        true = leak_params_true[mask]
        L_x = nn.functional.mse_loss(pred[:, 0], true[:, 0])
        L_q = nn.functional.mse_loss(pred[:, 1], true[:, 1])
        
        lambda_x = 15.0
        lambda_q = 5.0
        L_reg = lambda_x * L_x + lambda_q * L_q
    else:
        L_reg = torch.tensor(0.0, device=leak_params_pred.device)

    L_total = lambda_reg * L_reg + lambda_cls * L_cls
    return L_total, L_reg, L_cls

# ═══════════════════════════════════════════════════════════════
# PARTE 4 — LOOP DE ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════

def train_lstm(h5_path="lstm_dataset.h5",
               n_epochs=100,
               batch_size=64,
               lr=1e-3,
               hidden_size=128,
               n_layers=2,
               dropout=0.2,
               lambda_reg=10.0,
               lambda_cls=0.3,
               n_pressure_sensors=3,
               save_path=None):
    '''
    Entrena el LSTM y retorna modelo + historial.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Entrenando modelo para {n_pressure_sensors} sensores en: {device}")
    
    if save_path is None:
        save_path = f"checkpoints/lstm_best_{n_pressure_sensors}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n_channels = 5 if n_pressure_sensors == 3 else 4
    model = LeakLSTM(n_channels=n_channels, hidden_size=hidden_size, n_layers=n_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_loader, val_loader, _ = get_dataloaders(h5_path, batch_size=batch_size, n_pressure_sensors=n_pressure_sensors)
    
    history = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    patience_limit = 20
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        if epoch < 21:
            # Etapa 1: Solo clasificación
            for param in model.regression_head.parameters():
                param.requires_grad = False
            current_lambda_reg = 0.0
            current_lambda_cls = lambda_cls
        else:
            # Etapa 2: Clasificación + regresión
            for param in model.regression_head.parameters():
                param.requires_grad = True
            current_lambda_reg = lambda_reg
            current_lambda_cls = lambda_cls
            
            if epoch == 21:
                print("\n─── Transición a Etapa 2: Unfreezing regression_head y reseteando optimizador/patience ───")
                optimizer = torch.optim.Adam([
                    {"params": model.lstm.parameters(), "lr": 1e-4},
                    {"params": model.classification_head.parameters(), "lr": 1e-4},
                    {"params": model.regression_head.parameters(), "lr": 1e-3},
                ], weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
                best_val_loss = float('inf')
                patience_counter = 0

        model.train()
        train_loss_total = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            leak_params_true = y_batch[:, :2]
            has_leak_true = y_batch[:, 2]
            
            optimizer.zero_grad()
            leak_params_pred, has_leak_pred = model(X_batch)
            loss, _, _ = compute_lstm_loss(leak_params_pred, has_leak_pred, leak_params_true, has_leak_true, current_lambda_reg, current_lambda_cls)
            
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * X_batch.size(0)
            
        train_loss = train_loss_total / len(train_loader.dataset)
        
        model.eval()
        val_loss_total = 0.0
        all_val_x_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                leak_params_true = y_batch[:, :2]
                has_leak_true = y_batch[:, 2]
                
                leak_params_pred, has_leak_pred = model(X_batch)
                loss, _, _ = compute_lstm_loss(leak_params_pred, has_leak_pred, leak_params_true, has_leak_true, current_lambda_reg, current_lambda_cls)
                val_loss_total += loss.item() * X_batch.size(0)
                
                all_val_x_preds.append(leak_params_pred[:, 0].cpu().numpy())
                
        val_loss = val_loss_total / len(val_loader.dataset)
        scheduler.step(val_loss)
        
        val_x_preds = np.concatenate(all_val_x_preds)
        val_x_pred_std_m = np.std(val_x_preds) * cfg.PIPE_LENGTH
        
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            if epoch < 21:
                etapa_str = f"Etapa 1 | Epoch {epoch+1:2d}/21"
            else:
                etapa_str = f"Etapa 2 | Epoch {epoch+1:2d}/{n_epochs}"
            comment = "  ← debe crecer en Etapa 2" if epoch >= 21 else ""
            print(f"{etapa_str} | Train loss: {train_loss:.2e} | Val loss: {val_loss:.2e} | Best val: {best_val_loss:.2e} (epoch {best_epoch+1}) | x_pred_std: {val_x_pred_std_m:.1f} m{comment}")
            
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    training_time_s = time.time() - start_time
    
    # Cargar los mejores pesos para inferencia posterior
    model.load_state_dict(torch.load(save_path))
    
    return {
        "model": model,
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
        "training_time_s": training_time_s
    }

# ═══════════════════════════════════════════════════════════════
# PARTE 5 — INFERENCIA Y EVALUACIÓN
# ═══════════════════════════════════════════════════════════════

def run_lstm(scenario_id,
             noise_level="trivial",
             n_pressure_sensors=3,
             h5_path="lstm_dataset.h5",
             model_path=None):
    if model_path is None:
        model_path = f"checkpoints/lstm_best_{n_pressure_sensors}.pt"
        
    model = load_model(model_path)
    input_size = model.lstm.input_size
    n_pressure_sensors = 3 if input_size == 5 else 2
    device = next(model.parameters()).device

    with h5py.File(h5_path, 'r') as f:
        grp = f['test']
        X_all = grp['X'][:]
        y_all = grp['y'][:]
        meta = json.loads(grp['metadata'][()])

    idx = -1
    for i, m in enumerate(meta):
        if m['scenario_id'] == scenario_id and m['noise_name'] == noise_level:
            idx = i
            break

    if idx == -1:
        raise ValueError(f"Escenario {scenario_id} con ruido {noise_level} no encontrado en test.")

    X = X_all[idx:idx+1]
    if n_pressure_sensors == 2:
        X = X[:, :, [0, 2, 3, 4]]
    y = y_all[idx]

    start_t = time.time()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        leak_params_pred, has_leak_pred = model(X_t)
    inf_time_ms = (time.time() - start_t) * 1000.0

    leak_params_pred = leak_params_pred.cpu().numpy()[0]
    has_leak_prob = has_leak_pred.cpu().item()

    leak_detected = has_leak_prob > 0.5

    x_leak_pred = leak_params_pred[0] * float(cfg.PIPE_LENGTH) if leak_detected else None
    q_leak_pred = leak_params_pred[1] * float(cfg.Q_OUTLET) if leak_detected else 0.0

    has_leak = y[2] > 0.5
    x_leak_true = y[0] * float(cfg.PIPE_LENGTH) if has_leak else None
    q_leak_true = y[1] * float(cfg.Q_OUTLET) if has_leak else 0.0

    x_leak_error_km = None
    if has_leak and leak_detected and x_leak_pred is not None:
        x_leak_error_km = abs(x_leak_pred - x_leak_true) / 1000.0

    q_leak_error_pct = None
    if has_leak and leak_detected and q_leak_true > 0:
        q_leak_error_pct = abs(q_leak_pred - q_leak_true) / q_leak_true * 100.0

    return {
        "leak_detected":      leak_detected,
        "has_leak_prob":      has_leak_prob,
        "t_detection":        None,
        "x_leak_pred":        x_leak_pred,
        "x_leak_true":        x_leak_true,
        "x_leak_error_km":    x_leak_error_km,
        "q_leak_pred":        q_leak_pred,
        "q_leak_true":        q_leak_true,
        "q_leak_error_pct":   q_leak_error_pct,
        "method":             "lstm",
        "scenario_id":        scenario_id,
        "noise_level":        noise_level,
        "n_pressure_sensors": n_pressure_sensors,
        "inference_time_ms":  inf_time_ms,
        "has_leak":           has_leak
    }

def evaluate_lstm(noise_levels=None, n_sensors_list=None, model_path=None):
    if noise_levels is None:
        noise_levels = list(cfg.NOISE_LEVELS.keys())
    if n_sensors_list is None:
        n_sensors_list = cfg.N_PRESSURE_SENSOR_LEVELS

    if model_path is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(model_path, map_location=device)
        input_size = state_dict['lstm.weight_ih_l0'].shape[1]
        expected_sensors = 3 if input_size == 5 else 2
        n_sensors_list = [expected_sensors]

    from data_utils import list_scenarios
    scenarios = list_scenarios()

    rows = []
    total = len(scenarios) * len(noise_levels) * len(n_sensors_list)
    counter = 0

    for n_ps in n_sensors_list:
        for noise in noise_levels:
            noise_std = cfg.NOISE_LEVELS[noise]
            for _, sc in scenarios.iterrows():
                sid = sc['scenario_id']
                counter += 1
                print(f'  Corriendo LSTM | noise={noise:<12s} | n_p_sensors={n_ps} | escenario {sid:>2d} ({counter}/{total})')

                res = run_lstm(scenario_id=sid, noise_level=noise, n_pressure_sensors=n_ps, model_path=model_path)

                rows.append({
                    'scenario_id': sid,
                    'has_leak': res['has_leak'],
                    'x_leak_true': res['x_leak_true'] if res['x_leak_true'] is not None else sc['x_leak'],
                    'q_leak_true': res['q_leak_true'] if res['q_leak_true'] > 0 else sc['q_leak'],
                    'leak_size': sc['leak_size'],
                    'noise_level': noise,
                    'noise_std': noise_std,
                    'n_pressure_sensors': n_ps,
                    'leak_detected': res['leak_detected'],
                    't_detection': res['t_detection'],
                    'x_leak_pred': res['x_leak_pred'],
                    'x_leak_error_km': res['x_leak_error_km'],
                    'q_leak_pred': res['q_leak_pred'],
                    'q_leak_error_pct': res['q_leak_error_pct'],
                    'method': 'lstm',
                    'inference_time_ms': res['inference_time_ms'],
                    'has_leak_prob': res['has_leak_prob']
                })

    df = pd.DataFrame(rows)
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(cfg.RESULTS_DIR, 'baseline_lstm.csv')
    df.to_csv(csv_path, index=False)
    print(f'\n  Resultados guardados en: {csv_path}')
    return df

# ═══════════════════════════════════════════════════════════════
# PARTE 6 — VISUALIZACIÓN Y MÉTRICAS
# ═══════════════════════════════════════════════════════════════

def plot_lstm_results(df, history, save_dir="figs"):
    os.makedirs(save_dir, exist_ok=True)

    # Figura 1: Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history['epoch'], history['train_loss'], label='Train Loss')
    axes[0].plot(history['epoch'], history['val_loss'], label='Val Loss')
    best_ep = history['val_loss'].idxmin()
    axes[0].axvline(best_ep, color='r', linestyle='--', label=f'Best Epoch ({best_ep})')
    axes[0].set_title('Loss de Entrenamiento (LSTM)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss Total')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['epoch'], history['train_loss'])
    axes[1].plot(history['epoch'], history['val_loss'])
    axes[1].set_yscale('log')
    axes[1].axvline(best_ep, color='r', linestyle='--')
    axes[1].set_title('Loss (Log Scale)')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'lstm_training_curves.png'), dpi=150)
    plt.close(fig)

    # Figura 2: Error vs Noise
    noise_order = ['trivial', 'facil', 'moderado', 'dificil', 'muy_dificil']
    df_ok = df[(df['has_leak']) & (df['leak_detected']) & (df['x_leak_error_km'].notna())].copy()

    if not df_ok.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        markers = {2: 's', 3: 'o'}
        for ns in sorted(df_ok['n_pressure_sensors'].unique()):
            sub = df_ok[df_ok['n_pressure_sensors'] == ns]
            means = []
            valid_noises = []
            for noise in noise_order:
                vals = sub[sub['noise_level'] == noise]['x_leak_error_km']
                if len(vals) > 0:
                    means.append(vals.mean())
                    valid_noises.append(noise)
            if means:
                ax.plot(valid_noises, means, marker=markers.get(ns, 'o'), lw=2, ms=8, label=f'{ns} sensores presión')

        ax.set_xlabel('Nivel de Ruido')
        ax.set_ylabel('Error medio de localización [km]')
        ax.set_title('Baseline LSTM: Error de localización vs Ruido')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, 'lstm_error_vs_noise.png'), dpi=150)
        plt.close(fig)

    # Figura 3: Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, noise in enumerate(noise_order):
        sub = df[df['noise_level'] == noise]
        if sub.empty: continue
        y_true = sub['has_leak'].astype(int)
        y_pred = sub['leak_detected'].astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues', cbar=False)
        axes[i].set_title(f'Ruido: {noise}')
        axes[i].set_xlabel('Predicción')
        axes[i].set_ylabel('Real')
        axes[i].set_xticklabels(['No Fuga', 'Fuga'])
        axes[i].set_yticklabels(['No Fuga', 'Fuga'])
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'lstm_confusion_matrix.png'), dpi=150)
    plt.close(fig)

def print_summary(df):
    noise_order = ['trivial', 'facil', 'moderado', 'dificil', 'muy_dificil']
    n_ps_vals = df['n_pressure_sensors'].unique()
    n_ps_label = ', '.join(str(v) for v in sorted(n_ps_vals))

    print()
    print('══════════════════════════════════════════════════════')
    print('  RESULTADOS BASELINE: LSTM PURO')
    print(f'  Sensores de presión: {n_ps_label}  |  Caudalímetros: 2 (fijos)')
    print('══════════════════════════════════════════════════════')
    print(f'  {"Noise level":<14s} {"Det.rate":>8s} {"x_err(km)":>10s} {"q_err(%)":>10s}')
    print(f'  {"─" * 14} {"─" * 8} {"─" * 10} {"─" * 10}')

    for noise in noise_order:
        sub = df[df['noise_level'] == noise]
        if sub.empty: continue

        sub_leak = sub[sub['has_leak']]
        if len(sub_leak) > 0:
            det_rate = sub_leak['leak_detected'].mean() * 100.0
        else:
            det_rate = float('nan')

        sub_detected = sub_leak[sub_leak['leak_detected']]
        if len(sub_detected) > 0 and sub_detected['x_leak_error_km'].notna().any():
            x_err = sub_detected['x_leak_error_km'].mean()
            x_err_str = f'{x_err:.3f} km'
        else:
            x_err_str = '  N/A'

        if len(sub_detected) > 0 and sub_detected['q_leak_error_pct'].notna().any():
            q_err = sub_detected['q_leak_error_pct'].mean()
            q_err_str = f'{q_err:.1f}%'
        else:
            q_err_str = '  N/A'

        print(f'  {noise:<14s} {det_rate:>7.0f}% {x_err_str:>10s} {q_err_str:>10s}')

    sub_noleak = df[~df['has_leak']]
    if len(sub_noleak) > 0:
        fp_rate = sub_noleak['leak_detected'].mean() * 100.0
        print(f'\n  Falsos positivos (sin fuga): {fp_rate:.0f}% ({sub_noleak["leak_detected"].sum()}/{len(sub_noleak)})')
    
    inf_time = df['inference_time_ms'].mean()
    print(f'  Tiempo medio de inferencia: {inf_time:.2f} ms')
    print('══════════════════════════════════════════════════════\n')


# ═══════════════════════════════════════════════════════════════
# PARTE 7 — BLOQUE PRINCIPAL
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 1. Diagnóstico del modelo actual
    if os.path.exists("checkpoints/lstm_best.pt"):
        print("═══ DIAGNÓSTICO PRE-FIX ═══")
        diagnose_collapse("checkpoints/lstm_best.pt")

    # 2. Reentrenar con fixes
    print("═══ REENTRENANDO CON FIXES ═══")
    train_result = train_lstm(
        n_epochs=100,
        save_path="checkpoints/lstm_fixed.pt"
    )

    # 3. Diagnóstico post-fix
    print("═══ DIAGNÓSTICO POST-FIX ═══")
    diagnose_collapse("checkpoints/lstm_fixed.pt")

    # 4. Evaluación completa con modelo fixed
    df = evaluate_lstm(model_path="checkpoints/lstm_fixed.pt")

    # 5. Visualizaciones y resumen
    plot_lstm_results(df, train_result["history"])
    print_summary(df[df["n_pressure_sensors"] == 3])

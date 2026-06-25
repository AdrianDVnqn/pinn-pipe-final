"""
app.py -- Streamlit Dashboard for Wave-Injection PINN Leak Detection
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config as cfg
from simulator import run_moc, get_sensor_data
from wave_pinn import WaveLeakPINN, compute_loss, compute_k

# Constants
L_PIPE = float(cfg.PIPE_LENGTH)
T_TOTAL = float(cfg.T_TOTAL)
P_INLET = float(cfg.P_INLET)
Q_OUTLET = float(cfg.Q_OUTLET)
X_PRESSURE_SENSORS = [1000.0, 5000.0, 9000.0]
X_FLOW_METERS = [0.0, 10000.0]

st.set_page_config(page_title="Pipeline Leak Detector", layout="wide")

st.title("🛢️ Simulador de Detección de Fugas (Wave-PINN v2)")
st.markdown("""
Este simulador permite generar una fuga transitoria en un oleoducto virtual e intentar detectarla
utilizando una **Red Neuronal Informada por la Física (PINN)** en tiempo real.
""")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Configuración Física")
true_x = st.sidebar.slider("Posición de la Fuga (m)", min_value=1000, max_value=9000, value=6000, step=500)
true_q = st.sidebar.slider("Tamaño de la Fuga (m³/s)", min_value=0.005, max_value=0.050, value=0.030, step=0.005, format="%.3f")
noise_pa = st.sidebar.selectbox("Ruido en Sensores (Pa)", options=[0, 500, 2000, 8000, 25000, 50000], index=3)

st.sidebar.header("🧠 Configuración IA")
n_epochs = st.sidebar.slider("Epochs (Velocidad vs Precisión)", min_value=500, max_value=4000, value=1500, step=500)

if 'data_tensors' not in st.session_state:
    st.session_state.data_tensors = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Generar Fuga Falsa")
    if st.button("🌊 Inyectar Fuga y Ver Sensores", use_container_width=True):
        with st.spinner("Simulando onda transitoria (MOC)..."):
            # Generar datos usando simulator.py
            moc = run_moc(Q_leak=true_q, x_leak=true_x, t_leak=cfg.T_LEAK_START, noise_std=0.0)
            p_data = get_sensor_data(moc, X_PRESSURE_SENSORS, noise_std=noise_pa)
            q_data = get_sensor_data(moc, X_FLOW_METERS, noise_std=noise_pa * Q_OUTLET / P_INLET)
            
            st.session_state.raw_data = {
                't': moc['t'],
                'P_sensors': p_data['P_sensors'],
                'Q_sensors': q_data['Q_sensors']
            }
            
            # Prepare tensors for PINN
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            t_tensor = torch.tensor(moc['t'], dtype=torch.float32, device=device)
            P_sensors_t = torch.tensor(p_data['P_sensors'], dtype=torch.float32, device=device)
            Q_sensors_t = torch.tensor(q_data['Q_sensors'], dtype=torch.float32, device=device)
            
            t_bc = torch.linspace(0, T_TOTAL, 200, dtype=torch.float32, device=device)
            x_ic = torch.linspace(0, L_PIPE, 200, dtype=torch.float32, device=device)
            P_ss_ic = torch.tensor(
                P_INLET - cfg.FRICTION_FACTOR * cfg.FLUID_DENSITY * x_ic.cpu().numpy() * (Q_OUTLET * abs(Q_OUTLET)) / (2.0 * cfg.PIPE_DIAMETER * (np.pi*cfg.PIPE_DIAMETER**2/4.0)**2),
                dtype=torch.float32, device=device
            )
            
            st.session_state.data_tensors = {
                't': t_tensor, 'P_sensors': P_sensors_t, 'Q_sensors': Q_sensors_t,
                't_bc': t_bc, 'x_ic': x_ic, 't0_ic': torch.zeros_like(x_ic), 'P_ss_ic': P_ss_ic,
            }
        
        if st.session_state.raw_data is not None:
            st.success("¡Datos recolectados exitosamente!")
            
            # Simple Edge Detection (SCADA NPW Trigger)
            # Buscamos la derivada negativa más fuerte en los sensores
            P_sens = st.session_state.raw_data['P_sensors']
            t_arr = st.session_state.raw_data['t']
            
            # Filtro Industrial LTA/STA (Long-Term Average vs Short-Term Average)
            dt = t_arr[1] - t_arr[0]
            win_sta = max(1, int(5.0 / dt))   # Ventana corta: 5 seg
            win_lta = max(1, int(30.0 / dt))  # Ventana larga: 30 seg
            
            t_detected = None
            for i in range(len(X_PRESSURE_SENSORS)):
                P_series = pd.Series(P_sens[i])
                STA = P_series.rolling(win_sta, min_periods=1).mean().to_numpy()
                LTA = P_series.rolling(win_lta, min_periods=1).mean().to_numpy()
                
                # Diferencia: si el promedio corto cae muy por debajo del largo
                caida = LTA - STA
                
                # Ignoramos el periodo de inicialización del LTA (30s)
                idx_drop = np.where((caida > 15000) & (t_arr > 30.0))[0]
                if len(idx_drop) > 0:
                    t_drop = t_arr[idx_drop[0]]
                    if t_detected is None or t_drop < t_detected:
                        t_detected = t_drop
                        
            st.session_state.t_detected = t_detected
            
            if t_detected is not None:
                st.warning(f"🚨 **ALERTA SCADA**: Caída abrupta de presión detectada a los {t_detected:.1f} segundos.")
                st.info("Algoritmo rápido disparado. Aislado ventana de datos para análisis PINN de alta fidelidad.")
            else:
                st.error("El ruido es demasiado alto o la fuga muy pequeña para el trigger rápido SCADA. La PINN podría fallar.")
            
            # Plot Sensors
            fig, ax = plt.subplots(figsize=(8, 4))
            for i, pos in enumerate(X_PRESSURE_SENSORS):
                ax.plot(t_arr, P_sens[i] / 1e6, label=f'x={pos:.0f}m')
            
            if t_detected is not None:
                ax.axvline(t_detected, color='red', linestyle='--', linewidth=2, label='Trigger SCADA')
                ax.axvspan(max(0, t_detected - 30), min(T_TOTAL, t_detected + 60), color='red', alpha=0.1, label='Buffer PINN')
                
            ax.set_title("Mediciones de Presión en Tiempo Real")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Presión (MPa)")
            ax.legend(loc='upper right')
            st.pyplot(fig)


with col2:
    st.subheader("2. Detección Inteligente")
    if st.button("🤖 Ejecutar PINN", use_container_width=True, type="primary"):
        if st.session_state.data_tensors is None:
            st.error("Primero debés inyectar la fuga.")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = WaveLeakPINN().to(device)
            
            opt_mlp = torch.optim.Adam(model.network_params(), lr=1e-3)
            opt_phys = torch.optim.Adam(model.physical_params(), lr=5e-2)
            
            lambdas = {'P': 100.0, 'Q': 100.0, 'pde': 1.0, 'bc': 10.0, 'ic': 10.0}
            
            n_collocation = 5000
            x_col = torch.empty(n_collocation, dtype=torch.float32, device=device)
            t_col = torch.empty(n_collocation, dtype=torch.float32, device=device)
            
            phase1_epochs = int(0.4 * n_epochs)
            phase2_epochs = int(0.3 * n_epochs)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            m_phase = metric_col1.empty()
            m_x = metric_col2.empty()
            m_q = metric_col3.empty()
            
            for epoch in range(1, n_epochs + 1):
                k_current = compute_k(epoch, phase1_epochs)

                torch.manual_seed(epoch + 42)
                with torch.no_grad():
                    x_col.uniform_(0.0, L_PIPE)
                    t_col.uniform_(0.0, T_TOTAL)
                x_col.requires_grad_(True)
                t_col.requires_grad_(True)
                
                # Training logic (condensed)
                if epoch <= phase1_epochs:
                    phase = "Fase 1 (Alineación)"
                    opt_phys.zero_grad()
                    L_P_tensor = torch.tensor(0.0, device=device)
                    Nt = st.session_state.data_tensors['t'].shape[0]
                    for i, xs in enumerate(X_PRESSURE_SENSORS):
                        x_t = torch.full((Nt,), xs, dtype=torch.float32, device=device)
                        P_pred, _, _, _ = model(x_t, st.session_state.data_tensors['t'], k_current)
                        L_P_tensor += torch.mean((P_pred / P_INLET - st.session_state.data_tensors['P_sensors'][i] / P_INLET)**2)
                    L_P_tensor /= len(X_PRESSURE_SENSORS)
                    L_phase1 = lambdas['P'] * L_P_tensor
                    L_phase1.backward()
                    opt_phys.step()
                    
                elif epoch <= phase1_epochs + phase2_epochs:
                    phase = "Fase 2 (Fricción)"
                    opt_mlp.zero_grad()
                    progress = min(1.0, (epoch - phase1_epochs) / float(phase2_epochs))
                    lambdas['pde'] = 1.0 + (1000.0 - 1.0) * progress
                    L_total, comps = compute_loss(model, st.session_state.data_tensors, x_col, t_col, lambdas, k_current)
                    L_total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt_mlp.step()
                else:
                    phase = "Fase 3 (Fino)"
                    if epoch == phase1_epochs + phase2_epochs + 1:
                        for pg in opt_phys.param_groups:
                            pg['lr'] = 2e-3
                    opt_mlp.zero_grad()
                    opt_phys.zero_grad()
                    L_total, comps = compute_loss(model, st.session_state.data_tensors, x_col, t_col, lambdas, k_current)
                    L_total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt_mlp.step()
                    opt_phys.step()

                # Update UI periodically
                if epoch % 50 == 0 or epoch == n_epochs:
                    progress_bar.progress(epoch / n_epochs)
                    status_text.text(f"Entrenando Epoch {epoch}/{n_epochs}...")
                    m_phase.metric("Fase Activa", phase)
                    m_x.metric("Predicción x_leak", f"{model.x_leak.item():.0f} m")
                    m_q.metric("Predicción q_leak", f"{model.q_leak.item():.4f} m³/s")
            
            # Final Results
            x_pred = model.x_leak.item()
            q_pred = model.q_leak.item()
            err_x = abs(x_pred - true_x)
            
            st.success(f"✅ ¡Fuga Detectada!")
            st.markdown(f"### Posición Estimada: **{x_pred:.0f} m**")
            st.markdown(f"### Caudal Estimado: **{q_pred:.4f} m³/s**")
            
            if err_x < 200:
                st.info(f"Precisión excelente. Error absoluto de localización: {err_x:.0f}m ({err_x/L_PIPE*100:.1f}%)")
            else:
                st.warning(f"La detección tiene un margen de error mayor debido a la configuración actual. Error: {err_x:.0f}m")

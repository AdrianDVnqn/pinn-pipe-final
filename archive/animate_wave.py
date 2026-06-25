"""
animate_wave.py -- Animación de la propagación de onda de presión y caudal tras una fuga.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from simulator import run_moc
import config as cfg

def create_animation(x_leak=6000.0, q_leak=0.030, save_path='wave_animation.gif'):
    print(f"Generando simulación MOC para x_leak={x_leak}m, q_leak={q_leak}m³/s...")
    moc = run_moc(Q_leak=q_leak, x_leak=x_leak, t_leak=cfg.T_LEAK_START, noise_std=0.0)
    
    x = moc['x']
    t = moc['t']
    P = moc['P'] / 1e6  # MPa
    Q = moc['Q']
    
    # Submuestrear el tiempo para la animación (ej. 1 de cada 4 frames)
    step = 4
    t_anim = t[::step]
    P_anim = P[:, ::step]
    Q_anim = Q[:, ::step]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f'Simulación de Fuga Transitoria (x={x_leak}m, q={q_leak}m³/s)', fontsize=14)
    
    # Configurar Eje de Presión
    ax1.set_xlim(0, cfg.PIPE_LENGTH)
    ax1.set_ylim(np.min(P) - 0.1, np.max(P) + 0.1)
    ax1.set_ylabel('Presión (MPa)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.axvline(x_leak, color='r', linestyle='--', alpha=0.5, label='Fuga')
    ax1.legend(loc='upper right')
    
    # Configurar Eje de Caudal
    ax2.set_xlim(0, cfg.PIPE_LENGTH)
    ax2.set_ylim(np.min(Q) - 0.05, np.max(Q) + 0.05)
    ax2.set_xlabel('Distancia (m)')
    ax2.set_ylabel('Caudal (m³/s)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.axvline(x_leak, color='r', linestyle='--', alpha=0.5, label='Fuga')
    
    # Líneas a actualizar
    line_p, = ax1.plot([], [], 'b-', lw=2)
    line_q, = ax2.plot([], [], 'g-', lw=2)
    
    time_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    def init():
        line_p.set_data([], [])
        line_q.set_data([], [])
        time_text.set_text('')
        return line_p, line_q, time_text
    
    def update(frame):
        line_p.set_data(x, P_anim[:, frame])
        line_q.set_data(x, Q_anim[:, frame])
        
        current_t = t_anim[frame]
        status = "ESTADO ESTACIONARIO" if current_t < cfg.T_LEAK_START else "¡FUGA ACTIVA!"
        time_text.set_text(f'Tiempo: {current_t:.1f} s | {status}')
        
        if current_t >= cfg.T_LEAK_START:
            time_text.set_color('red')
        else:
            time_text.set_color('black')
            
        return line_p, line_q, time_text

    print(f"Renderizando {len(t_anim)} frames. Esto puede tardar unos minutos...")
    ani = animation.FuncAnimation(fig, update, frames=len(t_anim), init_func=init, blit=True)
    
    # Guardar usando pillow (GIF)
    ani.save(save_path, writer='pillow', fps=15)
    print(f"¡Animación guardada con éxito en: {save_path}!")

if __name__ == '__main__':
    create_animation()

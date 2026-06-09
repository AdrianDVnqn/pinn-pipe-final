import numpy as np
import matplotlib.pyplot as plt
import config as cfg


# ------------------------------------------------------------------
# Este archivo implementa un simulador de oleoducto 1D por MOC.
# La idea es separar de forma clara:
#   1) parametros fisicos y numericos,
#   2) estado estacionario analitico,
#   3) integracion transitoria por metodo de caracteristicas,
#   4) extraccion de datos en sensores,
#   5) tests de validacion y graficos.
# ------------------------------------------------------------------

# ==========================================================
# PARÁMETROS DEL SISTEMA
# ==========================================================
# Geometria del ducto.
L = float(cfg.PIPE_LENGTH)
D = float(cfg.PIPE_DIAMETER)
A = np.pi * D**2 / 4.0

# Propiedades del fluido y parametro de friccion Darcy-Weisbach.
rho = float(cfg.FLUID_DENSITY)
a = float(cfg.WAVE_SPEED)
f = float(cfg.FRICTION_FACTOR)

# Condiciones de contorno impuestas en los extremos.
P_in = float(cfg.P_INLET)
Q_out = float(cfg.Q_OUTLET)

# Parametros de la fuga que se usan por defecto en los tests.
x_leak_default = float(cfg.X_LEAK_VALUES[2])
Q_leak_default = float(cfg.Q_LEAK_VALUES[1])
t_leak_default = float(cfg.T_LEAK_START)

# Discretizacion espacial y temporal.
Nx = int(cfg.N_NODES)
dx = L / (Nx - 1)
# Con dt = dx/a se cumple CFL exacto para las caracteristicas.
dt = dx / a
T = float(cfg.T_TOTAL)


def build_grid():
    # Malla espacial uniforme y vector temporal completo de simulacion.
    x = np.linspace(0.0, L, Nx)
    t = np.arange(0.0, T + dt / 2.0, dt)
    return x, t


def steady_state_profile():
    # En estacionario, el caudal es uniforme y coincide con la condicion de salida.
    x = np.linspace(0.0, L, Nx)
    Q_ss = np.full(Nx, Q_out, dtype=float)
    # La presion cae de forma lineal por friccion. No se usa MOC aqui.
    P_ss = P_in - f * rho * x * (Q_out * np.abs(Q_out)) / (2.0 * D * A**2)
    return x, P_ss, Q_ss


def validate_steady_state(P_ss, Q_ss, x):
    # Verificacion analitica del perfil estacionario.
    if len(x) != len(P_ss):
        raise ValueError('x y P_ss deben tener la misma longitud.')
    v = Q_out / A
    expected_P_end = P_in - f * (L / D) * (rho * v**2 / 2.0)
    checks = {
        'P0': np.isclose(P_ss[0], P_in, atol=1e-12, rtol=0.0),
        'Pend': np.isclose(P_ss[-1], expected_P_end, atol=1e-9, rtol=0.0),
        'Quniform': np.allclose(Q_ss, Q_out, atol=1e-12, rtol=0.0),
    }
    # Si el perfil es perfectamente lineal, la segunda diferencia discreta es casi cero.
    second_diff = np.max(np.abs(np.diff(P_ss, n=2))) if len(P_ss) > 2 else 0.0
    checks['linear_profile'] = second_diff < 1e-9
    return checks


def run_moc(Q_leak=0.0, x_leak=x_leak_default, t_leak=t_leak_default, noise_std=0.0):
    '''
    Retorna dict con:
      - t:        array de tiempos [Nt]
      - x:        array de posiciones [Nx]
      - P:        array P(x,t) [Nx, Nt]
      - Q:        array Q(x,t) [Nx, Nt]
      - P_ss:     perfil estacionario [Nx]
      - Q_ss:     caudal estacionario [Nx]
      - dP:       P - P_ss (desviación) [Nx, Nt]
      - dQ:       Q - Q_ss (desviación) [Nx, Nt]
    '''
    # Construimos la malla de calculo y el estado estacionario inicial.
    x, t = build_grid()
    Nt = len(t)
    P_ss = P_in - f * rho * x * (Q_out * np.abs(Q_out)) / (2.0 * D * A**2)
    Q_ss = np.full(Nx, Q_out, dtype=float)

    # Arreglos de estado completo P(x,t) y Q(x,t).
    P = np.zeros((Nx, Nt), dtype=float)
    Q = np.zeros((Nx, Nt), dtype=float)

    # El problema arranca exactamente desde el estacionario para evitar transitorios artificiales.
    P[:, 0] = P_ss
    Q[:, 0] = Q_ss

    # B es la impedancia caracteristica y R agrupa la friccion del tramo dx.
    B = rho * a / A
    R = rho * f * dx / (2.0 * D * A**2)

    # Localizamos el nodo mas cercano a la fuga y el instante en que se activa.
    leak_idx = int(np.argmin(np.abs(x - x_leak)))
    leak_active_step = int(np.ceil(t_leak / dt))

    # Loop temporal principal del metodo de caracteristicas.
    for n in range(Nt - 1):
        P_next = P[:, n].copy()
        Q_next = Q[:, n].copy()

        # --------------------------------------------------------------
        # Condicion de contorno upstream:
        #   P(0,t) = P_in
        # y el caudal se obtiene usando la caracteristica entrante C-.
        # --------------------------------------------------------------
        CM = P[1, n] - B * Q[1, n] + R * Q[1, n] * np.abs(Q[1, n])
        P_next[0] = P_in
        Q_next[0] = (P_in - CM) / B

        # --------------------------------------------------------------
        # Nodos interiores sin fuga:
        #   combinamos las caracteristicas C+ y C- que llegan desde
        #   los vecinos izquierdo y derecho.
        # --------------------------------------------------------------
        for i in range(1, Nx - 1):
            # En torno a la fuga dejamos el cierre especial para un bloque aparte,
            # asi evitamos mezclar la formulacion standard con el nodo dividido.
            if i in (leak_idx - 1, leak_idx, leak_idx + 1) and n + 1 >= leak_active_step and Q_leak > 0.0:
                continue

            # Caracteristica que llega desde la izquierda.
            CP = P[i - 1, n] + B * Q[i - 1, n] - R * Q[i - 1, n] * np.abs(Q[i - 1, n])
            # Caracteristica que llega desde la derecha.
            CM = P[i + 1, n] - B * Q[i + 1, n] + R * Q[i + 1, n] * np.abs(Q[i + 1, n])
            # Solucion local del sistema lineal para el nodo interior.
            P_next[i] = 0.5 * (CP + CM)
            Q_next[i] = (CP - CM) / (2.0 * B)

        # --------------------------------------------------------------
        # Tratamiento especial del nodo de fuga y de sus vecinos inmediatos.
        # La presion es continua, pero el caudal se divide en dos ramas:
        #   Q_left  = caudal que llega al nodo desde la izquierda
        #   Q_right = caudal que sale hacia la derecha
        # y la diferencia entre ambos es la fuga Q_leak.
        # --------------------------------------------------------------
        if n + 1 >= leak_active_step and Q_leak > 0.0 and 1 <= leak_idx <= Nx - 2:
            # Caracteristica que llega al nodo de fuga desde la izquierda.
            CP_leak = P[leak_idx - 1, n] + B * Q[leak_idx - 1, n] - R * Q[leak_idx - 1, n] * np.abs(Q[leak_idx - 1, n])
            # Caracteristica que llega al nodo de fuga desde la derecha.
            CM_leak = P[leak_idx + 1, n] - B * Q[leak_idx + 1, n] + R * Q[leak_idx + 1, n] * np.abs(Q[leak_idx + 1, n])

            # Presion comun en el nodo de fuga con una correccion por la perdida de caudal.
            P_leak = 0.5 * (CP_leak + CM_leak) - 0.5 * B * Q_leak
            # Caudal aguas abajo del nodo de fuga.
            Q_right = (CP_leak - CM_leak - B * Q_leak) / (2.0 * B)
            # Caudal aguas arriba del nodo de fuga.
            Q_left = Q_right + Q_leak

            # Guardamos la presion del nodo y el caudal downstream.
            P_next[leak_idx] = P_leak
            Q_next[leak_idx] = Q_right

            # Nodo inmediatamente a la izquierda de la fuga.
            if leak_idx - 1 >= 1:
                CP_left = P[leak_idx - 2, n] + B * Q[leak_idx - 2, n] - R * Q[leak_idx - 2, n] * np.abs(Q[leak_idx - 2, n])
                CM_left = P_leak - B * Q_left + R * Q_left * np.abs(Q_left)
                P_next[leak_idx - 1] = 0.5 * (CP_left + CM_left)
                Q_next[leak_idx - 1] = (CP_left - CM_left) / (2.0 * B)

            # Nodo inmediatamente a la derecha de la fuga.
            if leak_idx + 1 <= Nx - 2:
                CP_right = P_leak + B * Q_right - R * Q_right * np.abs(Q_right)
                CM_right = P[leak_idx + 2, n] - B * Q[leak_idx + 2, n] + R * Q[leak_idx + 2, n] * np.abs(Q[leak_idx + 2, n])
                P_next[leak_idx + 1] = 0.5 * (CP_right + CM_right)
                Q_next[leak_idx + 1] = (CP_right - CM_right) / (2.0 * B)
        elif n + 1 >= leak_active_step and Q_leak > 0.0:
            # Caso de respaldo si la fuga cae en un borde numerico no deseado.
            CP = P[leak_idx - 1, n] + B * Q[leak_idx - 1, n] - R * Q[leak_idx - 1, n] * np.abs(Q[leak_idx - 1, n])
            CM = P[leak_idx + 1, n] - B * Q[leak_idx + 1, n] + R * Q[leak_idx + 1, n] * np.abs(Q[leak_idx + 1, n])
            P_next[leak_idx] = 0.5 * (CP + CM)
            Q_next[leak_idx] = (CP - CM) / (2.0 * B)

        # --------------------------------------------------------------
        # Condicion de contorno downstream:
        #   Q(L,t) = Q_out
        # y la presion se obtiene con la caracteristica entrante C+.
        # --------------------------------------------------------------
        CP = P[Nx - 2, n] + B * Q[Nx - 2, n] - R * Q[Nx - 2, n] * np.abs(Q[Nx - 2, n])
        Q_next[Nx - 1] = Q_out
        P_next[Nx - 1] = CP - B * Q_out

        # Guardamos el nuevo estado temporal.
        P[:, n + 1] = P_next
        Q[:, n + 1] = Q_next

    # Ruido opcional para emular mediciones sinteticas.
    if noise_std > 0.0:
        rng = np.random.default_rng(12345)
        P = P + rng.normal(0.0, noise_std, size=P.shape)
        Q = Q + rng.normal(0.0, noise_std / max(B, 1.0), size=Q.shape)

    # Desviaciones respecto del estado estacionario base.
    dP = P - P_ss[:, None]
    dQ = Q - Q_ss[:, None]

    return {
        't': t,
        'x': x,
        'P': P,
        'Q': Q,
        'P_ss': P_ss,
        'Q_ss': Q_ss,
        'dP': dP,
        'dQ': dQ,
        'B': B,
        'R': R,
        'leak_idx': leak_idx,
        'leak_active_step': leak_active_step,
    }


def _interp_at_x(x_grid, values, x_query):
    # Interpolacion lineal 1D sin SciPy.
    if x_query <= x_grid[0]:
        return values[0]
    if x_query >= x_grid[-1]:
        return values[-1]
    idx = int(np.searchsorted(x_grid, x_query))
    x0 = x_grid[idx - 1]
    x1 = x_grid[idx]
    w = (x_query - x0) / (x1 - x0)
    return (1.0 - w) * values[idx - 1] + w * values[idx]


def get_sensor_data(moc_result, sensor_positions, noise_std=0.0):
    '''
    sensor_positions: lista de posiciones en metros, ej [0, 5000, 10000]

    Retorna dict con:
      - t:          array de tiempos
      - P_sensors:  array [n_sensores, Nt] con ruido gaussiano
      - Q_sensors:  array [n_sensores, Nt] con ruido gaussiano
      - dP_sensors: desviaciones respecto al estacionario
      - sensor_idx: índices de nodo correspondientes a cada sensor
    '''
    # Extraemos la solucion completa del simulador.
    x = moc_result['x']
    t = moc_result['t']
    P = moc_result['P']
    Q = moc_result['Q']
    P_ss = moc_result['P_ss']

    # Prealocamos matrices de sensores.
    n_sensors = len(sensor_positions)
    Nt = len(t)
    P_sensors = np.zeros((n_sensors, Nt), dtype=float)
    Q_sensors = np.zeros((n_sensors, Nt), dtype=float)
    dP_sensors = np.zeros((n_sensors, Nt), dtype=float)
    sensor_idx = np.zeros(n_sensors, dtype=int)

    # Cada sensor se evalua por interpolacion espacial sobre toda la historia temporal.
    for s, xpos in enumerate(sensor_positions):
        sensor_idx[s] = int(np.argmin(np.abs(x - xpos)))
        for n in range(Nt):
            P_sensors[s, n] = _interp_at_x(x, P[:, n], xpos)
            Q_sensors[s, n] = _interp_at_x(x, Q[:, n], xpos)

        # Desviacion respecto al perfil estacionario interpolado en la misma posicion.
        P_ss_s = _interp_at_x(x, P_ss, xpos)
        dP_sensors[s, :] = P_sensors[s, :] - P_ss_s

    # Ruido gaussiano opcional sobre las mediciones de sensores.
    if noise_std > 0.0:
        rng = np.random.default_rng(54321)
        P_sensors = P_sensors + rng.normal(0.0, noise_std, size=P_sensors.shape)
        Q_sensors = Q_sensors + rng.normal(0.0, noise_std / max(moc_result['B'], 1.0), size=Q_sensors.shape)
        for s, xpos in enumerate(sensor_positions):
            # Si cambia la medicion, recalculamos la desviacion medida.
            P_ss_s = _interp_at_x(x, P_ss, xpos)
            dP_sensors[s, :] = P_sensors[s, :] - P_ss_s

    return {
        't': t,
        'P_sensors': P_sensors,
        'Q_sensors': Q_sensors,
        'dP_sensors': dP_sensors,
        'sensor_idx': sensor_idx,
        'sensor_positions': np.array(sensor_positions, dtype=float),
    }


def plot_steady_state(x, P_ss):
    # Grafico simple del perfil de referencia para inspeccion visual rapida.
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(x, P_ss / 1e6, linewidth=2)
    ax.set_title('Estado estacionario de presión')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('P_ss [MPa]')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def test_1_stationary_stability():
    # Caso base sin fuga: el sistema deberia quedarse practicamente congelado.
    result = run_moc(Q_leak=0.0, x_leak=x_leak_default, t_leak=t_leak_default, noise_std=0.0)
    x = result['x']
    t = result['t']
    dP = result['dP']
    sensor_positions = [0.0, L / 2.0, L]
    sensors = get_sensor_data(result, sensor_positions, noise_std=0.0)
    max_abs_dP = float(np.max(np.abs(dP)))

    # Cuatro vistas complementarias del mismo test.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    axes[0].plot(x, result['P_ss'] / 1e6)
    axes[0].set_title('Perfil estacionario P_ss(x)')
    axes[0].set_xlabel('x [m]')
    axes[0].set_ylabel('P [MPa]')
    axes[0].grid(True, alpha=0.3)

    for xpos, series in zip(sensor_positions, sensors['P_sensors']):
        axes[1].plot(t, series / 1e6, label=f'x={xpos:.0f} m')
    axes[1].set_title('P(t) en sensores, sin fuga')
    axes[1].set_xlabel('t [s]')
    axes[1].set_ylabel('P [MPa]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, np.max(np.abs(dP), axis=0))
    axes[2].set_title('Máximo |dP| en el dominio')
    axes[2].set_xlabel('t [s]')
    axes[2].set_ylabel('|dP| [Pa]')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, sensors['dP_sensors'].T)
    axes[3].set_title('dP(t) en sensores')
    axes[3].set_xlabel('t [s]')
    axes[3].set_ylabel('dP [Pa]')
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f'Test 1 - max(|dP|) = {max_abs_dP:.3e} Pa')
    fig.tight_layout()
    return {
        'result': result,
        'fig': fig,
        'max_abs_dP': max_abs_dP,
        'stable': max_abs_dP < 1.0,
    }


def test_2_wave_speed():
    # Caso con fuga: medimos la llegada de la perturbacion a ambos extremos.
    x_leak = x_leak_default
    t_leak = t_leak_default
    result = run_moc(Q_leak=Q_leak_default, x_leak=x_leak, t_leak=t_leak, noise_std=0.0)
    sensors = get_sensor_data(result, [0.0, L], noise_std=0.0)
    t = result['t']
    predicted_up = t_leak + x_leak / a
    predicted_down = t_leak + (L - x_leak) / a

    # En el extremo upstream usamos Q porque P(0,t) esta fijada por contorno.
    upstream_signal = np.abs(sensors['Q_sensors'][0] - Q_out)
    downstream_signal = np.abs(sensors['dP_sensors'][1])
    upstream_threshold = 1e-4
    downstream_threshold = 1.0
    upstream_idx = int(np.argmax(upstream_signal > upstream_threshold)) if np.any(upstream_signal > upstream_threshold) else int(np.argmax(upstream_signal))
    downstream_idx = int(np.argmax(downstream_signal > downstream_threshold)) if np.any(downstream_signal > downstream_threshold) else int(np.argmax(downstream_signal))
    observed_up = t[upstream_idx]
    observed_down = t[downstream_idx]

    # Separamos la senal upstream y downstream para visualizar el retardo de propagacion.
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(t, sensors['Q_sensors'][0] - Q_out, label='Sensor upstream Q(x=0) - Q_out')
    axes[0].axvline(predicted_up, color='k', linestyle='--', label=f'Predicho {predicted_up:.2f} s')
    axes[0].axvline(observed_up, color='r', linestyle=':', label=f'Observado {observed_up:.2f} s')
    axes[0].set_title('Onda upstream')
    axes[0].set_ylabel('dQ [m3/s]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, sensors['dP_sensors'][1], label='Sensor downstream x=L')
    axes[1].axvline(predicted_down, color='k', linestyle='--', label=f'Predicho {predicted_down:.2f} s')
    axes[1].axvline(observed_down, color='r', linestyle=':', label=f'Observado {observed_down:.2f} s')
    axes[1].set_title('Onda downstream')
    axes[1].set_xlabel('t [s]')
    axes[1].set_ylabel('dP [Pa]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return {
        'result': result,
        'fig': fig,
        'predicted_up': predicted_up,
        'predicted_down': predicted_down,
        'observed_up': observed_up,
        'observed_down': observed_down,
    }


def test_3_mass_balance():
    # Comparamos el balance global en ausencia y presencia de fuga.
    no_leak = run_moc(Q_leak=0.0, x_leak=x_leak_default, t_leak=t_leak_default, noise_std=0.0)
    leak = run_moc(Q_leak=Q_leak_default, x_leak=x_leak_default, t_leak=t_leak_default, noise_std=0.0)

    Q_no = no_leak['Q']
    Q_leak_arr = leak['Q']

    steady_err = float(max(abs(Q_no[0, -1] - Q_out), abs(Q_no[-1, -1] - Q_out)))
    steady_mass_ok = np.isclose(Q_no[0, -1], Q_no[-1, -1], atol=1e-6, rtol=0.0)

    # Para evaluar el efecto de la fuga medimos en un tramo tardio, cuando la senal ya se establecio.
    regime_mask = leak['t'] >= 150.0
    final_mass_diff = float(np.mean(Q_leak_arr[0, regime_mask] - Q_leak_arr[-1, regime_mask]))
    leak_mass_ok = np.isclose(final_mass_diff, Q_leak_default, atol=2e-3, rtol=0.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(no_leak['t'], Q_no[0], label='Q[0] sin fuga')
    axes[0].plot(no_leak['t'], Q_no[-1], label='Q[-1] sin fuga')
    axes[0].set_title('Balance de masa sin fuga')
    axes[0].set_ylabel('Q [m3/s]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(leak['t'], Q_leak_arr[0] - Q_leak_arr[-1], label='Q_in - Q_out')
    axes[1].axhline(Q_leak_default, color='k', linestyle='--', label='Q_leak esperada')
    axes[1].set_title('Balance de masa con fuga')
    axes[1].set_xlabel('t [s]')
    axes[1].set_ylabel('Caudal [m3/s]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return {
        'no_leak': no_leak,
        'leak': leak,
        'fig': fig,
        'steady_err': steady_err,
        'steady_mass_ok': steady_mass_ok,
        'final_mass_diff': final_mass_diff,
        'leak_mass_ok': leak_mass_ok,
    }


def test_4_snr():
    # El SNR se calcula sobre la senal limpia y luego se compara con mediciones con ruido.
    noise_std = 1000.0
    result_clean = run_moc(Q_leak=Q_leak_default, x_leak=x_leak_default, t_leak=t_leak_default, noise_std=0.0)
    result_noisy = run_moc(Q_leak=Q_leak_default, x_leak=x_leak_default, t_leak=t_leak_default, noise_std=0.0)

    sensors_clean = get_sensor_data(result_clean, [0.0, L / 2.0, L], noise_std=0.0)
    sensors_noisy = get_sensor_data(result_noisy, [0.0, L / 2.0, L], noise_std=noise_std)

    max_signal = float(np.max(np.abs(sensors_clean['dP_sensors'])))
    snr = max_signal / noise_std

    # La misma figura muestra curvas limpias y con ruido superpuestas.
    fig, axes = plt.subplots(1, 1, figsize=(12, 4))
    for idx, xpos in enumerate([0.0, L / 2.0, L]):
        axes.plot(result_clean['t'], sensors_clean['dP_sensors'][idx], label=f'limpio x={xpos:.0f} m')
        axes.plot(result_noisy['t'], sensors_noisy['dP_sensors'][idx], alpha=0.5, linestyle='--', label=f'ruido x={xpos:.0f} m')
    axes.set_title(f'Test 4 - SNR = {snr:.2f}')
    axes.set_xlabel('t [s]')
    axes.set_ylabel('dP [Pa]')
    axes.legend(ncol=2, fontsize=8)
    axes.grid(True, alpha=0.3)
    fig.tight_layout()
    return {
        'fig': fig,
        'snr': snr,
        'max_signal': max_signal,
        'noise_std': noise_std,
        'snr_ok': snr > 1.0,
    }


def run_all_tests():
    # Punto de entrada unico para correr toda la bateria de validaciones.
    x, P_ss, Q_ss = steady_state_profile()
    steady_checks = validate_steady_state(P_ss, Q_ss, x)
    steady_fig = plot_steady_state(x, P_ss)

    # Cada test se ejecuta por separado para facilitar diagnostico y trazabilidad.
    t1 = test_1_stationary_stability()
    t2 = test_2_wave_speed()
    t3 = test_3_mass_balance()
    t4 = test_4_snr()

    # Resumen numerico en consola para comprobar resultados sin abrir figuras.
    print('=== Validación estado estacionario ===')
    for key, value in steady_checks.items():
        print(f'{key}: {value}')
    print('=== Test 1 ===')
    print(f"max(|dP|) = {t1['max_abs_dP']:.6e} Pa | stable = {t1['stable']}")
    print('=== Test 2 ===')
    print(f"upstream predicho = {t2['predicted_up']:.3f} s | observado = {t2['observed_up']:.3f} s")
    print(f"downstream predicho = {t2['predicted_down']:.3f} s | observado = {t2['observed_down']:.3f} s")
    print('=== Test 3 ===')
    print(f"steady mass ok = {t3['steady_mass_ok']} | steady_err = {t3['steady_err']:.6e}")
    print(f"leak mass ok = {t3['leak_mass_ok']} | final Qin-Qout = {t3['final_mass_diff']:.6e}")
    print('=== Test 4 ===')
    print(f"SNR = {t4['snr']:.3f} | ok = {t4['snr_ok']}")

    # En modo interactivo esto abre las figuras; en modo headless no molesta.
    plt.show()
    return {
        'steady_checks': steady_checks,
        'steady_fig': steady_fig,
        'test_1': t1,
        'test_2': t2,
        'test_3': t3,
        'test_4': t4,
    }


if __name__ == '__main__':
    # Si se ejecuta como script, corrermos toda la bateria de tests y graficos.
    run_all_tests()
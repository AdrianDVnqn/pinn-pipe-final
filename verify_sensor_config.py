import numpy as np
from data_utils import get_training_data

def run_tests():
    print("Iniciando pruebas de configuración de sensores...\n")

    # TEST 1 — Shapes correctas
    data = get_training_data(8, "trivial", n_pressure_sensors=3)
    assert data["P_noisy"].shape[0] == 3,    "Error: P_noisy debe tener 3 sensores"
    assert data["Q_noisy"].shape[0] == 2,    "Error: Q_noisy debe tener 2 caudalímetros"
    assert data["x_flow_meters"][0] == 0,    "Error: primer caudalímetro en x=0"
    assert data["x_flow_meters"][1] == 10000,"Error: segundo caudalímetro en x=L"
    print("✓ TEST 1 — Shapes correctas")

    # TEST 2 — Q en extremos tiene señal de fuga
    t = data["t"]
    dQ_in  = data["dQ_noisy"][0, :]   # entrada
    dQ_out = data["dQ_noisy"][1, :]   # salida
    delta_Q = dQ_in - dQ_out
    max_signal = np.max(np.abs(delta_Q[t > 60]))
    assert max_signal > 0.001, "Error: no se detecta señal de fuga en delta_Q"
    print(f"✓ TEST 2 — Señal de fuga en delta_Q: max={max_signal:.4f} m³/s")

    # TEST 3 — P en sensores intermedios tiene señal
    dP_max = np.max(np.abs(data["dP_noisy"][:, data["t"] > 60]))
    assert dP_max > 1000, "Error: señal de presión demasiado baja"
    print(f"✓ TEST 3 — Señal de presión: max dP = {dP_max:.0f} Pa")

    # TEST 4 — Subconjunto de 2 sensores de presión
    data_2 = get_training_data(8, "trivial", n_pressure_sensors=2)
    data_3 = get_training_data(8, "trivial", n_pressure_sensors=3)
    assert data_2["P_noisy"].shape[0] == 2
    assert data_3["P_noisy"].shape[0] == 3
    assert data_2["Q_noisy"].shape[0] == 2   # caudalímetros no cambian
    assert data_3["Q_noisy"].shape[0] == 2   # caudalímetros no cambian
    print("✓ TEST 4 — Subconjuntos de sensores de presión correctos")

    print("\n4/4 tests pasaron — configuración de sensores correcta")

if __name__ == "__main__":
    run_tests()

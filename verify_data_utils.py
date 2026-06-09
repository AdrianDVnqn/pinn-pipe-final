import numpy as np

import config as cfg
from data_utils import get_training_data, list_scenarios, load_scenario


def main():
    passed = 0

    print('TEST 1 — Carga básica')
    scenario = load_scenario(5)
    print(
        f"id={scenario['id']}, has_leak={scenario['has_leak']}, x_leak={scenario['x_leak']}, "
        f"q_leak={scenario['q_leak']}, P_sensors={scenario['P_sensors'].shape}, "
        f"Q_sensors={scenario['Q_sensors'].shape}, dP_sensors={scenario['dP_sensors'].shape}, "
        f"P_full={scenario['P_full'].shape}"
    )
    passed += 1

    print('\nTEST 2 — Reproducibilidad del ruido')
    data_a = get_training_data(5, 'moderado')
    data_b = get_training_data(5, 'moderado')
    if np.array_equal(data_a['P_noisy'], data_b['P_noisy']):
        print('✓ Ruido reproducible')
        passed += 1
    else:
        print('✗ FALLO: ruido no reproducible')

    print('\nTEST 3 — Distintos niveles de ruido son distintos')
    data_facil = get_training_data(5, 'facil')
    data_dificil = get_training_data(5, 'dificil')
    std_facil = float(np.std(data_facil['P_noisy']))
    std_dificil = float(np.std(data_dificil['P_noisy']))
    print(f'std facil   = {std_facil:.6f}')
    print(f'std dificil = {std_dificil:.6f}')
    if std_dificil > std_facil:
        print('✓ Ruido más alto produce mayor std')
        passed += 1
    else:
        print('✗ FALLO: std no aumenta con el ruido')

    print('\nTEST 4 — Subconjunto de sensores')
    data_2s = get_training_data(5, 'moderado', n_sensors=2)
    data_3s = get_training_data(5, 'moderado', n_sensors=3)
    ok_shapes = data_2s['P_noisy'].shape[0] == 2 and data_3s['P_noisy'].shape[0] == 3
    ok_subsets = np.array_equal(data_2s['x_sensors_used'], np.asarray(cfg.SENSOR_SUBSETS[2])) and np.array_equal(
        data_3s['x_sensors_used'], np.asarray(cfg.SENSOR_SUBSETS[3])
    )
    print(f"shape 2 sensores = {data_2s['P_noisy'].shape}")
    print(f"shape 3 sensores = {data_3s['P_noisy'].shape}")
    print(f"x_sensors_used 2 = {data_2s['x_sensors_used'].tolist()}")
    print(f"x_sensors_used 3 = {data_3s['x_sensors_used'].tolist()}")
    if ok_shapes and ok_subsets:
        print('✓ Subconjunto de sensores correcto')
        passed += 1
    else:
        print('✗ FALLO: subconjunto de sensores incorrecto')

    print('\nTEST 5 — list_scenarios()')
    df = list_scenarios()
    print(df.to_string(index=False))
    ok_df = len(df) == 13 and not bool(df.loc[df['scenario_id'] == 0, 'has_leak'].iloc[0])
    if ok_df:
        print('✓ DataFrame correcto')
        passed += 1
    else:
        print('✗ FALLO: DataFrame incorrecto')

    print(f'\n{passed}/5 tests pasaron')


if __name__ == '__main__':
    main()

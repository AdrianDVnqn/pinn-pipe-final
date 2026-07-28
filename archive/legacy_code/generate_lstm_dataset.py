import os
import sys
import json
import h5py
import numpy as np
from tqdm import tqdm

import config as cfg
from simulator import run_moc, get_sensor_data
from config import NOISE_LEVELS

def generate_lstm_dataset():
    x_leak_train_vals = np.arange(500, 9501, 500)
    q_leak_train_vals = [0.003, 0.005, 0.008, 0.012, 0.015, 0.020, 0.025, 0.030]
    
    test_x_vals = [2000, 4000, 6000, 8000]
    test_q_vals = [0.005, 0.015, 0.030]
    test_combinations = [(x, q) for x in test_x_vals for q in test_q_vals]
    
    train_combinations = []
    for x in x_leak_train_vals:
        for q in q_leak_train_vals:
            if (x, q) not in test_combinations:
                train_combinations.append((x, q))
                
    for tx, tq in test_combinations:
        assert (tx, tq) not in train_combinations
        
    n_no_leak_train = 20
    t_leak = cfg.T_LEAK_START
    noise_levels_train = [500.0, 2000.0, 8000.0, 25000.0]
    n_augmentations = 5
    
    train_scenarios = []
    for (x, q) in train_combinations:
        for noise in noise_levels_train:
            for aug in range(n_augmentations):
                train_scenarios.append({
                    'x_leak': float(x),
                    'q_leak': float(q),
                    'has_leak': 1,
                    'noise_std': float(noise),
                    'seed': aug
                })
    for _ in range(n_no_leak_train):
        for noise in noise_levels_train:
            for aug in range(n_augmentations):
                train_scenarios.append({
                    'x_leak': float(cfg.X_LEAK_VALUES[2]), # Arbitrario
                    'q_leak': 0.0,
                    'has_leak': 0,
                    'noise_std': float(noise),
                    'seed': aug
                })
                
    test_scenarios = []
    base_scens = [(0, 0.0, 0.0, 0)]
    scen_id = 1
    for q in test_q_vals:
        for x in test_x_vals:
            base_scens.append((scen_id, float(x), float(q), 1))
            scen_id += 1
            
    for (sid, x, q, has_leak) in base_scens:
        for noise_name, noise_std in NOISE_LEVELS.items():
            test_scenarios.append({
                'scenario_id': sid,
                'x_leak': x,
                'q_leak': q,
                'has_leak': has_leak,
                'noise_std': float(noise_std),
                'noise_name': noise_name,
                'seed': 42 # Fijo para test
            })

    def process_scenario(s, sensor_positions_p, sensor_positions_q):
        moc = run_moc(Q_leak=s['q_leak'], x_leak=s['x_leak'], t_leak=t_leak, noise_std=0.0)
        rng = np.random.default_rng(s['seed'])
        
        sens_p = get_sensor_data(moc, sensor_positions_p, noise_std=0.0)
        P_noisy = sens_p['P_sensors'] + rng.normal(0.0, s['noise_std'], size=sens_p['P_sensors'].shape)
        
        sens_q = get_sensor_data(moc, sensor_positions_q, noise_std=0.0)
        q_noise_std = s['noise_std'] / moc['B']
        Q_noisy = sens_q['Q_sensors'] + rng.normal(0.0, q_noise_std, size=sens_q['Q_sensors'].shape)
        
        t = moc['t']
        mask_base = t < 40.0
        P_base = np.mean(P_noisy[:, mask_base], axis=1, keepdims=True)
        Q_base = np.mean(Q_noisy[:, mask_base], axis=1, keepdims=True)
        
        dP = P_noisy - P_base
        dQ = Q_noisy - Q_base
        
        dP_norm = dP / cfg.P_INLET
        dQ_norm = dQ / cfg.Q_OUTLET
        
        X = np.concatenate([dP_norm, dQ_norm], axis=0) # [5, Nt]
        X = X.T # [Nt, 5]
        
        x_norm = s['x_leak'] / cfg.PIPE_LENGTH
        q_norm = s['q_leak'] / cfg.Q_OUTLET
        y = np.array([x_norm, q_norm, s['has_leak']], dtype=np.float32)
        
        return X.astype(np.float32), y
        
    sensor_p = cfg.PRESSURE_SENSOR_POSITIONS
    sensor_q = cfg.FLOW_METER_POSITIONS
    
    with h5py.File("lstm_dataset.h5", "w") as f:
        grp_train = f.create_group("train")
        X_train = grp_train.create_dataset("X", (len(train_scenarios), 2401, 5), dtype=np.float32)
        y_train = grp_train.create_dataset("y", (len(train_scenarios), 3), dtype=np.float32)
        meta_train = []
        
        for i, s in enumerate(tqdm(train_scenarios, desc="Train")):
            X, y = process_scenario(s, sensor_p, sensor_q)
            X_train[i] = X
            y_train[i] = y
            meta_train.append(s)
            
        grp_train.create_dataset("metadata", data=json.dumps(meta_train))
        
        grp_test = f.create_group("test")
        X_test = grp_test.create_dataset("X", (len(test_scenarios), 2401, 5), dtype=np.float32)
        y_test = grp_test.create_dataset("y", (len(test_scenarios), 3), dtype=np.float32)
        meta_test = []
        
        for i, s in enumerate(tqdm(test_scenarios, desc="Test")):
            X, y = process_scenario(s, sensor_p, sensor_q)
            X_test[i] = X
            y_test[i] = y
            meta_test.append(s)
            
        grp_test.create_dataset("metadata", data=json.dumps(meta_test))

    print("\nDataset LSTM generado:")
    n_train_fuga = sum(1 for s in train_scenarios if s['has_leak'])
    n_train_nofuga = len(train_scenarios) - n_train_fuga
    print(f" - Train: {len(train_scenarios)} samples ({n_train_fuga} con fuga, {n_train_nofuga} sin fuga)")
    print(f" - Test:  {len(test_scenarios)} samples ({len(base_scens)} por nivel de ruido)")
    print(f" - Shape input: [Nt=2401, channels=5]")
    print(f" - Guardado en: lstm_dataset.h5")

if __name__ == "__main__":
    generate_lstm_dataset()

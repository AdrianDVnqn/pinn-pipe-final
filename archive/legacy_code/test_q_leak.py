import torch
from data_utils import get_training_data

data = get_training_data(12, 'trivial', 3) # large leak
Q_in = data['Q_noisy'][0, :]
Q_out = data['Q_noisy'][1, :]
q_leak_est = (Q_in[-50:] - Q_out[-50:]).mean()
print(f"True leak: {data['q_leak']}, estimated: {q_leak_est}")

data2 = get_training_data(10, 'dificil', 3) # small leak
Q_in2 = data2['Q_noisy'][0, :]
Q_out2 = data2['Q_noisy'][1, :]
q_leak_est2 = (Q_in2[-50:] - Q_out2[-50:]).mean()
print(f"True leak: {data2['q_leak']}, estimated: {q_leak_est2}")

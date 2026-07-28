from baseline_pressure_gradient import run_pressure_gradient

res = run_pressure_gradient(scenario_id=8, noise_level="trivial", n_pressure_sensors=3)
print("--- SCENARIO 8, TRIVIAL ---")
print("Gradients:", res["gradients"])
print("Mean gradient:", sum(res["gradients"])/len(res["gradients"]))
print("Anomaly score:", res["anomaly_score"])
print("Leak detected:", res["leak_detected"])
print("Q_in mean:", res["q_leak_pred"] + res["q_leak_true"]) # rough
print("q_leak_pred:", res["q_leak_pred"])
print("G_upstream_obs:", res["G_upstream_obs"])
print("G_downstream_obs:", res["G_downstream_obs"])
print("G_expected:", res["G_expected"])

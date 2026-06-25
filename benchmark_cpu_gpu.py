import time
import pandas as pd
from perfect_pinn_model import train_pinn

def run_benchmark():
    scenario_id = 6
    epochs = 500
    
    print("="*50)
    print(" BENCHMARK: CPU vs GPU ")
    print("="*50)
    print(f"Escenario: {scenario_id}")
    print(f"Epochs: {epochs}")
    print("-" * 50)
    
    results = []
    
    for device in ["cpu", "cuda"]:
        print(f"\n🚀 Corriendo en dispositivo: {device.upper()}")
        try:
            start_time = time.time()
            res = train_pinn(
                scenario_id=scenario_id,
                n_epochs=epochs,
                n_collocation=20000,
                verbose=False,
                use_lbfgs=False,
                force_device=device
            )
            end_time = time.time()
            duration = end_time - start_time
            ms_per_epoch = res["ms_per_epoch"]
            
            print(f"✅ Terminado en {duration:.2f} segundos ({ms_per_epoch:.2f} ms/epoch)")
            
            results.append({
                "Device": device.upper(),
                "Time (s)": duration,
                "ms / epoch": ms_per_epoch
            })
        except Exception as e:
            print(f"❌ Error corriendo en {device}: {e}")
            
    print("\n" + "="*50)
    print(" RESULTADOS FINALES ")
    print("="*50)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    if len(results) == 2:
        speedup = results[0]["Time (s)"] / results[1]["Time (s)"]
        print(f"\n⚡ Speedup GPU vs CPU: {speedup:.1f}x más rápido")

if __name__ == "__main__":
    run_benchmark()

import sys
from perfect_pinn_model import train_pinn

if __name__ == '__main__':
    result = train_pinn(
        scenario_id=8,
        noise_level="trivial",
        n_pressure_sensors=3,
        n_epochs=10_000,
        activation="leaky_relu",
        verbose=True,
    )
    print("FINISHED")

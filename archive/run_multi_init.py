import os
import sys
import torch
import logging
from perfect_pinn_model import train_pinn, plot_training_diagnostics

def setup_logger():
    logger = logging.getLogger('multi_init')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(ch)
    return logger

log = setup_logger()

def run_multi_init():
    scenario_id = 8
    noise_level = "trivial"
    n_pressure_sensors = 3
    
    inits = [2000.0, 4000.0, 6000.0, 8000.0]
    best_loss = float('inf')
    best_init = None
    
    log.info("Starting Multi-Initialization Search...")
    results = {}
    
    for init_x in inits:
        log.info(f"--- Testing initialization at {init_x}m ---")
        try:
            # Short training run
            res = train_pinn(
                scenario_id=scenario_id,
                noise_level=noise_level,
                n_pressure_sensors=n_pressure_sensors,
                n_epochs=1500,  # Just enough to escape initial bad regions
                initial_x_leak=init_x,
                use_lbfgs=False,
                verbose=False
            )
            
            final_loss = res['history'].iloc[-1]['loss_total']
            pred_x = res['x_leak_pred']
            log.info(f"Init {init_x}m -> Final x_leak: {pred_x:.0f}m | Loss: {final_loss:.3e}")
            
            results[init_x] = final_loss
            if final_loss < best_loss:
                best_loss = final_loss
                best_init = init_x
                
        except Exception as e:
            log.error(f"Failed at init {init_x}m: {e}")
            
    log.info(f"Best initialization found: {best_init}m with loss {best_loss:.3e}")
    
    log.info("--- Proceeding with full training on best initialization ---")
    
    # Run full training on the best initialization
    final_res = train_pinn(
        scenario_id=scenario_id,
        noise_level=noise_level,
        n_pressure_sensors=n_pressure_sensors,
        n_epochs=10000,
        initial_x_leak=best_init,
        use_lbfgs=True,
        lbfgs_epochs=2000,
        verbose=True
    )
    
    plot_training_diagnostics(final_res, save_dir='figs_multi_init')
    
    ckpt_name = f'pinn_s{scenario_id}_nP{n_pressure_sensors}_{noise_level}_multi_init.pt'
    torch.save(final_res['model'].state_dict(), os.path.join('checkpoints', ckpt_name))
    log.info(f'Training complete. Model saved to checkpoints/{ckpt_name}')
    
if __name__ == '__main__':
    run_multi_init()

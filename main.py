from Sim.backends import run_simulation, BaseSimulationBackend
from Sim.config.sys_config import HPSim_Config
import tyro

if __name__ == "__main__":
    # Load configuration
    config: HPSim_Config = tyro.cli(HPSim_Config)

    backend = BaseSimulationBackend.create_from_name(config.simulator_backend)
    print(f"Using {backend.get_display_name()} ({config.simulator_backend})...")
    run_simulation(config)

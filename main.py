from Sim.simulator import Simulator

from Sim.config.sys_config import HPSim_Config
import Sim.config.utils as utils
import logging
import sys
import tyro

if __name__ == "__main__":

    print("Starting the simulation...")
    
    # Load configuration
    config = tyro.cli(HPSim_Config)
    
    # Initialize simulator
    simulator = Simulator(config)
    simulator.run()

    # Run the simulation
    # simulator.run()
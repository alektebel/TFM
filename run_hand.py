#!/usr/bin/env python3

import os
import sys

def main():
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the src directory to Python path
    sys.path.append(os.path.join(script_dir, 'src'))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    # Import and run the simulation
    from simulation.prosthetic_sim import main as run_simulation
    run_simulation()

if __name__ == "__main__":
    main() 
import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import the policy module
from src.control.policy import HandPolicy

def main():
    print("Starting simulation...")
    
    # Load model
    model_path = os.path.join(project_root, "models", "scene_right.xml")
    print(f"Loading model from: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Create hand policy
    
    # Launch the viewer
    print("Launching viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initial position
        policy = HandPolicy(model, data, viewer)
        policy.data.qpos[0:3] = [0, 0, 0.3]  # x, y, z position
        
        # Simulation loop
        print("Starting simulation loop...")
        start_time = time.time()
        
        while viewer.is_running():
            # Calculate current time
            t = time.time() - start_time
            print(f"Time: {t}")
            
            # Move in a circular pattern
            radius = 0.3
            x = radius * np.cos(t)
            y = radius * np.sin(t)
            z = 0.3  # Keep height constant
            
            # Update position
            policy.data.qpos[0:3] = [x, y, z]
            
            # Step simulation and update viewer
            mujoco.mj_step(policy.model, policy.data)
            viewer.sync()
            
            # Control simulation speed
            time.sleep(0.01)
            
            # Example: Wave fingers every 10 seconds
            if int(t) % 10 == 0 and int(t) > 0:
                print("Waving fingers...")
                
                policy.wave_fingers(duration=4.0, frequency=1.0)
    
    print("Simulation ended")

if __name__ == "__main__":
    main()


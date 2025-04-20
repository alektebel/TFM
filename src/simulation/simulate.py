import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def main():
    print("Starting simulation...")
    
    # Load model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             "models", "scene_right.xml")
    print(f"Loading model from: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Launch the viewer
    print("Launching viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initial position
        data.qpos[0:2] = [0, 0]  # x, y position (z is fixed)
        
        # Simulation loop
        print("Starting simulation loop...")
        while viewer.is_running():
            # Move in a square pattern
            t = time.time()
            x = 0.3 * np.cos(t)
            y = 0.3 * np.sin(t)
            
            # Update position
            data.qpos[0] = x
            data.qpos[1] = y
            
            # Step simulation and update viewer
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Control simulation speed
            time.sleep(0.01)
    
    print("Simulation ended")

if __name__ == "__main__":
    main()

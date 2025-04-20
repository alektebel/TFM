import mujoco
import mujoco.viewer
import numpy as np
import time
from policy import HandPolicy

# Load the scene model (which includes the hand)
model = mujoco.MjModel.from_xml_path("models/scene.xml")
data = mujoco.MjData(model)

# Create the hand policy
policy = HandPolicy(model, data)

# Simulation parameters
duration = 10.0  # seconds
dt = 0.01  # time step
radius = 0.2  # radius of circular motion
angular_velocity = 2 * np.pi / duration  # one complete circle per duration

# Initial position
initial_pos = np.array([0.0, 0.0, 0.3])

# Create viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Reset simulation
    mujoco.mj_resetData(model, data)
    
    # Set initial position
    data.qpos[0:3] = initial_pos
    
    # Main simulation loop
    while viewer.is_running():
        # Calculate current time
        t = data.time
        
        # Calculate desired position in circular motion
        x = radius * np.cos(angular_velocity * t)
        y = radius * np.sin(angular_velocity * t)
        z = initial_pos[2]  # Keep height constant
        
        # Set position (maintaining orientation)
        data.qpos[0:3] = [x, y, z]
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Update viewer
        viewer.sync()
        
        # Small sleep to maintain real-time simulation
        time.sleep(max(0, dt - (time.time() - t)))
        
        # Example: Wave fingers every 10 seconds
        if int(t) % 10 == 0 and int(t) > 0:
            policy.wave_fingers(duration=5.0, frequency=1.0) 
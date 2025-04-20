import mujoco
import mujoco.viewer
import numpy as np
import time
import os

def main():
    print("Starting prosthetic hand simulation...")
    
    # Load model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             "models", "prosthetic_hand.xml")
    print(f"Loading model from: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Set initial position for the floating hand
    data.qpos[0:3] = [0, 0, 0.3]  # x, y, z position
    data.qpos[3:7] = [1, 0, 0, 0]  # quaternion orientation
    
    # Launch the viewer
    print("Launching viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Simulation loop
        print("Starting simulation loop...")
        print("\nControls:")
        print("- Space: Toggle auto-animation")
        print("- O: Open hand")
        print("- C: Close hand")
        print("- Q: Quit\n")
        
        # Initial state - hand open
        open_angles = {
            'wrist_flexion': 0,
            'thumb': 0,
            'index': 0,
            'middle': 0,
            'ring': 0,
            'pinky': 0
        }
        
        # Closed state
        closed_angles = {
            'wrist_flexion': 0,
            'thumb': 90,
            'index': 90,
            'middle': 90,
            'ring': 90,
            'pinky': 90
        }
        
        # Current state
        current_state = 'open'
        
        # Animation parameters
        animation_speed = 3.0  # degrees per step
        current_angles = open_angles.copy()
        
        # Auto-animate between states
        auto_animate = True
        animation_timer = 0
        last_time = time.time()
        
        # Position control parameters
        target_pos = np.array([0, 0, 0.3])  # Target position
        pos_kp = 100.0  # Position control gain
        pos_kd = 10.0   # Velocity damping gain
        
        while viewer.is_running():
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Auto-animate between open and closed states
            if auto_animate:
                animation_timer += dt
                if animation_timer > 2.0:  # Change state every 2 seconds
                    animation_timer = 0
                    if current_state == 'open':
                        current_state = 'closed'
                        print("Closing hand...")
                    else:
                        current_state = 'open'
                        print("Opening hand...")
            
            # Smoothly animate to target angles
            target_angles = open_angles if current_state == 'open' else closed_angles
            
            for joint_name in current_angles:
                current = current_angles[joint_name]
                target = target_angles[joint_name]
                
                if abs(current - target) > animation_speed:
                    if current < target:
                        current_angles[joint_name] += animation_speed
                    else:
                        current_angles[joint_name] -= animation_speed
                else:
                    current_angles[joint_name] = target
            
            # Apply finger controls
            for i, joint_name in enumerate(['wrist_flexion', 'thumb', 'index', 'middle', 'ring', 'pinky']):
                data.ctrl[i] = current_angles[joint_name]
            
            # Position control to keep the hand floating
            current_pos = data.qpos[0:3]
            current_vel = data.qvel[0:3]
            
            # Calculate position error and apply control force
            pos_error = target_pos - current_pos
            force = pos_kp * pos_error - pos_kd * current_vel
            
            # Apply the force to the free joint
            data.qfrc_applied[0:3] = force
            
            # Add some gentle motion to the target position
            t = current_time
            target_pos[0] = 0.1 * np.sin(0.5 * t)  # gentle x motion
            target_pos[1] = 0.1 * np.cos(0.3 * t)  # gentle y motion
            target_pos[2] = 0.3 + 0.05 * np.sin(0.2 * t)  # gentle z motion
            
            # Step simulation and update viewer
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Control simulation speed
            time.sleep(0.1)
    
    print("Simulation ended")

if __name__ == "__main__":
    main() 
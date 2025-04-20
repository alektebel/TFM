import numpy as np
from math import cos, sin

def calculate_ik(model, data, target_pos):
    """Simple geometric IK for 3-DOF arm"""
    # Input validation
    if not isinstance(target_pos, (list, np.ndarray)) or len(target_pos) != 3:
        raise ValueError("target_pos must be a 3D position vector")
    
    # Arm lengths (should match model)
    L1 = 0.3  # Upper arm
    L2 = 0.3  # Forearm
    L3 = 0.22  # Hand
    
    # Target position relative to shoulder
    x, y, z = target_pos
    
    # Check if target is reachable
    max_reach = L1 + L2
    if np.sqrt(x**2 + y**2 + z**2) > max_reach:
        # If unreachable, scale to maximum reach
        scale = max_reach / np.sqrt(x**2 + y**2 + z**2)
        x, y, z = x * scale, y * scale, z * scale
    
    # Shoulder flexion/extension (joint 0)
    shoulder_angle = np.arctan2(z, x)
    
    # Elbow flexion/extension (joint 1)
    # Using law of cosines
    D = (x**2 + z**2 - L1**2 - L2**2) / (2 * L1 * L2)
    D = np.clip(D, -1, 1)  # Avoid numerical errors
    elbow_angle = np.arccos(D)
    
    # Wrist rotation (joint 2) - simple orientation control
    wrist_angle = np.arctan2(y, x) * 0.5
    
    # Clip angles to joint limits
    shoulder_angle = np.clip(shoulder_angle, -np.pi, np.pi)
    elbow_angle = np.clip(elbow_angle, 0, np.pi)
    wrist_angle = np.clip(wrist_angle, -np.pi/2, np.pi/2)
    
    return np.array([shoulder_angle, elbow_angle, wrist_angle])

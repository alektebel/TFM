import numpy as np
import mujoco
import mujoco.viewer
import time

class HandPolicy:
    def __init__(self, model, data, viewer):
        self.model = model
        self.data = data
        self.viewer = viewer
        # Store joint indices for each finger
        self.finger_joints = {
            'first': ['ffj0', 'ffj1', 'ffj2', 'ffj3'],  # First finger joints
            'middle': ['mfj0', 'mfj1', 'mfj2', 'mfj3'],  # Middle finger joints
            'ring': ['rfj0', 'rfj1', 'rfj2', 'rfj3'],    # Ring finger joints
            'thumb': ['thj0', 'thj1', 'thj2', 'thj3']    # Thumb joints
        }
        
        # Store actuator indices for each finger
        self.finger_actuators = {
            'first': ['ffa0', 'ffa1', 'ffa2', 'ffa3'],  # First finger actuators
            'middle': ['mfa0', 'mfa1', 'mfa2', 'mfa3'],  # Middle finger actuators
            'ring': ['rfa0', 'rfa1', 'rfa2', 'rfa3'],    # Ring finger actuators
            'thumb': ['tha0', 'tha1', 'tha2', 'tha3']    # Thumb actuators
        }
        
        # Default positions for each finger (closed position)
        self.default_positions = {
            'first': [0.0, 0.0, 0.0, 0.0],
            'middle': [0.0, 0.0, 0.0, 0.0],
            'ring': [0.0, 0.0, 0.0, 0.0],
            'thumb': [0.5, 0.0, 0.0, 0.0]
        }
        
        # Open positions for each finger
        self.open_positions = {
            'first': [0.0, 1.0, 1.0, 1.0],
            'middle': [0.0, 1.0, 1.0, 1.0],
            'ring': [0.0, 1.0, 1.0, 1.0],
            'thumb': [0.5, 1.0, 1.0, 1.0]
        }
        
        # Wave animation state
        self.wave_start_time = None
        self.wave_duration = 0
        self.wave_frequency = 0
        self.is_waving = False
    
    def move_finger(self, finger_name, positions):
        """Move a specific finger to given positions for each joint."""
        if finger_name not in self.finger_actuators:
            raise ValueError(f"Unknown finger: {finger_name}")
            
        actuator_names = self.finger_actuators[finger_name]
        for i, actuator_name in enumerate(actuator_names):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            self.data.ctrl[actuator_id] = positions[i]
    
    def move_first_finger(self, positions=None):
        """Move the first finger to specified positions or open position."""
        if positions is None:
            positions = self.open_positions['first']
        self.move_finger('first', positions)
    
    def move_middle_finger(self, positions=None):
        """Move the middle finger to specified positions or open position."""
        if positions is None:
            positions = self.open_positions['middle']
        self.move_finger('middle', positions)
    
    def move_ring_finger(self, positions=None):
        """Move the ring finger to specified positions or open position."""
        if positions is None:
            positions = self.open_positions['ring']
        self.move_finger('ring', positions)
    
    def move_thumb(self, positions=None):
        """Move the thumb to specified positions or open position."""
        if positions is None:
            positions = self.open_positions['thumb']
        self.move_finger('thumb', positions)
    
    def close_all_fingers(self):
        """Close all fingers to default positions."""
        for finger in self.finger_actuators.keys():
            self.move_finger(finger, self.default_positions[finger])
    
    def open_all_fingers(self):
        """Open all fingers."""
        for finger in self.finger_actuators.keys():
            self.move_finger(finger, self.open_positions[finger])
    
    def wave_fingers(self, duration=5.0, frequency=1.0):
        """Create a wave motion with the fingers."""
        start_time = time.time()
        while time.time() - start_time < duration:
            t = time.time() - start_time
            # Wave pattern: first -> middle -> ring -> thumb
            phase = 2 * np.pi * frequency * t
            
            # First finger
            self.move_first_finger([0.0, 0.5 + 0.5 * np.sin(phase), 0.5 + 0.5 * np.sin(phase), 0.5 + 0.5 * np.sin(phase)])
            
            # Middle finger
            self.move_middle_finger([0.0, 0.5 + 0.5 * np.sin(phase + np.pi/2), 0.5 + 0.5 * np.sin(phase + np.pi/2), 0.5 + 0.5 * np.sin(phase + np.pi/2)])
            
            # Ring finger
            self.move_ring_finger([0.0, 0.5 + 0.5 * np.sin(phase + np.pi), 0.5 + 0.5 * np.sin(phase + np.pi), 0.5 + 0.5 * np.sin(phase + np.pi)])
            
            # Thumb
            self.move_thumb([0.5, 0.5 + 0.5 * np.sin(phase + 3*np.pi/2), 0.5 + 0.5 * np.sin(phase + 3*np.pi/2), 0.5 + 0.5 * np.sin(phase + 3*np.pi/2)])
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Update viewer

            self.viewer.sync()
            
            # Small sleep to maintain real-time simulation
            time.sleep(0.01)
        
        # Return to default position
        self.close_all_fingers()
    
    def update_wave(self):
        """Update the wave motion if active. Call this in your main simulation loop."""
        if not self.is_waving:
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.wave_start_time
        
        if elapsed_time >= self.wave_duration:
            self.is_waving = False
            self.close_all_fingers()
            print("Wave motion completed")
            return
            
        # Wave pattern: first -> middle -> ring -> thumb
        phase = 2 * np.pi * self.wave_frequency * elapsed_time
        
        # First finger
        self.move_first_finger([0.0, 0.5 + 0.5 * np.sin(phase), 0.5 + 0.5 * np.sin(phase), 0.5 + 0.5 * np.sin(phase)])
        
        # Middle finger
        self.move_middle_finger([0.0, 0.5 + 0.5 * np.sin(phase + np.pi/2), 0.5 + 0.5 * np.sin(phase + np.pi/2), 0.5 + 0.5 * np.sin(phase + np.pi/2)])
        
        # Ring finger
        self.move_ring_finger([0.0, 0.5 + 0.5 * np.sin(phase + np.pi), 0.5 + 0.5 * np.sin(phase + np.pi), 0.5 + 0.5 * np.sin(phase + np.pi)])
        
        # Thumb
        self.move_thumb([0.5, 0.5 + 0.5 * np.sin(phase + 3*np.pi/2), 0.5 + 0.5 * np.sin(phase + 3*np.pi/2), 0.5 + 0.5 * np.sin(phase + 3*np.pi/2)]) 
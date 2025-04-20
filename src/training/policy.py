# Move individual fingers
policy.move_first_finger()  # Opens first finger
policy.move_middle_finger([0.0, 0.5, 0.5, 0.5])  # Custom position

# Group actions
policy.open_all_fingers()
policy.close_all_fingers()

# Wave motion
policy.wave_fingers(duration=5.0, frequency=1.0)

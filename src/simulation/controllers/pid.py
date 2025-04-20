import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=10):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.prev_error = 0
        self.integral = 0
    
    def update(self, target, current, dt=0.01):
        error = target - current
        
        # Proportional term
        p = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i = self.ki * self.integral
        
        # Derivative term
        d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error
        
        return p + i + d

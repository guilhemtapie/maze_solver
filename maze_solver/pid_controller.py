"""
PID Controller Module

This module provides a PID controller implementation for controlling motor movements.
"""

import time
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PIDController:
    """
    PID Controller class for controlling a single axis.
    
    Attributes:
        kp (float): Proportional gain
        ki (float): Integral gain
        kd (float): Derivative gain
        integral (float): Accumulated integral error
        previous_error (float): Previous error value
        last_time (float): Last time the PID was calculated
        min_output (float): Minimum output value
        max_output (float): Maximum output value
        integral_limit (float): Limit for integral term to prevent windup
    """
    
    def __init__(self, kp=0.5, ki=0.1, kd=0.2, min_output=-100, max_output=100, integral_limit=100):
        """Initialize a PID controller."""
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0
        self.last_time = time.time()
        self.min_output = min_output
        self.max_output = max_output
        self.integral_limit = integral_limit
        
        logger.info(f"PID Controller initialized with kp={kp}, ki={ki}, kd={kd}")
    
    def calculate(self, error):
        """Calculate PID output based on the current error."""
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if dt == 0:
            dt = 0.001
        
        # Calculate terms
        p_term = self.kp * error
        
        self.integral += error * dt
        if self.integral > self.integral_limit:
            self.integral = self.integral_limit
        elif self.integral < -self.integral_limit:
            self.integral = -self.integral_limit
        i_term = self.ki * self.integral
        
        derivative = (error - self.previous_error) / dt
        d_term = self.kd * derivative
        
        self.previous_error = error
        
        # Calculate and limit output
        output = p_term + i_term + d_term
        if output > self.max_output:
            output = self.max_output
        elif output < self.min_output:
            output = self.min_output
        
        logger.debug(f"PID calculation: error={error:.2f}, p={p_term:.2f}, i={i_term:.2f}, d={d_term:.2f}, output={output:.2f}")
        return output
    
    def reset(self):
        """Reset the PID controller state."""
        self.integral = 0
        self.previous_error = 0
        self.last_time = time.time()
        logger.info("PID Controller reset")

# Default PID configuration
DEFAULT_PID_CONFIG = {
    'x_axis': {
        'kp': 0.3,
        'ki': 0.05,
        'kd': 0.1,
        'min_output': -100,
        'max_output': 100,
        'integral_limit': 50
    },
    'y_axis': {
        'kp': 0.3,
        'ki': 0.05,
        'kd': 0.1,
        'min_output': -100,
        'max_output': 100,
        'integral_limit': 50
    },
    'base_position': {
        'x': 490,
        'y': 485
    },
    'position_limits': {
        'min': 300,
        'max': 700
    }
}

# Create PID controllers
pid_x = PIDController(**DEFAULT_PID_CONFIG['x_axis'])
pid_y = PIDController(**DEFAULT_PID_CONFIG['y_axis'])

def adjust_motor_movement(x, y, goal):
    """
    Adjust motor movement based on current position and goal.
    
    Args:
        x (float): Current x position
        y (float): Current y position
        goal (tuple): Goal position (y, x)
        
    Returns:
        tuple: Motor positions (motor1_position, motor2_position)
    """
    try:
        # Calculate errors - goal is in (y, x) format
        error_x = goal[1] - x  # x error
        error_y = goal[0] - y  # y error
        
        # Calculate PID outputs
        output_x = pid_x.calculate(error_x)
        output_y = pid_y.calculate(error_y)
        
        # Convert PID outputs to motor positions
        base_position_x = DEFAULT_PID_CONFIG['base_position']['x']
        base_position_y = DEFAULT_PID_CONFIG['base_position']['y']
        
        # Invert the outputs because tilting the platform in the opposite direction
        # moves the ball in the desired direction
        motor1_position = int(base_position_x - output_x)  # Invert x output
        motor2_position = int(base_position_y - output_y)  # Invert y output
        
        # Ensure positions are within valid range
        position_limits = DEFAULT_PID_CONFIG['position_limits']
        
        if motor1_position < position_limits['min']:
            motor1_position = position_limits['min']
        elif motor1_position > position_limits['max']:
            motor1_position = position_limits['max']
            
        if motor2_position < position_limits['min']:
            motor2_position = position_limits['min']
        elif motor2_position > position_limits['max']:
            motor2_position = position_limits['max']
        
        # Add small random variation to prevent getting stuck
        if abs(error_x) < 5 and abs(error_y) < 5:
            motor1_position += np.random.randint(-2, 3)
            motor2_position += np.random.randint(-2, 3)
        
        logger.debug(f"Adjusted motor positions: {motor1_position}, {motor2_position}")
        return motor1_position, motor2_position
    
    except Exception as e:
        logger.error(f"Error adjusting motor movement: {str(e)}")
        return DEFAULT_PID_CONFIG['base_position']['x'], DEFAULT_PID_CONFIG['base_position']['y']

def reset_pid_controllers():
    """Reset both PID controllers to initial state."""
    pid_x.reset()
    pid_y.reset()
    logger.info("PID controllers reset") 
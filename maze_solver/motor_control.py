"""
Motor Control Module

This module provides functions for initializing and controlling AX12 servo motors.
"""

import sys
import os

# Add parent directory to path to import Ax12
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Ax12 import Ax12

import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default motor configuration
DEFAULT_CONFIG = {
    'device_name': '/dev/ttyUSB0',
    'baudrate': 1_000_000,
    'motor_ids': [1, 2],
    'moving_speed': 100,
    'initial_positions': [490, 485],
    'position_limits': [(300, 700), (300, 700)]
}

def initialize_motors(config=None):
    """
    Initialize motor connection and setup.
    
    Args:
        config (dict, optional): Configuration dictionary with the following keys:
            - device_name: Serial device name
            - baudrate: Serial baudrate
            - motor_ids: List of motor IDs
            - moving_speed: Initial moving speed
            
    Returns:
        tuple: Initialized motor objects
    
    Raises:
        Exception: If motor initialization fails
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    try:
        # Configure Ax12 connection
        Ax12.DEVICENAME = config.get('device_name', DEFAULT_CONFIG['device_name'])
        Ax12.BAUDRATE = config.get('baudrate', DEFAULT_CONFIG['baudrate'])
        Ax12.connect()
        
        # Get motor IDs
        motor_ids = config.get('motor_ids', DEFAULT_CONFIG['motor_ids'])
        if len(motor_ids) < 2:
            raise ValueError("At least 2 motor IDs must be provided")
        
        # Initialize motors
        motor1 = Ax12(motor_ids[0])
        motor2 = Ax12(motor_ids[1])
        
        # Set moving speed
        moving_speed = config.get('moving_speed', DEFAULT_CONFIG['moving_speed'])
        motor1.set_moving_speed(moving_speed)
        motor2.set_moving_speed(moving_speed)
        
        logger.info(f"Motors initialized with IDs {motor_ids} and speed {moving_speed}")
        return motor1, motor2
    
    except Exception as e:
        logger.error(f"Failed to initialize motors: {str(e)}")
        raise

def move_motors(motor1, motor2, goal_position1, goal_position2):
    """
    Move motors to specified positions.
    
    Args:
        motor1: First motor object
        motor2: Second motor object
        goal_position1 (int): Target position for motor1
        goal_position2 (int): Target position for motor2
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Apply position limits
        position_limits = DEFAULT_CONFIG['position_limits']
        
        # Limit motor1 position
        if goal_position1 < position_limits[0][0]:
            goal_position1 = position_limits[0][0]
            logger.warning(f"Motor1 position limited to minimum: {goal_position1}")
        elif goal_position1 > position_limits[0][1]:
            goal_position1 = position_limits[0][1]
            logger.warning(f"Motor1 position limited to maximum: {goal_position1}")
            
        # Limit motor2 position
        if goal_position2 < position_limits[1][0]:
            goal_position2 = position_limits[1][0]
            logger.warning(f"Motor2 position limited to minimum: {goal_position2}")
        elif goal_position2 > position_limits[1][1]:
            goal_position2 = position_limits[1][1]
            logger.warning(f"Motor2 position limited to maximum: {goal_position2}")
        
        # Set goal positions
        motor1.set_goal_position(goal_position1)
        motor2.set_goal_position(goal_position2)
        
        logger.debug(f"Motors moved to positions: {goal_position1}, {goal_position2}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to move motors: {str(e)}")
        return False

def move_motors_init_pos(motor1, motor2):
    """
    Move motors to initial positions.
    
    Args:
        motor1: First motor object
        motor2: Second motor object
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        initial_positions = DEFAULT_CONFIG['initial_positions']
        result = move_motors(motor1, motor2, initial_positions[0], initial_positions[1])
        
        if result:
            logger.info(f"Motors moved to initial positions: {initial_positions}")
        
        # Wait for motors to reach position
        time.sleep(0.5)
        
        return result
    
    except Exception as e:
        logger.error(f"Failed to move motors to initial positions: {str(e)}")
        return False

def set_motor_speed(motor1, motor2, speed1, speed2):
    """
    Set the moving speed of the motors.
    
    Args:
        motor1: First motor object
        motor2: Second motor object
        speed1 (int): Speed for motor1 (0-1023)
        speed2 (int): Speed for motor2 (0-1023)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        motor1.set_moving_speed(speed1)
        motor2.set_moving_speed(speed2)
        logger.info(f"Motor speeds set to: {speed1}, {speed2}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to set motor speeds: {str(e)}")
        return False 
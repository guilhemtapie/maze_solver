#!/usr/bin/env python3
"""
Entry point script to run the maze solver.
"""

import sys
import argparse
import time
from maze_solver.maze_solver import MazeSolver, DEFAULT_CONFIG
from maze_solver.motor_control import initialize_motors, move_motors, move_motors_init_pos

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Maze Solver')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--max-iterations', type=int, default=1000, help='Maximum number of iterations')
    parser.add_argument('--delay', type=float, default=0.1, help='Control loop delay')
    parser.add_argument('--motor-demo', action='store_true', help='Run a motor demo sequence')
    return parser.parse_args()

def run_motor_demo():
    """
    Run a simple motor demo to demonstrate motor functionality.
    
    This function:
    1. Initializes the motors
    2. Moves them to various positions in a pattern
    3. Returns to initial position
    """
    print("Starting motor demonstration...")
    try:
        # Initialize motors
        motor1, motor2 = initialize_motors()
        
        # Move to initial position to start
        print("Moving to initial position...")
        move_motors_init_pos(motor1, motor2)
        time.sleep(1)
        
        # Pattern demonstration
        print("Running motor pattern...")
        
        # Move motors to different positions
        positions = [
            # motor1, motor2
            (400, 500),  # Move right
            (500, 500),  # Move up
            (600, 500),  # Move left
            (500, 600),  # Move down
            (400, 600),  # Move diagonally
            (600, 400),  # Move in opposite diagonal
        ]
        
        for i, (pos1, pos2) in enumerate(positions):
            print(f"Movement {i+1}/{len(positions)}: Motor1={pos1}, Motor2={pos2}")
            move_motors(motor1, motor2, pos1, pos2)
            time.sleep(1.5)  # Give time to reach position
        
        # Return to initial position
        print("Returning to initial position...")
        move_motors_init_pos(motor1, motor2)
        time.sleep(1)
        
        print("Motor demonstration completed successfully!")
        return True
        
    except Exception as e:
        print(f"Motor demonstration failed: {str(e)}")
        return False

def main():
    """Main entry point for the maze solver application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check if motor demo is requested
    if args.motor_demo:
        success = run_motor_demo()
        return 0 if success else 1
    
    # Update configuration based on arguments
    config = DEFAULT_CONFIG.copy()
    config['control']['debug_visualization'] = args.visualize
    config['control']['max_iterations'] = args.max_iterations
    config['control']['control_loop_delay'] = args.delay
    
    # Create and run the maze solver
    solver = MazeSolver(config)
    success = solver.run()
    
    # Return appropriate exit code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 
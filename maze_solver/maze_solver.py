"""
Maze Solver Module

This is the main module that ties together all the components for solving a maze.
It handles the overall workflow of capturing images, detecting the ball and goal,
finding a path, and controlling the motors to navigate through the maze.
"""

import time
import numpy as np
from picamera2 import Picamera2, Preview
import matplotlib.pyplot as plt
from skimage import io, color
import logging
import os
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("maze_solver.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import functions from other modules
from .motor_control import initialize_motors, move_motors, move_motors_init_pos, set_motor_speed
from .image_processing import (
    initialize_camera, capture_image, load_image, crop_image, convert_to_grayscale,
    resize_image, enhance_walls, detect_ball, mark_points_on_image, plot_images,
    thicken_walls, crop_around_point, detect_walls
)
from .path_finding import (
    find_path, solve_maze, get_current_position, calculate_distance,
    find_next_target_point, visualize_path
)
from .pid_controller import adjust_motor_movement, reset_pid_controllers, update_pid_parameters

# Default configuration
DEFAULT_CONFIG = {
    'image_paths': {
        'first_image': "/home/labybille/Desktop/new_proj/code_python/images/first_image.jpg",
        'current_image_template': "/home/labybille/Desktop/new_proj/code_python/images/current_image{}.jpg"
    },
    'goal_detection': {
        'method': 'fixed',  # 'fixed', 'marker', or 'color'
        'fixed_position': (120, 120),  # (y, x) coordinates
        'marker_color': (0, 255, 0),  # Green
        'marker_size_range': (20, 50)
    },
    'ball_detection': {
        'min_radius': 20,  # Minimum radius of the ball in pixels
        'max_radius': 40,  # Maximum radius of the ball in pixels
        'color_range': {
            'blue': {
                'hue': (0.55, 0.65),  # Blue hue range in HSV
                'saturation': (0.4, 1.0),
                'value': (0.2, 0.8)
            }
        }
    },
    'control': {
        'distance_threshold': 5,  # Distance to consider a point reached
        'control_loop_delay': 0.1,  # Delay between control loop iterations
        'max_iterations': 1000,  # Maximum number of iterations
        'debug_visualization': False  # Whether to show debug visualizations
    }
}


class MazeSolver:
    """
    Main class for solving a maze using computer vision and motor control.
    
    Attributes:
        config (dict): Configuration dictionary
        camera (Picamera2): Camera object
        motor1 (Ax12): First motor object
        motor2 (Ax12): Second motor object
    """
    
    def __init__(self, config=None):
        """
        Initialize the maze solver.
        
        Args:
            config (dict, optional): Configuration dictionary
        """
        self.config = config if config is not None else DEFAULT_CONFIG
        self.camera = None
        self.motor1 = None
        self.motor2 = None
        logger.info("MazeSolver initialized")
    
    def setup(self):
        """
        Set up the hardware components (camera and motors).
        
        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            # Initialize camera
            logger.info("Initializing camera...")
            self.camera = initialize_camera()
            
            # Initialize motors
            logger.info("Initializing motors...")
            self.motor1, self.motor2 = initialize_motors()
            
            # Move motors to initial position
            move_motors_init_pos(self.motor1, self.motor2)
            time.sleep(1)
            
            logger.info("Setup completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            self.cleanup()
            return False
    
    def detect_goal(self, image):
        """
        Detect the goal position in the maze.
        
        Args:
            image (numpy.ndarray): Image of the maze
            
        Returns:
            tuple: (y, x) coordinates of the goal
        """
        try:
            goal_config = self.config['goal_detection']
            method = goal_config['method']
            
            if method == 'fixed':
                # Use fixed position
                goal = goal_config['fixed_position']
                logger.info(f"Using fixed goal position: {goal}")
                return goal
            
            elif method == 'marker':
                # TODO: Implement marker detection
                logger.warning("Marker detection not implemented, using fixed position")
                return goal_config['fixed_position']
            
            elif method == 'color':
                # TODO: Implement color-based detection
                logger.warning("Color detection not implemented, using fixed position")
                return goal_config['fixed_position']
            
            else:
                logger.warning(f"Unknown goal detection method: {method}, using fixed position")
                return goal_config['fixed_position']
        
        except Exception as e:
            logger.error(f"Goal detection failed: {str(e)}")
            # Return a default position
            height, width = image.shape[:2]
            return (height // 4, width // 4)
    
    def process_image(self, image):
        """
        Process the image to prepare it for path finding.
        
        Args:
            image (numpy.ndarray): Raw image from the camera
            
        Returns:
            tuple: (processed_image, original_scale_image, ball_position)
        """
        try:
            logger.info("Starting image processing...")
            
            # Crop the image
            logger.info("Cropping image...")
            img_cropped = crop_image(image)
            logger.info(f"Cropped image shape: {img_cropped.shape}")
            
            # Detect walls to create binary maze
            logger.info("Detecting walls...")
            maze_binary = detect_walls(img_cropped, plot_result=self.config['control']['debug_visualization'])
            logger.info(f"Wall detection result shape: {maze_binary.shape if maze_binary is not None else 'None'}")
            
            # Detect the ball position in the original scale image
            logger.info("Detecting ball...")
            cx, cy = detect_ball(img_cropped, self.config['control']['debug_visualization'])
            
            if len(cx) == 0 or len(cy) == 0:
                logger.warning("Ball not detected in the image")
                ball_position = None
            else:
                ball_position = (cy[0], cx[0])
                logger.info(f"Ball detected at position: {ball_position}")
            
            return maze_binary, img_cropped, ball_position
        
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            logger.exception("Full traceback:")  # This will print the full traceback
            return None, None, None
    
    def solve_maze_from_image(self, image):
        """
        Process an image and find a path through the maze.
        
        Args:
            image (numpy.ndarray): Image of the maze
            
        Returns:
            tuple: (path, passage_points, start_position, goal_position)
        """
        try:
            # Process the image
            maze_binary, img_gray, ball_position = self.process_image(image)
            
            if maze_binary is None or ball_position is None:
                logger.error("Failed to process image or detect ball")
                return None, None, None, None
            
            # Detect the goal
            goal_position = self.detect_goal(image)
            
            # Scale coordinates to match the resized image
            start_y = int(ball_position[0] * maze_binary.shape[0] / img_gray.shape[0])
            start_x = int(ball_position[1] * maze_binary.shape[1] / img_gray.shape[1])
            start_position = (start_y, start_x)
            
            goal_y = int(goal_position[0] * maze_binary.shape[0] / img_gray.shape[0])
            goal_x = int(goal_position[1] * maze_binary.shape[1] / img_gray.shape[1])
            goal_position_scaled = (goal_y, goal_x)
            
            # Get ball radius from configuration
            ball_radius = self.config['ball_detection']['min_radius']
            # Scale ball radius to match the binary maze dimensions
            ball_radius_scaled = int(ball_radius * maze_binary.shape[0] / img_gray.shape[0])
            
            # Find path through the maze
            path, passage_points = solve_maze(maze_binary, start_position, goal_position_scaled, ball_radius_scaled)
            
            if path is None:
                logger.error("No path found through the maze")
                return None, None, ball_position, goal_position
            
            # Scale path and passage points back to original image size
            path_original_scale = []
            for point in path:
                y = int(point[0] * img_gray.shape[0] / maze_binary.shape[0])
                x = int(point[1] * img_gray.shape[1] / maze_binary.shape[1])
                path_original_scale.append((y, x))
            
            passage_points_original_scale = []
            for point in passage_points:
                y = int(point[0] * img_gray.shape[0] / maze_binary.shape[0])
                x = int(point[1] * img_gray.shape[1] / maze_binary.shape[1])
                passage_points_original_scale.append((y, x))
            
            logger.info(f"Path found with {len(path_original_scale)} points and {len(passage_points_original_scale)} passage points")
            
            # Visualize the path if debug visualization is enabled
            if self.config['control']['debug_visualization']:
                visualize_path(maze_binary, path, passage_points, start_position)
            
            return path_original_scale, passage_points_original_scale, ball_position, goal_position
        
        except Exception as e:
            logger.error(f"Maze solving failed: {str(e)}")
            return None, None, None, None
    
    def navigate_maze(self):
        """
        Navigate through the maze by following the calculated path.
        
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        try:
            # Capture initial image
            image_path = capture_image(self.camera, 0)
            image = load_image(image_path)
            
            # Solve the maze
            path, passage_points, start_position, goal_position = self.solve_maze_from_image(image)
            
            if path is None or passage_points is None:
                logger.error("Failed to solve maze, cannot navigate")
                return False
            
            # Reset PID controllers
            reset_pid_controllers()
            
            # Initialize variables for navigation
            current_point_index = 0
            x, y = start_position[1], start_position[0]  # Convert from (y, x) to (x, y)
            xprec, yprec = x, y
            iteration = 0
            max_iterations = self.config['control']['max_iterations']
            distance_threshold = self.config['control']['distance_threshold']
            control_loop_delay = self.config['control']['control_loop_delay']
            
            # Main control loop
            logger.info("Starting navigation...")
            
            while current_point_index < len(passage_points) and iteration < max_iterations:
                # Capture current image
                image_path = capture_image(self.camera, iteration + 1)
                image = load_image(image_path)
                
                # Get current position
                x, y = get_current_position(image, prev_x=x, prev_y=y, xprec=xprec, yprec=yprec, is_initial=False)
                xprec, yprec = x, y
                
                # Get current target point
                target_point = passage_points[current_point_index]
                
                # Calculate motor positions using PID
                motor1_position, motor2_position = adjust_motor_movement(x, y, target_point)
                
                # Move motors
                move_motors(self.motor1, self.motor2, motor1_position, motor2_position)
                
                # Check if we've reached the current target point
                distance_to_target = calculate_distance(x, y, target_point[1], target_point[0])
                logger.debug(f"Distance to target: {distance_to_target:.2f}, threshold: {distance_threshold}")
                
                if distance_to_target < distance_threshold:
                    current_point_index += 1
                    logger.info(f"Reached point {current_point_index} of {len(passage_points)}")
                
                # Increment iteration counter
                iteration += 1
                
                # Delay for control loop rate
                time.sleep(control_loop_delay)
            
            # Check if we've reached the goal
            if current_point_index >= len(passage_points):
                logger.info("Navigation completed successfully!")
                return True
            else:
                logger.warning(f"Navigation stopped after {iteration} iterations without reaching the goal")
                return False
        
        except KeyboardInterrupt:
            logger.info("Navigation interrupted by user")
            return False
        
        except Exception as e:
            logger.error(f"Navigation failed: {str(e)}")
            return False
        
        finally:
            # Move motors to initial position
            move_motors_init_pos(self.motor1, self.motor2)
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.motor1 is not None and self.motor2 is not None:
                move_motors_init_pos(self.motor1, self.motor2)
            
            if self.camera is not None:
                self.camera.stop_preview()
                self.camera.stop()
            
            logger.info("Cleanup completed")
        
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
    
    def run(self):
        """
        Run the complete maze solving process.
        
        Returns:
            bool: True if successful, False otherwise
        """
        success = False
        
        try:
            # Setup hardware
            if not self.setup():
                return False
            
            # Navigate through the maze
            success = self.navigate_maze()
            
            return success
        
        except Exception as e:
            logger.error(f"Maze solver run failed: {str(e)}")
            return False
        
        finally:
            self.cleanup()


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
    return parser.parse_args()


def main():
    """Main entry point for the maze solver application."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Configure logging level
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Update configuration based on arguments
        config = DEFAULT_CONFIG.copy()
        config['control']['debug_visualization'] = args.visualize
        config['control']['max_iterations'] = args.max_iterations
        config['control']['control_loop_delay'] = args.delay
        
        # Create and run the maze solver
        solver = MazeSolver(config)
        success = solver.run()
        
        if success:
            logger.info("Maze solved successfully!")
            return 0
        else:
            logger.error("Failed to solve the maze")
            return 1
    
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    main() 
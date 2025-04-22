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
    thicken_walls, crop_around_point, detect_walls, capture_image_fast, fast_ball_detection
)
from .path_finding import (
    solve_maze, get_current_position, calculate_distance,
    find_next_target_point, visualize_path, AStar
)
from .pid_controller import adjust_motor_movement, reset_pid_controllers, update_pid_parameters

# Default configuration
DEFAULT_CONFIG = {
    'image_paths': {
        'first_image': "/home/labybille/Desktop/new_proj/code_python/images/first_image.jpg",
        'current_image_template': "/home/labybille/Desktop/new_proj/code_python/images/current_image{}.jpg"
    },
    'crop_dimensions': {
        'y_min': 80,    
        'y_max': 1000,    
        'x_min': 100,    
        'x_max': 900     
    },
    'goal_detection': {
        'method': 'fixed',  # 'fixed', 'marker', or 'color'
        'fixed_position': (200, 200),  # (y, x) coordinates
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
        'max_iterations': 1000,  # Maximum number of iterations
        'debug_visualization': False,  # Whether to show debug visualizations
        'compare_detection': True  # Whether to compare fast and classic ball detection
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
        maze_binary (numpy.ndarray): Binary maze representation
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
        self.maze_binary = None
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
        """Process a single image to detect walls and ball."""
        try:
            start_time = time.time()
            logger.info("Starting image processing...")
            
            # Save raw image for debugging
            plt.imsave('raw_camera_image.png', image)
            logger.info("Raw camera image saved")
            
            # Crop the image
            crop_start = time.time()
            logger.info("Cropping image...")
            img_cropped = crop_image(image)
            crop_time = time.time() - crop_start
            logger.info(f"Cropped image shape: {img_cropped.shape}")
            logger.info(f"=== Image cropping completed in {crop_time:.3f} seconds ===")
            
            # Detect the ball position first
            ball_start = time.time()
            logger.info("Detecting ball...")
            ball_x, ball_y = detect_ball(img_cropped, plot_result=self.config['control']['debug_visualization'], config=self.config)
            
            if not ball_x or not ball_y:
                logger.warning("Ball not detected in the image")
                ball_position = None
            else:
                ball_position = (ball_x[0], ball_y[0])
                logger.info(f"Ball detected at position: {ball_position}")
            ball_time = time.time() - ball_start
            logger.info(f"=== Ball detection completed in {ball_time:.3f} seconds ===")
            
            # Detect walls to create binary maze, passing the ball position
            walls_start = time.time()
            logger.info("Detecting walls...")
            maze_binary = detect_walls(img_cropped, ball_position=ball_position, 
                                     plot_result=self.config['control']['debug_visualization'])
            walls_time = time.time() - walls_start
            logger.info(f"Wall detection result shape: {maze_binary.shape if maze_binary is not None else 'None'}")
            logger.info(f"=== Wall detection completed in {walls_time:.3f} seconds ===")
            
            total_time = time.time() - start_time
            logger.info(f"=== Total image processing completed in {total_time:.3f} seconds ===")
            
            return maze_binary, img_cropped, ball_position
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return None, None, None

    def solve_maze_from_image(self, image):
        """Process an image and solve the maze."""
        try:
            start_time = time.time()
            
            # Process image
            process_start = time.time()
            maze_binary, cropped_image, ball_position = self.process_image(image)
            process_time = time.time() - process_start
            logger.info(f"Image processing total time: {process_time:.3f} seconds")
            
            if maze_binary is None or ball_position is None:
                logger.error("Failed to process image")
                return None, None
            
            # Convert to binary maze
            maze_start = time.time()
            self.maze_binary = maze_binary.astype(np.uint8)
            maze_time = time.time() - maze_start
            logger.info(f"Maze binary conversion time: {maze_time:.3f} seconds")
            
            # Save the binary maze array
            np.save('binary_maze.npy', self.maze_binary)
            logger.info("Saved binary maze array to binary_maze.npy")
            
            # Get start and goal positions
            if ball_position is None:
                logger.error("No ball position available for pathfinding")
                return None, None
                
            # ball_position is in (x, y) format, convert to (y, x) for maze coordinates
            start = (ball_position[1], ball_position[0])  # Convert from (x, y) to (y, x)
            
            # Use fixed goal position (200, 200)
            goal = (200, 200)
            logger.info(f"Using fixed goal position: {goal}")
            
            # Verify goal position is within bounds
            height, width = self.maze_binary.shape
            if not (0 <= goal[0] < height and 0 <= goal[1] < width):
                logger.error(f"Goal position {goal} is out of bounds. Maze shape is {self.maze_binary.shape}")
                return None, None
            
            # Debug logging for start and goal positions
            logger.info(f"Start position (y, x): {start}")
            logger.info(f"Goal position (y, x): {goal}")
            
            # Check if start or goal positions are on walls
            if self.maze_binary[start[0], start[1]]:
                logger.error(f"Start position {start} is on a wall")
                return None, None
            if self.maze_binary[goal[0], goal[1]]:
                logger.error(f"Goal position {goal} is on a wall")
                return None, None
            
            # Find path
            path_start = time.time()
            logger.info("Starting path finding...")
            path = solve_maze(self.maze_binary, start, goal)
            path_time = time.time() - path_start
            logger.info(f"=== Path finding completed in {path_time:.3f} seconds ===")
            
            if path is None or len(path) == 0:
                logger.warning("No path found from start to goal")
                return None, None
            
            logger.info(f"Path found with {len(path)} points")
            
            # Display path visualization if debug_visualization is enabled
            if self.config['control']['debug_visualization']:
                plt.figure(figsize=(10, 10))
                plt.imshow(self.maze_binary, cmap='gray')
                path_y, path_x = zip(*path)
                plt.plot(path_x, path_y, 'r-', linewidth=2)
                plt.plot(start[1], start[0], 'go', markersize=10, label='Start')
                plt.plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')
                plt.legend()
                plt.title("Path Found")
                plt.axis('off')
                plt.show()
                plt.close()
            
            total_time = time.time() - start_time
            logger.info(f"=== Total maze solving completed in {total_time:.3f} seconds ===")
            
            return path, ball_position
        
        except Exception as e:
            logger.error(f"Error solving maze: {str(e)}")
            return None, None
    
    def navigate_maze(self):
        """
        Navigate through the maze by following the calculated path.
        
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        try:
            # Capture initial image
            image_path = capture_image(self.camera, 0, self.config)
            image = load_image(image_path)
            
            # Solve the maze and get initial ball position
            path, ball_position = self.solve_maze_from_image(image)
            if path is None or ball_position is None:
                logger.error("Failed to solve maze or get ball position, cannot navigate")
                return False
            
            # Reset PID controllers
            reset_pid_controllers()
            
            # Initialize variables for navigation
            current_point_index = 0
            x, y = ball_position  # Use initial ball position
            xprec, yprec = x, y
            iteration = 0
            max_iterations = self.config['control']['max_iterations']
            distance_threshold = self.config['control']['distance_threshold']
            
            # Main control loop
            logger.info("Starting navigation...")
            
            while current_point_index < len(path) and iteration < max_iterations:
                loop_start = time.time()
                
                # Capture current image
                capture_start = time.time()
                image = capture_image_fast(self.camera, config=self.config)
                capture_time = time.time() - capture_start
                
                # Get current position using fast detection
                position_start = time.time()
                new_x, new_y = fast_ball_detection(image, prev_position=(x, y), config=self.config)
                if new_x is not None and new_y is not None:
                    x, y = new_x, new_y
                    xprec, yprec = x, y
                    logger.info(f"Ball detected at position: ({x}, {y})")
                else:
                    logger.warning("Ball detection failed, using previous position")
                    x, y = xprec, yprec
                position_time = time.time() - position_start
                
                # Get current target point
                target_point = path[current_point_index]
                logger.info(f"Current target point: {target_point}")
                
                # Visualize current position and target if debug_visualization is enabled
                if self.config['control']['debug_visualization']:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(self.maze_binary, cmap='gray')
                    plt.plot(x, y, 'bo', markersize=10, label='Current Position')
                    plt.plot(target_point[1], target_point[0], 'ro', markersize=10, label='Target Position')
                    plt.title(f"Navigation Step {iteration}")
                    plt.legend()
                    plt.axis('off')
                    plt.show()
                    plt.close()
                
                # Calculate motor positions using PID
                pid_start = time.time()
                # Pass target point directly since it's already in (y, x) format
                motor1_position, motor2_position = adjust_motor_movement(x, y, target_point)
                if motor1_position is None or motor2_position is None:
                    logger.error("PID calculation failed to produce valid motor positions")
                    break
                logger.info(f"Calculated motor positions: M1={motor1_position}, M2={motor2_position}")
                pid_time = time.time() - pid_start
                
                # Move motors
                motor_start = time.time()
                try:
                    move_motors(self.motor1, self.motor2, motor1_position, motor2_position)
                    logger.info("Motors moved successfully")
                except Exception as e:
                    logger.error(f"Failed to move motors: {str(e)}")
                    break
                motor_time = time.time() - motor_start
                
                # Check if we've reached the current target point
                distance_to_target = calculate_distance(x, y, target_point[1], target_point[0])
                logger.info(f"Distance to target: {distance_to_target:.2f} pixels")
                
                if distance_to_target < distance_threshold:
                    current_point_index += 1
                    logger.info(f"Reached point {current_point_index} of {len(path)}")
                
                # Calculate total loop time
                loop_time = time.time() - loop_start
                
                # Log timing information
                logger.info(f"Iteration {iteration} timing:")
                logger.info(f"  Image capture: {capture_time:.3f} seconds")
                logger.info(f"  Position detection: {position_time:.3f} seconds")
                logger.info(f"  PID calculation: {pid_time:.3f} seconds")
                logger.info(f"  Motor movement: {motor_time:.3f} seconds")
                logger.info(f"  Total loop time: {loop_time:.3f} seconds")
                
                # Increment iteration counter
                iteration += 1
            
            # Check if we've reached the goal
            if current_point_index >= len(path):
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
"""
Path Finding Module

This module provides functions for finding paths through a maze using the A* algorithm.
"""

import numpy as np
import heapq
import matplotlib.pyplot as plt
from skimage import color
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the detectbille function from image_processing
from .image_processing import detectbille

class AStar:
    """
    A* path finding algorithm implementation.
    
    Attributes:
        array (numpy.ndarray): 2D array representing the maze (0 = free space, 1 = wall)
        start (tuple): Starting position (y, x)
        goal (tuple): Goal position (y, x)
        neighbors (list): List of possible movement directions
    """
    
    def __init__(self, array, start, goal):
        """
        Initialize the A* path finder.
        
        Args:
            array (numpy.ndarray): 2D array representing the maze (0 = free space, 1 = wall)
            start (tuple): Starting position (y, x)
            goal (tuple): Goal position (y, x)
        """
        self.array = array
        self.start = start
        self.goal = goal
        # 8-directional movement: horizontal, vertical, and diagonal neighbors
        self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
    def heuristic(self, a, b):
        """
        Calculate the heuristic (Manhattan distance) between two points.
        
        Args:
            a (tuple): First point (y, x)
            b (tuple): Second point (y, x)
            
        Returns:
            float: Manhattan distance between the points
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def euclidean_distance(self, a, b):
        """
        Calculate the Euclidean distance between two points.
        
        Args:
            a (tuple): First point (y, x)
            b (tuple): Second point (y, x)
            
        Returns:
            float: Euclidean distance between the points
        """
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def is_valid_position(self, position):
        """
        Check if a position is valid (within bounds and not a wall).
        
        Args:
            position (tuple): Position to check (y, x)
            
        Returns:
            bool: True if the position is valid, False otherwise
        """
        y, x = position
        
        # Check if within bounds
        if 0 <= y < self.array.shape[0] and 0 <= x < self.array.shape[1]:
            # Check if not a wall
            return self.array[y][x] != 1
        
        return False
    
    def find_path(self):
        """
        Find a path from start to goal using the A* algorithm.
        
        Returns:
            list: List of positions forming the path, or empty list if no path found
        """
        try:
            # Initialize data structures
            close_set = set()
            came_from = {}
            gscore = {self.start: 0}
            fscore = {self.start: self.heuristic(self.start, self.goal)}
            oheap = []

            heapq.heappush(oheap, (fscore[self.start], self.start))
            
            while oheap:
                current = heapq.heappop(oheap)[1]

                # If we've reached the goal
                if current == self.goal:
                    data = []
                    while current in came_from:
                        data.append(current)
                        current = came_from[current]
                    
                    # Return the path in the correct order (start to goal)
                    path = data[::-1]
                    logger.info(f"Path found with {len(path)} steps")
                    return path

                close_set.add(current)
                
                # Check all neighbors
                for i, j in self.neighbors:
                    neighbor = current[0] + i, current[1] + j
                    
                    # Skip if not a valid position
                    if not self.is_valid_position(neighbor):
                        continue
                    
                    # Calculate tentative g score
                    tentative_g_score = gscore[current] + self.euclidean_distance(current, neighbor)
                    
                    # Skip if we've found a better path already
                    if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                        continue
                    
                    # This is the best path so far
                    if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, self.goal)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))
            
            # No path found
            logger.warning("No path found from start to goal")
            return []
        
        except Exception as e:
            logger.error(f"Error in A* path finding: {str(e)}")
            return []


def find_path(maze, start, goal):
    """
    Find a path through the maze using the A* algorithm.
    
    Args:
        maze (numpy.ndarray): 2D array representing the maze (0 = free space, 1 = wall)
        start (tuple): Starting position (y, x)
        goal (tuple): Goal position (y, x)
        
    Returns:
        list: List of positions forming the path, or empty list if no path found
    """
    try:
        astar = AStar(maze, start, goal)
        return astar.find_path()
    except Exception as e:
        logger.error(f"Error finding path: {str(e)}")
        return []


def find_passage_points(path, min_angle_change=0.5):
    """
    Find significant passage points in the path (points with direction changes).
    
    Args:
        path (list): List of positions forming the path
        min_angle_change (float): Minimum angle change to consider a point significant
        
    Returns:
        list: List of significant passage points
    """
    try:
        if len(path) < 3:
            logger.warning("Path too short to find passage points")
            return path
        
        # Convert path to array for easier calculations
        path_array = np.array(path)
        
        # Find points with significant changes in direction
        passage_points = []
        
        for i in range(1, len(path) - 1):
            prev_point = path[i-1]
            current_point = path[i]
            next_point = path[i+1]
            
            # Calculate vectors
            v1 = (current_point[0] - prev_point[0], current_point[1] - prev_point[1])
            v2 = (next_point[0] - current_point[0], next_point[1] - current_point[1])
            
            # Calculate angle between vectors
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            mag_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
            mag_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
            
            # Avoid division by zero
            if mag_v1 * mag_v2 > 0:
                cos_angle = dot_product / (mag_v1 * mag_v2)
                # Clamp to avoid numerical errors
                cos_angle = max(-1, min(1, cos_angle))
                angle = np.arccos(cos_angle)
                
                # If angle is significant, add as passage point
                if angle > min_angle_change:
                    passage_points.append(current_point)
        
        # Always include start and end points
        if path and path[0] not in passage_points:
            passage_points.insert(0, path[0])
        if path and path[-1] not in passage_points:
            passage_points.append(path[-1])
        
        logger.info(f"Found {len(passage_points)} passage points")
        return passage_points
    
    except Exception as e:
        logger.error(f"Error finding passage points: {str(e)}")
        return path if path else []


def solve_maze(maze, start, goal):
    """
    Find a solution path through the maze and identify key passage points.
    
    Args:
        maze (numpy.ndarray): 2D array representing the maze (0 = free space, 1 = wall)
        start (tuple): Starting position (y, x)
        goal (tuple): Goal position (y, x)
        
    Returns:
        tuple: (path, passage_points) or (None, None) if no path found
    """
    try:
        # Find path using A*
        path = find_path(maze, start, goal)
        
        if not path:
            logger.warning("No path found through maze")
            return None, None
        
        # Find passage points
        passage_points = find_passage_points(path)
        
        return path, passage_points
    
    except Exception as e:
        logger.error(f"Error solving maze: {str(e)}")
        return None, None


def get_current_position(image, prev_x=None, prev_y=None, is_initial=False):
    """
    Determine the current position of the ball in the maze.
    
    Args:
        image (numpy.ndarray): Current image of the maze
        prev_x (int, optional): Previous x position
        prev_y (int, optional): Previous y position
        is_initial (bool): Whether this is the initial position
        
    Returns:
        tuple: (x, y) position of the ball
    """
    try:
        if is_initial:
            # For initial position, just return the provided coordinates
            return prev_x, prev_y
        
        # Detect the ball
        cx, cy = detectbille(image, False)
        
        if len(cx) == 0 or len(cy) == 0:
            logger.warning("Ball not detected, using previous position")
            return prev_x, prev_y
        
        logger.debug(f"Ball detected at position: ({cx[0]}, {cy[0]})")
        return cx[0], cy[0]
    
    except Exception as e:
        logger.error(f"Error getting current position: {str(e)}")
        return prev_x, prev_y


def calculate_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points.
    
    Args:
        x1 (float): X coordinate of first point
        y1 (float): Y coordinate of first point
        x2 (float): X coordinate of second point
        y2 (float): Y coordinate of second point
        
    Returns:
        float: Euclidean distance between the points
    """
    try:
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    except Exception as e:
        logger.error(f"Error calculating distance: {str(e)}")
        return float('inf')


def find_next_target_point(current_x, current_y, path, current_target_index, distance_threshold=5):
    """
    Find the next target point in the path based on the current position.
    
    Args:
        current_x (float): Current x position
        current_y (float): Current y position
        path (list): List of path points
        current_target_index (int): Index of the current target point
        distance_threshold (float): Distance threshold to consider a point reached
        
    Returns:
        int: Index of the next target point
    """
    try:
        if not path or current_target_index >= len(path):
            logger.warning("Invalid path or target index")
            return max(0, min(current_target_index, len(path) - 1 if path else 0))
        
        # Calculate distance to current target point
        target_point = path[current_target_index]
        dist_to_target = calculate_distance(current_x, current_y, target_point[1], target_point[0])
        
        # If close enough to current target, move to next point
        if dist_to_target < distance_threshold:
            next_index = current_target_index + 1
            logger.info(f"Target point {current_target_index} reached, moving to point {next_index}")
            return min(next_index, len(path) - 1)
        
        return current_target_index
    
    except Exception as e:
        logger.error(f"Error finding next target point: {str(e)}")
        return current_target_index


def visualize_path(maze, path, passage_points=None, current_position=None):
    """
    Visualize the maze, path, and current position.
    
    Args:
        maze (numpy.ndarray): 2D array representing the maze
        path (list): List of positions forming the path
        passage_points (list, optional): List of significant passage points
        current_position (tuple, optional): Current position (x, y)
    """
    try:
        plt.figure(figsize=(10, 10))
        
        # Display the maze
        plt.imshow(maze, cmap='gray')
        
        # Plot the path
        if path:
            path_array = np.array(path)
            plt.plot(path_array[:, 1], path_array[:, 0], 'b-', linewidth=2)
        
        # Plot passage points
        if passage_points:
            passage_array = np.array(passage_points)
            plt.plot(passage_array[:, 1], passage_array[:, 0], 'ro', markersize=8)
        
        # Plot current position
        if current_position:
            plt.plot(current_position[0], current_position[1], 'go', markersize=10)
        
        plt.title('Maze Path Visualization')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        logger.error(f"Error visualizing path: {str(e)}")


# Aliases for backward compatibility
module = lambda x, y: np.sqrt(x**2 + y**2)
heuristic = lambda a, b: abs(a[0] - b[0]) + abs(a[1] - b[1])
astar = find_path
trouver_max_de_marge = lambda *args: None  # Placeholder
trouver_chemin_avec_le_max_de_marge = lambda *args: None  # Placeholder
trouver_point_de_passage = find_passage_points
solution_labyrinthe = solve_maze
position_actuelle = get_current_position
norme = calculate_distance
point_proche = find_next_target_point 
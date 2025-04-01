"""
Image Processing Module

This module provides functions for image acquisition, processing, and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
import scipy.ndimage as scnd
import skimage.morphology as morph
from PIL import Image
from picamera2 import Picamera2, Preview
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    'image_paths': {
        'first_image': "/home/labybille/Desktop/new_proj/code_python/images/first_image.jpg",
        'current_image_template': "/home/labybille/Desktop/new_proj/code_python/images/current_image{}.jpg"
    },
    'crop_dimensions': {
        'y_min': 100,    
        'y_max': 970,    
        'x_min': 150,    
        'x_max': 850     
    },
    'ball_detection': {
        'sigma': 0.5,            # Reduced to detect finer edges
        'low_threshold': 5,      # Lowered to detect more edges
        'high_threshold': 15,    # Lowered to detect more edges
        'min_radius': 28,        # Adjusted based on your ball size
        'max_radius': 36,        # Adjusted based on your ball size
        'radius_step': 1,
        'max_peaks': 1
    },
    'wall_detection': {
        'brown_hue_low': 0.05,   # Brown color hue range
        'brown_hue_high': 0.15,
        'saturation_low': 0.3,   # Saturation range for brown
        'saturation_high': 0.8,
        'value_low': 0.1,        # Value range for dark brown
        'value_high': 0.5,
        'closing_kernel': 5,     # Size of morphological operation kernels
        'opening_kernel': 3
    },
    'resize_dimensions': (1000, 1000)
}

# Add detect_walls to the list of exposed functions at the top of the file
__all__ = [
    'initialize_camera',
    'capture_image',
    'load_image',
    'crop_image',
    'convert_to_grayscale',
    'resize_image',
    'enhance_walls',
    'detect_ball',
    'mark_points_on_image',
    'plot_images',
    'thicken_walls',
    'crop_around_point',
    'detect_walls'
]

# Camera functions
def initialize_camera():
    """
    Initialize and start the camera.
    
    Returns:
        Picamera2: Initialized camera object
    """
    try:
        camera = Picamera2()
        camera.preview_configuration.main.size = (DEFAULT_CONFIG['resize_dimensions'][0], DEFAULT_CONFIG['resize_dimensions'][1])
        camera.configure("preview")
        camera.start()
        time.sleep(2)  # Allow camera to warm up
        logger.info("Camera initialized successfully")
        return camera
    except Exception as e:
        logger.error(f"Failed to initialize camera: {str(e)}")
        raise

def capture_image(camera, image_number=0, config=None):
    """
    Capture an image with the camera.
    
    Args:
        camera: Camera object
        image_number (int): Image number (0 for first image, >0 for subsequent images)
        config (dict, optional): Configuration dictionary
        
    Returns:
        str: Path to the captured image
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    try:
        if image_number == 0:
            image_path = config['image_paths']['first_image']
        else:
            image_path = config['image_paths']['current_image_template'].format(image_number)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        camera.capture_file(image_path)
        logger.info(f"Image captured: {image_path}")
        return image_path
    
    except Exception as e:
        logger.error(f"Failed to capture image: {str(e)}")
        raise

def load_image(image_path):
    """
    Load an image from file.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Loaded image
    """
    try:
        image = io.imread(image_path)
        logger.debug(f"Image loaded: {image_path}, shape: {image.shape}")
        return image
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {str(e)}")
        raise

# Image preprocessing functions
def crop_image(image, config=None):
    """
    Crop the input image.
    
    Args:
        image (numpy.ndarray): Input image
        config (dict, optional): Configuration dictionary
        
    Returns:
        numpy.ndarray: Cropped image
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    try:
        crop_dims = config['crop_dimensions']
        cropped = image[crop_dims['y_min']:crop_dims['y_max'], 
                        crop_dims['x_min']:crop_dims['x_max']]
        logger.debug(f"Image cropped from {image.shape} to {cropped.shape}")
        return cropped
    except Exception as e:
        logger.error(f"Failed to crop image: {str(e)}")
        return image  # Return original image if cropping fails

def convert_to_grayscale(image):
    """
    Convert image to grayscale.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        numpy.ndarray: Grayscale image
    """
    try:
        if len(image.shape) == 3 and image.shape[2] >= 3:
            gray = color.rgb2gray(image)
            logger.debug("Image converted to grayscale")
            return gray
        elif len(image.shape) == 2:
            logger.debug("Image already in grayscale")
            return image
        else:
            logger.warning(f"Unexpected image shape: {image.shape}")
            return image
    except Exception as e:
        logger.error(f"Failed to convert image to grayscale: {str(e)}")
        return image

def resize_image(image, target_size=None, config=None):
    """
    Resize image for processing.
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple, optional): Target size (width, height)
        config (dict, optional): Configuration dictionary
        
    Returns:
        numpy.ndarray: Resized image
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    if target_size is None:
        target_size = config['resize_dimensions']
    
    try:
        # Convert to PIL Image for resizing
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        img_pil = img_pil.resize(target_size)
        resized = np.array(img_pil) / 255.0
        logger.debug(f"Image resized from {image.shape} to {resized.shape}")
        return resized
    except Exception as e:
        logger.error(f"Failed to resize image: {str(e)}")
        return image

def enhance_walls(image, kernel_size=3):
    """
    Enhance walls in the image using binary dilation.
    
    Args:
        image (numpy.ndarray): Input binary image
        kernel_size (int): Size of the kernel for dilation
        
    Returns:
        numpy.ndarray: Enhanced image
    """
    try:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        enhanced = scnd.binary_dilation(image, kernel)
        logger.debug(f"Walls enhanced with kernel size {kernel_size}")
        return enhanced
    except Exception as e:
        logger.error(f"Failed to enhance walls: {str(e)}")
        return image

# Ball detection functions
def detect_ball(image, plot_result=False, config=None):
    """
    Detect balls in the image using color thresholding and blob detection.
    
    Args:
        image (numpy.ndarray): Input image (RGB or grayscale)
        plot_result (bool): Whether to plot the detection result
        config (dict, optional): Configuration dictionary
        
    Returns:
        tuple: Lists of x and y coordinates of detected balls
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    try:
        # Debug logging for input image
        logger.info(f"Input image shape: {image.shape}")
        logger.info(f"Input image dtype: {image.dtype}")
        
        # Ensure we have an RGB image
        if len(image.shape) == 2:  # If grayscale, convert to RGB
            image = color.gray2rgb(image)
        
        # Convert to HSV color space for better color segmentation
        hsv_image = color.rgb2hsv(image)
        
        # Define color range for blue ball
        # Blue in HSV has a hue around 0.55-0.65
        blue_low = np.array([0.55, 0.4, 0.2])  # Dark blue lower bound
        blue_high = np.array([0.65, 1.0, 0.8])  # Dark blue upper bound
        
        # Create mask for blue color
        mask = np.all((hsv_image >= blue_low) & (hsv_image <= blue_high), axis=2)
        
        # Apply morphological operations to clean up the mask
        # Use smaller kernels to avoid merging with wall regions
        mask = scnd.binary_closing(mask, structure=np.ones((3,3)))
        mask = scnd.binary_opening(mask, structure=np.ones((2,2)))
        
        # Find connected components (blobs)
        labeled_mask, num_features = scnd.label(mask)
        
        if num_features > 0:
            # Find the largest blob (assuming it's the ball)
            sizes = np.bincount(labeled_mask.ravel())[1:]
            largest_blob = np.argmax(sizes) + 1
            
            # Get the centroid of the largest blob
            cy, cx = np.mean(np.where(labeled_mask == largest_blob), axis=1)
            
            logger.info(f"Ball detected at position: ({int(cx)}, {int(cy)})")
            
            if plot_result:
                # Plot the detection result
                plt.figure(figsize=(15, 5))
                
                # Show original image
                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.axis('off')
                
                # Show color mask
                plt.subplot(1, 3, 2)
                plt.imshow(mask, cmap='gray')
                plt.title("Color Mask")
                plt.axis('off')
                
                # Show result
                result_img = image.copy()
                rr, cc = circle_perimeter(int(cy), int(cx), 20, shape=image.shape)
                result_img[rr, cc] = [0, 0, 1]  # Blue circle
                plt.subplot(1, 3, 3)
                plt.imshow(result_img)
                plt.title("Detection Result")
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
            
            return [int(cx)], [int(cy)]
        else:
            logger.warning("No ball detected")
            if plot_result:
                plt.figure(figsize=(15, 5))
                
                # Show original image
                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.axis('off')
                
                # Show color mask
                plt.subplot(1, 3, 2)
                plt.imshow(mask, cmap='gray')
                plt.title("Color Mask")
                plt.axis('off')
                
                plt.tight_layout()
                plt.show()
            
            return [], []
    
    except Exception as e:
        logger.error(f"Failed to detect ball: {str(e)}")
        logger.error(f"Error occurred with image shape: {image.shape}")
        return [], []

def plot_ball_detection(image, cy, cx, radii):
    """
    Plot the ball detection result.
    
    Args:
        image (numpy.ndarray): Input image
        cy (list): Y coordinates of detected balls
        cx (list): X coordinates of detected balls
        radii (list): Radii of detected balls
    """
    try:
        # Draw detected circles
        image_rgb = color.gray2rgb(image) if len(image.shape) == 2 else image.copy()
        
        for center_y, center_x, radius in zip(cy, cx, radii):
            circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
            try:
                image_rgb[circy, circx] = (220, 20, 20)
            except IndexError:
                pass
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb, cmap=plt.cm.gray)
        plt.title("Ball Detection Result")
        plt.show()
    
    except Exception as e:
        logger.error(f"Failed to plot ball detection: {str(e)}")

def mark_points_on_image(image, cx, cy, color=(255, 0, 0)):
    """
    Mark detected points on the image.
    
    Args:
        image (numpy.ndarray): Input image
        cx (list): X coordinates of points
        cy (list): Y coordinates of points
        color (tuple): RGB color for marking
        
    Returns:
        numpy.ndarray: Image with marked points
    """
    try:
        marked_image = np.copy(image)
        
        # Convert to RGB if grayscale
        if len(marked_image.shape) == 2:
            marked_image = color.gray2rgb(marked_image)
        
        # Draw circles at each point
        for i in range(len(cx)):
            rr, cc = circle_perimeter(cy[i], cx[i], 10)
            try:
                marked_image[rr, cc] = color
            except IndexError:
                pass
        
        return marked_image
    
    except Exception as e:
        logger.error(f"Failed to mark points on image: {str(e)}")
        return image

# Morphological operations
def create_structuring_element(shape="square", size=3):
    """
    Create a structuring element for morphological operations.
    
    Args:
        shape (str): Shape of the element ('square', 'diamond', 'disk')
        size (int): Size of the element
        
    Returns:
        numpy.ndarray: Structuring element
    """
    try:
        if shape == "square":
            return morph.square(size)
        elif shape == "diamond":
            return morph.diamond(size)
        elif shape == "disk":
            return morph.disk(size)
        else:
            logger.warning(f"Unknown shape: {shape}, using square")
            return morph.square(size)
    
    except Exception as e:
        logger.error(f"Failed to create structuring element: {str(e)}")
        return np.ones((size, size), dtype=bool)

def thicken_walls(binary_image, kernel_size=3):
    """
    Thicken walls in the maze using binary dilation.
    
    Args:
        binary_image (numpy.ndarray): Binary image of the maze
        kernel_size (int): Size of the kernel for dilation
        
    Returns:
        numpy.ndarray: Image with thickened walls
    """
    try:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thickened = scnd.binary_dilation(binary_image, kernel)
        logger.debug(f"Walls thickened with kernel size {kernel_size}")
        return thickened
    
    except Exception as e:
        logger.error(f"Failed to thicken walls: {str(e)}")
        return binary_image

# Utility functions
def plot_images(images, titles=None, cmap='gray'):
    """
    Display multiple images.
    
    Args:
        images (list): List of images to display
        titles (list, optional): List of titles for each image
        cmap (str): Colormap to use
    """
    try:
        n = len(images)
        plt.figure(figsize=(15, 5))
        
        for i, img in enumerate(images):
            plt.subplot(1, n, i+1)
            plt.imshow(img, cmap=cmap)
            
            if titles is not None and i < len(titles):
                plt.title(titles[i])
                
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        logger.error(f"Failed to plot images: {str(e)}")

def crop_around_point(image, cy, cx, size=50):
    """
    Crop image around a specific point.
    
    Args:
        image (numpy.ndarray): Input image
        cy (int): Y coordinate of the center point
        cx (int): X coordinate of the center point
        size (int): Half-size of the crop window
        
    Returns:
        numpy.ndarray: Cropped image
    """
    try:
        # Calculate crop boundaries
        y_min = max(0, cy - size)
        y_max = min(image.shape[0], cy + size)
        x_min = max(0, cx - size)
        x_max = min(image.shape[1], cx + size)
        
        # Crop the image
        cropped = image[y_min:y_max, x_min:x_max]
        logger.debug(f"Image cropped around point ({cx}, {cy}) with size {size}")
        return cropped
    
    except Exception as e:
        logger.error(f"Failed to crop around point: {str(e)}")
        return image

# Aliases for backward compatibility
appelcamera = capture_image
acquisition_image = initialize_camera
rehaussement_murs = enhance_walls
recadrage = crop_image
image_grise = convert_to_grayscale
plot = plot_images
image_detecte = mark_points_on_image
stucturing_element = create_structuring_element
detectbille = detect_ball
redim_image = resize_image
eppaissir_les_murs = thicken_walls
decoupage = crop_around_point

def detect_walls(image, plot_result=False, config=None):
    """
    Detect walls in the maze using color thresholding.
    
    Args:
        image (numpy.ndarray): Input image (RGB or grayscale)
        plot_result (bool): Whether to plot the detection result
        config (dict, optional): Configuration dictionary
        
    Returns:
        numpy.ndarray: Binary mask where walls are True (1) and paths are False (0)
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    try:
        logger.info("Starting wall detection...")
        # Ensure we have an RGB image
        if len(image.shape) == 2:
            image = color.gray2rgb(image)
        
        # Convert to HSV color space for better color segmentation
        hsv_image = color.rgb2hsv(image)
        
        # Define multiple color ranges for brown walls to catch different shades
        # Dark brown
        brown_low1 = np.array([0.05, 0.3, 0.1])
        brown_high1 = np.array([0.15, 0.9, 0.6])
        
        # Lighter brown
        brown_low2 = np.array([0.05, 0.2, 0.2])
        brown_high2 = np.array([0.15, 0.7, 0.7])
        
        # Create masks for both brown ranges
        mask1 = np.all((hsv_image >= brown_low1) & (hsv_image <= brown_high1), axis=2)
        mask2 = np.all((hsv_image >= brown_low2) & (hsv_image <= brown_high2), axis=2)
        
        # Combine masks
        mask = mask1 | mask2
        
        # Apply morphological operations to clean up the mask
        # First close small gaps in walls
        mask = scnd.binary_closing(mask, structure=np.ones((7,7)))
        
        # Remove small noise
        mask = scnd.binary_opening(mask, structure=np.ones((3,3)))
        
        # Fill holes in walls
        mask = scnd.binary_fill_holes(mask)
        
        # One final closing to ensure wall continuity
        mask = scnd.binary_closing(mask, structure=np.ones((5,5)))
        
        logger.info(f"Wall detection completed. Mask shape: {mask.shape}")
        
        if plot_result:
            plt.figure(figsize=(15, 5))
            
            # Show original image
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')
            
            # Show color mask
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap='gray')
            plt.title("Wall Mask")
            plt.axis('off')
            
            # Show result
            result_img = image.copy()
            result_img[mask] = [0, 0, 0]  # Mark walls in black
            plt.subplot(1, 3, 3)
            plt.imshow(result_img)
            plt.title("Detection Result")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return mask
    
    except Exception as e:
        logger.error(f"Wall detection failed: {str(e)}")
        return None 
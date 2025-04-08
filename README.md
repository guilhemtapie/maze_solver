# Maze Solver Project

This project implements a robot that can solve a maze using computer vision and motor control. The robot uses a camera to detect the maze, finds a path through the maze, and controls motors to navigate through it.

## Project Structure

The code has been organized into several modules for better maintainability:

- `maze_solver/`: A Python package containing the maze solver code

  - `motor_control.py`: Functions for initializing and controlling the motors
  - `image_processing.py`: Functions for image acquisition, processing, and analysis
  - `path_finding.py`: Implementation of the A\* algorithm and other path-finding utilities
  - `pid_controller.py`: PID control implementation for smooth motor movement
  - `maze_solver.py`: Main application that ties everything together
  - `__init__.py`: Package initialization file

- `Ax12.py`: Library for controlling AX12 servo motors
- `run_maze_solver.py`: Entry point script to run the maze solver

## Dependencies

See `maze_solver/requirements.txt` for a list of dependencies.

## Hardware Requirements

- Raspberry Pi (or compatible single-board computer)
- Camera module
- Dynamixel AX-12 servo motors
- USB2AX interface (or compatible)

## Usage

1. Connect the hardware components (camera and motors).
2. Ensure all dependencies are installed.
3. Run the main application:

```bash
python run_maze_solver.py
```

Command-line options:

- `--debug`: Enable debug mode with more verbose logging
- `--visualize`: Enable visualization of the maze and path
- `--max-iterations`: Set the maximum number of control loop iterations (default: 1000)
- `--delay`: Set the control loop delay in seconds (default: 0.1)

## How It Works

1. The camera captures an image of the maze.
2. Image processing is used to identify the maze structure, the current position of the ball, and the goal position.
3. The A\* algorithm finds a path through the maze.
4. The PID controller guides the motors to navigate the ball through the maze.
5. The process repeats until the ball reaches the goal.

## Customization

You can adjust various parameters in the code to optimize for your specific setup:

- PID parameters in `pid_controller.py`
- Image processing parameters in `image_processing.py`
- Path finding parameters in `path_finding.py`

## Contributors 🚀

Guihlem Tapie, Ilias El Ganaoui, Vincent Barrier 
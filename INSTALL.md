# Installation Guide

This guide will help you install and set up the Maze Solver project.

## Prerequisites

- Python 3.6 or higher
- pip (Python package installer)
- Raspberry Pi (or compatible single-board computer)
- Camera module
- Dynamixel AX-12 servo motors
- USB2AX interface (or compatible)

## Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/maze_solver.git
   cd maze_solver
   ```

2. Install the package and its dependencies:

   ```bash
   pip install -e .
   ```

   This will install the package in development mode, allowing you to make changes to the code without reinstalling.

3. Connect the hardware:

   - Connect the camera module to the Raspberry Pi
   - Connect the USB2AX interface to the Raspberry Pi
   - Connect the AX-12 servo motors to the USB2AX interface

4. Configure the serial port:

   - Make sure the serial port is accessible by the user:
     ```bash
     sudo usermod -a -G dialout $USER
     ```
   - You may need to log out and log back in for this change to take effect.

5. Test the installation:
   ```bash
   python run_maze_solver.py --debug
   ```
   This should start the maze solver in debug mode. If everything is set up correctly, you should see log messages indicating that the camera and motors have been initialized.

## Troubleshooting

### Camera Issues

If you encounter issues with the camera:

- Make sure the camera is properly connected
- Enable the camera interface in Raspberry Pi configuration:
  ```bash
  sudo raspi-config
  ```
  Navigate to "Interfacing Options" > "Camera" and enable it.

### Motor Issues

If you encounter issues with the motors:

- Check the connections between the motors and the USB2AX interface
- Make sure the USB2AX interface is properly connected to the Raspberry Pi
- Check that the correct serial port is specified in the configuration (default is '/dev/ttyUSB0')
- Ensure that the user has permission to access the serial port

### Import Errors

If you encounter import errors:

- Make sure you have installed the package with `pip install -e .`
- Check that all dependencies are installed
- If you're still having issues, try installing the dependencies manually:
  ```bash
  pip install -r maze_solver/requirements.txt
  ```

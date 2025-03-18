from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("maze_solver/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="maze_solver",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for solving mazes using computer vision and motor control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/maze_solver",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "maze_solver=maze_solver.maze_solver:main",
        ],
    },
) 
# UCSD ECE276B PR3 

## Overview
In this assignment, you will implement a controller for a car robot to track a trajectory.

## Dependencies
This starter code was tested with: python 3.7, matplotlib 3.4, and numpy 1.20. 

First create the conda enviroment and activate it using 
```
conda env create --name proj2 --file=environment.yml
conda activate proj2
```

## Starter code
### 1. main.py
This file contains examples of how to generate control inputs from a simple P controller and apply the control on a car model. This simple controller does not work well. Your task is to replace the P controller with your own controller using CEC and GPI as described in the project statement.

### 2. utils.py
This file contains code to visualize the desired trajectory, robot's trajectory, and obstacles.

### 3. state.py
The file contains code for the motion models

### 4. cec.py
The file contains the code for the CEC controller. It can be run using 
```
python cec.py
```

### 5. gpi.py
The file contains the code for the GPI controller. It can be run using
```
python gpi.py
```


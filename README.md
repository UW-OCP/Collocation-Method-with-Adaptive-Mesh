Collocation Solver Adaptive Mesh Refinement
===
###### tags: `nonlinear optimal control problem`, `boundary value problem`, `collocation method`, `adaptive mesh refinement`, `Python`, `CUDA`

> A software to solve nonlinear optimal control problems using collocation method with ability for adaptively refining the mesh, which implemented in Python and CUDA. 

> The article is under publication. :smile: 

> The instruction of the solver is presented below. :arrow_down: 

## :memo: Set up

The solver needs to run on machine equipped with targeted Nvidia **GPU** and **CUDA** installed. 

The installation of CUDA can be found at https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html.

The solver can be ran on **Python3** with the following necessary packages.

- Python (>=3.5.0)
- Numpy (>=1.11.0)
- Sympy (>=1.1)
- Matplotlib (>=3.00)
- Numba (>=0.48)
- Libgcc
- Cudatoolkit

Set up Python with CUDA with installing needed dependencies by
```bash
sudo apt-get install build-essential freeglut3-dev libxmu-dev libxi-dev git cmake libqt4-dev libphonon-dev libxml2-dev libxslt1-dev libqtwebkit-dev libboost-all-dev python-setuptools libboost-python-dev libboost-thread-dev -y
```

## :incoming_envelope:  Development Guide

The solver can be used with the command in terminal as
```bash
./run_adaptive_collocation.sh ocp_example
```
### Input
- **ocp_example**:  a positional argument which is a plain text file defining the optimal control problem (OCP) to be solved with necessary fields for the solver. Definition syntax can be found in the repo `Multiple_Shooting_Solver_CUDA`. Examples can be found in the repo `ocp_test_problems`.

## About the solver

*The solver is developed by Dynamic Systems Modeling and Controls Laboratory at University of Washington.*

![](https://i.imgur.com/kQSpFjN.png)

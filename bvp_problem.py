# Created by OCP.py
from math import *
from numba import cuda
from BVPDAEReadWriteData import bvpdae_read_data, bvpdae_write_data
import numpy as np


class BvpDae:

	def __init__(self):
		T = 1000
		self.size_y = 2
		self.size_z = 1
		self.size_p = 2
		self.size_inequality = 0
		self.size_sv_inequality = 0
		self.output_file = 'exarao01.data'
		self.tolerance = 1e-06
		self.maximum_nodes = 2000
		self.maximum_newton_iterations = 200
		self.maximum_mesh_refinements = 10
		self.N = 1001
		self.t_initial = 0
		self.t_final = 1
		self.T0 = np.linspace(self.t_initial, self.t_final, self.N)
		self.Y0 = np.ones((self.N, self.size_y), dtype=np.float64)
		self.Z0 = np.ones((self.N, self.size_z), dtype=np.float64)
		self.P0 = np.ones(self.size_p, dtype=np.float64)


@cuda.jit(device=True)
def _abvp_f(_y, _z, _p, _f):
	T = 1000
	y = _y[0]
	_lambda1 = _y[1]
	u = _z[0]
	_f[0] = T*(u - pow(y, 3))
	_f[1] = 3*T*_lambda1*pow(y, 2) - 2*T*y


@cuda.jit(device=True)
def _abvp_g(_y, _z, _p, _alpha, _g):
	T = 1000
	y = _y[0]
	_lambda1 = _y[1]
	u = _z[0]
	_g[0] = T*_lambda1 + 2*T*u


def _abvp_r(_y0, _y1, _p, _r):
	T = 1000
	_kappa_i1 = _p[0]
	_kappa_f1 = _p[1]
	y = _y0[0]
	_lambda1 = _y0[1]
	# initial conditions
	_r[0] = y - 1.0
	_r[1] = _kappa_i1 + _lambda1
	# final conditions
	y = _y1[0]
	_lambda1 = _y1[1]
	_r[2] = y - 1.5
	_r[3] = -_kappa_f1 + _lambda1


@cuda.jit(device=True)
def _abvp_Df(_y, _z, _p, _Df):
	T = 1000
	y = _y[0]
	_lambda1 = _y[1]
	u = _z[0]
	_Df[0][0] = -3*T*pow(y, 2)
	_Df[0][2] = T
	_Df[1][0] = 6*T*_lambda1*y - 2*T
	_Df[1][1] = 3*T*pow(y, 2)


@cuda.jit(device=True)
def _abvp_Dg(_y, _z, _p, _alpha, _Dg):
	T = 1000
	y = _y[0]
	_lambda1 = _y[1]
	u = _z[0]
	_Dg[0][1] = T
	_Dg[0][2] = 2*T


def _abvp_Dr(_y0, _y1, _p, _Dr):
	T = 1000
	_kappa_i1 = _p[0]
	_kappa_f1 = _p[1]
	y = _y0[0]
	_lambda1 = _y0[1]
	# initial conditions
	_Dr[0][0] = 1
	_Dr[1][1] = 1
	_Dr[1][4] = 1
	# final conditions
	y = _y1[0]
	_lambda1 = _y1[1]
	_Dr[2][2] = 1
	_Dr[3][3] = 1
	_Dr[3][5] = -1


def abvp_Dg(_y, _z, _p, _alpha, _Dg):
	T = 1000
	y = _y[0]
	_lambda1 = _y[1]
	u = _z[0]
	_Dg[0][1] = T
	_Dg[0][2] = 2*T


'''

# Rao and Mease , Eigenvector approximate dichotomic basis method 
# for solving hyper-sensitive optimal control ptoblems, 
# Optimal Control Applications and Methods, 23, 2002, pp. 215-238

# This is test problem arao01 (example 8.1) from John T. Betts, A collection of optimal control test problems, November 17, 2015


Constants = [T = 1000]; # solve with T = 10000 and Nodes = 9001
StateVariables = [y];
ControlVariables = [u];
InitialConstraints = [y - 1.0];
TerminalConstraints = [y - 1.5];
CostFunctional = T*(y*y + u*u);
DifferentialEquations = [(-y*y*y + u)*T];
FinalTime = 1;
Nodes = 1001;
Tolerance = 1.0e-6;
OutputFile = "exarao01.data";


'''


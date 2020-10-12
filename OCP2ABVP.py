"""This file implements the OCP class."""
# TODO: remove unused variables in the C code
import sys
from sympy import *
from sympy import ccode

_raw_line_number_ = 0


class OCP(object):
    """The OCP class"""
    def __init__(self):
        self.x = []
        self.u = []
        self.p = []
        self.phi = 0
        self.L = 0
        self.f = []
        self.c = []
        self.d = []
        self.s = []
        self.Gamma = []
        self.Psi = []
        self.t_i = 0
        self.t_f = 1
        self.nodes = 100
        self.input_file = ""
        self.output_file = "ocp.data"
        self.tolerance = 1.0e-6
        self.maximum_nodes = 2000
        self.maximum_newton_iterations = 200
        self.method = "mshootdae"
        self.constants = None
        self.constants_value = None
        self.display = 0
        self.maximum_mesh_refinements = 10
        self.state_estimate = None
        self.control_estimate = None
        self.parameter_estimate = None

    def make_abvp(self):
        """
            Define the variables for OCP problem.
            y : ODE variables, [x lambda gamma]
            z : DAE variables, [u theta mu nu]
                theta : variables for equality constraints
                mu : lagrange multipliers for inequality constraints
                nu : slack variables for inequality constraints
            alpha : continuation parameter
        """

        o = self

        # y = [x, lambda, gamma]
        nx = len(o.x) # number of state variables
        np = len(o.p) # number of paraters
        y = [0] * (2*nx + np)
        for i in range(nx):
            y[i] = o.x[i]
            y[i+nx] = Symbol('_lambda%d' % (i+1))
        for i in range(np):
            y[2*nx+i] = Symbol('_gamma%d' % (i+1))

        # z = [u, theta, mu, nu]
        nu = len(o.u)  # number of control varaibles
        nc = len(o.c)  # number of equality constraints
        nd = len(o.d)  # number of inequality constraints
        ns = len(o.s)  # number of state variable inequality constraints
        z = [0] * (nu + nc + 2*nd + 2*ns)
        for i in range(nu):
            z[i] = o.u[i]
        for i in range(nc):
            z[nu+i] = Symbol('_theta%d' % (i+1))
        for i in range(nd):
            z[nu + nc + i] = Symbol('_mu%d' % (i+1))
            z[nu + nc + nd + ns + i] = Symbol('_nu%d' % (i+1))
        for i in range(ns):
            z[nu + nc + nd + i] = Symbol('_xi%d' % (i+1))
            z[nu + nc + nd + ns + nd + i] = Symbol('_sigma%d' % (i+1))

        # p = [p, nu_i, nu_f]
        ngamma = len(o.Gamma) # number of initial constraints
        npsi = len(o.Psi) # number of final constraints
        p = [0] * (np + ngamma + npsi)
        for i in range(np):
            p[i] = o.p[i]
        for i in range(ngamma):
            p[np+i] = Symbol('_kappa_i%d' % (i+1))
        for i in range(npsi):
            p[np+ngamma+i] = Symbol('_kappa_f%d' % (i+1))

        # alpha : continuation parameter
        _alpha = Symbol('_alpha')

        # form the Hamiltonian
        H = self.L
        for i in range(len(o.x)):
            H = H + y[nx+i]*o.f[i]
        # second order necessary condition term
        if nd > 0 or ns > 0:
            for i in range(len(o.u)):
                H = H + _alpha * o.u[i] * o.u[i] / 2
        # for i in range(len(o.u)):
        #     H = H + _alpha * o.u[i] * o.u[i] / 2
        for i in range(len(o.c)):
            H = H + z[nu+i]*o.c[i]
        for i in range(len(o.d)):
            H = H + z[nu+nc+i]*o.d[i]
        for i in range(len(o.s)):
            H = H + z[nu + nc + nd + i] * o.s[i]

        # print H

        # form the states, co-states and parameters : f
        #
        # f = [H_lambda; -H_x; H_p]
        #
        f = [0] * (2*nx + np)
        for i in range(nx):
            f[i] = diff(H, y[nx+i]) # = o.f[i]
            f[nx+i] = -diff(H, o.x[i])
        for i in range(np):
            f[2*nx+i] = diff(H, o.p[i])

        # form the stationary condition and complementarity conditions : g
        #
        # g = [H_u; H_theta; H_mu+nu; mu*nu]
        #
        g = [0] * (nu + nc + 2 * nd + 2 * ns)
        for i in range(nu):
            g[i] = diff(H, o.u[i])
        for i in range(nc):
            g[nu+i] = diff(H, z[nu+i])  # =  o.c[i]
        for i in range(nd):
            # g[nu+nc+i] = diff(H, z[nu+nc+i]) + z[nu+nc+nd+i]  # = o.d[i] + z[nu+nc+nd+i]
            g[nu + nc + i] = diff(H, z[nu + nc + i]) + z[nu + nc + nd + ns + i] - _alpha * z[nu + nc + i]
            # g[nu+nc+nd+i] = z[nu+nc+i]*z[nu+nc+nd+i]
            g[nu + nc + nd + ns + i] = z[nu + nc + i] + z[nu + nc + nd + ns + i] - \
                                       sqrt(z[nu + nc + i]**2 + z[nu + nc + nd + ns + i]**2 + 2 * _alpha)
        for i in range(ns):
            g[nu + nc + nd + i] = diff(H, z[nu + nc + nd + i]) + \
                                  z[nu + nc + nd + ns + nd + i] - _alpha * z[nu + nc + nd + i]
            g[nu + nc + nd + ns + nd + i] = z[nu + nc + nd + i] + z[nu + nc + nd + ns + nd + i] - \
                                            sqrt(z[nu + nc + nd + i]**2 + z[nu + nc + nd + ns + nd+ i]**2 + 2 * _alpha)

        # form the boundary conditions : r
        #
        # r_initial = [Gamma; lambda + Gamma_x*nu_i; gamma]
        # f_final = [Psi; lambda - phi_x - Psi_x*nu_f; gamma + phi_p + Psi_p*nu_f]
        #
        ri = [0] * (ngamma + nx + np)
        for i in range(ngamma):
            ri[i] = o.Gamma[i]
        Gamma_x_nu_i = [0] * nx
        if ngamma > 0:
            for i in range(nx):
                for j in range(ngamma):
                    Gamma_x_nu_i[i] += diff(o.Gamma[j], y[i]) * p[np+j]
        for i in range(nx):
            ri[ngamma+i] = y[nx+i] + Gamma_x_nu_i[i]
        for i in range(np):
            ri[ngamma+nx+i] = y[2*nx+i]

        rf = [0] * (npsi + nx + np)
        for i in range(npsi):
            rf[i] = o.Psi[i]
        Psi_x_nu_f = [0] * nx
        if npsi > 0:
            for i in range(nx):
                for j in range(npsi):
                    Psi_x_nu_f[i] += diff(o.Psi[j], y[i]) * p[np+ngamma+j]
        for i in range(nx):
            rf[npsi+i] = y[nx+i] - diff(o.phi, y[i]) - Psi_x_nu_f[i]
        if np > 0:
            Psi_p_nu_f = [0] * np
            for i in range(np):
                for j in range(npsi):
                    Psi_p_nu_f[i] += diff(o.Psi[j], p[i]) * p[np+ngamma+j]
            for i in range(np):
                rf[npsi+nx+i] = y[2*nx+i] + diff(o.phi, p[i]) + Psi_p_nu_f[i]

        o.write_abvp_c_code(y, z, p, f, g, ri, rf, len(o.d), len(o.s))

    def write_abvp_c_code(self, y, z, p, f, g, ri, rf, n_ineq, n_svineq):
        o = self
        '''
            construct a BVP_DAE class
        '''

        # constant values defined in the OCP file
        constants = ''
        if o.constants:
            for i in range(len(o.constants)):
                # constants += '\t{0} {1}\n'.format(o.constants[i], o.constants_value[i])
                # add a whitespace after the equal sign in the constant value string
                constants_value = ''
                for j in range(len(o.constants_value[i])):
                    constants_value += o.constants_value[i][j]
                    if o.constants_value[i][j] == '=':
                        constants_value += ' '
                constants += '\t{0} {1}\n'.format(o.constants[i], constants_value)

        # f(y,z,p)
        fg_vars = ''
        for i in range(len(y)):
            fg_vars += '\t{0} = _y[{1}]\n'.format(y[i], i)
        for i in range(len(z)):
            fg_vars += '\t{0} = _z[{1}]\n'.format(z[i], i)
        for i in range(len(o.p)):
            fg_vars += '\t{0} = _p[{1}]\n'.format(o.p[i], i)

        abvp_f = '\n@cuda.jit(device=True)\n'
        abvp_f += 'def _abvp_f(_y, _z, _p, _f):\n'
        if o.constants:
            abvp_f += constants
        abvp_f += fg_vars
        for i in range(len(f)):
            abvp_f += '\t_f[{0}] = {1}\n'.format(i, ccode(f[i]))
        # abvp_f += '\n'

        # g(y,z,p)
        abvp_g = '\n@cuda.jit(device=True)\n'
        abvp_g += 'def _abvp_g(_y, _z, _p, _alpha, _g):\n'
        if o.constants:
            abvp_g += constants
        abvp_g += fg_vars
        for i in range(len(g)):
            abvp_g += '\t_g[{0}] = {1}\n'.format(i, ccode(g[i]))
        # abvp_g += '\n'

        # r(y0, y1, p)
        ri_vars = ''
        for i in range(len(p)):
            ri_vars += '\t{0} = _p[{1}]\n'.format(p[i], i)
        for i in range(len(y)):
            ri_vars += '\t{0} = _y0[{1}]\n'.format(y[i], i)
        abvp_r = '\ndef _abvp_r(_y0, _y1, _p, _r):\n'
        if o.constants:
            abvp_r += constants
        abvp_r += ri_vars
        abvp_r += '\t# initial conditions\n'
        for i in range(len(ri)):
            abvp_r += '\t_r[{0}] = {1}\n'.format(i, ccode(ri[i]))

        rf_vars = ''
        for i in range(len(y)):
            rf_vars += '\t{0} = _y1[{1}]\n'.format(y[i], i)
        nri = len(ri)
        abvp_r += '\t# final conditions\n'
        abvp_r += rf_vars
        for i in range(len(rf)):
            abvp_r += '\t_r[{0}] = {1}\n'.format(i+nri, ccode(rf[i]))
        # abvp_r += '\n'

        # df(y, z, p)
        abvp_Df = '\n@cuda.jit(device=True)\n'
        abvp_Df += 'def _abvp_Df(_y, _z, _p, _Df):\n'
        if o.constants:
            abvp_Df += constants
        abvp_Df += fg_vars
        ny = len(y)
        nz = len(z)
        np = len(p)
        for i in range(len(f)):
            for j in range(len(y)):
                df = diff(f[i], y[j])
                if df != 0:
                    abvp_Df += '\t_Df[{0}][{1}] = {2}\n'.format(i, j, ccode(df))
            for j in range(len(z)):
                df = diff(f[i], z[j])
                if df != 0:
                    abvp_Df += '\t_Df[{0}][{1}] = {2}\n'.format(i, ny+j, ccode(df))
            for j in range(len(p)):
                df = diff(f[i], p[j])
                if df != 0:
                    abvp_Df += '\t_Df[{0}][{1}] = {2}\n'.format(i, ny+nz+j, ccode(df))
        # abvp_Df += '\n'

        # dg(y, z, p)
        abvp_Dg = '\n@cuda.jit(device=True)\n'
        abvp_Dg += 'def _abvp_Dg(_y, _z, _p, _alpha, _Dg):\n'
        abvp_Dg_normal = '\ndef abvp_Dg(_y, _z, _p, _alpha, _Dg):\n'
        if o.constants:
            abvp_Dg += constants
            abvp_Dg_normal += constants
        abvp_Dg += fg_vars
        abvp_Dg_normal += fg_vars
        for i in range(len(g)):
            for j in range(len(y)):
                df = diff(g[i], y[j])
                if df != 0:
                    abvp_Dg += '\t_Dg[{0}][{1}] = {2}\n'.format(i, j, ccode(df))
                    abvp_Dg_normal += '\t_Dg[{0}][{1}] = {2}\n'.format(i, j, ccode(df))
            for j in range(len(z)):
                df = diff(g[i], z[j])
                if df != 0:
                    abvp_Dg += '\t_Dg[{0}][{1}] = {2}\n'.format(i, ny+j, ccode(df))
                    abvp_Dg_normal += '\t_Dg[{0}][{1}] = {2}\n'.format(i, ny+j, ccode(df))
            for j in range(len(p)):
                df = diff(g[i], p[j])
                if df != 0:
                    abvp_Dg += '\t_Dg[{0}][{1}] = {2}\n'.format(i, ny+nz+j, ccode(df))
                    abvp_Dg_normal += '\t_Dg[{0}][{1}] = {2}\n'.format(i, ny+nz+j, ccode(df))
        # abvp_Dg += '\n'

        # dr(y0, y1, p)
        abvp_Dr = '\ndef _abvp_Dr(_y0, _y1, _p, _Dr):\n'
        if o.constants:
            abvp_Dr += constants
        abvp_Dr += ri_vars
        abvp_Dr += '\t# initial conditions\n'
        for i in range(len(ri)):
            for j in range(len(y)):
                dr = diff(ri[i], y[j])
                if dr != 0:
                    abvp_Dr += '\t_Dr[{0}][{1}] = {2}\n'.format(i, j, ccode(dr))
            for j in range(len(p)):
                dr = diff(ri[i], p[j])
                if dr != 0:
                    abvp_Dr += '\t_Dr[{0}][{1}] = {2}\n'.format(i, 2*ny+j, ccode(dr))
        nri = len(ri)
        abvp_Dr += '\t# final conditions\n'
        abvp_Dr += rf_vars
        for i in range(len(rf)):
            for j in range(len(y)):
                dr = diff(rf[i], y[j])
                if dr != 0:
                    abvp_Dr += '\t_Dr[{0}][{1}] = {2}\n'.format(i+nri, ny+j, ccode(dr))
            for j in range(len(p)):
                dr = diff(rf[i], p[j])
                if dr != 0:
                    abvp_Dr += '\t_Dr[{0}][{1}] = {2}\n'.format(i+nri, 2*ny+j, ccode(dr))
        # abvp_Dr += '\n'

        # main()
        abvp_main = '\tdef __init__(self):\n'
        # constant values defined in the constructor function
        solution_constructor_constants = ''
        if o.constants:
            for i in range(len(o.constants)):
                # constants += '\t{0} {1}\n'.format(o.constants[i], o.constants_value[i])
                # add a whitespace after the equal sign in the constant value string
                constants_value = ''
                for j in range(len(o.constants_value[i])):
                    constants_value += o.constants_value[i][j]
                    if o.constants_value[i][j] == '=':
                        constants_value += ' '
                solution_constructor_constants += '\t\t{0} {1}\n'.format(o.constants[i], constants_value)
            abvp_main += solution_constructor_constants
        abvp_main += '\t\tself.size_y = {0}\n'.format(len(y))
        abvp_main += '\t\tself.size_z = {0}\n'.format(len(z))
        abvp_main += '\t\tself.size_p = {0}\n'.format(len(p))
        abvp_main += '\t\tself.size_inequality = {0}\n'.format(n_ineq)
        abvp_main += '\t\tself.size_sv_inequality = {0}\n'.format(n_svineq)
        abvp_main += '\t\tself.output_file = \'{0}\'\n'.format(o.output_file)
        abvp_main += '\t\tself.tolerance = {0}\n'.format(o.tolerance)
        abvp_main += '\t\tself.maximum_nodes = {0}\n'.format(o.maximum_nodes)
        abvp_main += '\t\tself.maximum_newton_iterations = {0}\n'.format(o.maximum_newton_iterations)
        abvp_main += '\t\tself.maximum_mesh_refinements = {0}\n'.format(o.maximum_mesh_refinements)
        # abvp_main += '\t_m->display = {0};\n'.format(o.display)
        if o.input_file != "":
            abvp_main += '\t\terror, t0, y0, z0, p0 = bvpdae_read_data("{0}")\n'.format(o.input_file)
            # if read file fails
            abvp_main += '\t\tif error != 0:\n'
            abvp_main += '\t\t\tprint("Unable to read input file!")\n'
            abvp_main += '\t\t\tself.N = {0}\n'.format(o.nodes)
            abvp_main += '\t\t\tself.t_initial = {0}\n'.format(o.t_i)
            abvp_main += '\t\t\tself.t_final = {0}\n'.format(o.t_f)
            abvp_main += '\t\t\tself.T0 = np.linspace(self.t_initial, self.t_final, self.N)\n'
            abvp_main += '\t\t\tself.Y0 = np.ones((self.N, self.size_y), dtype=np.float64)\n'
            abvp_main += '\t\t\tself.Z0 = np.ones((self.N, self.size_z), dtype=np.float64)\n'
            if np > 0:
                abvp_main += '\t\t\tself.P0 = np.ones((self.size_p), dtype = np.float64)\n'
            else:
                abvp_main += '\t\t\tself.P0 = np.ones((0), dtype=np.float64)\n'
            if o.state_estimate or o.control_estimate or o.parameter_estimate:
                abvp_main += '\t\t\tself._solution_estimate(self.T0, self.Y0, self.Z0, self.P0)\n'
            abvp_main += '\t\tif error == 0:\n'
            abvp_main += '\t\t\tprint("Read input file!")\n'
            abvp_main += '\t\t\tself.N = t0.shape[0]\n'
            abvp_main += '\t\t\tself.T0 = t0\n'
            abvp_main += '\t\t\tself.Y0 = None\n'
            abvp_main += '\t\t\tif self.size_y > 0:\n'
            abvp_main += '\t\t\t\tself.Y0 = np.ones((self.N, self.size_y), dtype=np.float64)\n'
            abvp_main += '\t\t\tself.Z0 = None\n'
            abvp_main += '\t\t\tif self.size_z > 0:\n'
            abvp_main += '\t\t\t\tself.Z0 = np.ones((self.N, self.size_z), dtype=np.float64)\n'
            abvp_main += '\t\t\tself.P0 = None\n'
            abvp_main += '\t\t\tif self.size_p > 0:\n'
            abvp_main += '\t\t\t\tself.P0 = np.ones(self.size_p, dtype=np.float64)\n'
            abvp_main += '\t\t\tself._pack_YZP(self.Y0, self.Z0, self.P0, y0, z0, p0)\n'
            if o.state_estimate or o.control_estimate or o.parameter_estimate:
                abvp_main += '\t\t\tself._solution_estimate(self.T0, self.Y0, self.Z0, self.P0)\n'
        else:
            abvp_main += '\t\tself.N = {0}\n'.format(o.nodes)
            abvp_main += '\t\tself.t_initial = {0}\n'.format(o.t_i)
            abvp_main += '\t\tself.t_final = {0}\n'.format(o.t_f)
            abvp_main += '\t\tself.T0 = np.linspace(self.t_initial, self.t_final, self.N)\n'
            abvp_main += '\t\tself.Y0 = np.ones((self.N, self.size_y), dtype=np.float64)\n'
            abvp_main += '\t\tself.Z0 = np.ones((self.N, self.size_z), dtype=np.float64)\n'
            if np > 0:
                abvp_main += '\t\tself.P0 = np.ones(self.size_p, dtype=np.float64)\n'
            else:
                abvp_main += '\t\tself.P0 = np.ones((0), dtype = np.float64)\n'
            if o.state_estimate or o.control_estimate or o.parameter_estimate:
                abvp_main += '\t\tself._solution_estimate(self.T0, self.Y0, self.Z0, self.P0)\n'

        # pack data from the input file
        if o.input_file != "":
            abvp_main += '\n'
            abvp_main += '\tdef _pack_YZP(self, _Y, _Z, _P, y0, z0, p0):\n'
            abvp_main += '\t\tif _Y is not None and y0 is not None:\n'
            # abvp_main += '\t\t\t_n = y0.shape[0] if y0.shape[0] < _Y.shape[0] else _Y.shape[0]\n'
            abvp_main += '\t\t\t_n = self.N\n'
            abvp_main += '\t\t\t_m = y0.shape[1] if y0.shape[1] < _Y.shape[1] else _Y.shape[1]\n'
            abvp_main += '\t\t\tfor i in range(_n):\n'
            abvp_main += '\t\t\t\tfor j in range(_m):\n'
            abvp_main += '\t\t\t\t\t_Y[i][j] = y0[i][j]\n'
            abvp_main += '\t\tif _Z is not None and z0 is not None:\n'
            # abvp_main += '\t\t\t_n = z0.shape[0] if z0.shape[0] < _Z.shape[0] else _Z.shape[0]\n'
            abvp_main += '\t\t\t_n = self.N\n'
            abvp_main += '\t\t\t# only read in enough dtat to fill the controls\n'
            abvp_main += '\t\t\t_m = z0.shape[1] if z0.shape[1] < {0} else {1}\n'.format(len(o.u), len(o.u))
            abvp_main += '\t\t\tfor i in range(_n):\n'
            abvp_main += '\t\t\t\tfor j in range(_m):\n'
            abvp_main += '\t\t\t\t\t_Z[i][j] = z0[i][j]\n'
            abvp_main += '\t\tif _P is not None and p0 is not None:\n'
            abvp_main += '\t\t\t_n = p0.shape[0] if p0.shape[0] < _P.shape[0] else _P.shape[0]\n'
            abvp_main += '\t\t\tfor i in range(_n):\n'
            abvp_main += '\t\t\t\t_P[i] = p0[i]\n'
        # initial solution estimate
        if o.state_estimate or o.control_estimate or o.parameter_estimate:
            abvp_main += '\n'
            abvp_main += '\tdef _solution_estimate(self, _T, _Y, _Z, _P):\n'
            # constants defined in the constructor part
            abvp_main += solution_constructor_constants
            abvp_main += '\t\tN = _T.shape[0]\n'
            abvp_main += '\t\tfor i in range(N):\n'
            abvp_main += '\t\t\tt = _T[i]\n'
            if o.state_estimate:
                for j in range(len(o.state_estimate)):
                    abvp_main += '\t\t\t_Y[i][{0}] = {1}\n'.format(j, o.state_estimate[j])
            if o.control_estimate:
                for j in range(len(o.control_estimate)):
                    abvp_main += '\t\t\t_Z[i][{0}] = {1}\n'.format(j, o.control_estimate[j])
            abvp_main += '\n'
            abvp_main += '\t\tif _P.shape[0] != 0:\n'
            abvp_main += '\t\t\tfor i in range(self.size_p):\n'
            if o.parameter_estimate:
                for j in range(len(o.parameter_estimate)):
                    abvp_main += '\t\t\t\t_P[{0}] = {1}\n'.format(j, o.parameter_estimate[j])
            else:
                abvp_main += '\t\t\t\t_P0 = np.ones(self.size_p, dtype=np.float64)\n'

        # header, import necessary packages and class header
        abvp_header = '# Created by OCP.py\n'
        abvp_header += 'from math import *\n'
        abvp_header += 'from numba import cuda\n'
        abvp_header += 'from BVPDAEReadWriteData import bvpdae_read_data, bvpdae_write_data\n'
        abvp_header += 'import numpy as np\n\n\n'
        abvp_header += 'class BvpDae:\n'

        '''
        abvp_header += 'double _rho_ = 1.0e3;\n\n'
        abvp_header += 'double _mu_ = 1.0e-1;\n\n'
        '''
        print(abvp_header)
        print(abvp_main)
        print(abvp_f)
        print(abvp_g)
        print(abvp_r)
        print(abvp_Df)
        print(abvp_Dg)
        print(abvp_Dr)
        print(abvp_Dg_normal)
        # print (abvp_main)


def _make_variables_py0(line, typ):
    rline = ''
    k = 0
    np = line.count(',') + 1
    rline += '_o.%s = [0] * %d\n' % (typ,np)
    i = line.find('[')
    while (line[i] != ']'):
        p = ''
        while ((line[i + 1] != ',')
                and (line[i + 1] != ']')):
            p += line[i + 1]
            i += 1
        rline += '%s = Symbol(\'%s\')\n' % (p, p)
        rline += '_o.%s[%d] = %s\n' % (typ, k, p)
        i += 1
        k += 1

    return (rline, k)


def _make_constants_py0(line):
    global _raw_line_number_
    rline = ''
    k = 0
    np = line.count(',') + 1
    rline += '_o.constants = [0] * %d\n' % np
    rline += '_o.constants_value = [0] * %d\n' % np
    i = line.find('[')
    try:
        while line[i] != ']':
            p = ''
            pv = ''
            while line[i + 1] != '=':  # get the constants
                p += line[i + 1]
                i += 1
            while ((line[i + 1] != ',')
                    and (line[i + 1] != ']')):  # get the value
                pv += line[i + 1]
                i += 1
            rline += '%s = Symbol(\'%s\')\n' % (p, p)
            rline += '_o.constants[%d] = \'%s\'\n' % (k, p)
            rline += '_o.constants_value[%d] = \'%s\'\n' % (k, pv)
            i += 1
            k += 1
    except:
        print('Syntax error on line: ', _raw_line_number_)
        print('Exception: ', sys.exc_type, sys.exc_value)
        raise SyntaxError

    return rline


def _line_to_ocp(line, contain_variable, variable_set, variable_dict):
    global _n_, _m1_, _m2_, _m3_, _m4_
    rline = ''
    if line.find('Constants') == 0:
        rline = '# ' + line + '\n'
        vline = _make_constants_py0(line)
        rline += vline
    elif line.find('StateVariables') == 0:
        rline = '# ' + line + '\n'
        (vline, _n_) = _make_variables_py0(line, 'x')
        rline += vline
    elif line.find('ControlVariables') == 0:
        rline = '# ' + line + '\n'
        (vline, _n_) = _make_variables_py0(line, 'u')
        rline += vline
    elif line.find('ParameterVariables') == 0:
        rline = '# ' + line + '\n'
        (vline, _n_) = _make_variables_py0(line, 'p')
        rline += vline
    elif line.find('TerminalPenalty') == 0:
        line = substitute_variable(line, contain_variable, variable_set, variable_dict)
        rline = line.replace('TerminalPenalty', '_o.phi')
    elif line.find('CostFunctional') == 0:
        line = substitute_variable(line, contain_variable, variable_set, variable_dict)
        rline = line.replace('CostFunctional', '_o.L')
    elif line.find('InitialConstraints') == 0:
        line = substitute_variable(line, contain_variable, variable_set, variable_dict)
        rline = line.replace('InitialConstraints', '_o.Gamma')
    elif line.find('TerminalConstraints') == 0:
        line = substitute_variable(line, contain_variable, variable_set, variable_dict)
        rline = line.replace('TerminalConstraints', '_o.Psi')
    elif line.find('DifferentialEquations') == 0:
        line = substitute_variable(line, contain_variable, variable_set, variable_dict)
        rline = line.replace('DifferentialEquations', '_o.f')
    elif line.find('InequalityConstraints') == 0:
        line = substitute_variable(line, contain_variable, variable_set, variable_dict)
        rline = line.replace('InequalityConstraints', '_o.d')
    elif line.find('StateVariableInequalityConstraints') == 0:
        line = substitute_variable(line, contain_variable, variable_set, variable_dict)
        rline = line.replace('StateVariableInequalityConstraints', '_o.s')
    elif line.find('EqualityConstraints') == 0:
        line = substitute_variable(line, contain_variable, variable_set, variable_dict)
        rline = line.replace('EqualityConstraints', '_o.c')
    elif line.find('InitialTime') == 0:
        rline = line.replace('InitialTime', '_o.t_i')
    elif line.find('FinalTime') == 0:
        rline = line.replace('FinalTime', '_o.t_f')
    elif line.find('Nodes') == 0:
        rline = line.replace('Nodes', '_o.nodes')
    elif line.find('Tolerance') == 0:
        rline = line.replace('Tolerance', '_o.tolerance')
    elif line.find('InputFile') == 0:
        rline = line.replace('InputFile', '_o.input_file')
    elif line.find('OutputFile') == 0:
        rline = line.replace('OutputFile', '_o.output_file')
    elif line.find('Display') == 0:
        rline = line.replace('Display', '_o.display')
    elif line.find('MaximumMeshRefinements') == 0:
        rline = line.replace('MaximumMeshRefinements', '_o.maximum_mesh_refinements')
    elif line.find('MaximumNewtonIterations') == 0:
        rline = line.replace('MaximumNewtonIterations', '_o.maximum_newton_iterations')
    elif line.find('MaximumNodes') == 0:
        rline = line.replace('MaximumNodes', '_o.maximum_nodes')
    elif line.find('StateEstimate') == 0:
        rline = line.replace('StateEstimate', '_o.state_estimate')
    elif line.find('ControlEstimate') == 0:
        rline = line.replace('ControlEstimate', '_o.control_estimate')
    elif line.find('ParameterEstimate') == 0:
        rline = line.replace('ParameterEstimate', '_o.parameter_estimate')
    elif line.find('Variables') == 0:
        rline = '# ' + line + '\n'
    else:
        rline = line
    return rline


def _ocp_translate(inpt, typ):
    """
        Translate the OCP script file
        If typ == 0 -> C MSHOOTDAE
        If typ == 1 -> C RKOCP
        If typ == 2 -> C ROWOCP
        If typ == 3 -> FORTRAN TOMP
    """
    # check whether the file contains variables
    contain_variable, variable_number, variable_set, variable_dict = make_variable_dict(inpt)

    global _raw_line_number_
    fid = open(inpt, 'r')
    s = ''
    rawline = ''
    # raw lines from the input file
    _raw_lines_ = ''
    while 1:
        line = fid.readline()
        _raw_lines_ += line
        # print (_raw_lines_)
        _raw_line_number_ += 1
        if not line:
            break
        indx = line.find('#')
        if (indx < 0):
            rline = line
        else:
            rline = line[0: indx]
        for i in range(len(rline)):
            # ignore newline, return and white spaces
            if (rline[i] != '\n') and (rline[i] != '\r') and (rline[i] != '\t') and (rline[i] != ' '):
                rawline += rline[i]
                if rline[i] == ';':
                    if contain_variable:
                        # if the raw line contains the variable definition, comment it out
                        for variable in variable_set:
                            # comment out the variable definition line and make sure
                            if rawline.find(variable) == 0 and \
                                    rawline[0: len(variable) + 1] == variable + '=':
                                rawline = "# " + rawline
                                break
                    # print rawline
                    rawline = _line_to_ocp(rawline, contain_variable, variable_set, variable_dict)
                    # print rawline
                    s += rawline + '\n'
                    rawline = ''
    r = ''
    r += 'from OCP2ABVP import *\n'
    r += 'NO = 0\n'
    r += 'YES = 1\n'
    r += 't = Symbol(\'t\')\n'
    r += '_o = OCP()\n'
    s += '_o.make_abvp()\n'
    t = '%s\n%s' % (r, s)

    # print('r:')
    # print(r)
    # print('r done.')
    # print('s:')
    # print(s)
    # print('s done.')
    # print('t')
    # print(t)
    # print('t done')
    # print('Raw lines:')
    # print(_raw_lines_)
    # print('raw lines done!')

    exec(t)
    print('\n\'\'\'\n')
    print(_raw_lines_)
    print('\'\'\'\n')


# deal with variables
def substitute_variable(rawline, contain_variable, variable_set, variable_dict):
    if contain_variable:
        # if ocp contains variable, substitute the variable in the expression first
        for variable in variable_set:
            if variable in variable_dict:
                # rawline = rawline.replace(variable, variable_dict[variable])
                # make sure the replaced string only contains the variable string
                if rawline.find(variable) >= 0:
                    ii = 0
                    while ii + len(variable) < len(rawline):
                        if rawline[ii] == variable[0]:
                            if rawline[ii: ii + len(variable)] == variable and \
                                    (ii + len(variable) >= len(rawline) or
                                     (not rawline[ii + len(variable)].isalnum())) and \
                                    (ii == 0 or (not rawline[ii - 1].isalnum())):
                                rawline = rawline[0: ii] + variable_dict[variable] + \
                                          rawline[ii + len(variable):]
                                ii += len(variable)
                        ii += 1
    return rawline


def make_variable_set(line):
    variable_set = set()
    i = line.find('[')
    while line[i] != ']':
        p = ''
        while (line[i + 1] != ',') and (line[i + 1] != ']'):
            p += line[i + 1]
            i += 1
        variable_set.add(p)
        i += 1
    return variable_set


def clean_variable_dict(variable_set, variable_dict):
    # recursively substitute the expression in the key of other variables
    for variable in variable_set:
        substitute_key(variable, variable_set, variable_dict)


def substitute_key(cur_variable, variable_set, variable_dict):
    for variable in variable_set:
        if cur_variable in variable_dict and cur_variable != variable:
            cur_key = variable_dict[cur_variable]
            if cur_key.find(variable) >= 0:
                ii = 0
                while ii + len(variable) < len(variable_dict[cur_variable]):
                    if variable_dict[cur_variable][ii] == variable[0]:
                        # make sure the character before and after are not letter or numbers
                        # so that the  varaible is the real variable alone
                        if variable_dict[cur_variable][ii: ii + len(variable)] == variable and (
                                ii + len(variable) >= len(variable_dict[cur_variable]) or (
                                not variable_dict[cur_variable][ii + len(variable)].isalnum())) and (
                                ii == 0 or (
                                not variable_dict[cur_variable][ii - 1].isalnum())):
                            substitute_key(variable, variable_set, variable_dict)
                            new_key = variable_dict[cur_variable].replace(variable, variable_dict[variable])
                            variable_dict[cur_variable] = new_key
                            ii += len(variable)
                    ii += 1
                # substitute_key(variable, variable_set, variable_dict)
                # new_key = variable_dict[cur_variable].replace(variable, variable_dict[variable])
                # variable_dict[cur_variable] = new_key
    return


def make_variable_dict(fname):
    raw_line_num = 0
    contain_variable = False
    variable_number = 0
    variable_set = None
    variable_dict = {}
    with open(fname, 'r') as f:
        for line in f:
            raw_line = ''
            raw_line_num += 1
            # print(line, end='')
            index = line.find('#')
            if index < 0:
                rline = line
            else:
                rline = line[0: index]
            for i in range(len(rline)):
                # ignore newline, return and white spaces
                if (rline[i] != '\n') and (rline[i] != '\r') and (rline[i] != '\t') and (rline[i] != ' '):
                    raw_line += rline[i]
                    if rline[i] == ';':
                        if raw_line.find('Variables') == 0:
                            contain_variable = True
                            variable_number = line.count(',') + 1
                            # make the set for all the variable strings
                            variable_set = make_variable_set(raw_line)
                        elif contain_variable:
                            # check if the line contains the variable
                            for variable in variable_set:
                                if raw_line.find(variable) == 0:
                                    i = raw_line.find('=')
                                    if i >= 0 and raw_line[0: len(variable)] == variable and raw_line[
                                            len(variable)] == '=':
                                        # expression may contain the variable and that line should be ignored
                                        j = raw_line.find(';')
                                        variable_dict[variable] = '(' + raw_line[i + 1: j] + ')'
                        raw_line = ''
    if contain_variable:
        # clean the variable dict recursively
        clean_variable_dict(variable_set, variable_dict)
    return contain_variable, variable_number, variable_set, variable_dict


if __name__ == '__main__':
    try:
        ocp = sys.argv[1]
    except:
        print('Unable to read input file')
        print('Exception: ', sys.exc_type, sys.exc_value)

    _ocp_translate(ocp, 0)

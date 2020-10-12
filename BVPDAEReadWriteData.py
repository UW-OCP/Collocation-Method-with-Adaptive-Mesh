import numpy as np


"""
    Input:
        fname: a string which is the file name to read
    Output:
        If the file contains the right content, returns t0, y0, z0, p0 of the BVP-DAEs
        error: 0 if the file exists
               1 if the file does not exist
"""


def bvpdae_read_data(fname):
    try:
        with open(fname, 'r') as f:
            error = 0
            # read the number of time nodes and the time span variable
            line = f.readline()
            line = f.readline()
            N = int(line)
            line = f.readline()
            line = f.readline()
            list_tspan = line.rstrip().split(' ')
            t0 = np.zeros(N, dtype=np.float64)
            for i in range(N):
                t0[i] = float(list_tspan[i])
            # read the number of y variables and the y variables
            line = f.readline()
            line = f.readline()
            n_y = int(line)
            line = f.readline()
            y0 = np.zeros((N, n_y), dtype=np.float64)
            for i in range(N):
                line = f.readline()
                list_y0 = line.rstrip().split(' ')
                for j in range(n_y):
                    y0[i, j] = float(list_y0[j])
            # read the number of y variables and the y variables
            line = f.readline()
            line = f.readline()
            n_z = int(line)
            line = f.readline()
            z0 = np.zeros((N, n_z), dtype=np.float64)
            for i in range(N):
                line = f.readline()
                list_z0 = line.rstrip().split(' ')
                for j in range(n_z):
                    z0[i, j] = float(list_z0[j])
            # read the number of p variables and the p variables
            line = f.readline()
            line = f.readline()
            n_p = int(line)
            line = f.readline()
            line = f.readline()
            list_p0 = line.rstrip().split(' ')
            p0 = np.zeros(n_p, dtype = np.float64)
            for i in range(n_p):
                p0[i] = float(list_p0[i])
    except:
        error = 1
        t0 = []
        y0 = []
        z0 = []
        p0 =[]
    return error, t0, y0, z0, p0


def bvpdae_write_data(fname, N, n_y, n_z, n_p, t_span, y0, z0, p0):
    try:
        with open(fname, 'w') as f:
            error = 0
            # write number of time nodes and the time span in one line
            f.write('nt:\n{}\nt:\n'.format(N))
            for i in range(N):
                f.write('{} '.format(t_span[i]))
            f.write('\n')
            # write number of y variables and the y variables
            f.write('ny:\n{}\ny:\n'.format(n_y))
            for i in range(N):
                for j in range(n_y):
                    f.write('{} '.format(y0[i, j]))
                f.write('\n')
            # write number of z variables and the z variables
            f.write('nz:\n{}\nz:\n'.format(n_z))
            for i in range(N):
                for j in range(n_z):
                    f.write('{} '.format(z0[i, j]))
                f.write('\n')
            # write number of p variables and the p variables in one line
            f.write('np:\n{}\np:\n'.format(n_p))
            for i in range(n_p):
                f.write('{} '.format(p0[i]))
    except:
        error = 1
    return error

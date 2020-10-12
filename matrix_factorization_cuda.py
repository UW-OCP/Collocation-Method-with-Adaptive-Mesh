from math import *
from numba import cuda, jit


@cuda.jit(device=True)
def qr(a, cpy, q, r, v, vv, beta, w_t, u_t, eps):
    """
        QR factorization of self via Householder transformation.
        Returns Q, R such that a = Q*R.
        a : matrix, size: n x m
        cpy : matrix, size: n x m
        q : matrix, size: n x n
        r: matrix, size n x m
        v : matrix, size: n x m
        vv: vector, size: n(max size)
        beta : vector, size: m
        w_t: vector, size: m(max size)
        u_t: vector, size: n(max size)
    """
    n, m = cpy.shape
    if m > n:
        raise TypeError('qr requires a matrix with columns <= rows')
    # copy a matrix
    for i in range(n):
        for j in range(m):
            cpy[i, j] = a[i, j]
    col = m
    # set q as the identity matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                q[i, j] = 1
            else:
                q[i, j] = 0
    # process each column
    for k in range(0, m):
        for i in range(k, n):
            vv[i] = cpy[i, k]
        if vv[k] >= 0:
            s = 1.0
        else:
            s = -1.0
        vv_dot = 0
        for i in range(k, n):
            vv_dot += vv[i] * vv[i]
        g = -s * sqrt(vv_dot)
        vv[k] = vv[k] - g
        vs = 0
        for i in range(k, n):
            vs += vv[i] * vv[i]
        if vs < eps:
            col = k
            break
        b = -2.0 / vs
        for i in range(k, m):
            w_t[i] = 0
            for j in range(k, n):
                w_t[i] += vv[j] * cpy[j, i]
        for i in range(k, n):
            for j in range(k, m):
                cpy[i, j] += b * vv[i] * w_t[j]
        beta[k] = b
        for i in range(k, n):
            v[i, k] = vv[i]
    for i in range(m):
        for j in range(m):
            r[i, j] = cpy[i, j]
    # set the other elements as 0s
    for i in range(m, n):
        for j in range(m):
            r[i, j] = 0
    for k in range(col - 1, -1, -1):
        for i in range(k, n):
            vv[i] = v[i, k]
        for i in range(k, n):
            u_t[i] = 0
            for j in range(k, n):
                u_t[i] += vv[j] * q[j, i]
        for i in range(k, n):
            for j in range(k, n):
                q[i, j] += beta[k] * vv[i] * u_t[j]
    return


'''
        LU factorization of self.  Returns L, U, P.
        P * a = L * U
'''


@cuda.jit(device=True)
def lu(a, cpy, P, L, U, eps):
    n, m = a.shape
    n_P, m_P = P.shape
    n_L, m_L = L.shape
    n_U, m_U = U.shape
    if n != m or n_P != m_P or n != n_P or n_L != m_L or n != n_L or n_U != m_U or n != n_U:
        raise TypeError('The input to lu should be all square matrices!')
    # copy a
    for i in range(n):
        for j in range(m):
            cpy[i, j] = a[i, j]
    # P = np.eye(n)
    # set P as the identity matrix
    for i in range(n_P):
        for j in range(m_P):
            if i == j:
                P[i, j] = 1
            else:
                P[i, j] = 0
    # L = np.zeros((n, n), dtype=np.float64)
    # set L as the zero matrix
    for i in range(n_L):
        for j in range(m_L):
            L[i, j] = 0
    # U = np.zeros((n, n), dtype=np.float64)
    for i in range(n_U):
        for j in range(m_U):
            U[i, j] = 0
    # for each column
    for j in range(n):
        # for i in xrange(j):
        for i in range(j):
            s = 0
            # for k in xrange(i):
            for k in range(i):
                # s = s + L.element[i][k] * U.element[k][j]
                s += L[i, k] * U[k, j]
            U[i, j] = cpy[i, j] - s
        max_u = 0
        row_max = j
        # for i in xrange(j, n):
        for i in range(j, n):
            s = 0
            # for k in xrange(j):
            for k in range(j):
                # s = s + L.element[i][k] * U.element[k][j]
                s += L[i, k] * U[k, j]
            U[i, j] = cpy[i, j] - s
            if i == j:
                max_u = float(abs(U[i, i]))
            else:
                max_ui = float(abs(U[i, j]))
                if max_ui > max_u:
                    max_u = max_ui
                    row_max = i
        if row_max != j:  # FIX: this moves too much irrelevant data
            # cpy.swap_rows(j + 1, row_max + 1)  # set U = A
            swap_rows_mat(cpy, j, row_max)
            # L.swap_rows(j + 1, row_max + 1)  # only need to swap columns 1 to j-1
            swap_rows_mat(L, j, row_max)
            # U.swap_rows(j + 1, row_max + 1)  # don't need this if U = A
            swap_rows_mat(U, j, row_max)
            # P.swap_rows(j + 1, row_max + 1)  # store P as a 1d array
            swap_rows_mat(P, j, row_max)
        max_u = U[j, j]
        if float(abs(max_u)) < 10.0 * eps:
            print('lu(): singular matrix')
        # for i in xrange(j + 1, n):
        for i in range(j + 1, n):
            if float(abs(max_u)) >= 10.0 * eps:
                L[i, j] = float(U[i, j]) / float(max_u)
            U[i, j] = 0
        L[j, j] = 1
    return


'''
    Swap the elements of the ith and jth row of the matrix a.
'''


@cuda.jit(device=True)
def swap_rows_mat(a, i, j):
    n, m = a.shape
    if i == j:
        return
    for k in range(m):
        tmp = a[i, k]
        a[i, k] = a[j, k]
        a[j, k] = tmp
    return


'''
    Swap the elements of the ith and jth row of the vector a.
'''


@cuda.jit(device=True)
def swap_rows_vec(a, i, j):
    n = a.shape
    if len(n) > 1:
        raise TypeError('The input should be a vector!')
    if i == j:
        return
    tmp = a[i]
    a[i] = a[j]
    a[j] = tmp
    return


'''
    Forward substitution.
    Solve the linear system: Ly = b.
    L is assumed to be lower triangular. (Sanity check)
    b is assumed to be a vector.
    y is assumed to be a vector.
'''


@cuda.jit(device=True)
def forward_solve_vec(L, b, y, eps):
    n, m = L.shape
    n_b = b.shape
    n_y = y.shape
    min_diag = 1
    for i in range(n):
        min_diag = min(min_diag, abs(L[i, i]))
    if len(n_b) > 1 or len(n_y) > 1:
        raise TypeError('The input should be a vector in forward_solve_vec!')
    if (n != m) or (n_b[0] != n) or (m != n_y[0]) or (min_diag < eps):
        raise TypeError('Invalid input to forward_solve!')
    for i in range(n_b[0]):
        # zero y vector
        y[i] = 0
        for k in range(i):
            y[i] += L[i, k] * y[k]
        y[i] = float(b[i] - y[i]) / float(L[i, i])
    return


'''
    Backward substitution.
    Solve the linear system: Ux = y.
    U is assumed to be upper triangular. (Sanity check)
    y is assumed to be a vector.
    x is assumed to be a vector.
'''


@cuda.jit(device=True)
def backward_solve_vec(U, y, x, eps):
    n, m = U.shape
    n_y = y.shape
    n_x = x.shape
    min_diag = 1
    for i in range(n):
        min_diag = min(min_diag, abs(U[i, i]))
    if len(n_y) > 1 or (len(n_x) > 1):
        raise TypeError('The input should be a vector in backward_solve_vec!')
    if (n != m) or (n_y[0] != n) or (m != n_x[0]) or (min_diag < eps):
        raise TypeError('Invalid input to backward_solve!')
    # for i in xrange(y.rows - 1, -1, -1):
    for i in range(n_y[0] - 1, -1, -1):
        # zero x vector
        x[i] = 0
        for k in range(i, n_y[0]):
            x[i] += U[i, k] * x[k]
        x[i] = float(y[i] - x[i]) / float(U[i, i])
    return


'''
    Forward substitution.
    Solve the linear system: Lx = b.
    L is assumed to be lower triangular.
    b is assumed to be a matrix.
    x is assumed to be a matrix.
'''


@cuda.jit(device=True)
def forward_solve_mat(L, b, y, eps):
    n_L, m_L = L. shape
    n_b, m_b = b.shape
    n_y, m_y = y.shape
    min_diag = 1
    for i in range(n_L):
        min_diag = min(min_diag, abs(L[i, i]))
    if (n_L != m_L) or (n_b != n_L) or (m_L != n_y) or (min_diag < eps):
        raise TypeError('Invalid input to forward_solve_mat!')
    # solve each column
    for j in range(m_b):
        for i in range(n_b):
            # zero y matrix
            y[i, j] = 0
            for k in range(i):
                y[i, j] += L[i, k] * y[k, j]
            y[i, j] = float(b[i, j] - y[i, j]) / float(L[i, i])
    return


'''
    Backward substitution.
    Solve the linear system: Ux = y.
    U is assumed to be upper triangular.
    y is assumed to be a matrix.
'''


@cuda.jit(device=True)
def backward_solve_mat(U, y, x, eps):
    n_U, m_U = U.shape
    n_y, m_y = y.shape
    n_x, m_x = x.shape
    min_diag = 1
    for i in range(n_U):
        min_diag = min(min_diag, abs(U[i, i]))
    if (n_U != m_U) or (n_y != n_U) or (m_U != n_x) or (min_diag < eps):
        raise TypeError('Invalid input to backward_solve_mat!')
    for j in range(m_y):
        for i in range(n_y - 1, -1, -1):
            x[i, j] = 0
            for k in range(i, n_y):
                x[i, j] += U[i, k] * x[k, j]
            x[i, j] = float(y[i, j] - x[i, j]) / float(U[i, i])
    return


'''
    solve the linear system A*x = b
    b is supposed to be a vector.
    x is supposed to be a vector.
'''


@cuda.jit(device=True)
def lu_solve_vec(A, cpy, P, L, U, b, c, y, x, eps):
    n_A, m_A = A.shape
    n_b = b.shape
    n_c = c.shape
    n_y = y.shape
    n_x = x.shape
    if len(n_b) > 1 or len(n_c) > 1 or len(n_y) > 1 or len(n_x) > 1:
        raise TypeError('The input to lu_solve_vec should be a vector!')
    if n_A != n_b[0]:
        raise TypeError('Invalid input to lu_solver_vec!')
    lu(A, cpy, P, L, U, eps)
    for i in range(n_A):
        c[i] = 0
        for j in range(m_A):
            c[i] += P[i, j] * b[j]
    # forward_solve_vec(L, b, y, eps)
    forward_solve_vec(L, c, y, eps)
    backward_solve_vec(U, y, x, eps)
    return


'''
    solve the linear system A*x = b
    b is supposed to be a matrix.
    x is supposed to be a matrix.
'''


@cuda.jit(device=True)
def lu_solve_mat(A, cpy, P, L, U, b, c, y, x, eps):
    n_A, m_A = A.shape
    n_b, m_b = b.shape
    n_c, m_c = c.shape
    n_y, m_y = y.shape
    n_x, m_x = x.shape
    if n_A != n_b or m_A != n_x or m_b != m_x or n_x != n_y or m_A != n_c:
        raise TypeError('Invalid input to lu_solver_vec!')
    lu(A, cpy, P, L, U, eps)
    for i in range(n_A):
        for j in range(m_A):
            c[i, j] = 0
            for k in range(n_b):
                c[i, j] += P[i, k] * b[k, j]
    forward_solve_mat(L, c, y, eps)
    backward_solve_mat(U, y, x, eps)
    return

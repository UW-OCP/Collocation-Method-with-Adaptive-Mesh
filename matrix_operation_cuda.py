from numba import cuda


'''
    Self implemented various matrix operations that can be ran on CUDA device.
'''


'''
    Perform the matrix multiplication on device.
    Input matrix: d_a, d_b
    Output matrix: d_c = np.dot(d_a, d_b)
'''


@cuda.jit(device=True)
def mat_mul(d_a, d_b, d_c):
    r_a, c_a = d_a.shape
    r_b, c_b = d_b.shape
    r_c, c_c = d_c.shape
    if c_a != r_b or r_a != r_c or c_b != c_c:
        raise TypeError('Matrix dimensions are not consistent!')
    for i in range(r_a):
        for j in range(c_b):
            d_c[i, j] = 0
            for k in range(c_a):
                d_c[i, j] += d_a[i, k] * d_b[k, j]
    return


'''
    Perform the matrix vector multiplication on device.
    Input matrix: d_a
    Input vector: d_b
    Output matrix: d_c = np.dot(d_a, d_b)
'''


@cuda.jit(device=True)
def mat_vec_mul(d_a, d_b, d_c):
    r_a, c_a = d_a.shape
    r_b = d_b.shape
    r_c = d_c.shape
    if c_a != r_b[0] or r_a != r_c[0]:
        raise TypeError('Matrix and vector dimensions are not consistent!')
    for i in range(r_a):
        d_c[i] = 0
        for j in range(c_a):
            d_c[i] += d_a[i, j] * d_b[j]
    return


'''
    set equal to the two matrices
    Input matrix: d_a
    Output matrix: d_b
'''
@cuda.jit(device=True)
def set_equal_mat(d_a, d_b):
    r_a, c_a = d_a.shape
    r_b, c_b = d_b.shape
    if r_a != r_b or c_a != c_b:
        raise TypeError('Matrix dimensions are not consistent!')
    for i in range(r_b):
        for j in range(c_b):
            d_b[i, j] = d_a[i, j]
    return


'''
    Set equal to the two vectors
    Input vector: d_a
    Output vector: d_b
'''
@cuda.jit(device=True)
def set_equal_vec(d_a, d_b):
    l_a = d_a.shape
    l_b = d_b.shape
    if l_a != l_b:
        raise TypeError('Vector dimensions are not consistent!')
    for i in range(l_b[0]):
        d_b[i] = d_a[i]
    return


'''
    Vertically stack the two matrices to the result matrix.
    Input:
        d_a: top matrix
        d_b: bottom matrix
        d_c: result matrix from the two matrices
'''
@cuda.jit(device=True)
def vstack(d_a, d_b, d_c):
    r_a, c_a = d_a.shape
    r_b, c_b = d_b.shape
    r_c, c_c = d_c.shape
    if c_a != c_b or c_b != c_c or r_c != (r_a + r_b):
        raise TypeError('Matrix dimensions are not consistent!')
    # set the d_a matrix first
    for i in range(r_a):
        for j in range(c_a):
            d_c[i, j] = d_a[i, j]
    # set the d_b matrix
    for i in range(r_b):
        for j in range(c_b):
            d_c[r_a + i, j] = d_b[i, j]
    return


'''
    Perform the matrix transpose on device.
    Input matrix: d_a
    Transpose matrix: d_b, d_b = d_a.T
'''
@cuda.jit(device=True)
def mat_trans(d_a, d_b):
    r_a, c_a = d_a.shape
    r_b, c_b = d_b.shape
    if r_a != c_b or c_a != r_b:
        raise TypeError('Matrix dimensions are not consistent!')
    for i in range(r_b):
        for j in range(r_a):
            d_b[i, j] = d_a[j, i]
    return


'''
    Block matrix multiplication with the top zero block.
    Input matrix:
        d_q: big matrix of the left part of the multiplication
        d_b: bottom block of the right part of the multiplication
    Output matrix:
        d_c: top block of the multiplication product
        d_d: bottom block of the multiplication product
'''
@cuda.jit(device=True)
def block_mat_mat_mul_top_zero(d_q, d_b, d_c, d_d):
    r_q, c_q = d_q.shape
    r_b, c_b = d_b.shape
    r_c, c_c = d_c.shape
    r_d, c_d = d_d.shape
    # sanity check
    if c_c != c_d or r_q != c_q or r_q != (r_c + r_d) or c_b != c_c:
        raise TypeError('Matrix dimensions are not consistent!')
    r_a = c_q - r_b
    # get the top block of the product d_c
    for i in range(r_c):
        for j in range(c_c):
            d_c[i, j] = 0
            # top block is zero block, no need to perform the multiplication
            # multiplication with the bottom block d_b
            for k in range(r_b):
                d_c[i, j] += d_q[i, k + r_a] * d_b[k, j]
    # get the bottom block of the product d_d
    for i in range(r_d):
        for j in range(c_d):
            d_d[i, j] = 0
            # top block is zero block, no need to perform the multiplication
            # multiplication with the bottom block d_b
            for k in range(r_b):
                d_d[i, j] += d_q[i + r_c, k + r_a] * d_b[k, j]


'''
    Block matrix multiplication with the bottom zero block.
    Input matrix:
        d_q: big matrix of the left part of the multiplication
        d_a: top block of the right part of the multiplication
    Output matrix:
        d_c: top block of the multiplication product
        d_d: bottom block of the multiplication product
'''
@cuda.jit(device=True)
def block_mat_mat_mul_bot_zero(d_q, d_a, d_c, d_d):
    r_q, c_q = d_q.shape
    r_a, c_a = d_a.shape
    r_c, c_c = d_c.shape
    r_d, c_d = d_d.shape
    # sanity check
    if c_c != c_d or r_q != c_q or r_q != (r_c + r_d) or c_a != c_c:
        raise TypeError('Matrix dimensions are not consistent!')
    r_b = c_q - r_a
    # get the top block of the product d_c
    for i in range(r_c):
        for j in range(c_c):
            d_c[i, j] = 0
            # multiplication with the top block d_a
            for k in range(r_a):
                d_c[i, j] += d_q[i, k] * d_a[k, j]
            # bottom block is zero block, no need to perform the multiplication
    # get the bottom block of the product d_d
    for i in range(r_d):
        for j in range(c_d):
            d_d[i, j] = 0
            # multiplication with the top block d_a
            for k in range(r_a):
                d_d[i, j] += d_q[i + r_c, k] * d_a[k, j]
            # bottom block is zero block, no need to perform the multiplication


'''
    Block matrix multiplication.
    Input matrix:
        d_q: big matrix of the left part of the multiplication
        d_a: top block of the right part of the multiplication
        d_b: bottom block of the right part of the multiplication
    Output matrix:
        d_c: top block of the multiplication product
        d_d: bottom block of the multiplication product
'''
@cuda.jit(device=True)
def block_mat_mat_mul(d_q, d_a, d_b, d_c, d_d):
    r_q, c_q = d_q.shape
    r_a, c_a = d_a.shape
    r_b, c_b = d_b.shape
    r_c, c_c = d_c.shape
    r_d, c_d = d_d.shape
    # sanity check
    if c_a != c_b or c_c != c_d or r_q != c_q or c_q != (r_a + r_b) or r_q != (r_c + r_d) or c_a != c_c:
        raise TypeError('Matrix dimensions are not consistent!')
    # get the top block of the product d_c
    for i in range(r_c):
        for j in range(c_c):
            d_c[i, j] = 0
            # multiplication with the top block d_a
            for k in range(r_a):
                d_c[i, j] += d_q[i, k] * d_a[k, j]
            # multiplication with the bottom block d_b
            for k in range(r_b):
                d_c[i, j] += d_q[i, k + r_a] * d_b[k, j]
    # get the bottom block of the product d_d
    for i in range(r_d):
        for j in range(c_d):
            d_d[i, j] = 0
            # multiplication with the top block d_a
            for k in range(r_a):
                d_d[i, j] += d_q[i + r_c, k] * d_a[k, j]
            # multiplication with the bottom block d_b
            for k in range(r_b):
                d_d[i, j] += d_q[i + r_c, k + r_a] * d_b[k, j]


'''
    Block matrix vector multiplication.
    Input matrix:
        d_q: big matrix of the left part of the multiplication
    Input vector:
        d_a: top block of the right part of the multiplication
        d_b: bottom block of the right part of the multiplication
    Output vector:
        d_c: top block of the multiplication product
        d_d: bottom block of the multiplication product
'''
@cuda.jit(device=True)
def block_mat_vec_mul(d_q, d_a, d_b, d_c, d_d):
    r_q, c_q = d_q.shape
    r_a = d_a.shape
    r_b = d_b.shape
    r_c = d_c.shape
    r_d = d_d.shape
    # sanity check
    if r_q != c_q or c_q != (r_a[0] + r_b[0]) or r_q != (r_c[0] + r_d[0]):
        raise TypeError('Matrix vector dimensions are not consistent!')
    # get the top block of the product d_c
    for i in range(r_c[0]):
        d_c[i] = 0
        # multiplication with the top block d_a
        for k in range(r_a[0]):
            d_c[i] += d_q[i, k] * d_a[k]
        # multiplication with the bottom block d_b
        for k in range(r_b[0]):
            d_c[i] += d_q[i, k + r_a[0]] * d_b[k]
    # get the bottom block of the product d_d
    for i in range(r_d[0]):
        d_d[i] = 0
        # multiplication with the top block d_a
        for k in range(r_a[0]):
            d_d[i] += d_q[i + r_c[0], k] * d_a[k]
        # multiplication with the bottom block d_b
        for k in range(r_b[0]):
            d_d[i] += d_q[i + r_c[0], k + r_a[0]] * d_b[k]

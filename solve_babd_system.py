from numba import cuda
import babd_node
import matrix_operation_cuda
import matrix_factorization_cuda
import numpy as np
import sys


TPB = 32


# start the implementation of partition factorization of the Jacobian


'''
    Partition the Jabobian matrix into M partitions, qr decomposition is performed 
    in each partition and generate the final BABD system to solve.
    Input:
        size_y: number of ODE variables of the BVP-DAE problem
        size_p: number of parameter variables of the BVP-DAE problem
        M: number of partitions
        N: number of time nodes of the system
        A : the A matrix element in the reduced Jacobian matrix in a row dominant matrix form
            dimension: (N - 1) * size_y x size_y
            each size_y x size_y corresponds to a matrix block at a time node
        C : the C matrix element in the reduced Jacobian matrix in a row dominant matrix form
            dimension: (N - 1) * size_y x size_y
            each size_y x size_y corresponds to a matrix block at a time node
        H : the H matrix element in the reduced Jacobian matrix in a row dominant matrix form
            dimension: (N - 1) * size_y x size_p
            each size_y x size_p corresponds to a matrix block at a time node
        b : the b vector element of the residual in the reduced BABD system in a row dominant matrix form
            dimension: (N - 1) x size_y
            each row vector with size size_y corresponds to a vector block at a time node
    Output:
        index: the index of the time node in each partition
               r_start = index[i] is the start index of the partition
               r_end = index[i + 1] - 1 is the end index of the partition
               index[0] = 1 which is the first node of the mesh
               index[-1] = N - 2 which is the second last node of the mesh
        R: the R matrix element from the partition factorization which contains 
           upper triangular matrix from the qr decomposition in a row dominant matrix form
           dimension: (N - 1) * size_y x size_y
           each size_y x size_y matrix block corresponds to the R matrix block at a time node
        E: the E matrix element from the partition factorization from the qr decomposition 
           in a row dominant matrix form
           dimension: (N - 1) * size_y x size_y
           each size_y x size_y matrix block corresponds to the E matrix block at a time node
        J_reduced: the J matrix element from the partition factorization from the qr decomposition 
                   in a row dominant matrix form
                   dimension: (N - 1) * size_y x size_p
                   each size_y x size_p matrix block corresponds to the J matrix block at a time node
        G: the G matrix element from the partition factorization from the qr decomposition 
           in a row dominant matrix form
           dimension: (N - 1) * size_y x size_y
           each size_y x size_y matrix block corresponds to the G matrix block at a time node
        A_tilde: the A_tilde matrix element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M * size_y x size_y
                 each size_y x size_y matrix block corresponds to the A_tilde matrix block 
                 at the boundary of each partition
        C_tilde: the C_tilde matrix element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M * size_y x size_y
                 each size_y x size_y matrix block corresponds to the C_tilde matrix block 
                 at the boundary of each partition
        H_tilde: the H_tilde matrix element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M * size_y x size_p
                 each size_y x size_p matrix block corresponds to the H_tilde matrix block 
                 at the boundary of each partition
        b_tilde: the b_tilde vector element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M x size_y
                 each size_y vector block corresponds to the b_tilde vector block at the boundary of each partition
        d: the d vector element from the partition factorization from the qr decomposition in a row dominant matrix form
           dimension: (N - 1) x size_y
           each size_y vector block corresponds to the d vector block at the boundary of each partition

        q_i : matrix from qr decomposition
            size: 2 * size_y x 2 * size
        q : big row dominated matrix
            size: (N - 1) * 2 * size_y x 2 * size_y
        q_t : container to hold the transpose of each q_i matrix
            size: (N - 1) * 2 * size_y x 2 * size_y
        r_i : matrix from qr decomposition
            size: 2 * size_y x size_y
        r : big row dominated matrix
            size: (N - 1) * 2 * size_y x size_y
'''


def partition_factorization_parallel(size_y, size_p, M, N, A, C, H, b):
    # compute the grid dimension of the warp
    grid_dims = (M + TPB - 1) // TPB
    # number of partitions to use: M
    # integer number of time nodes in each partition
    # M partitions / threads, (N - 1) time nodes divided into M blocks
    num_thread = (N - 1) // M
    indices = []
    for i in range(M):
        indices.append(i * num_thread)
    # final node to be processed is the (N - 2)th node
    # the final node is the (N - 1) = (N - 2) + 1 th node
    indices.append(N - 1)
    index = np.array(indices)
    # transfer the memory from CPU to GPU
    d_index = cuda.to_device(index)
    d_A = cuda.to_device(A)
    d_C = cuda.to_device(C)
    d_H = cuda.to_device(H)
    d_b = cuda.to_device(b)

    # holders for each intermediate matrix
    d_q = cuda.device_array(((N - 1) * 2 * size_y, 2 * size_y), dtype=np.float64)
    d_q_t = cuda.device_array(((N - 1) * 2 * size_y, 2 * size_y), dtype=np.float64)
    d_r = cuda.device_array(((N - 1) * 2 * size_y, size_y), dtype=np.float64)
    # holders for output variables
    d_r_j = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    # holders for each intermediate matrix
    d_C_tilde = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_G_tilde = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_H_tilde = cuda.device_array(((N - 1) * size_y, size_p), dtype=np.float64)
    d_b_tilde = cuda.device_array(((N - 1), size_y), dtype=np.float64)
    # holders for output variables
    d_E = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_J = cuda.device_array(((N - 1) * size_y, size_p), dtype=np.float64)
    d_G = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_d = cuda.device_array(((N - 1), size_y), dtype=np.float64)
    d_A_tilde_r_end = cuda.device_array((M * size_y, size_y), dtype=np.float64)
    d_C_tilde_r_end = cuda.device_array((M * size_y, size_y), dtype=np.float64)
    d_H_tilde_r_end = cuda.device_array((M * size_y, size_p), dtype=np.float64)
    d_b_tilde_r_end = cuda.device_array((M, size_y), dtype=np.float64)
    # container to hold the vstack of the C and A matrices
    d_C_A = cuda.device_array(((N - 1) * 2 * size_y, size_y), dtype=np.float64)  # row dominated matrix
    # container to hold the intermediate variables for qr decomposition
    d_cpy = cuda.device_array(((N - 1) * 2 * size_y, size_y), dtype=np.float64)  # row dominated matrix
    d_v = cuda.device_array(((N - 1) * 2 * size_y, size_y), dtype=np.float64)  # row dominated matrix
    d_vv = cuda.device_array(((N - 1), 2 * size_y), dtype=np.float64)  # row dominated vector
    d_beta = cuda.device_array(((N - 1), size_y), dtype=np.float64)  # row dominated vector
    d_w_t = cuda.device_array(((N - 1), size_y), dtype=np.float64)  # row dominated vector
    d_u_t = cuda.device_array(((N - 1), 2 * size_y), dtype=np.float64)  # row dominated vector
    # machine precision
    eps = sys.float_info.epsilon
    # perform the parallel partition factorization on GPUs
    partition_factorization_kernel[grid_dims, TPB](
        eps, d_index, size_y, size_p, M, d_A, d_C, d_H, d_b, d_q, d_q_t, d_r, d_r_j, d_C_tilde, d_E, d_G, d_J, d_d,
        d_G_tilde, d_H_tilde, d_b_tilde, d_A_tilde_r_end, d_C_tilde_r_end, d_H_tilde_r_end, d_b_tilde_r_end, d_C_A,
        d_cpy, d_v, d_vv, d_beta, d_w_t, d_u_t)
    cuda.synchronize()
    return index, d_r_j.copy_to_host(), d_E.copy_to_host(), d_J.copy_to_host(), d_G.copy_to_host(), \
        d_d.copy_to_host(), d_A_tilde_r_end.copy_to_host(), d_C_tilde_r_end.copy_to_host(), \
        d_H_tilde_r_end.copy_to_host(), d_b_tilde_r_end.copy_to_host()


'''
    Kernel for performing the partition factorization on each partition.
'''


@cuda.jit()
def partition_factorization_kernel(
        eps, d_index, size_y, size_p, M, d_A, d_C, d_H, d_b, d_q, d_q_t, d_r, d_r_j, d_C_tilde, d_E, d_G, d_J, d_d,
        d_G_tilde, d_H_tilde, d_b_tilde, d_A_tilde_r_end, d_C_tilde_r_end, d_H_tilde_r_end, d_b_tilde_r_end, d_C_A,
        d_cpy, d_v, d_vv, d_beta, d_w_t, d_u_t):
    i = cuda.grid(1)
    if i < M:
        r_start = d_index[i]
        r_end = d_index[i + 1] - 1
        # set the initial condition
        start_row_index = r_start * size_y
        end_row_index = start_row_index + size_y
        # C_tilde_{r_start} = C_{r_start}
        matrix_operation_cuda.set_equal_mat(d_C[start_row_index: end_row_index, 0: size_y],
                                            d_C_tilde[start_row_index: end_row_index, 0: size_y])
        # G_tilde_{r_start} = A_{r_start}
        matrix_operation_cuda.set_equal_mat(d_A[start_row_index: end_row_index, 0: size_y],
                                            d_G_tilde[start_row_index: end_row_index, 0: size_y])
        # H_tilde_{r_start} = H_{r_start}
        matrix_operation_cuda.set_equal_mat(d_H[start_row_index: end_row_index, 0: size_p],
                                            d_H_tilde[start_row_index: end_row_index, 0: size_p])
        # b_tilde_{r_start} = b_{r_start}
        matrix_operation_cuda.set_equal_vec(d_b[r_start, 0: size_y], d_b_tilde[r_start, 0: size_y])
        for j in range(r_start, r_end):
            # index to access matrix A and C
            # start row index for jth element
            start_row_index_cur = j * size_y
            # end row index for jth element
            end_row_index_cur = start_row_index_cur + size_y
            # start row index for (j + 1)th element
            start_row_index_next = (j + 1) * size_y
            # end row index for (j + 1)th element
            end_row_index_next = start_row_index_next + size_y
            # start row index for C_A
            start_row_index_C_A = j * 2 * size_y
            # end row index for C_A
            end_row_index_C_A = start_row_index_C_A + 2 * size_y
            matrix_operation_cuda.vstack(d_C_tilde[start_row_index_cur: end_row_index_cur, 0: size_y],
                                         d_A[start_row_index_next: end_row_index_next, 0: size_y],
                                         d_C_A[start_row_index_C_A: end_row_index_C_A, 0: size_y])
            # qr_cuda(a, cpy, q, r, v, vv, beta, w_t, u_t, eps)
            # qr decomposition of the C_A matrix
            matrix_factorization_cuda.qr(d_C_A[start_row_index_C_A: end_row_index_C_A, 0: size_y],
                                         d_cpy[start_row_index_C_A: end_row_index_C_A, 0: size_y],
                                         d_q[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y],
                                         d_r[start_row_index_C_A: end_row_index_C_A, 0: size_y],
                                         d_v[start_row_index_C_A: end_row_index_C_A, 0: size_y],
                                         d_vv[j, 0: 2 * size_y],
                                         d_beta[j, 0: size_y],
                                         d_w_t[j, 0: size_y],
                                         d_u_t[j, 0: 2 * size_y],
                                         eps)
            # obtain the transpose of the q matrix
            matrix_operation_cuda.mat_trans(d_q[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y],
                                            d_q_t[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y])
            # [E_j; C_tilde_{j + 1} = Q.T * [0; C_{j + 1}]
            # block_mat_mat_mul_top_zero(d_q, d_b, d_c, d_d)
            matrix_operation_cuda.block_mat_mat_mul_top_zero(
                d_q_t[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y],
                d_C[start_row_index_next: end_row_index_next, 0: size_y],
                d_E[start_row_index_cur: end_row_index_cur, 0: size_y],
                d_C_tilde[start_row_index_next: end_row_index_next, 0: size_y])
            # [G_j; G_tilde_{j + 1} = Q.T * [G_tilde_{j}; 0]
            # block_mat_mat_mul_bot_zero(d_q, d_a, d_c, d_d)
            matrix_operation_cuda.block_mat_mat_mul_bot_zero(
                d_q_t[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y],
                d_G_tilde[start_row_index_cur: end_row_index_cur, 0: size_y],
                d_G[start_row_index_cur: end_row_index_cur, 0: size_y],
                d_G_tilde[start_row_index_next: end_row_index_next, 0: size_y])
            # [J_j; H_tilde_{j + 1} = Q.T * [H_tilde_{j}; H_{j + 1}]
            # block_mat_mat_mul(d_q, d_a, d_b, d_c, d_d)
            matrix_operation_cuda.block_mat_mat_mul(d_q_t[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y],
                                                    d_H_tilde[start_row_index_cur: end_row_index_cur, 0: size_p],
                                                    d_H[start_row_index_next: end_row_index_next, 0: size_p],
                                                    d_J[start_row_index_cur: end_row_index_cur, 0: size_p],
                                                    d_H_tilde[start_row_index_next: end_row_index_next, 0: size_p])
            # [d_j; b_tilde_{j + 1} = Q.T * [b_tilde_{j}; b_{j + 1}]
            # block_mat_vec_mul(d_q, d_a, d_b, d_c, d_d)
            matrix_operation_cuda.block_mat_vec_mul(d_q_t[start_row_index_C_A: end_row_index_C_A, 0: 2 * size_y],
                                                    d_b_tilde[j, 0: size_y],
                                                    d_b[j + 1, 0: size_y],
                                                    d_d[j, 0: size_y],
                                                    d_b_tilde[j + 1, 0: size_y])
            # save R_j
            matrix_operation_cuda.set_equal_mat(d_r[start_row_index_C_A: start_row_index_C_A + size_y, 0: size_y],
                                                d_r_j[start_row_index_cur: end_row_index_cur, 0: size_y])
        # start index for r_end
        start_row_index_end = r_end * size_y
        # end index for r_end
        end_row_index_end = start_row_index_end + size_y
        # set A_tilde_r_end, C_tilde_r_end, H_tilde_r_end, d_b_tilde_r_end
        matrix_operation_cuda.set_equal_mat(d_G_tilde[start_row_index_end: end_row_index_end, 0: size_y],
                                            d_A_tilde_r_end[i * size_y: (i + 1) * size_y, 0: size_y])
        matrix_operation_cuda.set_equal_mat(d_C_tilde[start_row_index_end: end_row_index_end, 0: size_y],
                                            d_C_tilde_r_end[i * size_y: (i + 1) * size_y, 0: size_y])
        matrix_operation_cuda.set_equal_mat(d_H_tilde[start_row_index_end: end_row_index_end, 0: size_p],
                                            d_H_tilde_r_end[i * size_y: (i + 1) * size_y, 0: size_p])
        matrix_operation_cuda.set_equal_vec(d_b_tilde[r_end, 0: size_y], d_b_tilde_r_end[i, 0: size_y])


# finish the implementation of partition factorization of the Jacobian


# start the implementation of solving the partitioned BABD system in sequential
# this part is copied from the sequential solver and should be updated?


'''
    Construct the reduced BABD system.
    Partition the Jabobian matrix into M partitions, qr decomposition is performed 
    in each partition and generate the final BABD system to solve.
    Input:
        size_y: number of ODE variables of the BVP-DAE problem
        size_z: number of DAE variables of the BVP-DAE problem
        size_p: number of parameter variables of the BVP-DAE problem
        m: number of collocation points used in the algorithm
        M: number of partitions
        A_tilde: the A_tilde matrix element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M * size_y x size_y
                 each size_y x size_y matrix block corresponds to the A_tilde matrix block 
                 at the boundary of each partition
        C_tilde: the C_tilde matrix element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M * size_y x size_y
                 each size_y x size_y matrix block corresponds to the C_tilde matrix block 
                 at the boundary of each partition
        H_tilde: the H_tilde matrix element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M * size_y x size_p
                 each size_y x size_p matrix block corresponds to the H_tilde matrix block 
                 at the boundary of each partition
        b_tilde: the b_tilde vector element from the partition factorization at the boundary location of each partition
                 in a row dominant matrix form
                 dimension: M x size_y
                 each size_y vector block corresponds to the b_tilde vector block at the boundary of each partition
        B_0: derivatives of boundary conditions w.r.t. ODE variables at initial time
             dimension: size_y + size_p x size_y
        B_n: derivatives of boundary conditions w.r.t. ODE variables at final time
             dimension: size_y + size_p x size_y
        V_n: derivatives of boundary conditions w.r.t. parameter varaibels
             dimension: size_y + size_p x size_p
        r_bc : boundary conditions of the system in vector form
            dimension: size_y + size_p
    Output:
        sol: self designed data structure used to solve the BABD system which contains the necessary elements
             in the reduced Jacobian matrix.
'''


def construct_babd_mshoot(size_y, size_z, size_p, M, A_tilde, C_tilde, H_tilde, b_tilde, B_0, B_n, V_n, b_n):
    sol = []
    for i in range(M):
        node_i = babd_node.MultipleShootingNode(size_y, size_z, size_p)
        node_i.set_A(A_tilde[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: (size_y + size_z)])
        node_i.set_C(C_tilde[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: (size_y + size_z)])
        node_i.set_H(H_tilde[i * (size_y + size_z): (i + 1) * (size_y + size_z), 0: (size_y + size_z)])
        node_i.set_b(b_tilde[i, 0: size_y + size_z])
        sol.append(node_i)
    node_n = babd_node.MultipleShootingNode(size_y, size_z, size_p)
    node_n.set_B_n(B_n)
    node_n.set_H_n(V_n)
    node_n.set_b_n(b_n)
    sol.append(node_n)
    sol[0].set_B_0(B_0)
    return sol


'''
    Solve the reduced Jacobian system in sequential with qr decomposition.
    size_s: size of the matrix block in the Jacobian system.
    size_p: size of the parameter block in the Jacobian system.
    N: number of blocks in the Jacobian system.
    sol: self designed data structure with necessary elements of the Jacobian system
'''


def qr_decomposition(size_s, size_p, N, sol):
    sol[0].C_tilda = sol[0].C
    sol[0].G_tilda = sol[0].A
    sol[0].H_tilda = sol[0].H
    sol[0].b_tilda = sol[0].b

    for i in range(N - 2):
        C_tilda_A = np.concatenate((sol[i].C_tilda, sol[i + 1].A), axis=0)
        Q, R = np.linalg.qr(C_tilda_A, mode='complete')
        sol[i].R = R[0: size_s, :]
        zero_C = np.concatenate((np.zeros((size_s, size_s), dtype=np.float64), sol[i + 1].C), axis=0)
        EC = np.dot(Q.T, zero_C)
        sol[i].E = EC[0: size_s, 0: size_s]
        sol[i + 1].C_tilda = EC[size_s: 2 * size_s, 0: size_s]
        G_tilda_zero = np.concatenate((sol[i].G_tilda, np.zeros((size_s, size_s), dtype=np.float64)), axis=0)
        GG = np.dot(Q.T, G_tilda_zero)
        sol[i].G = GG[0: size_s, 0: size_s]
        sol[i + 1].G_tilda = GG[size_s: 2 * size_s, 0: size_s]
        H_tilda_H = np.concatenate((sol[i].H_tilda, sol[i + 1].H), axis=0)
        JH = np.dot(Q.T, H_tilda_H)
        sol[i].K = JH[0: size_s, 0: size_p]
        sol[i + 1].H_tilda = JH[size_s: 2 * size_s, 0: size_p]
        b_tilda_b = np.concatenate((sol[i].b_tilda, sol[i + 1].b), axis=0)
        db = np.dot(Q.T, b_tilda_b)
        sol[i].d = db[0: size_s]
        sol[i + 1].b_tilda = db[size_s: 2 * size_s]
    final_block_up = np.concatenate((sol[N - 2].C_tilda, sol[N - 2].G_tilda, sol[N - 2].H_tilda), axis=1)
    H_n = sol[N - 1].H_n
    final_block_down = np.concatenate((sol[N - 1].B_n, sol[0].B_0, H_n), axis=1)
    final_block = np.concatenate((final_block_up, final_block_down), axis=0)
    Q, R = np.linalg.qr(final_block, mode='complete')
    sol[N - 2].R = R[0: size_s, 0: size_s]
    sol[N - 2].G = R[0: size_s, size_s: 2 * size_s]
    sol[N - 2].K = R[0: size_s, 2 * size_s: 2 * size_s + size_p]
    sol[N - 1].R = R[size_s: 2 * size_s, size_s: 2 * size_s]
    sol[N - 1].K = R[size_s: 2 * size_s, 2 * size_s: 2 * size_s + size_p]
    sol[N - 1].Rp = R[2 * size_s: 2 * size_s + size_p, 2 * size_s: 2 * size_s + size_p]

    b_n = sol[N - 1].b_n
    b_tilda_b = np.concatenate((sol[N - 2].b_tilda, b_n), axis=0)
    d = np.dot(Q.T, b_tilda_b)
    sol[N - 2].d = d[0: size_s]
    sol[N - 1].d = d[size_s: 2 * size_s]
    sol[N - 1].dp = d[2 * size_s: 2 * size_s + size_p]


'''
    Perform the backward substitution to solve the upper triangular matrix system 
    to get the solution from the linear system of Newton's method.
    N: number of blocks of the system.
    sol: self designed data structure with necessary elements
'''


def backward_substitution(N, sol):
    try:
        delta_p = np.linalg.solve(sol[N - 1].Rp, sol[N - 1].dp)
    except np.linalg.linalg.LinAlgError:
        print("Matrix Rp at the (N - 1)th node is singular!")
        delta_p = np.zeros(sol[N - 1].dp.shape)
    sol[N - 1].delta_p = delta_p

    try:
        delta_s1 = np.linalg.solve(sol[N - 1].R, (sol[N - 1].d - np.dot(sol[N - 1].K, delta_p)))
    except np.linalg.linalg.LinAlgError:
        print("Matrix R at the (N - 1)th node is singular!")
        delta_s1 = np.zeros(sol[N - 1].d.shape)
    sol[0].set_delta_s(delta_s1)

    b_sN = sol[N - 2].d - np.dot(sol[N - 2].G, delta_s1) - np.dot(sol[N - 2].K, delta_p)
    try:
        delta_sN = np.linalg.solve(sol[N - 2].R, b_sN)
    except np.linalg.linalg.LinAlgError:
        print("Matrix R at the (N - 2)th node is singular!")
        delta_sN = np.zeros(b_sN.shape)
    sol[N - 1].set_delta_s(delta_sN)

    for i in range(N - 2, 0, -1):
        b_si = sol[i - 1].d - np.dot(sol[i - 1].E, sol[i + 1].delta_s) - np.dot(sol[i - 1].G, delta_s1) - np.dot(
            sol[i - 1].K, delta_p)
        try:
            delta_si = np.linalg.solve(sol[i - 1].R, b_si)
        except np.linalg.linalg.LinAlgError:
            print("Matrix R at the ({})th node is singular!".format(i - 1))
            delta_si = np.zeros(b_si.shape)
        sol[i].set_delta_s(delta_si)


# finish the implementation of solving the partitioned BABD system in sequential


"""
Recover the solution to the BABD system after backward_substitution from the data structure
"""


def recover_babd_solution(M, size_y, size_z, size_p, sol):
    # obtain the solution from the reduced BABD system
    delta_s = np.zeros((M + 1, size_y + size_z), dtype=np.float64)
    for i in range(M + 1):
        delta_s[i, :] = sol[i].delta_s
    delta_p = np.copy(sol[M].delta_p[0: size_p])
    return delta_s, delta_p


# start the implementation of the parallel backward substitution


'''
    Obtain the solution of the BABD system using partition backward substitution in parallel.
    Input:
        size_y: number of ODE variables of the BVP-DAE problem
        size_p: number of parameter variables of the BVP-DAE problem
        M: number of partitions
        N: number of time nodes of the system
        index: the index of the time node in each partition
               r_start = index[i] is the start index of the partition
               r_end = index[i + 1] - 1 is the end index of the partition
               index[0] = 0 which is the first node of the mesh
               index[-1] = N - 2 which is the second last node of the mesh
        delta_s_r: solution to the reduced BABD system which is the solution at the boudary
                   of each partition of the BABD system in a row dominant matrix form
                   dimension: (M + 1) x size_y
                   delta_s_r[0] = delta_s[0] which is the solution at the start of the first partition which is also the
                   first node
                   delta_s_r[1] = delta_s[r_1 + 1] which is the solution at the start of the second partition, which is 
                   also the node after the last node of the first partition
                   delta_s_r[-1] = delta_s[r_M + 1] which is the solution at the node after the final partition, which 
                   is also the final node (index: N - 1)
        delta_p: solution of the parameter variables to the reduced BABD system in vector form
                 dimension: size_p
        R: the R matrix element from the partition factorization which contains 
           upper triangular matrix from the qr decomposition in a row dominant matrix form
           dimension: (N - 1) * size_y x size_y
           each size_y x size_y matrix block corresponds to the R matrix block at a time node
        G: the G matrix element from the partition factorization from the qr decomposition 
           in a row dominant matrix form
           dimension: (N - 1) * size_y x size_y
           each size_y x size_y matrix block corresponds to the G matrix block at a time node
        E: the E matrix element from the partition factorization from the qr decomposition 
           in a row dominant matrix form
           dimension: (N - 1) * size_y x size_y
           each size_y x size_y matrix block corresponds to the E matrix block at a time node
        J: the J matrix element from the partition factorization from the qr decomposition 
                   in a row dominant matrix form
                   dimension: (N - 1) * size_y x size_p
                   each size_y x size_p matrix block corresponds to the J matrix block at a time node
        d: the d vector element from the partition factorization from the qr decomposition in a row dominant matrix form
           dimension: (N - 1) x size_y
           each size_y vector block corresponds to the d vector block at the boundary of each partition
    Output:
        delta_s: solution to the BABD system in a row dominant matrix form.
                 dimension: N x size_y
                 each size size_y vector block corresponds to the delta_s vector block at a time node
'''


def partition_backward_substitution_parallel(size_y, size_p, M, N, index, delta_s_r, delta_p, R, G, E, J, d):
    # compute the grid dimension of the voxel model
    grid_dims = (M + TPB - 1) // TPB
    # transfer memory from CPU to GPU
    d_index = cuda.to_device(index)
    d_delta_s_r = cuda.to_device(delta_s_r)
    d_delta_p = cuda.to_device(delta_p)
    d_R = cuda.to_device(R)
    d_G = cuda.to_device(G)
    d_E = cuda.to_device(E)
    d_J = cuda.to_device(J)
    d_d = cuda.to_device(d)
    # holder for output variable
    d_delta_s = cuda.device_array((N, size_y), dtype=np.float64)
    # holder for intermediate matrix vector multiplication variables
    d_G_delta_s = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_E_delta_s = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_J_delta_p = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # holder for the right hand side of the linear system to solve in BABD system
    d_vec = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # holder for the intermediate variables in lu decomposition
    d_P = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_L = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_U = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_cpy = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_c = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_y = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # machine precision
    eps = sys.float_info.epsilon
    partition_backward_substitution_kernel[grid_dims, TPB](eps, size_y, size_p, M, d_index, d_delta_s_r, d_delta_p, d_R,
                                                           d_G, d_E, d_J, d_d, d_G_delta_s, d_E_delta_s, d_J_delta_p,
                                                           d_vec, d_P, d_L, d_U, d_cpy, d_c, d_y, d_delta_s)
    return d_delta_s.copy_to_host()


'''
    Kernel for computing the solution to BABD system which performs the partition backward substitution.
'''


# d_delta_s_r: (M + 1) x size_y
# d_delta_s: N x size_y
@cuda.jit()
def partition_backward_substitution_kernel(eps, size_y, size_p, M, d_index, d_delta_s_r, d_delta_p, d_R, d_G, d_E, d_J,
                                           d_d, d_G_delta_s, d_E_delta_s, d_J_delta_p, d_vec, d_P, d_L, d_U, d_cpy, d_c,
                                           d_y, d_delta_s):
    i = cuda.grid(1)
    if i < M:
        r_start = d_index[i]
        r_end = d_index[i + 1] - 1
        for k in range(size_y):
            d_delta_s[r_start, k] = d_delta_s_r[i, k]
            d_delta_s[r_end + 1, k] = d_delta_s_r[i + 1, k]
        for j in range(r_end, r_start, -1):
            # set the matrix R_j as the upper triangular
            # eliminate the machine error
            for k in range(size_y):
                for l in range(k):
                    d_R[(j - 1) * size_y + k, l] = 0
            # compute G_j * delta_s[r_start, :]
            # compute E_j * delta_s[j + 1, :]
            # compute J_j * delta_p
            matrix_operation_cuda.mat_vec_mul(d_G[(j - 1) * size_y: j * size_y, 0: size_y],
                                              d_delta_s[r_start, 0: size_y],
                                              d_G_delta_s[j, 0: size_y])
            matrix_operation_cuda.mat_vec_mul(d_E[(j - 1) * size_y: j * size_y, 0: size_y], d_delta_s[j + 1, 0: size_y],
                                              d_E_delta_s[j, 0: size_y])
            matrix_operation_cuda.mat_vec_mul(d_J[(j - 1) * size_y: j * size_y, 0: size_p], d_delta_p,
                                              d_J_delta_p[j, 0: size_y])
            for k in range(size_y):
                d_vec[j, k] = d_d[j - 1, k] - d_G_delta_s[j, k] - d_E_delta_s[j, k] - d_J_delta_p[j, k]
            matrix_factorization_cuda.lu_solve_vec(d_R[(j - 1) * size_y: j * size_y, 0: size_y],
                                                   d_cpy[(j - 1) * size_y: j * size_y, 0: size_y],
                                                   d_P[(j - 1) * size_y: j * size_y, 0: size_y],
                                                   d_L[(j - 1) * size_y: j * size_y, 0: size_y],
                                                   d_U[(j - 1) * size_y: j * size_y, 0: size_y],
                                                   d_vec[j, 0: size_y], d_c[j, 0: size_y], d_y[j, 0: size_y],
                                                   d_delta_s[j, 0: size_y], eps)

# finish the implementation of the parallel backward substitution

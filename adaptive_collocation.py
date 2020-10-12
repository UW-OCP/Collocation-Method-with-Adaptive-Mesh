import sys
import time
from math import *

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, float64

import bvp_problem
import collocation_coefficients
import gauss_coefficients
import matrix_factorization_cuda
import matrix_operation_cuda
from BVPDAEReadWriteData import bvpdae_write_data
from bvp_problem import _abvp_f, _abvp_g, _abvp_r, _abvp_Df, _abvp_Dg, _abvp_Dr
import pathlib
import solve_babd_system
import mesh_strategies


# TODO: adaptive size
TPB_N = 16  # threads per block in time dimension, must be bigger than (m_max - m_min + 1)
N_shared = TPB_N + 1
TPB_m = 16  # threads per block in collocation dimension
TPB = 32  # threads per block for 1d kernel
m_collocation = 0
global_m_min = 0
global_m_max = 0
global_m_range = 0
global_m_sum = 0
global_size_y = 0
global_size_z = 0
global_size_p = 0
global_y_shared_size = 0
residual_type = 1
scale_by_time = True
scale_by_initial = False
residual_compute_type = "nodal"
save_result = False


def collocation_solver_parallel(m_init=3, mesh_strategy="adaptive"):
    global global_size_y, global_size_z, global_size_p, \
        m_collocation, TPB_m, global_y_shared_size, \
        global_m_min, global_m_max, global_m_range, global_m_sum
    # construct the bvp-dae problem
    # obtain the initial input
    bvp_dae = bvp_problem.BvpDae()
    size_y = bvp_dae.size_y
    global_size_y = size_y
    size_z = bvp_dae.size_z
    global_size_z = size_z
    size_p = bvp_dae.size_p
    global_size_p = size_p
    size_inequality = bvp_dae.size_inequality
    size_sv_inequality = bvp_dae.size_sv_inequality
    output_file = bvp_dae.output_file
    example_name = output_file.split('.')[0]

    t_span0 = bvp_dae.T0
    N = t_span0.shape[0]
    y0 = bvp_dae.Y0
    z0 = bvp_dae.Z0
    p0 = bvp_dae.P0
    # copy the data
    para = np.copy(p0)
    t_span = np.copy(t_span0)

    # parameters for numerical solvers
    tol = bvp_dae.tolerance
    max_iter = bvp_dae.maximum_newton_iterations
    max_iter = 500
    max_mesh = bvp_dae.maximum_mesh_refinements
    max_nodes = bvp_dae.maximum_nodes
    max_nodes = 4000
    min_nodes = 3
    max_linesearch = 20
    alpha = 0.1  # continuation parameter
    if size_inequality > 0 or size_sv_inequality > 0:
        alpha_m = 1e-6
    else:
        alpha_m = 0.1
    beta = 0.9  # scale factor
    # specify collocation coefficients
    m_min = 3
    m_max = 9
    global_m_min = m_min
    if residual_compute_type == "nodal":
        # extra when computing the nodal residual
        # with extra collocation point in each interval
        global_m_max = m_max + 1
        global_m_range = m_max - m_min + 2
        global_y_shared_size = TPB_N * (m_max + 1)
    else:
        global_m_max = m_max
        global_m_range = m_max - m_min + 1
        global_y_shared_size = TPB_N * m_max
    # m_init = 5  # number of collocation points
    # m_init2 = 4
    m_collocation = m_max
    # minimum number of power of 2 as the TPB in m direction
    pos = ceil(log(m_max, 2))
    TPB_m = max(int(pow(2, pos)), 2)  # at least two threads in y direction
    # parameters for mesh
    thres_remove = 1
    thres_add = 1
    rho = 2 * thres_add
    m_d = 2
    m_i = 2
    decay_rate_thres = 0.25

    M = 8  # number of blocks used to solve the BABD system in parallel
    success_flag = 1
    max_residual = 1 / tol

    # benchmark data
    initial_input_time, initial_input_count, residual_time, residual_count, \
        jacobian_time, jacobian_count, reduce_jacobian_time, reduce_jacobian_count, \
        recover_babd_time, recover_babd_count, segment_residual_time, segment_residual_count = benchmark_data_init()

    solver_start_time = time.time()

    # initial setup for the mesh of collocation points
    m_N, m_accumulate = collocation_points_init(m_init, m_min, m_max, N)
    # m_N, m_accumulate = collocation_points_init_2(m_init, m_init2, m_min, m_max, N)
    # m_max + 1 as computing the residual needs m + 1 collocation points
    m_sum, a_m, b_m, c_m = collocation_coefficients_init(m_min, m_max + 1)
    mesh_before = np.zeros(N - 1)
    global_m_sum = m_sum[-1]

    start_time_initial_input = time.time()
    # form the initial input of the solver
    y, y_dot, z_tilde = form_initial_input_parallel(
        size_y, size_z, size_p, m_min, m_max, m_N, m_accumulate, c_m, N, y0, z0, para)
    y_tilde = np.zeros((m_accumulate[-1], size_y))
    initial_input_time += (time.time() - start_time_initial_input)
    initial_input_count += 1

    for alpha_iter in range(max_iter):
        print("Continuation iteration: {}, solving alpha = {}".format(alpha_iter, alpha))
        mesh_it = 0
        iter_time = 0
        for iter_time in range(max_iter):
            start_time_residual = time.time()
            # compute the residual
            norm_f_q, y_tilde, f_a, f_b, r_bc = compute_f_q_parallel(
                size_y, size_z, size_p, m_min, m_max, m_N, m_accumulate, m_sum, N, a_m, b_m,
                t_span, y, y_dot, z_tilde, para, alpha)
            print("\tnorm: {0:.8f}".format(norm_f_q))
            residual_time += (time.time() - start_time_residual)
            residual_count += 1
            if norm_f_q < tol:
                print('\talpha = {}, solution is found. Number of nodes: {}'.format(alpha, N))
                break
            start_time_jacobian = time.time()
            # compute each necessary element in the Jacobian matrix
            J, V, D, W, B_0, B_n, V_n = construct_jacobian_parallel(
                size_y, size_z, size_p, m_min, m_max, N,
                m_N, m_accumulate, m_sum, a_m, b_m,
                t_span, y, y_tilde, z_tilde, para, alpha)
            jacobian_time += (time.time() - start_time_jacobian)
            jacobian_count += 1
            start_time_reduce_jacobian = time.time()
            # compute each necessary element in the reduced BABD system
            A, C, H, b = reduce_jacobian_parallel(
                size_y, size_z, size_p, m_max, N,
                m_N, m_accumulate,
                W, D, J, V, f_a, f_b)
            reduce_jacobian_time += (time.time() - start_time_reduce_jacobian)
            reduce_jacobian_count += 1
            # solve the BABD system
            # perform the partition factorization on the Jacobian matrix with qr decomposition
            index, R, E, J_reduced, G, d, A_tilde, C_tilde, H_tilde, b_tilde = \
                solve_babd_system.partition_factorization_parallel(size_y, size_p, M, N, A, C, H, b)
            # construct the partitioned Jacobian system
            sol = solve_babd_system.construct_babd_mshoot(
                size_y, 0, size_p, M, A_tilde, C_tilde, H_tilde, b_tilde, B_0, B_n, V_n, -r_bc)
            # perform the qr decomposition to transfer the system
            solve_babd_system.qr_decomposition(size_y, size_p, M + 1, sol)
            # perform the backward substitution to obtain the solution to the linear system of Newton's method
            solve_babd_system.backward_substitution(M + 1, sol)
            # obtain the solution from the reduced BABD system
            delta_s_r, delta_para = solve_babd_system.recover_babd_solution(M, size_y, 0, size_p, sol)
            # get the solution to the BABD system
            delta_y = solve_babd_system.partition_backward_substitution_parallel(
                size_y, size_p, M, N, index, delta_s_r, delta_para, R, G, E, J_reduced, d)
            start_time_recover_babd = time.time()
            # recover delta_k from the reduced BABD system
            delta_k, delta_y_dot, delta_z_tilde = recover_delta_k_parallel(
                size_y, size_z, size_p, m_max, N, m_N, m_accumulate, delta_y, delta_para, f_a, J, V, W)
            recover_babd_time += (time.time() - start_time_recover_babd)
            recover_babd_count += 1
            # line search
            alpha0 = 1
            line_search = 0
            for line_search in range(max_linesearch):
                y_new = y + alpha0 * delta_y
                y_dot_new = y_dot + alpha0 * delta_y_dot
                z_tilde_new = z_tilde + alpha0 * delta_z_tilde
                para_new = para + alpha0 * delta_para
                start_time_residual = time.time()
                norm_f_q_new, _, _, _, _ = compute_f_q_parallel(
                    size_y, size_z, size_p, m_min, m_max, m_N, m_accumulate, m_sum, N, a_m, b_m,
                    t_span, y_new, y_dot_new, z_tilde_new, para_new, alpha)
                residual_time += (time.time() - start_time_residual)
                residual_count += 1
                if norm_f_q_new < norm_f_q:
                    y = y_new
                    y_dot = y_dot_new
                    z_tilde = z_tilde_new
                    para = para_new
                    break
                alpha0 /= 2
            if line_search >= (max_linesearch - 1):
                print("\tLine search fails.")
                start_time_segment_residual = time.time()
                if residual_compute_type == "nodal":
                    residual, residual_collocation, max_residual, max_y_error, max_z_error = \
                        compute_residual_nodal(size_y, size_z, size_p, m_min, m_max, N, m_N, m_accumulate,
                                           c_m, t_span, y, y_dot, z_tilde, para, alpha, tol, m_sum, a_m, b_m, M)
                else:
                    residual, residual_collocation, max_residual = \
                        compute_segment_residual_collocation_parallel(
                            size_y, size_z, size_p, m_min, m_max, N, m_N, m_accumulate,
                            c_m, t_span, y, y_dot, z_tilde, para, alpha, tol)
                print('\tresidual error = {}, number of nodes = {}.'.format(max_residual, N))
                z = recover_solution_parallel(size_z, m_max, N, m_N, m_accumulate, z_tilde)
                if mesh_strategy == "adaptive":
                    N, t_span, y, z, m_N, m_accumulate = \
                        mesh_strategies.adaptive_remesh(
                            size_y, size_z, size_p, m_min, m_max, m_init, N, alpha,
                            m_N, m_accumulate, t_span, y, z, para, y_tilde, y_dot, z_tilde,
                            residual, residual_collocation,
                            thres_remove, thres_add, tol)
                else:
                    N, t_span, y, z = \
                        mesh_strategies.normal_remesh_add_only(
                            size_y, size_z, N, t_span, y, z, residual, thres_remove, thres_add)
                    # update the collocation points distribution
                    m_N, m_accumulate = collocation_points_init(m_init, m_min, m_max, N)
                    # m_N, m_accumulate = collocation_points_init_2(m_init, m_init2, m_min, m_max, N)
                segment_residual_time += (time.time() - start_time_segment_residual)
                segment_residual_count += 1
                mesh_it += 1
                start_time_initial_input = time.time()
                y, y_dot, z_tilde = form_initial_input_parallel(
                    size_y, size_z, size_p, m_min, m_max, m_N, m_accumulate, c_m, N, y, z, para)
                initial_input_time += (time.time() - start_time_initial_input)
                initial_input_count += 1
                print("\tRemeshed the problem. Number of nodes = {}".format(N))
                if mesh_sanity_check(N, mesh_it, max_mesh, max_nodes, min_nodes):
                    print("\talpha = {}, number of nodes is beyond limit after remesh!".format(
                        alpha))
                    success_flag = 0
                    break
        # check whether the iteration exceeds the maximum number
        if alpha_iter >= (max_iter - 1) or iter_time >= (max_iter - 1) \
                or N > max_nodes or mesh_it > max_mesh or N < min_nodes:
            print("\talpha = {}, reach the maximum iteration numbers and the problem does not converge!".format(alpha))
            success_flag = 0
            break
        start_time_segment_residual = time.time()
        if residual_compute_type == "nodal":
            residual, residual_collocation, max_residual, max_y_error, max_z_error = \
                compute_residual_nodal(size_y, size_z, size_p, m_min, m_max, N, m_N, m_accumulate,
                                       c_m, t_span, y, y_dot, z_tilde, para, alpha, tol, m_sum, a_m, b_m, M)
        else:
            residual, residual_collocation, max_residual = \
                compute_segment_residual_collocation_parallel(
                    size_y, size_z, size_p, m_min, m_max, N, m_N, m_accumulate,
                    c_m, t_span, y, y_dot, z_tilde, para, alpha, tol)
        segment_residual_time += (time.time() - start_time_segment_residual)
        segment_residual_count += 1
        print('\tresidual error = {}, number of nodes = {}.'.format(max_residual, N))
        if max_residual > 1:
            z = recover_solution_parallel(size_z, m_max, N, m_N, m_accumulate, z_tilde)
            if mesh_strategy == "adaptive":
                N, t_span, y, z, m_N, m_accumulate = \
                    mesh_strategies.adaptive_remesh(
                        size_y, size_z, size_p, m_min, m_max, m_init, N, alpha,
                        m_N, m_accumulate, t_span, y, z, para, y_tilde, y_dot, z_tilde,
                        residual, residual_collocation,
                        thres_remove, thres_add, tol)
            else:
                N, t_span, y, z = \
                    mesh_strategies.normal_remesh_add_only(
                        size_y, size_z, N, t_span, y, z, residual, thres_remove, thres_add)
                # update the collocation points distribution
                m_N, m_accumulate = collocation_points_init(m_init, m_min, m_max, N)
                # m_N, m_accumulate = collocation_points_init_2(m_init, m_init2, m_min, m_max, N)
            mesh_it += 1
            start_time_initial_input = time.time()
            y, y_dot, z_tilde = form_initial_input_parallel(
                size_y, size_z, size_p, m_min, m_max, m_N, m_accumulate, c_m, N, y, z, para)
            initial_input_time += (time.time() - start_time_initial_input)
            initial_input_count += 1
            print("\tRemeshed the problem. Number of nodes = {}".format(N))
            if mesh_sanity_check(N, mesh_it, max_mesh, max_nodes, min_nodes):
                print("\talpha = {}, number of nodes is beyond limit after remesh!".format(
                    alpha))
                success_flag = 0
                break
        else:
            print("\talpha = {}, solution is found with residual error = {}. Number of nodes = {}".format(
                alpha, max_residual, N))
            if alpha <= alpha_m:
                print("Final solution is found, alpha = {}. Number of nodes: {}".format(alpha, N))
                break
            alpha *= beta
    total_time = time.time() - solver_start_time
    print("Maximum residual: {}".format(max_residual))
    print("Elapsed time: {}".format(total_time))
    # recover the final solution
    z = recover_solution_parallel(size_z, m_max, N, m_N, m_accumulate, z_tilde)
    # write benchmark result
    benchmark_dir = "./benchmark_performance/"
    # create the directory
    pathlib.Path(benchmark_dir).mkdir(0o755, parents=True, exist_ok=True)
    benchmark_file = benchmark_dir + example_name + "_parallel_benchmark_M_{}.data".format(M)
    write_benchmark_result(benchmark_file,
                           initial_input_time, initial_input_count,
                           residual_time, residual_count,
                           jacobian_time, jacobian_count,
                           reduce_jacobian_time, reduce_jacobian_count,
                           recover_babd_time, recover_babd_count,
                           segment_residual_time, segment_residual_count,
                           total_time)
    # if alpha <= alpha_m and success_flag:
    if alpha <= alpha_m:
        if N >= max_nodes:
            N_saved = 101
            # if the number of nodes is too many, just take part of them to save
            index_saved = np.linspace(0, N - 1, num=N_saved, dtype=int)
            # write solution to the output file
            error = bvpdae_write_data(
                output_file, N_saved, size_y, size_z, size_p,
                t_span[index_saved], y[index_saved, :], z[index_saved, :], para)
        else:
            # directly write solution to the output file
            error = bvpdae_write_data(output_file, N, size_y, size_z, size_p, t_span, y, z, para)
        if error != 0:
            print('Write file failed.')
    # record the solved example
    with open("test_results_mesh_strategy_{}_residual_{}_m_init_{}.txt".format(
            mesh_strategy, residual_compute_type, m_init), 'a') as f:
        if alpha <= alpha_m and success_flag:
            f.write("{} solved successfully. alpha = {}. Elapsed time: {}(s). "
                    "Total number of time nodes: {}. Total number of collocation points: {}.\n".format(
                        example_name, alpha, total_time, N, m_accumulate[-1]))
            print("Problem solved successfully.")
        else:
            f.write("{} solved unsuccessfully. alpha = {}. Elapsed time: {}(s). "
                    "Total number of time nodes: {}. Total number of collocation points: {}.\n".format(
                        example_name, alpha, total_time, N, m_accumulate[-1]))
            print("Problem solved unsuccessfully.")
    # plot the result
    plot_result(size_y, size_z, t_span, y, z)
    return


def collocation_points_init(m_init, m_min, m_max, N):
    """
    Initial collocation points set up for adaptive collocation methods.
    :param m_init: initial global number of collocation points used
    :param m_min: minimum number of collocation points allowed
    :param m_max: maximum number of collocation points allowed
    :param N: number of time nodes
    :return:
        m_N: number of collocation points in each interval,
             size: (N - 1,)
        m_accumulate: accumulated number of collocation points prior to each interval;
                      range(m_accumulate[i], m_accumulate[i + 1]) corresponds to
                      the collocation points in interval i;
                      m_accumulate[-1] holds the total number of collocation points in the mesh
                      size: (N, )
    """
    # sanity check
    if m_min > m_max:
        raise TypeError("Minimum number of collocation points given is bigger than the maximum given!")
    m_N = m_init * np.ones(N - 1, dtype=int)  # total of N - 1 intervals
    m_accumulate = np.zeros(N, dtype=int)  # accumulated collocation points used
    for i in range(N):
        m_accumulate[i] = i * m_init
    return m_N, m_accumulate


def lobatto_weights_init(m_min, m_max):
    # sanity check
    if m_min > m_max:
        raise TypeError("Minimum number of collocation points given is bigger than the maximum given!")
    w_m = np.zeros((m_max - m_min + 1, m_max))  # coefficients w for each m
    for m in range(m_min, m_max + 1):
        lobatto_coef = collocation_coefficients.lobatto(m)
        w = lobatto_coef.w
        for j in range(m):
            w_m[m - m_min, j] = w[j]
    return w_m


def collocation_points_init_2(m_init, m_init2, m_min, m_max, N):
    """
    Initial collocation points set up for adaptive collocation methods.
    :param m_init: initial global number of collocation points used
    :param m_min: minimum number of collocation points allowed
    :param m_max: maximum number of collocation points allowed
    :param N: number of time nodes
    :return:
        m_N: number of collocation points in each interval,
             size: (N - 1,)
        m_accumulate: accumulated number of collocation points prior to each interval;
                      range(m_accumulate[i], m_accumulate[i + 1]) corresponds to
                      the collocation points in interval i;
                      m_accumulate[-1] holds the total number of collocation points in the mesh
                      size: (N, )
    """
    # sanity check
    if m_min > m_max:
        raise TypeError("Minimum number of collocation points given is bigger than the maximum given!")
    m_N = np.ones(N - 1, dtype=int)  # total of N - 1 intervals
    m_accumulate = np.zeros(N, dtype=int)  # accumulated collocation points used
    for i in range(N - 1):
        if i < N // 2:
            m_N[i] = m_init
            m_accumulate[i + 1] = m_accumulate[i] + m_init
        else:
            m_N[i] = m_init2
            m_accumulate[i + 1] = m_accumulate[i] + m_init2
    return m_N, m_accumulate


def collocation_coefficients_init(m_min, m_max):
    """
    Generate the coefficients for all the collocation points.
    :param m_min: minimum number of collocation points allowed
    :param m_max: maximum number of collocation points allowed
    :return:
        m_sum: accumulated stages for each collocation point;
                   range(m_sum[m - m_min], m_sum[m - m_min + 1]) corresponds to the stages of collocation point m
                   size: (m_max - m_min + 2, )
        a_m: coefficients a in row dominant order;
             a_m[m_sum[m - m_min]: m_sum[m - m_min], :] corresponds to the coefficients of collocation m
             shape: (m_sum[-1], m_max)
        b_m: coefficients b in row dominant order;
             b_m[m_sum[m - m_min], :] corresponds to the coefficients of collocation m
             shape: (m_max - m_min + 1, m_max)
        c_m: coefficients c in row dominant order;
             c_m[m_sum[m - m_min], :] corresponds to the coefficients of collocation m
             shape: (m_max - m_min + 1, m_max)
    """
    # sanity check
    if m_min > m_max:
        raise TypeError("Minimum number of collocation points given is bigger than the maximum given!")
    m_sum = np.zeros(m_max - m_min + 2, dtype=int)
    for m in range(m_min, m_max + 1):
        m_sum[m - m_min + 1] = m_sum[m - m_min] + m
    a_m = np.zeros((m_sum[-1], m_max))  # coefficients b for each m
    b_m = np.zeros((m_max - m_min + 1, m_max))  # coefficients b for each m
    c_m = np.zeros((m_max - m_min + 1, m_max))  # coefficients c for each m
    for m in range(m_min, m_max + 1):
        rk = collocation_coefficients.lobatto(m)
        a = rk.A
        b = rk.b
        c = rk.c
        for j in range(m):
            for k in range(m):
                a_m[m_sum[m - m_min] + j, k] = a[j, k]
            b_m[m - m_min, j] = b[j]
            c_m[m - m_min, j] = c[j]
    return m_sum, a_m, b_m, c_m


def gauss_coefficients_init(m_min, m_max):
    """
    Generate the coefficients for all the collocation points.
    :param m_min: minimum number of collocation points allowed
    :param m_max: maximum number of collocation points allowed
    :return:
        tau_m: time coefficients of different gauss points used in row dominant order;
             tau_m[m_sum[m - m_min], :] corresponds to the coefficients of (m + 1) gauss points
             shape: (m_max - m_min + 1, m_max + 1)
        w_m: weight coefficients of different gauss points used in row dominant order;
             w_m[m_sum[m - m_min], :] corresponds to the coefficients of (m + 1) gauss points
             shape: (m_max - m_min + 1, m_max + 1)
    """
    # sanity check
    if m_min > m_max:
        raise TypeError("Minimum number of collocation points given is bigger than the maximum given!")
    tau_m = np.zeros((m_max - m_min + 1, m_max + 1))  # coefficients b for each m
    w_m = np.zeros((m_max - m_min + 1, m_max + 1))  # coefficients c for each m
    for m in range(m_min, m_max + 1):
        gauss_coef = gauss_coefficients.gauss(m + 1)
        tau = gauss_coef.t
        w = gauss_coef.w
        for j in range(m + 1):
            tau_m[m - m_min, j] = tau[j]
            w_m[m - m_min, j] = w[j]
    return tau_m, w_m

# start implementations for forming initial input


def form_initial_input_parallel(size_y, size_z, size_p, m_min, m_max, m_N, m_accumulate, c_m, N, y0, z0, p0):
    """
    Form the initial input for the collocation algorithm. The inputs for the solver
    are  usually just ODE variables, DAE variables, and parameter variables. However,
    the inputs to the collocation solver should be ODE variables at each time node, the
    derivatives of the ODE variables and the value of DAE variables at each collocation
    point.
    :param size_y: number of ODE variables of the BVP-DAE problem
    :param size_z: number of DAE variables of the BVP-DAE problem
    :param size_p: number of parameter variables of the BVP-DAE problem
    :param m_min: minimum number of collocation points allowed
    :param m_max: maximum number of collocation points allowed
    :param m_N: number of collocation points in each time interval
    :param m_accumulate: accumulated number of collocation points in each time interval
    :param c_m: coefficients of the collocation points used;
                from m_min to m_max in row dominant order;
                size: (m_max - m_min + 1) * m_max, with zeros in empty spaces
    :param N: number of time nodes
    :param y0: values of the ODE variables in matrix form
               dimension: N x size_y, where each row corresponds the values at each time node
    :param z0: values of the DAE variables in matrix form
               dimension: N x size_z, where each row corresponds the values at each time node
    :param p0: values of the parameter variables in vector form
               dimension: N x size_z, where each row corresponds the values at each time node
    :return:
        y0: values of the ODE variables in matrix form
               dimension: N x size_y, where each row corresponds the values at each time node
        y_dot: values of the derivatives of ODE variables in row dominant matrix form
            dimension: (N - 1) * m x size_y, where each row corresponds the values at each time node.
            The index for the jth collocation point from the ith time node is i * m + j.
        z_tilde: values of the DAE variables in row dominant matrix form
           dimension: (N - 1) * m x size_z, where each row corresponds the values at each time node.
           The index for the jth collocation point from the ith time node is i * m + j.
    """
    # warp dimension for CUDA kernel
    grid_dims_1d = ((N - 1) + TPB_N - 1) // TPB_N
    block_dims_1d = TPB_N
    # transfer memory from CPU to GPU
    d_m_N = cuda.to_device(m_N)
    d_m_accumulate = cuda.to_device(m_accumulate)
    d_c_m = cuda.to_device(c_m)
    d_y0 = cuda.to_device(y0)
    d_z0 = cuda.to_device(z0)
    d_p0 = cuda.to_device(p0)
    # create holder for temporary variables
    d_y_temp = cuda.device_array((m_accumulate[-1], size_y), dtype=np.float64)
    # create holder for output variables
    d_y_dot = cuda.device_array((m_accumulate[-1], size_y), dtype=np.float64)
    d_z_tilde = cuda.device_array((m_accumulate[-1], size_z), dtype=np.float64)
    form_initial_input_kernel[grid_dims_1d, block_dims_1d](
        size_y, size_z, size_p, m_min, m_max, N, d_m_N, d_m_accumulate, d_c_m,
        d_y0, d_z0, d_p0, d_y_temp, d_y_dot, d_z_tilde)
    # transfer the memory back to CPU
    y_dot = d_y_dot.copy_to_host()
    z_tilde = d_z_tilde.copy_to_host()
    return y0, y_dot, z_tilde


@cuda.jit
def form_initial_input_kernel(
        size_y, size_z, size_p, m_min, m_max, N, d_m_N, d_m_accumulate, d_c_m,
        d_y0, d_z0, d_p0, d_y_temp, d_y_dot, d_z_tilde):
    """
    Kernel function for forming initial input.
    :param size_y:
    :param size_z:
    :param size_p:
    :param m_min:
    :param m_max:
    :param N:
    :param d_m_N:
    :param d_m_accumulate:
    :param d_c_m:
    :param d_y0:
    :param d_z0:
    :param d_p0:
    :param d_y_temp:
    :param d_y_dot:
    :param d_z_tilde:
    :return:
    """
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    shared_d_c_m = cuda.shared.array(shape=(global_m_range, global_m_max), dtype=float64)
    shared_d_y0 = cuda.shared.array(shape=(N_shared, global_size_y), dtype=float64)
    shared_d_z0 = cuda.shared.array(shape=(N_shared, global_size_z), dtype=float64)
    shared_d_p0 = cuda.shared.array(shape=(global_size_p, ), dtype=float64)

    # cuda thread index
    i = cuda.grid(1)
    tx = cuda.threadIdx.x  # thread index in x direction

    if i >= (N - 1):
        return

    # only need 1 dimensional memory load here
    for j in range(size_y):
        shared_d_y0[tx + 1, j] = d_y0[i + 1, j]
    for j in range(size_z):
        shared_d_z0[tx + 1, j] = d_z0[i + 1, j]
    for j in range(size_p):
        shared_d_p0[j] = d_p0[j]
    if tx == 0:
        # let the first index to load the coefficients
        # the reason is that the number of threads in the block may be less than (m_max - m_min + 1)
        # so we can not let each thread to load the corresponding row, which is not scallable
        for j in range(m_max - m_min + 1):
            # load coefficients c if the thread index is in range
            for k in range(m_min + j):
                shared_d_c_m[j, k] = d_c_m[j, k]
        # load the additional column in shared memory using the first thread
        for j in range(size_y):
            shared_d_y0[0, j] = d_y0[i, j]
        for j in range(size_z):
            shared_d_z0[0, j] = d_z0[i, j]
    cuda.syncthreads()  # finish the loading here

    m = d_m_N[i]
    for j in range(d_m_accumulate[i], d_m_accumulate[i + 1]):
        for k in range(size_y):
            d_y_temp[j, k] = (1 - shared_d_c_m[m - m_min, j - d_m_accumulate[i]]) * shared_d_y0[tx, k] + \
                             shared_d_c_m[m - m_min, j - d_m_accumulate[i]] * shared_d_y0[tx + 1, k]
        for k in range(size_z):
            d_z_tilde[j, k] = (1 - shared_d_c_m[m - m_min, j - d_m_accumulate[i]]) * shared_d_z0[tx, k] + \
                              shared_d_c_m[m - m_min, j - d_m_accumulate[i]] * shared_d_z0[tx + 1, k]
        _abvp_f(d_y_temp[j, 0: size_y], d_z_tilde[j, 0: size_z], shared_d_p0[0: size_p],
                d_y_dot[j, 0: size_y])
    return

# finish implementations for forming initial input


# start implementation for computing the residual of the system


def compute_f_q_parallel(size_y, size_z, size_p, m_min, m_max, m_N, m_accumulate, m_sum, N,
                         a_m, b_m, t_span, y, y_dot, z_tilde, p, alpha):
    """
    Compute the residual of the BVP-DAE using collocation method.
    :param size_y: number of ode variables of the problem
    :param size_z: number of algebraic variables
    :param size_p: number of parameter variables
    :param m_min: minimum number of collocation points allowed
    :param m_max: maximum number of collocation points allowed
    :param m_N: number of collocation points in each interval
    :param m_accumulate: accumulated collocation points in each interal
    :param m_sum: accumulated collocation sums in collocation coefficients
    :param N: number of time nodes of the problem
    :param a_m: stack coefficients a in lobbatto method in feasible range [m_min, m_max]
                in row dominated order
                shape: (m_sum[-1], m_max)
    :param b_m: stack coefficients b in lobbatto method in feasible range [m_min, m_max]
                in row dominated order
                shape: (m_max - m_min + 1, m_max)
    :param t_span: time span of the mesh
    :param y: values of the ODE variables in matrix form
           dimension: N x size_y, where each row corresponds the values at each time node
    :param y_dot: values of the derivatives of ODE variables in row dominant matrix form
            shape: (m_accumulate[-1], size_y), where each row corresponds the values at each time node.
            The index for the jth collocation point from the ith time node is i * m + j.
    :param z_tilde: values of the DAE variables in row dominant matrix form
           dimension: (m_accumulate[-1]m, size_z), where each row corresponds the values at each time node.
           The index for the jth collocation point from the ith time node is i * m + j.
    :param p: values of the parameter variables in vector form
           shape: (size_p, )
    :param alpha: continuation parameter of the Newton method
    :return:
        norm_f_q : infinite norm of the residual
        y_tilde: values of ODE variables y at each collocation point in row dominant matrix for
            shape: (m_accumulate[-1], size_y), where each row corresponds the values at each collocation point
        f_a : matrix of the residual f_a for each time node in row dominant matrix form
            shape: ((N - 1), m_max * (size_y + size_z)), where each row corresponds the values at each time node
        f_b : matrix of the residual f_b for each time node in row dominant matrix form
            shape: (N - 1, size_y), where each row corresponds the values at each time node
        r_bc : boundary conditions of the system in vector form
            shape: (size_y + size_p, )
    """
    # calculate the y values on collocation points
    # combine those two kernels into one maybe?
    # grid dimension of the warp of CUDA
    grid_dims_1d = ((N - 1) + TPB_N - 1) // TPB_N
    block_dims_1d = TPB_N
    # transfer memory from CPU to GPU
    d_m_N = cuda.to_device(m_N)
    d_m_accumulate = cuda.to_device(m_accumulate)
    d_m_sum = cuda.to_device(m_sum)
    d_a_m = cuda.to_device(a_m)
    d_b_m = cuda.to_device(b_m)
    d_t_span = cuda.to_device(t_span)
    d_y = cuda.to_device(y)
    d_y_dot = cuda.to_device(y_dot)
    # container to hold temporary variables
    d_sum_j = cuda.device_array((m_accumulate[-1], size_y), dtype=np.float64)
    # container to hold output variables y_tilde
    d_y_tilde = cuda.device_array((m_accumulate[-1], size_y), dtype=np.float64)
    # calculate the y variables at collocation points with the kernel function
    collocation_update_kernel[grid_dims_1d, block_dims_1d](
        size_y, m_min, m_max, N, d_m_N, d_m_accumulate, d_m_sum, d_a_m, d_t_span, d_y, d_y_dot, d_sum_j, d_y_tilde)
    # load the memory back from GPU to CPU
    y_tilde = d_y_tilde.copy_to_host()
    # transfer memory from CPU to GPU
    d_z_tilde = cuda.to_device(z_tilde)
    d_p = cuda.to_device(p)
    # container to hold temporary variables
    d_sum_i = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # container to hold derivatives
    d_r_h = cuda.device_array((m_accumulate[-1], size_y), dtype=np.float64)
    d_r_g = cuda.device_array((m_accumulate[-1], size_z), dtype=np.float64)
    # container to hold residuals, be careful about the dimensions
    d_f_a = cuda.device_array((N - 1, m_max * (size_y + size_z)), dtype=np.float64)
    d_f_b = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # calculate the f_a and f_b at each time node with the kernel function
    compute_f_q_kernel1[grid_dims_1d, block_dims_1d](size_y, size_z, m_max, N, d_m_N, d_m_accumulate,
                                                     d_y_dot, d_y_tilde, d_z_tilde, d_p, alpha, d_r_h, d_r_g, d_f_a)
    compute_f_q_kernel2[grid_dims_1d, block_dims_1d](
        size_y, m_min, m_max, N, d_m_N, d_m_accumulate, d_b_m, d_t_span, d_y, d_y_dot, d_sum_i, d_f_b)
    # load the memory back from GPU to CPU
    f_a = d_f_a.copy_to_host()
    f_b = d_f_b.copy_to_host()
    # calculate the boundary conditions
    r_bc = np.zeros((size_y + size_p), dtype=np.float64)
    # this boundary function is currently on CPU
    _abvp_r(y[0, 0: size_y], y[N - 1, 0: size_y], p, r_bc)

    # return the norm of the residual directly,
    # no need to form the residual as the infinity norm is used here
    norm_f_a = cuda_infinity_norm(d_f_a.reshape((N - 1) * m_max * (size_y + size_z), order='C'))
    norm_f_b = cuda_infinity_norm(d_f_b.reshape((N - 1) * size_y, order='C'))
    norm_r = np.linalg.norm(r_bc, np.inf)
    norm_f_q = max(norm_f_a, norm_f_b, norm_r)
    return norm_f_q, y_tilde, f_a, f_b, r_bc


'''
    Kernel function to compute each part of the residual of the system.
    d_f_a: N - 1 x m * (size_y + size_z)
    d_f_b: N - 1 x size_y
    d_r_h: (N - 1) * m x size_y
    d_r_g: (N - 1) * m x size_z
'''


@cuda.jit
def compute_f_q_kernel1(size_y, size_z, m_max, N, d_m_N, d_m_accumulate,
                        d_y_dot, d_y_tilde, d_z_tilde, d_p, alpha, d_r_h, d_r_g, d_f_a):
    i = cuda.grid(1)
    if i < (N - 1):
        m = d_m_N[i]
        # zero initialize the f_a
        for j in range(m_max * (size_y + size_z)):
            d_f_a[i, j] = 0.0
        for j in range(m):
            _abvp_f(d_y_tilde[d_m_accumulate[i] + j, 0: size_y], d_z_tilde[d_m_accumulate[i] + j, 0: size_z], d_p,
                    d_r_h[d_m_accumulate[i] + j, 0: size_y])
            _abvp_g(
                d_y_tilde[d_m_accumulate[i] + j, 0: size_y], d_z_tilde[d_m_accumulate[i] + j, 0: size_z], d_p, alpha,
                d_r_g[d_m_accumulate[i] + j, 0: size_z])
            # calculate the residual $h - y_dot$ on each collocation point
            for k in range(size_y):
                d_r_h[d_m_accumulate[i] + j, k] -= d_y_dot[d_m_accumulate[i] + j, k]
            # copy the result to f_a of the collocation point to the corresponding position
            start_index_y = j * (size_y + size_z)
            start_index_z = start_index_y + size_y
            # copy the residual of h and g to the corresponding positions
            for k in range(size_y):
                d_f_a[i, start_index_y + k] = d_r_h[d_m_accumulate[i] + j, k]
            for k in range(size_z):
                d_f_a[i, start_index_z + k] = d_r_g[d_m_accumulate[i] + j, k]
    return


@cuda.jit
def compute_f_q_kernel2(size_y, m_min, m_max, N, d_m_N, d_m_accumulate, d_b_m, d_t_span, d_y, d_y_dot, d_sum_i, d_f_b):
    shared_d_b_m = cuda.shared.array(shape=(global_m_range, global_m_max), dtype=float64)
    shared_d_y = cuda.shared.array(shape=(N_shared, global_size_y), dtype=float64)
    # cuda thread index
    i = cuda.grid(1)
    tx = cuda.threadIdx.x  # thread index in x direction

    if i >= (N - 1):
        return

    if tx == 0:
        # load coefficients b using the first thread in x direction
        for m in range(m_max - m_min + 1):
            for j in range(m + m_min):
                shared_d_b_m[m, j] = d_b_m[m, j]
    # only need 1 dimensional memory load here
    for j in range(size_y):
        shared_d_y[tx + 1, j] = d_y[i + 1, j]
    if tx == 0:
        # load the additional column in shared memory using the first thread
        for j in range(size_y):
            shared_d_y[0, j] = d_y[i, j]
    cuda.syncthreads()  # finish the loading here

    m = d_m_N[i]
    # initialize d_sum_i as zeros
    for k in range(size_y):
        d_sum_i[i, k] = 0
    for j in range(m):
        for k in range(size_y):
            d_sum_i[i, k] += shared_d_b_m[m - m_min, j] * d_y_dot[d_m_accumulate[i] + j, k]
    delta_t_i = d_t_span[i + 1] - d_t_span[i]
    for k in range(size_y):
        d_f_b[i, k] = shared_d_y[tx + 1, k] - shared_d_y[tx, k] - delta_t_i * d_sum_i[i, k]
    return


'''
    Kernel method for computing the values of y variables on each collocation point.
'''


@cuda.jit
def collocation_update_kernel(size_y, m_min, m_max, N, d_m_N, d_m_accumulate, d_m_sum,
                              d_a_m, d_t_span, d_y, d_y_dot, d_sum_j, d_y_tilde):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    shared_d_a = cuda.shared.array(shape=(global_m_sum, global_m_max), dtype=float64)
    shared_d_y = cuda.shared.array(shape=(TPB_N, global_size_y), dtype=float64)
    shared_d_y_dot = cuda.shared.array(shape=(global_y_shared_size, global_size_y), dtype=float64)

    # cuda thread index
    i = cuda.grid(1)
    tx = cuda.threadIdx.x  # thread index in x direction
    bx = cuda.blockIdx.x  # block index in x direction

    if i >= (N - 1):
        return

    m = d_m_N[i]
    if tx == 0:
        # load coefficients a to shared memory using the first thread in x direction
        for m_index in range(m_max - m_min + 1):
            # load 2d coefficients a if the thread index is in range
            for j in range(d_m_sum[m_index], d_m_sum[m_index + 1]):
                for k in range(m_min + m_index):
                    shared_d_a[j, k] = d_a_m[j, k]
    # load d_y to shared memory
    for l in range(size_y):
        # load d_y to shared memory using the first thread in y direction
        shared_d_y[tx, l] = d_y[i, l]
    # load d_y_dot to shared memory
    for j in range(d_m_accumulate[i], d_m_accumulate[i + 1]):
        for l in range(size_y):
            # load d_y_dot to shared memory
            # each thread loads the all corresponding collocation points
            shared_d_y_dot[j - d_m_accumulate[bx * TPB_N], l] = d_y_dot[j, l]
    cuda.syncthreads()

    # if tx == 0 and bx == 0:
    #     from pdb import set_trace
    #     set_trace()

    delta_t_i = d_t_span[i + 1] - d_t_span[i]
    m_start = d_m_sum[m - m_min]
    for j in range(m):
        # loop j for each collocation point t_ij
        # zero the initial value
        for l in range(size_y):
            d_sum_j[d_m_accumulate[i] + j, l] = 0
        # loop k to perform the integral on all collocation points
        for k in range(m):
            # loop l to loop over all the y variables
            for l in range(size_y):
                d_sum_j[d_m_accumulate[i] + j, l] += \
                    shared_d_a[m_start + j, k] * shared_d_y_dot[d_m_accumulate[i] + k - d_m_accumulate[bx * TPB_N], l]
        # loop l to loop over all the y variables to update the result
        for l in range(size_y):
            d_y_tilde[d_m_accumulate[i] + j, l] = shared_d_y[tx, l] + delta_t_i * d_sum_j[d_m_accumulate[i] + j, l]
    return


# finish the implementation of computing the residual


# start the implementation of constructing the Jacobian matrix


def construct_jacobian_parallel(size_y, size_z, size_p, m_min, m_max, N, m_N, m_accumulate, m_sum,
                                a_m, b_m, t_span, y, y_tilde, z_tilde, p, alpha):
    """
    Compute each small matrix elements in the Jacobian of the system.
    :param size_y: number of ode variables of the problem
    :param size_z: number of algebraic variables
    :param size_p: number of parameter variables
    :param m_min: minimum number of collocation points allowed
    :param m_max: maximum number of collocation points allowed
    :param N: number of time nodes of the problem
    :param m_N: number of collocation points in each interval
    :param m_accumulate: accumulated collocation points in each interval
    :param m_sum: accumulated collocation sums in collocation coefficients
    :param a_m: stack coefficients a in lobatto method in feasible range [m_min, m_max]
                in row dominated order
                shape: (m_sum[-1], m_max)
    :param b_m: stack coefficients b in lobbatto method in feasible range [m_min, m_max]
                in row dominated order
                shape: (m_max - m_min + 1, m_max)
    :param t_span: time span of the mesh
    :param y: values of the ODE variables in matrix form
    :param y_tilde: values of ODE variables y at each collocation point in row dominant matrix for
            shape: (m_accumulate[-1], size_y), where each row corresponds the values at each collocation point
    :param z_tilde: values of the DAE variables in row dominant matrix form
           dimension: (m_accumulate[-1], size_z), where each row corresponds the values at each time node.
           The index for the jth collocation point from the ith time node is i * m + j.
    :param p: values of the parameter variables in vector form
           shape: (size_p, )
    :param alpha: continuation parameter of the Newton method
    :return:
        J: the J matrix element in the Jacobian matrix in row dominant matrix form
           shape: (m_accumulate[-1] * (size_y + size_z), size_y)
           each m * (size_y + size_z) x size_y corresponds to a matrix block at a time node
        V: the V matrix element in the Jacobian matrix in row dominant matrix form
           shape: (m_accumulate[-1] * (size_y + size_z), size_p)
           each m * (size_y + size_z) x size_p corresponds to a matrix block at a time node
        D: the D matrix element in the Jacobian matrix in row dominant matrix form
           shape: ((N - 1) * size_y, m_max * (size_y + size_z))
           each size_y x m * (size_y + size_z) corresponds to a matrix block at a time node
        W: the D matrix element in the Jacobian matrix in row dominant matrix form
           shape: (m_accumulate[-1] * (size_y + size_z), m_max * (size_y + size_z))
           each m * (size_y + size_z) x m * (size_y + size_z) corresponds to a matrix block at a time node
        B_0: derivatives of boundary conditions w.r.t. ODE variables at initial time
             dimension: size_y + size_p x size_y
        B_n: derivatives of boundary conditions w.r.t. ODE variables at final time
             dimension: size_y + size_p x size_y
        V_n: derivatives of boundary conditions w.r.t. parameter varaibels
             dimension: size_y + size_p x size_p
    """
    # grid dimension of the kernel of CUDA
    grid_dims_1d = ((N - 1) + TPB_N - 1) // TPB_N
    block_dims_1d = TPB_N
    # transfer memory from CPU to GPU
    d_m_N = cuda.to_device(m_N)
    d_m_accumulate = cuda.to_device(m_accumulate)
    d_m_sum = cuda.to_device(m_sum)
    d_a_m = cuda.to_device(a_m)
    d_b_m = cuda.to_device(b_m)
    d_t_span = cuda.to_device(t_span)
    d_p = cuda.to_device(p)
    d_y_tilde = cuda.to_device(y_tilde)
    d_z_tilde = cuda.to_device(z_tilde)
    # container to hold the derivatives
    '''
        large row dominated matrix 
        start_row_index : i * m * size_y + j * size_y
        end_row_index : start_index + size_y
        d_d_h[start_row_index : end_row_index, :] can access the derivatives of the ODE
        equations at the jth collocation node of the ith time span
        d_d_g[start_row_index : end_row_index, :] can access the derivatives of the DAE
        equations at the jth collocation node of the ith time span
    '''
    # no zero initialize here, initialize it in the kernel
    d_d_h = cuda.device_array((size_y * m_accumulate[-1], (size_y + size_z + size_p)), dtype=np.float64)
    d_d_g = cuda.device_array((size_z * m_accumulate[-1], (size_y + size_z + size_p)), dtype=np.float64)
    '''
        large row dominant matrix 
        start_index : i * m * (size_y + size_z + size_p) + j * (size_y + size_z)
        end_index : start_index + (size_y + size_z + size_p)
        d_J[start_index : end_index, 0 : size_y] can access the elements associated with
        the jth collocation node of the ith time span
        d_V[start_index : end_index, 0 : size_p] can access the elements associated with
        the jth collocation node of the ith time span
    '''
    # holder for the output variables
    d_J = cuda.device_array(((size_y + size_z) * m_accumulate[-1], size_y), dtype=np.float64)
    d_V = cuda.device_array(((size_y + size_z) * m_accumulate[-1], size_p), dtype=np.float64)
    d_W = cuda.device_array(((size_y + size_z) * m_accumulate[-1], m_max * (size_y + size_z)), dtype=np.float64)
    # no zero initialization, initialize it in the kernel
    d_D = cuda.device_array(((N - 1) * size_y, m_max * (size_y + size_z)), dtype=np.float64)
    # construct the J, V, W, and D matrix on CUDA kernel
    construct_jacobian_kernel[grid_dims_1d, block_dims_1d](size_y, size_z, size_p, m_min, N,
                                                           d_m_N, d_m_accumulate, d_m_sum, d_a_m, d_b_m,
                                                           d_t_span, d_y_tilde, d_z_tilde, d_p, alpha,
                                                           d_d_h, d_d_g, d_J, d_V, d_D, d_W)
    # compute the derivative of the boundary conditions
    d_r = np.zeros(((size_y + size_p), (size_y + size_y + size_p)), dtype=np.float64)
    y_i = y[0, :]  # y values at the initial time
    y_f = y[N - 1, :]  # y values at the final time
    _abvp_Dr(y_i, y_f, p, d_r)
    B_0 = d_r[0: size_y + size_p, 0: size_y]  # B_1 in the paper
    B_n = d_r[0: size_y + size_p, size_y: size_y + size_y]  # B_N in the paper
    V_n = d_r[0: size_y + size_p, size_y + size_y: size_y + size_y + size_p]  # V_N in the paper
    return d_J.copy_to_host(), d_V.copy_to_host(), d_D.copy_to_host(), d_W.copy_to_host(), B_0, B_n, V_n


@cuda.jit()
def construct_jacobian_kernel(size_y, size_z, size_p, m_min, N,
                              d_m_N, d_m_accumulate, d_m_sum, d_a_m, d_b_m,
                              d_t_span, d_y_tilde, d_z_tilde, d_p, alpha,
                              d_d_h, d_d_g, d_J, d_V, d_D, d_W):
    """
    Kernel function for computing each element J, V, D, W in the Jacobian matrix
    :param size_y:
    :param size_z:
    :param size_p:
    :param m_min:
    :param N:
    :param d_m_N:
    :param d_m_accumulate:
    :param d_m_sum:
    :param d_a_m:
    :param d_b_m:
    :param d_t_span:
    :param d_y_tilde:
    :param d_z_tilde:
    :param d_p:
    :param alpha:
    :param d_d_h:
    :param d_d_g:
    :param d_J:
    :param d_V:
    :param d_D:
    :param d_W:
    :return:
        d_d_h : size_y x m * (N - 1) * (size_y + size_z + size_p)
        d_d_g : size_z x m * (N - 1) * (size_y + size_z + size_p)
        d_J : m * (size_y + size_z) * (N - 1) x size_y
        d_V : m * (size_y + size_z) * (N - 1) x size_p
        d_D : (N - 1) * size_y x m * (size_y + size_z)
        d_W : m * (size_y + size_z) * (N - 1) x m * (size_y + size_z)
    """
    i = cuda.grid(1)
    if i >= (N - 1):
        return

    m = d_m_N[i]
    m_start = d_m_sum[m - m_min]  # start index in a coefficients

    delta_t_i = d_t_span[i + 1] - d_t_span[i]
    for j in range(m):
        # the block index for each derivative of d_h
        start_row_index_d_h = d_m_accumulate[i] * size_y + j * size_y
        end_row_index_d_h = start_row_index_d_h + size_y
        # zero initialize the derivative matrix
        for row in range(start_row_index_d_h, end_row_index_d_h):
            for col in range(0, size_y + size_z + size_p):
                d_d_h[row, col] = 0
        # compute the derivatives
        _abvp_Df(d_y_tilde[d_m_accumulate[i] + j, 0: size_y], d_z_tilde[d_m_accumulate[i] + j, 0: size_z], d_p,
                 d_d_h[start_row_index_d_h: end_row_index_d_h, 0: size_y + size_z + size_p])
        # the block index for each derivative of d_g
        start_row_index_d_g = d_m_accumulate[i] * size_z + j * size_z
        end_row_index_d_g = start_row_index_d_g + size_z
        # zero initialize the derivative matrix
        for row in range(start_row_index_d_g, end_row_index_d_g):
            for col in range(0, size_y + size_z + size_p):
                d_d_g[row, col] = 0
        # compute the derivatives
        _abvp_Dg(d_y_tilde[d_m_accumulate[i] + j, 0: size_y], d_z_tilde[d_m_accumulate[i] + j, 0: size_z], d_p, alpha,
                 d_d_g[start_row_index_d_g: end_row_index_d_g, 0: size_y + size_z + size_p])
        '''
            indexing for each derivatives
            h_y = d_d_h[start_row_index_d_h: end_row_index_d_h, 0: size_y])
            h_z = d_d_h[start_row_index_d_h: end_row_index_d_h, size_y: size_y + size_z])
            h_p = d_d_h[start_row_index_d_h: end_row_index_d_h, size_y + size_z: size_y + size_z + size_p])
            g_y = d_d_g[start_row_index_d_g: end_row_index_d_g, 0: size_y]
            g_z = d_d_g[start_row_index_d_g: end_row_index_d_g, size_y: size_y + size_z]
            g_p = d_d_g[start_row_index_d_g: end_row_index_d_g, size_y + size_z: size_y + size_z + size_p]
        '''
        # construct the J and V matrix
        start_index_JV_h = d_m_accumulate[i] * (size_y + size_z) + j * (size_y + size_z)
        start_index_JV_g = start_index_JV_h + size_y
        for row in range(size_y):
            for col in range(size_y):
                d_J[start_index_JV_h + row, col] = d_d_h[start_row_index_d_h + row, col]
            for col in range(size_p):
                d_V[start_index_JV_h + row, col] = d_d_h[start_row_index_d_h + row, size_y + size_z + col]
        for row in range(size_z):
            for col in range(size_y):
                d_J[start_index_JV_g + row, col] = d_d_g[start_row_index_d_g + row, col]
            for col in range(size_p):
                d_V[start_index_JV_g + row, col] = d_d_g[start_row_index_d_g + row, size_y + size_z + col]
        # construct the D matrix
        start_row_index_D = i * size_y
        start_col_index_D = j * (size_y + size_z)
        for row in range(size_y):
            for col in range(size_y + size_z):
                if row == col:
                    d_D[start_row_index_D + row, start_col_index_D + col] = delta_t_i * d_b_m[m - m_min, j]
                else:
                    d_D[start_row_index_D + row, start_col_index_D + col] = 0.0
        # construct the W matrix
        # j associates the corresponding row block
        # start_row_index_W = i * m * (size_y + size_z) + j * (size_y + size_z)
        # loop through the m column blocks
        # each column block is size (size_y + size_z) x (size_y + size_z)
        for k in range(m):
            # start row index for the top block in W matrix
            start_row_index_W_top = d_m_accumulate[i] * (size_y + size_z) + j * (size_y + size_z)
            # start row index for the bottom block in W matrix
            start_row_index_W_bot = start_row_index_W_top + size_y
            # start column index for the left block in W matrix
            start_col_index_W_left = k * (size_y + size_z)
            # start column index for the right block in W matrix
            start_col_index_W_right = start_col_index_W_left + size_y
            # for the diagonal block
            if k == j:
                # top left block: -I + delta_t_i * a[j, k] * h_y
                for ii in range(size_y):
                    for jj in range(size_y):
                        # diagonal element
                        if ii == jj:
                            d_W[start_row_index_W_top + ii, start_col_index_W_left + jj] = \
                                -1.0 + delta_t_i * d_a_m[m_start + j, k] * d_d_h[start_row_index_d_h + ii, jj]
                        else:
                            d_W[start_row_index_W_top + ii, start_col_index_W_left + jj] = \
                                delta_t_i * d_a_m[m_start + j, k] * d_d_h[start_row_index_d_h + ii, jj]
                # top right block: h_z
                for ii in range(size_y):
                    for jj in range(size_z):
                        d_W[start_row_index_W_top + ii, start_col_index_W_right + jj] = \
                            d_d_h[start_row_index_d_h + ii, size_y + jj]
                # bottom left block: delta_t_i * a[j, k] * g_y
                for ii in range(size_z):
                    for jj in range(size_y):
                        d_W[start_row_index_W_bot + ii, start_col_index_W_left + jj] = \
                            delta_t_i * d_a_m[m_start + j, k] * d_d_g[start_row_index_d_g + ii, jj]
                # bottom right block: g_z
                for ii in range(size_z):
                    for jj in range(size_z):
                        d_W[start_row_index_W_bot + ii, start_col_index_W_right + jj] = \
                            d_d_g[start_row_index_d_g + ii, size_y + jj]
            else:
                # top left block: delta_t_i * a[j, k] * h_y
                for ii in range(size_y):
                    for jj in range(size_y):
                        d_W[start_row_index_W_top + ii, start_col_index_W_left + jj] = \
                            delta_t_i * d_a_m[m_start + j, k] * d_d_h[start_row_index_d_h + ii, jj]
                # top right block: 0s
                for ii in range(size_y):
                    for jj in range(size_z):
                        d_W[start_row_index_W_top + ii, start_col_index_W_right + jj] = 0
                # bottom left block: delta_t_i * a[j, k] * g_y
                for ii in range(size_z):
                    for jj in range(size_y):
                        d_W[start_row_index_W_bot + ii, start_col_index_W_left + jj] = \
                            delta_t_i * d_a_m[m_start + j, k] * d_d_g[start_row_index_d_g + ii, jj]
                # bottom right block: 0s
                for ii in range(size_z):
                    for jj in range(size_z):
                        d_W[start_row_index_W_bot + ii, start_col_index_W_right + jj] = 0
    return


def reduce_jacobian_parallel(size_y, size_z, size_p, m_max, N, m_N, m_accumulate, W, D, J, V, f_a, f_b):
    """
    Construct the reduced BABD system with a self-implemented LU factorization solver.
    :param size_y: number of ode variables of the problem
    :param size_z: number of algebraic variables
    :param size_p: number of parameter variables
    :param m_max: maximum number of collocation points allowed
    :param N: number of time nodes of the problem
    :param m_N: number of collocation points in each interval
    :param m_accumulate: accumulated collocation points in each interval
    :param W: the D matrix element in the Jacobian matrix in row dominant matrix form
           shape: (m_accumulate[-1] * (size_y + size_z), m_max * (size_y + size_z))
           each m * (size_y + size_z) x m * (size_y + size_z) corresponds to a matrix block at a time node
    :param D: he D matrix element in the Jacobian matrix in row dominant matrix form
           shape: ((N - 1) * size_y, m_max * (size_y + size_z))
           each size_y x m * (size_y + size_z) corresponds to a matrix block at a time node
    :param J: the J matrix element in the Jacobian matrix in row dominant matrix form
           shape: (m_accumulate[-1] * (size_y + size_z), size_y)
           each m * (size_y + size_z) x size_y corresponds to a matrix block at a time node
    :param V: the V matrix element in the Jacobian matrix in row dominant matrix form
           shape: (m_accumulate[-1] * (size_y + size_z), size_p)
           each m * (size_y + size_z) x size_p corresponds to a matrix block at a time node
    :param f_a: matrix of the residual f_a for each time node in row dominant matrix form
            shape: ((N - 1), m_max * (size_y + size_z)), where each row corresponds the values at each time node
    :param f_b: matrix of the residual f_b for each time node in row dominant matrix form
            shape: (N - 1, size_y), where each row corresponds the values at each time node
    :return:
        A : the A matrix element in the reduced Jacobian matrix in a row dominant matrix form
            shape: ((N - 1) * size_y, size_y)
            each size_y x size_y corresponds to a matrix block at a time node
        C : the C matrix element in the reduced Jacobian matrix in a row dominant matrix form
            shape: ((N - 1) * size_y, size_y)
            each size_y x size_y corresponds to a matrix block at a time node
        H : the H matrix element in the reduced Jacobian matrix in a row dominant matrix form
            shape: ((N - 1) * size_y, size_p)
            each size_y x size_p corresponds to a matrix block at a time node
        b : the b vector element of the residual in the reduced BABD system in a row dominant matrix form
            shape: (N - 1, size_y)
            each row vector with size size_y corresponds to a vector block at a time node
    """
    # grid dimension of the kernel of CUDA
    grid_dims_1d = ((N - 1) + TPB_N - 1) // TPB_N
    block_dims_1d = TPB_N
    # transfer memory from CPU to GPU
    d_m_N = cuda.to_device(m_N)
    d_m_accumulate = cuda.to_device(m_accumulate)
    d_W = cuda.to_device(W)
    d_D = cuda.to_device(D)
    d_J = cuda.to_device(J)
    d_V = cuda.to_device(V)
    d_f_a = cuda.to_device(f_a)
    d_f_b = cuda.to_device(f_b)
    # holder for output variables
    d_A = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_C = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_H = cuda.device_array(((N - 1) * size_y, size_p), dtype=np.float64)
    d_b = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # holder for the intermediate variables in lu decomposition
    d_P = cuda.device_array((m_accumulate[-1] * (size_y + size_z), m_max * (size_y + size_z)), dtype=np.float64)
    d_L = cuda.device_array((m_accumulate[-1] * (size_y + size_z), m_max * (size_y + size_z)), dtype=np.float64)
    d_U = cuda.device_array((m_accumulate[-1] * (size_y + size_z), m_max * (size_y + size_z)), dtype=np.float64)
    d_cpy = cuda.device_array((m_accumulate[-1] * (size_y + size_z), m_max * (size_y + size_z)), dtype=np.float64)
    d_c_J = cuda.device_array((m_accumulate[-1] * (size_y + size_z), size_y), dtype=np.float64)
    d_y_J = cuda.device_array((m_accumulate[-1] * (size_y + size_z), size_y), dtype=np.float64)
    d_x_J = cuda.device_array((m_accumulate[-1] * (size_y + size_z), size_y), dtype=np.float64)
    d_D_W_J = cuda.device_array(((N - 1) * size_y, size_y), dtype=np.float64)
    d_c_V = cuda.device_array((m_accumulate[-1] * (size_y + size_z), size_p), dtype=np.float64)
    d_y_V = cuda.device_array((m_accumulate[-1] * (size_y + size_z), size_p), dtype=np.float64)
    d_x_V = cuda.device_array((m_accumulate[-1] * (size_y + size_z), size_p), dtype=np.float64)
    d_c_f_a = cuda.device_array((N - 1, m_max * (size_y + size_z)), dtype=np.float64)
    d_y_f_a = cuda.device_array((N - 1, m_max * (size_y + size_z)), dtype=np.float64)
    d_x_f_a = cuda.device_array((N - 1, m_max * (size_y + size_z)), dtype=np.float64)
    d_D_W_f_a = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # machine precision
    eps = sys.float_info.epsilon
    # perform the jacobian reduction in parallel with the GPU kernel
    reduce_jacobian_parallel_kernel[grid_dims_1d, block_dims_1d](
        eps, size_y, size_z, size_p, N,
        d_m_N, d_m_accumulate,
        d_W, d_D, d_J, d_V, d_f_a, d_f_b,
        d_P, d_L, d_U, d_cpy, d_c_J, d_y_J, d_x_J, d_D_W_J,
        d_c_V, d_y_V, d_x_V, d_c_f_a, d_y_f_a, d_x_f_a, d_D_W_f_a,
        d_A, d_C, d_H, d_b)
    return d_A.copy_to_host(), d_C.copy_to_host(), d_H.copy_to_host(), d_b.copy_to_host()


@cuda.jit()
def reduce_jacobian_parallel_kernel(eps, size_y, size_z, size_p, N,
                                    d_m_N, d_m_accumulate,
                                    d_W, d_D, d_J, d_V, d_f_a, d_f_b,
                                    d_P, d_L, d_U, d_cpy, d_c_J, d_y_J, d_x_J, d_D_W_J,
                                    d_c_V, d_y_V, d_x_V, d_c_f_a, d_y_f_a, d_x_f_a, d_D_W_f_a,
                                    d_A, d_C, d_H, d_b):
    """
    Kernel function for computing each element A, C, H, b in the reduced Jacobian matrix.
    :param eps:
    :param size_y:
    :param size_z:
    :param size_p:
    :param N:
    :param d_m_N:
    :param d_m_accumulate:
    :param d_W:
    :param d_D:
    :param d_J:
    :param d_V:
    :param d_f_a:
    :param d_f_b:
    :param d_P:
    :param d_L:
    :param d_U:
    :param d_cpy:
    :param d_c_J:
    :param d_y_J:
    :param d_x_J:
    :param d_D_W_J:
    :param d_c_V:
    :param d_y_V:
    :param d_x_V:
    :param d_c_f_a:
    :param d_y_f_a:
    :param d_x_f_a:
    :param d_D_W_f_a:
    :param d_A:
    :param d_C:
    :param d_H:
    :param d_b:
    :return:
    """
    i = cuda.grid(1)
    if i < (N - 1):
        m = d_m_N[i]
        # start row index of the W element
        start_row_index_W = d_m_accumulate[i] * (size_y + size_z)
        # end row index of the W element
        end_row_index_W = start_row_index_W + m * (size_y + size_z)
        # start row index of the D element
        start_row_index_D = i * size_y
        # end row index of the D element
        end_row_index_D = start_row_index_D + size_y
        # start row index of the J element
        start_row_index_J = d_m_accumulate[i] * (size_y + size_z)
        # end row index of the J element
        end_row_index_J = start_row_index_J + m * (size_y + size_z)
        # start row index of the V element
        start_row_index_V = d_m_accumulate[i] * (size_y + size_z)
        # end row index of the V element
        end_row_index_V = start_row_index_V + m * (size_y + size_z)
        # start row index for A, C, and H
        start_row_index_ACH = i * size_y
        # end row index for A, C, and H
        end_row_index_ACH = start_row_index_ACH + size_y
        # perform LU decomposition of matrix W and save the results
        # P * W = L * U
        matrix_factorization_cuda.lu(
            d_W[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_cpy[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_P[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_L[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_U[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            eps)
        # A = -I + D * W^(-1) * J
        # compute W^(-1) * J = X first
        # J = W * X => P * J = P * W * X = L * U * X => L^{-1} * P * J = U * X => U^{-1} * L^{-1} * P * J = X
        # compute P * J first, the result of the product is saved in d_c_J
        matrix_operation_cuda.mat_mul(
            d_P[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_J[start_row_index_J: end_row_index_J, 0: size_y],
            d_c_J[start_row_index_J: end_row_index_J, 0: size_y])
        # first, forward solve the linear system L * (U * X) = (P * J), and the result is saved in d_y_J
        matrix_factorization_cuda.forward_solve_mat(
            d_L[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_c_J[start_row_index_J: end_row_index_J, 0: size_y],
            d_y_J[start_row_index_J: end_row_index_J, 0: size_y],
            eps)
        # then, backward solve the linear system U * X = Y, and the result is saved in d_x_J
        # X = W^(-1) * J
        matrix_factorization_cuda.backward_solve_mat(
            d_U[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_y_J[start_row_index_J: end_row_index_J, 0: size_y],
            d_x_J[start_row_index_J: end_row_index_J, 0: size_y],
            eps)
        # perform D * X
        matrix_operation_cuda.mat_mul(
            d_D[start_row_index_D: end_row_index_D, 0: m * (size_y + size_z)],
            d_x_J[start_row_index_J: end_row_index_J, 0: size_y],
            d_D_W_J[start_row_index_D: end_row_index_D, 0: size_y])
        # final step, A = -I + D * X
        # nested for loops, row-wise first, column-wise next
        for j in range(size_y):
            for k in range(size_y):
                if j == k:
                    d_A[start_row_index_ACH + j, k] = -1.0 + d_D_W_J[start_row_index_D + j, k]
                else:
                    d_A[start_row_index_ACH + j, k] = d_D_W_J[start_row_index_D + j, k]
        # H = D * W^(-1) * V
        # compute W^(-1) * V = X first
        # compute P * V first, the result of the product is saved in d_c_V
        matrix_operation_cuda.mat_mul(
            d_P[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_V[start_row_index_V: end_row_index_V, 0: size_p],
            d_c_V[start_row_index_V: end_row_index_V, 0: size_p])
        # first, forward solve the linear system L * (U * X) = (P * V), and the result is saved in d_y_V
        matrix_factorization_cuda.forward_solve_mat(
            d_L[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_c_V[start_row_index_V: end_row_index_V, 0: size_p],
            d_y_V[start_row_index_V: end_row_index_V, 0: size_p],
            eps)
        # then, backward solve the linear system U * X = Y, and the result is saved in d_x_V
        # X = W^(-1) * V
        matrix_factorization_cuda.backward_solve_mat(
            d_U[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_y_V[start_row_index_V: end_row_index_V, 0: size_p],
            d_x_V[start_row_index_V: end_row_index_V, 0: size_p],
            eps)
        # final step, perform D * X, then we get the results H
        matrix_operation_cuda.mat_mul(
            d_D[start_row_index_D: end_row_index_D, 0: m * (size_y + size_z)],
            d_x_V[start_row_index_V: end_row_index_V, 0: size_p],
            d_H[start_row_index_ACH: end_row_index_ACH, 0: size_p])
        # C = I
        # nested for loops, row-wise first, column-wise next
        for j in range(size_y):
            for k in range(size_y):
                if j == k:
                    d_C[start_row_index_ACH + j, k] = 1.0
                else:
                    d_C[start_row_index_ACH + j, k] = 0.0
        # b = -f_b - D * W^(-1) * f_a
        # compute W^(-1) * f_a = X first
        # compute P * f_a first, the result of the product is saved in d_c_f_a
        matrix_operation_cuda.mat_vec_mul(
            d_P[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_f_a[i, 0: m * (size_y + size_z)],
            d_c_f_a[i, 0: m * (size_y + size_z)])
        # first, forward solve the linear system L * (U * X) = (P * f_a), and the result is saved in d_y_f_a
        matrix_factorization_cuda.forward_solve_vec(
            d_L[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_c_f_a[i, 0: m * (size_y + size_z)],
            d_y_f_a[i, 0: m * (size_y + size_z)],
            eps)
        # then, backward solve the linear system U * X = Y, and the result is saved in d_x_f_a
        # X = W^(-1) * f_a
        matrix_factorization_cuda.backward_solve_vec(
            d_U[start_row_index_W: end_row_index_W, 0: m * (size_y + size_z)],
            d_y_f_a[i, 0: m * (size_y + size_z)],
            d_x_f_a[i, 0: m * (size_y + size_z)],
            eps)
        # perform D * X
        matrix_operation_cuda.mat_vec_mul(
            d_D[start_row_index_D: end_row_index_D, 0: m * (size_y + size_z)],
            d_x_f_a[i, 0: m * (size_y + size_z)],
            d_D_W_f_a[i, 0: size_y])
        # final step, b = -f_b - D * W^(-1) * f_a
        for j in range(size_y):
            d_b[i, j] = -d_f_b[i, j] - d_D_W_f_a[i, j]
    return

# finish the implementation of constructing the Jacobian matrix


# start the implementation of the parallel recovering the delta_k


def recover_delta_k_parallel(size_y, size_z, size_p, m_max, N, m_N, m_accumulate, delta_y, delta_p, f_a, J, V, W):
    """
    Recover the delta_k of the search direction from the reduced BABD system.
    :param size_y: number of ode variables of the problem
    :param size_z: number of algebraic variables
    :param size_p: number of parameter variables
    :param m_max: maximum number of collocation points allowed
    :param N: number of time nodes of the problem
    :param m_N: number of collocation points in each interval
    :param m_accumulate: accumulated collocation points in each interval
    :param delta_y: search direction of the ode variables obtained from Newton's method
                    shape: (N, size_y)
    :param delta_p: search direction of the parameter variables obtained from Newton's method
                    shape: (size_p, )
    :param f_a: matrix of the residual f_a for each time node in row dominant matrix form
            shape: ((N - 1), m_max * (size_y + size_z)), where each row corresponds the values at each time node
    :param J: the J matrix element in the Jacobian matrix in row dominant matrix form
           shape: (m_accumulate[-1] * (size_y + size_z), size_y)
           each m * (size_y + size_z) x size_y corresponds to a matrix block at a time node
    :param V: V: the V matrix element in the Jacobian matrix in row dominant matrix form
           shape: (m_accumulate[-1] * (size_y + size_z), size_p)
           each m * (size_y + size_z) x size_p corresponds to a matrix block at a time node
    :param W: the D matrix element in the Jacobian matrix in row dominant matrix form
           shape: (m_accumulate[-1] * (size_y + size_z), m_max * (size_y + size_z))
           each m * (size_y + size_z) x m * (size_y + size_z) corresponds to a matrix block at a time node
    :return:
         delta_k: solution of the search direction of y_dot and z variables of the system recovered from the reduced BABD
                 system
                 shape: ((N - 1), m_max * (size_y + size_z))
                 each size (size_y + size_z) vector corresponds to the search direction at each time node
        delta_y_dot: solution of the search direction of y_dot from corresponding position at delta_k
                     shape: (m_accumulate[-1], size_y)
                     each size size_y row vector corresponds to the search direction at the corresponding collocation
                     point. The index for the jth collocation point from the ith time node is i * x + j
        delta_z_tilde: solution of the search direction of z_tilde from corresponding position at delta_k
                     shape: (m_accumulate[-1], size_z)
                     each size size_z row vector corresponds to the search direction at the corresponding collocation
                     point. The index for the jth collocation point from the ith time node is i * x + j
    """
    # grid dimension of the kernel of CUDA
    grid_dims_1d = ((N - 1) + TPB_N - 1) // TPB_N
    block_dims_1d = TPB_N
    # transfer memory from CPU to GPU
    d_m_N = cuda.to_device(m_N)
    d_m_accumulate = cuda.to_device(m_accumulate)
    d_delta_y = cuda.to_device(delta_y)
    d_delta_p = cuda.to_device(delta_p)
    d_f_a = cuda.to_device(f_a)
    d_J = cuda.to_device(J)
    d_V = cuda.to_device(V)
    d_W = cuda.to_device(W)
    # holder for output variable
    d_delta_k = cuda.device_array((N - 1, m_max * (size_y + size_z)), dtype=np.float64)
    d_delta_y_dot = cuda.device_array((m_accumulate[-1], size_y), dtype=np.float64)
    d_delta_z_tilde = cuda.device_array((m_accumulate[-1], size_z), dtype=np.float64)
    # holder for intermediate matrix vector multiplication variables
    d_J_delta_y = cuda.device_array((N - 1, m_max * (size_y + size_z)), dtype=np.float64)
    d_V_delta_p = cuda.device_array((N - 1, m_max * (size_y + size_z)), dtype=np.float64)
    # holder for the right hand side of the linear system to solve in BABD system
    d_vec = cuda.device_array((N - 1, m_max * (size_y + size_z)), dtype=np.float64)
    # holder for the intermediate variables in lu decomposition
    d_P = cuda.device_array((m_accumulate[-1] * (size_y + size_z), m_max * (size_y + size_z)), dtype=np.float64)
    d_L = cuda.device_array((m_accumulate[-1] * (size_y + size_z), m_max * (size_y + size_z)), dtype=np.float64)
    d_U = cuda.device_array((m_accumulate[-1] * (size_y + size_z), m_max * (size_y + size_z)), dtype=np.float64)
    d_cpy = cuda.device_array((m_accumulate[-1] * (size_y + size_z), m_max * (size_y + size_z)), dtype=np.float64)
    d_c = cuda.device_array((N - 1, m_max * (size_y + size_z)), dtype=np.float64)
    d_y = cuda.device_array((N - 1, m_max * (size_y + size_z)), dtype=np.float64)
    # machine precision
    eps = sys.float_info.epsilon
    recover_delta_k_kernel[grid_dims_1d, block_dims_1d](
        eps, size_y, size_z, size_p, N, d_m_N, d_m_accumulate,
        d_delta_y, d_delta_p, d_f_a, d_J, d_V, d_W,
        d_J_delta_y, d_V_delta_p, d_vec, d_P, d_L, d_U, d_cpy, d_c, d_y,
        d_delta_k, d_delta_y_dot, d_delta_z_tilde)
    return d_delta_k.copy_to_host(), d_delta_y_dot.copy_to_host(), d_delta_z_tilde.copy_to_host()


@cuda.jit()
def recover_delta_k_kernel(eps, size_y, size_z, size_p, N, d_m_N, d_m_accumulate,
                           d_delta_y, d_delta_p, d_f_a, d_J, d_V, d_W,
                           d_J_delta_y, d_V_delta_p, d_vec, d_P, d_L, d_U, d_cpy, d_c, d_y,
                           d_delta_k, d_delta_y_dot, d_delta_z_tilde):
    """
    Kernel function of recovering delta_k from the reduced BABD system.
    :param eps:
    :param size_y:
    :param size_z:
    :param size_p:
    :param N:
    :param d_m_N:
    :param d_m_accumulate:
    :param d_delta_y:
    :param d_delta_p:
    :param d_f_a:
    :param d_J:
    :param d_V:
    :param d_W:
    :param d_J_delta_y:
    :param d_V_delta_p:
    :param d_vec:
    :param d_P:
    :param d_L:
    :param d_U:
    :param d_cpy:
    :param d_c:
    :param d_y:
    :param d_delta_k:
    :param d_delta_y_dot:
    :param d_delta_z_tilde:
    :return:
    """
    i = cuda.grid(1)
    if i < (N - 1):
        m = d_m_N[i]
        # start row index of the J element
        start_row_index_JVW = d_m_accumulate[i] * (size_y + size_z)
        # end row index of the J element
        end_row_index_JVW = start_row_index_JVW + m * (size_y + size_z)
        matrix_operation_cuda.mat_vec_mul(
            d_J[start_row_index_JVW: end_row_index_JVW, 0: size_y],
            d_delta_y[i, 0: size_y],
            d_J_delta_y[i, 0: m * (size_y + size_z)])
        matrix_operation_cuda.mat_vec_mul(
            d_V[start_row_index_JVW: end_row_index_JVW, 0: size_p],
            d_delta_p,
            d_V_delta_p[i, 0: m * (size_y + size_z)])
        for j in range(m * (size_y + size_z)):
            d_vec[i, j] = -d_f_a[i, j] - d_J_delta_y[i, j] - d_V_delta_p[i, j]
        matrix_factorization_cuda.lu_solve_vec(
            d_W[start_row_index_JVW: end_row_index_JVW, 0: m * (size_y + size_z)],
            d_cpy[start_row_index_JVW: end_row_index_JVW, 0: m * (size_y + size_z)],
            d_P[start_row_index_JVW: end_row_index_JVW, 0: m * (size_y + size_z)],
            d_L[start_row_index_JVW: end_row_index_JVW, 0: m * (size_y + size_z)],
            d_U[start_row_index_JVW: end_row_index_JVW, 0: m * (size_y + size_z)],
            d_vec[i, 0: m * (size_y + size_z)],
            d_c[i, 0: m * (size_y + size_z)],
            d_y[i, 0: m * (size_y + size_z)],
            d_delta_k[i, 0: m * (size_y + size_z)],
            eps)
        for j in range(m):
            start_index_y_collocation = j * (size_y + size_z)
            start_index_z_collocation = start_index_y_collocation + size_y
            for k in range(size_y):
                d_delta_y_dot[d_m_accumulate[i] + j, k] = d_delta_k[i, start_index_y_collocation + k]
            for k in range(size_z):
                d_delta_z_tilde[d_m_accumulate[i] + j, k] = d_delta_k[i, start_index_z_collocation + k]
    return


# start the implementation of computing segment residual on each time node


'''
    Input:
        size_y: number of ODE variables.
        m: number of collocation points used
        t: time during the time span
        L: collocation weights vector
           dimension: m
        y_dot: value of the derivative of the ODE variables in time span [t_j, t_(j + 1)]
                dimension: m x size_y
    Output:
        y_dot_ret: returned value of the derivative of the ODE variables at time t
               dimension: size_y
'''


@cuda.jit(device=True)
def get_y_dot(size_y, m, t, L, y_dot, y_dot_ret):
    # ydot = sum_{k=1,m} L_k(t) * ydot_j[k]
    # zero initialization
    if L.shape[0] != m:
        print("Input L size is wrong! Supposed to be", m, "instead of", L.shape[0], "columns!")
        raise Exception("Input L size is wrong!")
    if y_dot.shape[0] != m:
        print("Input y_dot size is wrong! Supposed to be", m, "instead of", y_dot.shape[0], "rows!")
        raise Exception("Input y_dot size is wrong!")
    if y_dot.shape[1] != size_y:
        print("Input y_dot size is wrong! Supposed to be", size_y, "instead of", y_dot.shape[1], "columns!")
        raise Exception("Input y_dot size is wrong!")
    if y_dot_ret.shape[0] != size_y:
        print("Input y_dot_ret size is wrong! Supposed to be", size_y, "instead of", y_dot_ret.shape[0], "columns!")
        raise Exception("Input y_dot_ret size is wrong!")
    for i in range(size_y):
        y_dot_ret[i] = 0
    # compute the collocation weights at time t
    collocation_coefficients.compute_L(m, t, L)
    # perform collocation integration
    for i in range(m):
        for j in range(size_y):
            y_dot_ret[j] += L[i] * y_dot[i, j]
    return


'''
    Input:
        size_z: number of DAE variables.
        m: number of collocation points used
        t: time during the time span
        L: collocation weights vector
           dimension: m
        z_tilde: value of the DAE variables in time span [t_j, t_(j + 1)]
                 dimension: m x size_z
    Output:
        z: returned value of the derivative of the DAE variables at time t
           dimension: size_z
'''


@cuda.jit(device=True)
def get_z(size_z, m, t, L, z_tilde, z):
    # z = sum_{k=1,m} L_k(t) * z_j[k]
    if L.shape[0] != m:
        print("Input L size is wrong! Supposed to be", m, "instead of", L.shape[0], "columns!")
        raise Exception("Input L size is wrong!")
    if z_tilde.shape[0] != m:
        print("Input z_tilde size is wrong! Supposed to be", m, "instead of", z_tilde.shape[0], "rows!")
        raise Exception("Input z_tilde size is wrong!")
    if z_tilde.shape[1] != size_z:
        print("Input z_tilde size is wrong! Supposed to be", size_z, "instead of", z_tilde.shape[1], "columns!")
        raise Exception("Input z_tilde size is wrong!")
    if z.shape[0] != size_z:
        print("Input z size is wrong! Supposed to be", size_z, "instead of", z.shape[0], "columns!")
        raise Exception("Input z size is wrong!")
    # zero initialization
    for i in range(size_z):
        z[i] = 0
    # compute the collocation weights at time t
    collocation_coefficients.compute_L(m, t, L)
    for i in range(m):
        for j in range(size_z):
            z[j] += L[i] * z_tilde[i, j]
    return


'''
    Input:
        size_y: number of ODE variables.
        m: number of collocation points used
        t: time during the time span
        delta_t: time interval of the time span
        I: collocation weights vector
           dimension: m
        y_dot: value of the derivative of the ODE variables in time span [t_j, t_(j + 1)]
               dimension: m x size_y
    Output:
        y_ret: returned value of the ODE variables at time t
               dimension: size_y
'''


@cuda.jit(device=True)
def get_y(size_y, m, t, delta_t, I, y, y_dot, y_ret):
    # y = sy + delta*sum_{k=1,m} I_k(t) * ydot_jk
    if I.shape[0] != m:
        print("Input I size is wrong! Supposed to be", m, "instead of", I.shape[0], "columns!")
        raise Exception("Input I size is wrong!")
    if y_dot.shape[0] != m:
        print("Input y_dot size is wrong! Supposed to be", m, "instead of", y_dot.shape[0], "rows!")
        raise Exception("Input y dot size is wrong!")
    if y_dot.shape[1] != size_y:
        print("Input y_dot size is wrong! Supposed to be", size_y, "instead of", y_dot.shape[1], "columns!")
        raise Exception("Input y dot size is wrong!")
    if y_ret.shape[0] != size_y:
        print("Input y_ret size is wrong! Supposed to be", size_y, "instead of", y_ret.shape[0], "columns!")
        raise Exception("Input y_ret size is wrong!")
    # copy the y values
    for i in range(size_y):
        y_ret[i] = y[i]
    # compute the collocation weights at time t
    collocation_coefficients.compute_I(m, t, I)
    for i in range(m):
        for j in range(size_y):
            y_ret[j] += delta_t * I[i] * y_dot[i, j]
    return


def compute_segment_residual_parallel(size_y, size_z, size_p, m_min, m_max, N, m_N, m_accumulate,
                                      tau_m, w_m, t_span, y, y_dot, z_tilde, p, alpha, tol):
    """
    Compute the segment residual using Gaussian quadrature rule on Gauss points.
    :param size_y: number of ode variables of the problem
    :param size_z: number of algebraic variables
    :param size_p: number of parameter variables
    :param m_min: minimum number of collocation points allowed
    :param m_max: maximum number of collocation points allowed
    :param N: number of time nodes of the problem
    :param m_N: number of collocation points in each interval
    :param m_accumulate: accumulated collocation points in each interval
    :param tau_m: time coefficients of different gauss points used in row dominant order;
             tau_m[m_sum[m - m_min], :] corresponds to the coefficients of (m + 1) gauss points
             shape: (m_max - m_min + 1, m_max + 1)
    :param w_m: weight coefficients of different gauss points used in row dominant order;
             w_m[m_sum[m - m_min], :] corresponds to the coefficients of (m + 1) gauss points
             shape: (m_max - m_min + 1, m_max + 1)
    :param t_span: time span of the mesh
    :param y: values of the ODE variables in matrix form
    :param y_dot: values of the derivatives of ODE variables in row dominant matrix form
            shape: (m_accumulate[-1], size_y), where each row corresponds the values at each time node.
            The index for the jth collocation point from the ith time node is m_accumulate[i] + j.
    :param z_tilde: values of the DAE variables in row dominant matrix form
           dimension: (m_accumulate[-1], size_z), where each row corresponds the values at each time node.
           The index for the jth collocation point from the ith time node is m_accumulate[i] + j.
    :param p: values of the parameter variables in vector form
           shape: (size_p, )
    :param alpha: continuation parameter
    :param tol: numerical tolerance
    :return:
        residual: residual error evaluated for each time interval
                shape: (N, )
        max_residual: maximum residual error
    """
    # grid dimension of the kernel of CUDA
    grid_dims_1d = ((N - 1) + TPB_N - 1) // TPB_N
    block_dims_1d = TPB_N
    # tranfer memory from CPU to GPU
    d_m_N = cuda.to_device(m_N)
    d_m_accumulate = cuda.to_device(m_accumulate)
    d_tau_m = cuda.to_device(tau_m)
    d_w_m = cuda.to_device(w_m)
    d_t_span = cuda.to_device(t_span)
    d_y = cuda.to_device(y)
    d_y_dot = cuda.to_device(y_dot)
    d_z_tilde = cuda.to_device(z_tilde)
    d_p = cuda.to_device(p)
    # holder for the output variables
    # remember to zero initialization in the kernel
    d_residual = cuda.device_array(N, dtype=np.float64)
    # holder for the intermediate variables
    d_h_res = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_g_res = cuda.device_array((N - 1, size_z), dtype=np.float64)
    # d_r = cuda.device_array(size_y + size_p, dtype=np.float64)
    d_L = cuda.device_array((N - 1, m_max), dtype=np.float64)
    d_I = cuda.device_array((N - 1, m_max), dtype=np.float64)
    d_y_temp = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_z_temp = cuda.device_array((N - 1, size_z), dtype=np.float64)
    d_y_dot_temp = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # need reduction here maybe?
    d_rho_h = cuda.device_array(N - 1, dtype=np.float64)
    d_rho_g = cuda.device_array(N - 1, dtype=np.float64)

    compute_segment_residual_kernel[grid_dims_1d, block_dims_1d](
        size_y, size_z, m_min, N, alpha, tol,
        d_m_N, d_m_accumulate, d_tau_m, d_w_m, d_t_span,
        d_y, d_y_dot, d_z_tilde, d_y_temp, d_y_dot_temp, d_z_temp,
        d_p, d_L, d_I, d_h_res, d_g_res, d_rho_h, d_rho_g,
        residual_type, scale_by_time, scale_by_initial,
        d_residual)
    # copy the memory back to CPU
    rho_h = d_rho_h.copy_to_host()
    rho_g = d_rho_g.copy_to_host()
    residual = d_residual.copy_to_host()
    max_rho_r = 0
    # compute the residual at the boundary
    if (size_y + size_p) > 0:
        r = np.zeros((size_y + size_p), dtype=np.float64)
        _abvp_r(y[0, 0: size_y], y[N - 1, 0: size_y], p, r)
        max_rho_r = np.linalg.norm(r, np.inf)
        residual[N - 1] = max_rho_r / tol
    max_rho_h = np.amax(rho_h)
    max_rho_g = np.amax(rho_g)
    max_residual = np.amax(residual)
    if residual_type == 2:
        print('\tres: |h|: {}, |g|: {}, |r|: {}'.format(sqrt(max_rho_h) / tol, sqrt(max_rho_g) / tol, max_rho_r / tol))
    else:
        print('\tres: |h|: {}, |g|: {}, |r|: {}'.format(max_rho_h / tol, max_rho_g / tol, max_rho_r / tol))
    return residual, max_residual


'''
    Kernel function for computing segment residual.
'''


@cuda.jit()
def compute_segment_residual_kernel(size_y, size_z, m_min, N, alpha, tol,
                                    d_m_N, d_m_accumulate, d_tau_m, d_w_m, d_t_span,
                                    d_y, d_y_dot, d_z_tilde, d_y_temp, d_y_dot_temp, d_z_temp,
                                    d_p, d_L, d_I, d_h_res, d_g_res, d_rho_h, d_rho_g,
                                    residual_type, scale_by_time, scale_by_initial,
                                    d_residual):
    j = cuda.grid(1)  # cuda thread index
    if j < (N - 1):
        m = d_m_N[j]  # number of collocation points of the interval j
        delta_t_j = d_t_span[j + 1] - d_t_span[j]
        d_rho_h[j] = 0
        d_rho_g[j] = 0
        for i in range(m + 1):
            # compute y_dot at gaussian points
            get_y_dot(
                size_y, m, d_tau_m[m - m_min, i], d_L[j, 0: m],
                d_y_dot[d_m_accumulate[j]: d_m_accumulate[j + 1], 0: size_y],
                d_y_dot_temp[j, 0: size_y])
            # compute y at gaussian points
            get_y(
                size_y, m, d_tau_m[m - m_min, i], delta_t_j, d_I[j, 0: m], d_y[j, 0: size_y],
                d_y_dot[d_m_accumulate[j]: d_m_accumulate[j + 1], 0: size_y],
                d_y_temp[j, 0: size_y])
            # compute z at gaussian points
            get_z(
                size_z, m, d_tau_m[m - m_min, i], d_L[j, 0: m],
                d_z_tilde[d_m_accumulate[j]: d_m_accumulate[j + 1], 0: size_z],
                d_z_temp[j, 0: size_z])
            # compute h
            _abvp_f(d_y_temp[j, 0: size_y], d_z_temp[j, 0: size_z], d_p, d_h_res[j, 0: size_y])
            # h(y,z,p) - ydot
            for k in range(size_y):
                d_h_res[j, k] -= d_y_dot_temp[j, k]
            # d_rho_h[j] += np.dot(h_res, h_res) * w[i]
            for k in range(size_y):
                if residual_type == 2:
                    d_rho_h[j] += d_w_m[m - m_min, i] * d_h_res[j, k] * d_h_res[j, k]
                elif residual_type == 1:
                    d_rho_h[j] += d_w_m[m - m_min, i] * abs(d_h_res[j, k])
                elif residual_type == 0:
                    d_rho_h[j] = max(d_rho_h[j], d_w_m[m - m_min, i] * abs(d_h_res[j, k]))
                else:
                    print("\tNorm type invalid!")
                if scale_by_time:
                    d_rho_h[j] *= delta_t_j
                # d_rho_h[j] += delta_t_j * d_w[i] * d_h_res[j, k] * d_h_res[j, k]
            if size_z > 0:
                _abvp_g(d_y_temp[j, 0: size_y], d_z_temp[j, 0: size_z], d_p, alpha, d_g_res[j, 0: size_z])
                # rho_g += np.dot(g_res, g_res) * w[i]
                for k in range(size_z):
                    if residual_type == 2:
                        d_rho_g[j] += d_w_m[m - m_min, i] * d_g_res[j, k] * d_g_res[j, k]
                    elif residual_type == 1:
                        d_rho_g[j] += d_w_m[m - m_min, i] * abs(d_g_res[j, k])
                    elif residual_type == 0:
                        d_rho_g[j] = max(d_rho_g[j], d_w_m[m - m_min, i] * abs(d_h_res[j, k]))
                    else:
                        print("\tNorm type invalid!")
                    if scale_by_time:
                        d_rho_g[j] *= delta_t_j
                    # d_rho_g[j] += delta_t_j * d_w[i] * d_g_res[j, k] * d_g_res[j, k]
        if residual_type == 2:
            d_residual[j] = sqrt(d_rho_h[j] + d_rho_g[j]) / tol
        elif residual_type == 1:
            d_residual[j] = (abs(d_rho_h[j]) + abs(d_rho_g[j])) / tol
        elif residual_type == 0:
            d_residual[j] = max(d_rho_h[j], d_rho_g[j]) / tol
    return


def compute_segment_residual_collocation_parallel(size_y, size_z, size_p, m_min, m_max, N, m_N, m_accumulate,
                                                  c_m, t_span, y, y_dot, z_tilde, p, alpha, tol):
    """
    Compute the segment residual on m + 1 collocation points on relative scale.
    :param size_y: number of ode variables of the problem
    :param size_z: number of algebraic variables
    :param size_p: number of parameter variables
    :param m_min: minimum number of collocation points allowed
    :param m_max: maximum number of collocation points allowed
    :param N: number of time nodes of the problem
    :param m_N: number of collocation points in each interval
    :param m_accumulate: accumulated collocation points in each interval
    :param c_m: collocation time coefficients
    :param t_span: time span of the mesh
    :param y: values of the ODE variables in matrix form
    :param y_dot: alues of the derivatives of ODE variables in row dominant matrix form
            shape: (m_accumulate[-1], size_y), where each row corresponds the values at each time node.
            The index for the jth collocation point from the ith time node is m_accumulate[i] + j.
    :param z_tilde: values of the DAE variables in row dominant matrix form
           dimension: (m_accumulate[-1], size_z), where each row corresponds the values at each time node.
           The index for the jth collocation point from the ith time node is m_accumulate[i] + j.
    :param p: values of the parameter variables in vector form
           shape: (size_p, )
    :param alpha: continuation parameter
    :param tol: numerical tolerance
    :return:
        residual: residual error evaluated for each time interval
                shape: (N, )
        residual_collocation: residual error on each collocation point
        max_residual: maximum residual error
    """
    # grid dimension of the kernel of CUDA
    grid_dims_1d = ((N - 1) + TPB_N - 1) // TPB_N
    block_dims_1d = TPB_N
    # tranfer memory from CPU to GPU
    d_m_N = cuda.to_device(m_N)
    d_m_accumulate = cuda.to_device(m_accumulate)
    d_c_m = cuda.to_device(c_m)
    d_t_span = cuda.to_device(t_span)
    d_y = cuda.to_device(y)
    d_y_dot = cuda.to_device(y_dot)
    d_z_tilde = cuda.to_device(z_tilde)
    d_p = cuda.to_device(p)
    # holder for the output variables
    # remember to zero initialization in the kernel
    d_residual = cuda.device_array(N, dtype=np.float64)
    d_residual_collocation = cuda.device_array((N, m_max + 1), dtype=np.float64)
    # holder for the intermediate variables
    d_h_res = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_g_res = cuda.device_array((N - 1, size_z), dtype=np.float64)
    # d_r = cuda.device_array(size_y + size_p, dtype=np.float64)
    d_L = cuda.device_array((N - 1, m_max), dtype=np.float64)
    d_I = cuda.device_array((N - 1, m_max), dtype=np.float64)
    d_y_temp = cuda.device_array((N - 1, size_y), dtype=np.float64)
    d_z_temp = cuda.device_array((N - 1, size_z), dtype=np.float64)
    d_y_dot_temp = cuda.device_array((N - 1, size_y), dtype=np.float64)
    # need reduction here maybe?
    d_rho_h = cuda.device_array(N - 1, dtype=np.float64)
    d_rho_g = cuda.device_array(N - 1, dtype=np.float64)
    compute_segment_residual_collocation_kernel[grid_dims_1d, block_dims_1d](
        size_y, size_z, m_min, m_max, N, alpha, tol,
        d_m_N, d_m_accumulate, d_c_m, d_t_span,
        d_y, d_y_dot, d_z_tilde, d_y_temp, d_y_dot_temp, d_z_temp,
        d_p, d_L, d_I, d_h_res, d_g_res, d_rho_h, d_rho_g,
        d_residual, d_residual_collocation)
    # copy the memory back to CPU
    residual = d_residual.copy_to_host()
    residual_collocation = d_residual_collocation.copy_to_host()
    max_rho_r = 0
    # compute the residual at the boundary
    if (size_y + size_p) > 0:
        r = np.zeros((size_y + size_p), dtype=np.float64)
        _abvp_r(y[0, 0: size_y], y[N - 1, 0: size_y], p, r)
        max_rho_r = np.linalg.norm(r, np.inf) / tol
        residual[N - 1] = max_rho_r
    max_rho_h = cuda_infinity_norm(d_rho_h) / tol
    max_rho_g = cuda_infinity_norm(d_rho_g) / tol
    max_residual = np.amax(residual)
    print('\tres: |h|: {}, |g|: {}, |r|: {}'.format(max_rho_h, max_rho_g, max_rho_r))
    return residual, residual_collocation, max_residual


'''
    Kernel function for computing segment residual.
'''


@cuda.jit()
def compute_segment_residual_collocation_kernel(size_y, size_z, m_min, m_max, N, alpha, tol,
                                                d_m_N, d_m_accumulate, d_c_m, d_t_span,
                                                d_y, d_y_dot, d_z_tilde, d_y_temp, d_y_dot_temp, d_z_temp,
                                                d_p, d_L, d_I, d_h_res, d_g_res, d_rho_h, d_rho_g,
                                                d_residual, d_residual_collocation):
    j = cuda.grid(1)  # cuda thread index
    if j < (N - 1):
        m = d_m_N[j]  # number of collocation points in the jth interval
        delta_t_j = d_t_span[j + 1] - d_t_span[j]
        d_rho_h[j] = 0.0
        d_rho_g[j] = 0.0
        d_residual[j] = 0.0
        for k in range(m_max):
            d_residual_collocation[j, k] = 0.0
        # compute the residual at (m + 1) collocation points
        for i in range(m + 1):
            # compute y_dot at gaussian points
            get_y_dot(size_y, m, d_c_m[m + 1 - m_min, i], d_L[j, 0: m],
                      d_y_dot[d_m_accumulate[j]: d_m_accumulate[j + 1], 0: size_y],
                      d_y_dot_temp[j, 0: size_y])
            # compute y at gaussian points
            get_y(size_y, m, d_c_m[m + 1 - m_min, i], delta_t_j, d_I[j, 0: m], d_y[j, 0: size_y],
                  d_y_dot[d_m_accumulate[j]: d_m_accumulate[j + 1], 0: size_y],
                  d_y_temp[j, 0: size_y])
            # compute z at gaussian points
            get_z(size_z, m, d_c_m[m + 1 - m_min, i], d_L[j, 0: m],
                  d_z_tilde[d_m_accumulate[j]: d_m_accumulate[j + 1], 0: size_z],
                  d_z_temp[j, 0: size_z])
            # compute h
            _abvp_f(d_y_temp[j, 0: size_y], d_z_temp[j, 0: size_z], d_p, d_h_res[j, 0: size_y])
            # h(y,z,p) - ydot
            for k in range(size_y):
                # E[j, k] = y_dot_tilde[j, k] - y_dot[j, k]
                d_h_res[j, k] -= d_y_dot_temp[j, k]
                # e^{h, j}_{i, k} = abs(E[j, k]) / (1 + y_dot[i, k])
                # e^h_j = max e^j_{i, k}
                residual_scaled = abs(d_h_res[j, k]) / (1 + abs(d_y_dot_temp[j, k]))
                d_rho_h[j] = max(d_rho_h[j], residual_scaled)
                d_residual_collocation[j, i] = max(d_residual_collocation[j, i], residual_scaled)
            if size_z > 0:
                _abvp_g(d_y_temp[j, 0: size_y], d_z_temp[j, 0: size_z], d_p, alpha, d_g_res[j, 0: size_z])
                for k in range(size_z):
                    # e^{g, j}_{i, k} = abs(E[j, k]) / (1 + z[i, k])
                    # e^g_j = max e^j_{i, k}
                    residual_scaled = abs(d_g_res[j, k]) / (1 + abs(d_z_temp[j, k]))
                    d_rho_g[j] = max(d_rho_g[j], residual_scaled)
                    d_residual_collocation[j, i] = max(d_residual_collocation[j, i], residual_scaled)
        # d_residual[j] = max(d_rho_h[j], d_rho_g[j]) / tol
        d_residual[j] = max(d_rho_h[j], d_rho_g[j]) / tol
    return


# finish the implementation of computing segment residual on each time node


# start the implementation of recovering solution


def recover_solution_parallel(size_z, m_max, N, m_N, m_accumulate, z_tilde):
    """
    Recover the solution to the BVP-DAE problem.
    :param size_z: number of algebraic variables
    :param m_max: maximum number of collocation points allowed
    :param N: number of time nodes of the problem
    :param m_N: number of collocation points in each interval
    :param m_accumulate: accumulated collocation points in each interval
    :param z_tilde: values of the DAE variables in row dominant matrix form
           dimension: (m_accumulate[-1], size_z), where each row corresponds the values at each time node.
           The index for the jth collocation point from the ith time node is i * m + j.
    :return:
        z: values of algebraic variables at each time node in row dominant matrix form
            shape: (N, size_z)
    """
    # grid dimension of the kernel of CUDA
    grid_dims_1d = (N + TPB_N - 1) // TPB_N
    block_dims_1d = TPB_N
    # tranfer memory from CPU to GPU
    d_m_N = cuda.to_device(m_N)
    d_m_accumulate = cuda.to_device(m_accumulate)
    d_z_tilde = cuda.to_device(z_tilde)
    # holder for output variables
    d_z = cuda.device_array((N, size_z), dtype=np.float64)
    # holder for intermediate variables
    d_L = cuda.device_array((N, m_max), dtype=np.float64)
    # execute the kernel function
    recover_solution_kernel[grid_dims_1d, block_dims_1d](size_z, N, d_m_N, d_m_accumulate, d_z_tilde, d_L, d_z)
    # return the ouput
    return d_z.copy_to_host()


@cuda.jit()
def recover_solution_kernel(size_z, N, d_m_N, d_m_accumulate, d_z_tilde, d_L, d_z):
    i = cuda.grid(1)  # cuda thread index
    if i < (N - 1):
        m = d_m_N[i]
        t = 0
        collocation_coefficients.compute_L(m, t, d_L[i, 0: m])
        # zero initialization
        for k in range(size_z):
            d_z[i, k] = 0
        # loop through all the collocation points
        for j in range(m):
            # loop through all the Z variables
            for k in range(size_z):
                d_z[i, k] += d_L[i, j] * d_z_tilde[d_m_accumulate[i] + j, k]
    # for the last time node
    if i == (N - 1):
        m = d_m_N[i - 1]  # number of collocation points of the last interval
        t = 1
        collocation_coefficients.compute_L(m, t, d_L[i, 0: m])
        # zero initialization
        for k in range(size_z):
            d_z[i, k] = 0
        # loop through all the collocation points
        for j in range(m):
            # loop through all the Z variables
            for k in range(size_z):
                d_z[i, k] += d_L[i, j] * d_z_tilde[d_m_accumulate[i - 1] + j, k]
    return


# finish the implementation of recovering solution


# start the implementation of remesh


def remesh(size_y, size_z, N, tspan, y0, z0, residual):
    """
    Remesh the problem
    :param size_y: number of y variables
    :param size_z: number of z variables
    :param N: number of time nodes in the current mesh
    :param tspan: time span of the current mesh
    :param y0: values of the differential variables in matrix form
    :param z0: values of the algebraic variables in matrix form
    :param residual: residual error evaluated for each time interval
    :return:
        N_New : number of time nodes of the new mesh .
        tspan_New : new time span of the problem.
        y0_New : values of the differential variables in new mesh in matrix form
        z0_New : values of the algebraic variables in in new mesh in matrix form
    """
    N_Temp = 0
    tspan_Temp = []
    y0_Temp = []
    z0_Temp = []
    residual_Temp = []

    # Deleting Nodes
    i = 0
    # Record the number of the deleted nodes
    k_D = 0

    thresholdDel = 1e-2
    while i < N - 4:
        res_i = residual[i]
        if res_i <= thresholdDel:
            res_i_Plus1 = residual[i + 1]
            res_i_Plus2 = residual[i + 2]
            res_i_Plus3 = residual[i + 3]
            res_i_Plus4 = residual[i + 4]
            if res_i_Plus1 <= thresholdDel and res_i_Plus2 <= thresholdDel and res_i_Plus3 <= thresholdDel and \
                    res_i_Plus4 <= thresholdDel:
                # append the 1st, 3rd, and 5th node
                # 1st node
                tspan_Temp.append(tspan[i])
                y0_Temp.append(y0[i, :])
                z0_Temp.append(z0[i, :])
                residual_Temp.append(residual[i])
                # 3rd node
                tspan_Temp.append(tspan[i + 2])
                y0_Temp.append(y0[i + 2, :])
                z0_Temp.append(z0[i + 2, :])
                residual_Temp.append(residual[i + 2])
                # 5th node
                tspan_Temp.append(tspan[i + 4])
                y0_Temp.append(y0[i + 4, :])
                z0_Temp.append(z0[i + 4, :])
                residual_Temp.append(residual[i + 4])
                # delete 2 nodes
                k_D += 2
                # add 3 nodes to the total number
                N_Temp += 3
                # ignore those five nodes
                i += 5
            else:
                # directly add the node
                tspan_Temp.append(tspan[i])
                y0_Temp.append(y0[i, :])
                z0_Temp.append(z0[i, :])
                residual_Temp.append(residual[i])
                N_Temp += 1
                i += 1
        else:
            # directly add the node
            tspan_Temp.append(tspan[i])
            y0_Temp.append(y0[i, :])
            z0_Temp.append(z0[i, :])
            residual_Temp.append(residual[i])
            N_Temp += 1
            i += 1
    '''
        if the previous loop stop at the ith node which is bigger than (N - 4), those last
        few nodes left are added manually, if the last few nodes have already been processed,
        the index i should be equal to N, then nothing needs to be done
    '''
    if i < N:
        '''
            add the last few nodes starting from i to N - 1, which
            is a total of (N - i) nodes
        '''
        for j in range(N - i):
            # append the N - 4 + j node
            tspan_Temp.append(tspan[i + j])
            y0_Temp.append(y0[i + j, :])
            z0_Temp.append(z0[i + j, :])
            residual_Temp.append(residual[i + j])
            N_Temp += 1
    # convert from list to numpy arrays for the convenience of indexing
    tspan_Temp = np.array(tspan_Temp)
    y0_Temp = np.array(y0_Temp)
    z0_Temp = np.array(z0_Temp)
    residual_Temp = np.array(residual_Temp)
    # lists to hold the outputs
    N_New = 0
    tspan_New = []
    y0_New = []
    z0_New = []
    residual_New = []
    # Adding Nodes
    i = 0
    # Record the number of the added nodes
    k_A = 0

    while i < N_Temp - 1:
        res_i = residual_Temp[i]
        if res_i > 1:
            if res_i > 10:
                # add three uniformly spaced nodes
                # add the time point of new nodes
                delta_t = (tspan_Temp[i + 1] - tspan_Temp[i]) / 4
                t_i = tspan_Temp[i]
                t_i_Plus1 = t_i + delta_t
                t_i_Plus2 = t_i + 2 * delta_t
                t_i_Plus3 = t_i + 3 * delta_t
                tspan_New.append(t_i)
                tspan_New.append(t_i_Plus1)
                tspan_New.append(t_i_Plus2)
                tspan_New.append(t_i_Plus3)
                # add the residuals of the new nodes
                delta_res = (residual_Temp[i + 1] - residual_Temp[i]) / 4
                res_i_Plus1 = res_i + delta_res
                res_i_Plus2 = res_i + 2 * delta_res
                res_i_Plus3 = res_i + 3 * delta_res
                residual_New.append(res_i)
                residual_New.append(res_i_Plus1)
                residual_New.append(res_i_Plus2)
                residual_New.append(res_i_Plus3)
                # add the ys of the new nodes
                y0_i = y0_Temp[i, :]
                y0_i_Next = y0_Temp[i + 1, :]
                delta_y0 = (y0_i_Next - y0_i) / 4
                y0_i_Plus1 = y0_i + delta_y0
                y0_i_Plus2 = y0_i + 2 * delta_y0
                y0_i_Plus3 = y0_i + 3 * delta_y0
                y0_New.append(y0_i)
                y0_New.append(y0_i_Plus1)
                y0_New.append(y0_i_Plus2)
                y0_New.append(y0_i_Plus3)
                # add the zs of the new nodes
                z0_i = z0_Temp[i, :]
                z0_i_Next = z0_Temp[i + 1, :]
                delta_z0 = (z0_i_Next - z0_i) / 4
                z0_i_Plus1 = z0_i + delta_z0
                z0_i_Plus2 = z0_i + 2 * delta_z0
                z0_i_Plus3 = z0_i + 3 * delta_z0
                z0_New.append(z0_i)
                z0_New.append(z0_i_Plus1)
                z0_New.append(z0_i_Plus2)
                z0_New.append(z0_i_Plus3)
                # update the index
                # 1 original node + 3 newly added nodes
                N_New += 4
                k_A += 3
                i += 1
            else:
                # add one node to the middle
                # add the time point of the new node
                delta_t = (tspan_Temp[i + 1] - tspan_Temp[i]) / 2
                t_i = tspan_Temp[i]
                t_i_Plus1 = t_i + delta_t
                tspan_New.append(t_i)
                tspan_New.append(t_i_Plus1)
                # add the residual of the new node
                delta_res = (residual_Temp[i + 1] - residual_Temp[i]) / 2
                res_i_Plus1 = res_i + delta_res
                residual_New.append(res_i)
                residual_New.append(res_i_Plus1)
                # add the y of the new node
                y0_i = y0_Temp[i, :]
                y0_i_Next = y0_Temp[i + 1, :]
                delta_y0 = (y0_i_Next - y0_i) / 2
                y0_i_Plus1 = y0_i + delta_y0
                y0_New.append(y0_i)
                y0_New.append(y0_i_Plus1)
                # add the z of the new node
                z0_i = z0_Temp[i, :]
                z0_i_Next = z0_Temp[i + 1, :]
                delta_z0 = (z0_i_Next - z0_i) / 2
                z0_i_Plus1 = z0_i + delta_z0
                z0_New.append(z0_i)
                z0_New.append(z0_i_Plus1)
                # update the index
                # 1 original node + 1 newly added node
                N_New += 2
                k_A += 1
                i += 1
        else:
            # add the current node only
            # add the time node of the current node
            t_i = tspan_Temp[i]
            tspan_New.append(t_i)
            # add the residual of the current node
            residual_New.append(res_i)
            # add the y of the current node
            y0_i = y0_Temp[i, :]
            y0_New.append(y0_i)
            # add the z of the current node
            z0_i = z0_Temp[i, :]
            z0_New.append(z0_i)
            # update the index
            # 1 original node only
            N_New += 1
            i += 1
    # add the final node
    tspan_New.append(tspan_Temp[N_Temp - 1])
    y0_New.append(y0_Temp[N_Temp - 1, :])
    z0_New.append(z0_Temp[N_Temp - 1, :])
    residual_New.append(residual_Temp[N_Temp - 1])
    N_New += 1
    # convert from list to numpy arrays for the convenience of indexing
    tspan_New = np.array(tspan_New)
    y0_New = np.array(y0_New)
    z0_New = np.array(z0_New)
    print("\tDelete nodes: {}; Add nodes: {}; Number of nodes after mesh: {}".format(k_D, k_A, N_New))
    # return the output
    return N_New, tspan_New, y0_New, z0_New


def hp_remesh(size_y, size_z, m_min, m_max, m_init, N,
              m_N, m_accumulate, c_m, tspan, y0, z0,
              residual, residual_collocation,
              thres_remove, thres_add, rho, m_d, m_i, tol):
    """
    Use hp mesh to remesh the problem
    :param size_y: number of y variables
    :param size_z: number of z variables.
    :param m_min: minimum number of collocation points allowed
    :param m_max: maximum number of collocation points allowed
    :param m_init: initial number of collocation points used
    :param N: number of time nodes in the current mesh
    :param m_N: number of collocation points in each interval in the current mesh
    :param m_accumulate: accumulated collocation points at each time node in the current mesh
    :param c_m: coefficients c of lobatto collocation points
    :param tspan: time span of the current mesh
    :param y0: values of the differential variables in the current mesh
    :param z0: values of the algebraic variables in the current mesh
    :param residual: residual error evaluated for each time interval
    :param residual_collocation: residual error on collocation points evaluated for each time interval
    :param thres_remove: threshold to remove nodes from the mesh
    :param thres_add: threshold to add nodes to the mesh
    :param m_d: number of collocation points decreased in each interval
    :param m_i: number of collocation points increased in each interval
    :param rho: threshold for uniform error
    :param tol: numerical tolerances
    :return:
        N_new: number of time nodes in the new mesh
        tspan_new: time span of the new mesh
        y0_new: values of the differential variables in the new mesh
        z0_new: values of the algebraic variables in the new mesh
        m_N_new: number of collocation points in each interval in the new mesh
        m_accumulate_new: accumulated collocation points at each time node in the new mesh
    """
    N_temp = 0
    tspan_temp = []
    y0_temp = []
    z0_temp = []
    residual_temp = []
    residual_collocation_temp = []
    m_N_temp = []

    # remove Nodes
    i = 0
    # record the number of removed nodes
    n_r = 0
    m_r = 0

    while i < N - 5:
        res_i = residual[i]
        if res_i <= thres_remove:
            res_i_Plus1 = residual[i + 1]
            res_i_Plus2 = residual[i + 2]
            res_i_Plus3 = residual[i + 3]
            res_i_Plus4 = residual[i + 4]
            if res_i_Plus1 <= thres_remove and res_i_Plus2 <= thres_remove and res_i_Plus3 <= thres_remove and \
                    res_i_Plus4 <= thres_remove:
                # append the 1st, 3rd, and 5th node
                # check the 2nd and the 4th node
                # if they use the m_min then remove, or decrese the number of collocation points by m_d
                # 1st node
                tspan_temp.append(tspan[i])
                y0_temp.append(y0[i, :])
                z0_temp.append(z0[i, :])
                residual_temp.append(residual[i])
                residual_collocation_temp.append(residual_collocation[i, :])
                m_N_temp.append(m_N[i])
                N_temp += 1
                # 2nd node
                if m_N[i + 1] > m_min:
                    tspan_temp.append(tspan[i + 1])
                    y0_temp.append(y0[i + 1, 0: size_y])
                    z0_temp.append(z0[i + 1, 0: size_z])
                    residual_temp.append(residual[i + 1])
                    residual_collocation_temp.append(residual_collocation[i + 1, :])
                    m_add = max(m_N[i + 1] - m_d, m_min)
                    m_N_temp.append(m_add)
                    m_r += (m_N[i + 1] - m_add)
                    N_temp += 1
                else:
                    n_r += 1
                    m_r += (m_N[i + 1])
                # 3rd node
                tspan_temp.append(tspan[i + 2])
                y0_temp.append(y0[i + 2, 0: size_y])
                z0_temp.append(z0[i + 2, 0: size_z])
                residual_temp.append(residual[i + 2])
                residual_collocation_temp.append(residual_collocation[i + 2, :])
                m_N_temp.append(m_N[i + 2])
                N_temp += 1
                # 4th node
                if m_N[i + 3] > m_min:
                    tspan_temp.append(tspan[i + 3])
                    y0_temp.append(y0[i + 3, 0: size_y])
                    z0_temp.append(z0[i + 3, 0: size_z])
                    residual_temp.append(residual[i + 3])
                    residual_collocation_temp.append(residual_collocation[i + 3, :])
                    m_add = max(m_N[i + 3] - m_d, m_min)
                    m_N_temp.append(m_add)
                    m_r += m_N[i + 3] - m_add
                    N_temp += 1
                else:
                    n_r += 1
                    m_r += (m_N[i + 3])
                # 5th node
                tspan_temp.append(tspan[i + 4])
                y0_temp.append(y0[i + 4, 0: size_y])
                z0_temp.append(z0[i + 4, 0: size_z])
                residual_temp.append(residual[i + 4])
                residual_collocation_temp.append(residual_collocation[i + 4, :])
                m_N_temp.append(m_N[i + 4])
                N_temp += 1
                # ignore those five nodes
                i += 5
            else:
                # directly add the node
                tspan_temp.append(tspan[i])
                y0_temp.append(y0[i, 0: size_y])
                z0_temp.append(z0[i, 0: size_z])
                residual_temp.append(residual[i])
                residual_collocation_temp.append(residual_collocation[i, :])
                m_N_temp.append(m_N[i])
                N_temp += 1
                i += 1
        else:
            # directly add the node
            tspan_temp.append(tspan[i])
            y0_temp.append(y0[i, 0: size_y])
            z0_temp.append(z0[i, 0: size_z])
            residual_temp.append(residual[i])
            residual_collocation_temp.append(residual_collocation[i, :])
            m_N_temp.append(m_N[i])
            N_temp += 1
            i += 1
    '''
        if the previous loop stop at the ith node which is bigger than (N - 4), those last
        few nodes left are added directly, if the last few nodes have already been processed,
        the index i should be equal to N, then nothing needs to be done
    '''
    if i < N:
        '''
            add the last few nodes starting from i to N - 1, which
            is a total of (N - i) nodes
        '''
        for j in range(N - i):
            # append the N - 4 + j node
            tspan_temp.append(tspan[i + j])
            y0_temp.append(y0[i + j, 0: size_y])
            z0_temp.append(z0[i + j, 0: size_z])
            residual_temp.append(residual[i + j])
            N_temp += 1
            if (i + j) != (N - 1):
                # no collocation residual for the last node
                residual_collocation_temp.append(residual_collocation[i + j, :])
                m_N_temp.append(m_N[i + j])

    # convert from list to numpy arrays for the convenience of indexing
    tspan_temp = np.array(tspan_temp)
    y0_temp = np.array(y0_temp)
    z0_temp = np.array(z0_temp)
    residual_temp = np.array(residual_temp)
    residual_collocation_temp = np.array(residual_collocation_temp)
    m_N_temp = np.array(m_N_temp)

    # lists to hold the outputs
    N_new = 0
    tspan_new = []
    y0_new = []
    z0_new = []
    m_N_new = []
    # Adding Nodes
    i = 0
    # Record the number of the added nodes
    n_a = 0
    m_a = 0

    while i < N_temp - 1:
        res_i = residual_temp[i]
        m = m_N_temp[i]
        if res_i > thres_add:
            m_add = 0
            # detect it is uniform type or non-uniform
            if res_i > rho:
                # non-uniform type
                index_add = []  # index to add time node at the point
                for j in range(1, m):
                    # start from index 1 as the initial node will be added automatically
                    # loop through the inner collocation points except the boundary points
                    if (residual_collocation_temp[i, j] / tol) > rho:
                        if j == 1 and residual_collocation_temp[i, j] >= residual_collocation_temp[i, j + 1]:
                            index_add.append(j)
                        elif j == (m - 1) and residual_collocation_temp[i, j] >= residual_collocation_temp[i, j - 1]:
                            index_add.append(j)
                        elif residual_collocation_temp[i, j] >= residual_collocation_temp[i, j + 1] and \
                                residual_collocation_temp[i, j] >= residual_collocation_temp[i, j - 1]:
                            index_add.append(j)
                # add the initial time node first
                delta_t = tspan_temp[i + 1] - tspan_temp[i]
                t_init = tspan_temp[i]
                tspan_new.append(t_init)
                y0_init = y0_temp[i, 0: size_y]
                y0_next = y0_temp[i + 1, 0: size_y]
                y0_new.append(y0_init)
                z0_init = z0_temp[i, 0: size_z]
                z0_next = z0_temp[i + 1, 0: size_z]
                z0_new.append(z0_init)
                # for divided intervals, starts with m_init for all the intervals
                m_N_new.append(m_init)
                m_add += m_init
                N_new += 1
                for index in index_add:
                    c = c_m[m + 1 - m_min, index]
                    t_cur = t_init + c * delta_t
                    tspan_new.append(t_cur)
                    y0_cur = (1 - c) * y0_init + c * y0_next
                    y0_new.append(y0_cur)
                    z0_cur = (1 - c) * z0_init + c * z0_next
                    z0_new.append(z0_cur)
                    m_N_new.append(m_init)
                    m_add += m_init
                    N_new += 1
                    n_a += 1
                m_a += (m_add - m)
            else:
                # uniform type
                delta_t = tspan_temp[i + 1] - tspan_temp[i]
                t_init = tspan_temp[i]
                y0_init = y0_temp[i, 0: size_y]
                y0_next = y0_temp[i + 1, 0: size_y]
                z0_init = z0_temp[i, 0: size_z]
                z0_next = z0_temp[i + 1, 0: size_z]
                # increase the number of collocation points
                # if exceeds the maximum allowed, divide the interval into two
                if (m + m_i) > m_max:
                    # add the initial time node first
                    # add the current node with current number of collocation points
                    tspan_new.append(t_init)
                    y0_new.append(y0_init)
                    z0_new.append(z0_init)
                    m_N_new.append(m)
                    N_new += 1
                    # add the middle time node with current number of collocation points
                    t_mid = t_init + 0.5 * delta_t
                    y0_mid = (y0_init + y0_next) / 2
                    z0_mid = (z0_init + z0_next) / 2
                    tspan_new.append(t_mid)
                    y0_new.append(y0_mid)
                    z0_new.append(z0_mid)
                    m_N_new.append(m)
                    N_new += 1
                    m_a += m
                else:
                    tspan_new.append(t_init)
                    y0_new.append(y0_init)
                    z0_new.append(z0_init)
                    m_N_new.append(m + m_i)
                    N_new += 1
                    m_a += m_i
        else:
            # directly add the node
            t_init = tspan_temp[i]
            tspan_new.append(t_init)
            y0_init = y0_temp[i, 0: size_y]
            y0_new.append(y0_init)
            z0_init = z0_temp[i, 0: size_z]
            z0_new.append(z0_init)
            m_N_new.append(m)
            N_new += 1
        i += 1  # update the loop index
    # add the final node
    tspan_new.append(tspan_temp[-1])
    y0_new.append(y0_temp[-1, :])
    z0_new.append(z0_temp[-1, :])
    N_new += 1
    # convert from list to numpy arrays for the convenience of indexing
    tspan_new = np.array(tspan_new)
    y0_new = np.array(y0_new)
    z0_new = np.array(z0_new)
    m_N_new = np.array(m_N_new)
    # generate the new accumulate collocation array
    m_accumulate_new = np.zeros(N_new, dtype=int)  # accumulated collocation points used
    for i in range(1, N_new):
        m_accumulate_new[i] = m_accumulate_new[i - 1] + m_N_new[i - 1]
    print("\tRemove time nodes: {}; Add time nodes: {}; "
          "Number of nodes before mesh: {}, after mesh: {}".format(n_r, n_a, N, N_new))
    print("\tRemove collocation points: {}; Add collocation points: {};\n"
          "\tPrevious total number of collocation points: {}, new total number of collocation points: {}.".format(
            m_r, m_a, m_accumulate[-1], m_accumulate_new[-1]))
    # return the output
    return N_new, tspan_new, y0_new, z0_new, m_N_new, m_accumulate_new


def hp_remesh2(size_y, size_z, m_min, m_max, m_init, N,
              m_N, m_accumulate, c_m, tspan, y0, z0,
              residual, residual_collocation,
              thres_remove, thres_add, m_d, m_i, rho, tol):
    """
    Use hp mesh to remesh the problem
    :param size_y: number of y variables
    :param size_z: number of z variables.
    :param m_min: minimum number of collocation points allowed
    :param m_max: maximum number of collocation points allowed
    :param m_init: initial number of collocation points used
    :param N: number of time nodes in the current mesh
    :param m_N: number of collocation points in each interval in the current mesh
    :param m_accumulate: accumulated collocation points at each time node in the current mesh
    :param c_m: coefficients c of lobatto collocation points
    :param tspan: time span of the current mesh
    :param y0: values of the differential variables in the current mesh
    :param z0: values of the algebraic variables in the current mesh
    :param residual: residual error evaluated for each time interval
    :param residual_collocation: residual error on collocation points evaluated for each time interval
    :param thres_remove: threshold to remove nodes from the mesh
    :param thres_add: threshold to add nodes to the mesh
    :param m_d: number of collocation points decreased in each interval
    :param m_i: number of collocation points increased in each interval
    :param rho: threshold for uniform error
    :param tol: numerical tolerances
    :return:
        N_new: number of time nodes in the new mesh
        tspan_new: time span of the new mesh
        y0_new: values of the differential variables in the new mesh
        z0_new: values of the algebraic variables in the new mesh
        m_N_new: number of collocation points in each interval in the new mesh
        m_accumulate_new: accumulated collocation points at each time node in the new mesh
    """
    N_temp = 0
    tspan_temp = []
    y0_temp = []
    z0_temp = []
    residual_temp = []
    residual_collocation_temp = []
    m_N_temp = []

    # remove Nodes
    i = 0
    # record the number of removed nodes
    n_r = 0

    # add the initial node first
    tspan_temp.append(tspan[0])
    y0_temp.append(y0[0, :])
    z0_temp.append(z0[0, :])
    N_temp += 1

    # leave out the last two intervals first
    while i < N - 2:
        # check the two consecutive intervals
        if residual[i] <= thres_remove and residual[i + 1] <= thres_remove:
            # check the number of collocation points
            if m_N[i] == m_min and m_N[i + 1] == m_min:
                # if the collocation points in both intervals are the minimum allowed,
                # delete the time node between the two intervals
                n_r += 1
                # append the second time node only
                tspan_temp.append(tspan[i + 2])
                y0_temp.append(y0[i + 2, 0: size_y])
                z0_temp.append(z0[i + 2, 0: size_z])
                residual_temp.append(residual[i + 1])
                residual_collocation_temp.append(residual_collocation[i + 1, :])
                m_N_temp.append(m_min)
                N_temp += 1
            else:
                # or adjust the collocation points in the intervals
                # adjust the number of collocation points used
                # the first time node
                tspan_temp.append(tspan[i + 1])
                y0_temp.append(y0[i + 1, 0: size_y])
                z0_temp.append(z0[i + 1, 0: size_z])
                residual_temp.append(residual[i])
                residual_collocation_temp.append(residual_collocation[i, :])
                m_N_temp.append(max(m_N[i] - m_d, m_min))
                N_temp += 1
                # the second time node
                tspan_temp.append(tspan[i + 2])
                y0_temp.append(y0[i + 2, 0: size_y])
                z0_temp.append(z0[i + 2, 0: size_z])
                residual_temp.append(residual[i + 1])
                residual_collocation_temp.append(residual_collocation[i + 1, :])
                m_N_temp.append(max(m_N[i + 1] - m_d, m_min))
                N_temp += 1
            i += 2
        else:
            # append the time node directly
            tspan_temp.append(tspan[i + 1])
            y0_temp.append(y0[i + 1, :])
            z0_temp.append(z0[i + 1, :])
            residual_temp.append(residual[i])
            residual_collocation_temp.append(residual_collocation[i, :])
            m_N_temp.append(m_N[i])
            N_temp += 1
            i += 1

    # check whether the last node is added
    if i < N - 1:
        for j in range((N - 1) - i):
            # append the time node directly
            tspan_temp.append(tspan[i + j + 1])
            y0_temp.append(y0[i + j + 1, :])
            z0_temp.append(z0[i + j + 1, :])
            residual_temp.append(residual[i + j])
            residual_collocation_temp.append(residual_collocation[i + j, :])
            if residual[i + j] <= thres_remove:
                m_N_temp.append(max(m_N[i + j] - m_d, m_min))
            else:
                m_N_temp.append(m_N[i + j])
            N_temp += 1

    # convert from list to numpy arrays for the convenience of indexing
    tspan_temp = np.array(tspan_temp)
    y0_temp = np.array(y0_temp)
    z0_temp = np.array(z0_temp)
    residual_temp = np.array(residual_temp)
    residual_collocation_temp = np.array(residual_collocation_temp)
    m_N_temp = np.array(m_N_temp)

    # lists to hold the outputs
    N_new = 0
    tspan_new = []
    y0_new = []
    z0_new = []
    m_N_new = []
    # Adding Nodes
    i = 0
    # Record the number of the added nodes
    n_a = 0

    while i < N_temp - 1:
        res_i = residual_temp[i]
        m = m_N_temp[i]
        if res_i > thres_add:
            # detect it is uniform type or non-uniform
            if res_i > rho:
                # non-uniform type
                index_add = []  # index to add time node at the point
                for j in range(1, m):
                    # start from index 1 as the initial node will be added automatically
                    # loop through the inner collocation points except the boundary points
                    if (residual_collocation_temp[i, j] / tol) > rho:
                        if j == 1 and residual_collocation_temp[i, j] >= residual_collocation_temp[i, j + 1]:
                            index_add.append(j)
                        elif j == (m - 1) and residual_collocation_temp[i, j] >= residual_collocation_temp[i, j - 1]:
                            index_add.append(j)
                        elif residual_collocation_temp[i, j] >= residual_collocation_temp[i, j + 1] and \
                                residual_collocation_temp[i, j] >= residual_collocation_temp[i, j - 1]:
                            index_add.append(j)
                # add the initial time node first
                delta_t = tspan_temp[i + 1] - tspan_temp[i]
                t_init = tspan_temp[i]
                tspan_new.append(t_init)
                y0_init = y0_temp[i, 0: size_y]
                y0_next = y0_temp[i + 1, 0: size_y]
                y0_new.append(y0_init)
                z0_init = z0_temp[i, 0: size_z]
                z0_next = z0_temp[i + 1, 0: size_z]
                z0_new.append(z0_init)
                # for divided intervals, starts with m_init for all the intervals
                m_N_new.append(m_init)
                N_new += 1
                for index in index_add:
                    c = c_m[m + 1 - m_min, index]
                    t_cur = t_init + c * delta_t
                    tspan_new.append(t_cur)
                    y0_cur = (1 - c) * y0_init + c * y0_next
                    y0_new.append(y0_cur)
                    z0_cur = (1 - c) * z0_init + c * z0_next
                    z0_new.append(z0_cur)
                    m_N_new.append(m_init)
                    N_new += 1
                    n_a += 1
            else:
                # uniform type
                delta_t = tspan_temp[i + 1] - tspan_temp[i]
                t_init = tspan_temp[i]
                y0_init = y0_temp[i, 0: size_y]
                y0_next = y0_temp[i + 1, 0: size_y]
                z0_init = z0_temp[i, 0: size_z]
                z0_next = z0_temp[i + 1, 0: size_z]
                # increase the number of collocation points
                # if exceeds the maximum allowed, divide the interval into two
                if (m + m_i) > m_max:
                    # add the initial time node first
                    # add the current node with current number of collocation points
                    tspan_new.append(t_init)
                    y0_new.append(y0_init)
                    z0_new.append(z0_init)
                    m_N_new.append(m)
                    N_new += 1
                    # add the middle time node with current number of collocation points
                    t_mid = t_init + 0.5 * delta_t
                    y0_mid = (y0_init + y0_next) / 2
                    z0_mid = (z0_init + z0_next) / 2
                    tspan_new.append(t_mid)
                    y0_new.append(y0_mid)
                    z0_new.append(z0_mid)
                    m_N_new.append(m)
                    N_new += 1
                    n_a += 1
                else:
                    tspan_new.append(t_init)
                    y0_new.append(y0_init)
                    z0_new.append(z0_init)
                    m_N_new.append(m + m_i)
                    N_new += 1
        else:
            # directly add the node
            t_init = tspan_temp[i]
            tspan_new.append(t_init)
            y0_init = y0_temp[i, 0: size_y]
            y0_new.append(y0_init)
            z0_init = z0_temp[i, 0: size_z]
            z0_new.append(z0_init)
            m_N_new.append(m)
            N_new += 1
        i += 1  # update the loop index
    # add the final node
    tspan_new.append(tspan_temp[-1])
    y0_new.append(y0_temp[-1, :])
    z0_new.append(z0_temp[-1, :])
    N_new += 1
    # convert from list to numpy arrays for the convenience of indexing
    tspan_new = np.array(tspan_new)
    y0_new = np.array(y0_new)
    z0_new = np.array(z0_new)
    m_N_new = np.array(m_N_new)
    # generate the new accumulate collocation array
    m_accumulate_new = np.zeros(N_new, dtype=int)  # accumulated collocation points used
    for i in range(1, N_new):
        m_accumulate_new[i] = m_accumulate_new[i - 1] + m_N_new[i - 1]
    print("\tOriginal number of nodes: {}; "
          "Remove time nodes: {}; "
          "Add time nodes: {}; "
          "Number of nodes after mesh: {}".format(N, n_r, n_a, N_new))
    print("\tPrevious total number of collocation points: {}, new total number of collocation points: {}.".format(
        m_accumulate[-1], m_accumulate_new[-1]))
    # return the output
    return N_new, tspan_new, y0_new, z0_new, m_N_new, m_accumulate_new


# finish the implementation of remesh


# start the implementation of plot


def plot_result(size_y, size_z, t_span, y, z):
    for i in range(size_y):
        fig, ax = plt.subplots()
        ax.plot(t_span, y[:, i])
        ax.set(xlabel='time', ylabel='ODE variable %s' % (i + 1),
               title='{}_{}'.format('ODE variable', (i + 1)))
        ax.grid()
        plt.show()
    for i in range(size_z):
        fig, ax = plt.subplots()
        ax.plot(t_span, z[:, i])
        ax.set(xlabel='time', ylabel='DAE variable %s' % (i + 1),
               title='{}_{}'.format('DAE variable', (i + 1)))
        ax.grid()
        plt.show()


# finish the implementation of plot


@cuda.reduce
def cuda_infinity_norm(a, b):
    """
    Use @reduce decorator for converting a simple binary operation into a reduction kernel.
    :param a:
    :param b:
    :return:
    """
    return max(abs(a), abs(b))


def benchmark_data_init():
    # benchmark data
    initial_input_time = initial_input_count = \
        residual_time = residual_count = \
        jacobian_time = jacobian_count = \
        reduce_jacobian_time = reduce_jacobian_count = \
        recover_babd_time = recover_babd_count = \
        segment_residual_time = segment_residual_count = 0
    return initial_input_time, initial_input_count, residual_time, residual_count, \
        jacobian_time, jacobian_count, reduce_jacobian_time, reduce_jacobian_count, \
        recover_babd_time, recover_babd_count, segment_residual_time, segment_residual_count


def write_benchmark_result(fname,
                           initial_input_time, initial_input_count,
                           residual_time, residual_count,
                           jacobian_time, jacobian_count,
                           reduce_jacobian_time, reduce_jacobian_count,
                           recover_babd_time, recover_babd_count,
                           segment_residual_time, segment_residual_count,
                           total_time):
    try:
        with open(fname, 'w') as f:
            f.write('Initial input time: {}\n'.format(initial_input_time))
            f.write('Initial input counts: {}\n'.format(initial_input_count))
            f.write('Residual time: {}\n'.format(residual_time))
            f.write('Residual counts: {}\n'.format(residual_count))
            f.write('Jacobian time: {}\n'.format(jacobian_time))
            f.write('Jacobian counts: {}\n'.format(jacobian_count))
            f.write('Reduce Jacobian time: {}\n'.format(reduce_jacobian_time))
            f.write('Reduce Jacobian counts: {}\n'.format(reduce_jacobian_count))
            f.write('Recover BABD time: {}\n'.format(recover_babd_time))
            f.write('Recover BABD counts: {}\n'.format(recover_babd_count))
            f.write('Segment Residual time: {}\n'.format(segment_residual_time))
            f.write('Segment Residual counts: {}\n'.format(segment_residual_count))
            f.write('Total time: {}\n'.format(total_time))
    except OSError:
        print('Write time failed!')


def mesh_sanity_check(N, mesh_it, max_mesh, max_nodes, min_nodes):
    if mesh_it > max_mesh:
        print("\tReach maximum number of mesh refinements allowed.")
        return True
    if N > max_nodes:
        print('\tReach maximum number of nodes allowed.')
        return True
    elif N < min_nodes:
        print('\tReach minimum number of nodes allowed.')
        return True
    return False


"""
new implementation
"""
# from adaptive_collocation import get_y_dot, get_y, get_z, \
#     compute_f_q_parallel, construct_jacobian_parallel, reduce_jacobian_parallel, recover_delta_k_parallel


def compute_residual_nodal(size_y, size_z, size_p, m_min, m_max, N, m_N_original, m_accumulate_original,
                           c_m, t_span, y, y_dot, z_tilde, p, alpha, tol, m_sum, a_m, b_m, M):
    print("\tCompute nodal residual:")
    y_tilde_interpolated, z_tilde_interpolated, \
    y_additional, y_dot_additional, z_tilde_additional, para_additional, \
    m_N_additional, m_accumulate_additional = \
        generate_nodal_intput_parallel(
            size_y, size_z, m_min, m_max, N, m_N_original, m_accumulate_original, c_m, t_span, y, y_dot, z_tilde, p)
    # compute the residual
    norm_f_q, y_tilde_additional, f_a, f_b, r_bc = compute_f_q_parallel(
        size_y, size_z, size_p, m_min, m_max + 1, m_N_additional, m_accumulate_additional, m_sum, N, a_m, b_m,
        t_span, y_additional, y_dot_additional, z_tilde_additional, para_additional, alpha)
    # compute each necessary element in the Jacobian matrix
    J, V, D, W, B_0, B_n, V_n = construct_jacobian_parallel(
        size_y, size_z, size_p, m_min, m_max + 1, N,
        m_N_additional, m_accumulate_additional, m_sum, a_m, b_m,
        t_span, y_additional, y_tilde_additional, z_tilde_additional, para_additional, alpha)
    # compute each necessary element in the reduced BABD system
    A, C, H, b = reduce_jacobian_parallel(
        size_y, size_z, size_p, m_max + 1, N,
        m_N_additional, m_accumulate_additional,
        W, D, J, V, f_a, f_b)
    # solve the BABD system
    # perform the partition factorization on the Jacobian matrix with qr decomposition
    index, R, E, J_reduced, G, d, A_tilde, C_tilde, H_tilde, b_tilde = \
        solve_babd_system.partition_factorization_parallel(size_y, size_p, M, N, A, C, H, b)
    # construct the partitioned Jacobian system
    sol = solve_babd_system.construct_babd_mshoot(
        size_y, 0, size_p, M, A_tilde, C_tilde, H_tilde, b_tilde, B_0, B_n, V_n, -r_bc)
    # perform the qr decomposition to transfer the system
    solve_babd_system.qr_decomposition(size_y, size_p, M + 1, sol)
    # perform the backward substitution to obtain the solution to the linear system of Newton's method
    solve_babd_system.backward_substitution(M + 1, sol)
    # obtain the solution from the reduced BABD system
    delta_s_r, delta_para = solve_babd_system.recover_babd_solution(M, size_y, 0, size_p, sol)
    # get the solution to the BABD system
    delta_y = solve_babd_system.partition_backward_substitution_parallel(
        size_y, size_p, M, N, index, delta_s_r, delta_para, R, G, E, J_reduced, d)
    # recover delta_k from the reduced BABD system
    delta_k, delta_y_dot, delta_z_tilde = recover_delta_k_parallel(
        size_y, size_z, size_p, m_max + 1, N, m_N_additional, m_accumulate_additional,
        delta_y, delta_para, f_a, J, V, W)
    residual, residual_collocation, max_residual, max_y_error, max_z_error = compute_nodal_residual(
        size_y, size_z, size_p, m_min, m_max + 1, N, tol,
        t_span, m_N_additional, m_accumulate_additional, y_additional, y_dot_additional, z_tilde_additional,
        delta_y, delta_y_dot, delta_z_tilde, delta_para)
    return residual, residual_collocation, max_residual, max_y_error, max_z_error


def compute_nodal_residual(size_y, size_z, size_p, m_min, m_max, N, tol,
                           t_span, m_N, m_accumulate, y, y_dot, z_tilde,
                           delta_y, delta_y_dot, delta_z_tilde, delta_para):
    # initialize the gauss coefficients
    w_m = lobatto_weights_init(m_min, m_max)
    residual = np.zeros(N)
    residual_collocation = np.zeros((N, m_max))
    # compute the normalizer
    y_normalizer = np.max(np.abs(y), axis=0)
    y_dot_normalizer = np.max(np.abs(y_dot), axis=0)
    y_normalizer = np.maximum(y_normalizer, y_dot_normalizer)
    z_normalizer = np.max(np.abs(z_tilde), axis=0)
    y_error = np.zeros((N, size_y))
    z_error = np.zeros((N, size_z))
    for j in range(N - 1):
        m = m_N[j]  # number of collocation points in the jth interval
        delta_j = t_span[j + 1] - t_span[j]
        # compute the y error first
        # initialization
        y_error[j, 0: size_y] = np.abs(delta_y[j, 0: size_y])
        # delta_y_error = np.zeros(size_y)
        for i in range(m):
            y_error[j, 0: size_y] += delta_j * w_m[m - m_min, i] * np.abs(delta_y_dot[m_accumulate[j] + i, 0: size_y])
            # y_error[j, 0: size_y] += delta_j * w_m[m - m_min, i] * delta_y_dot[m_accumulate[j] + i, 0: size_y]
            # delta_y_error += delta_j * w_m[m - m_min, i] * delta_y_dot[m_accumulate[j] + i, 0: size_y]
            z_error[j, 0: size_z] = np.maximum(z_error[j, 0: size_z],
                                               np.abs(delta_z_tilde[m_accumulate[j] + i, 0: size_z]))
        # y_error[j, 0: size_y] += np.abs(delta_y_error)
        # normalization
        y_error[j, 0: size_y] = y_error[j, 0: size_y] / (1 + y_normalizer)
        z_error[j, 0: size_z] = z_error[j, 0: size_z] / (1 + z_normalizer)
        residual[j] = np.maximum(np.amax(y_error[j, 0: size_y]), np.amax(z_error[j, 0: size_z])) / tol
    y_error[N - 1, 0: size_y] = np.abs(delta_y[N - 1, 0: size_y]) / (1 + y_normalizer)
    z_error[N - 1, 0: size_z] = np.abs(delta_z_tilde[m_accumulate[-1] - 1, 0: size_z]) / (1 + z_normalizer)
    residual[N - 1] = np.maximum(np.amax(y_error[N - 1, 0: size_y]), np.amax(z_error[N - 1, 0: size_z])) / tol
    # residual[N - 1] = np.maximum(np.amax(np.abs(delta_y[N - 1, 0: size_y])),
    #                              np.amax(np.abs(delta_z_tilde[m_accumulate[-1] - 1, 0: size_z]))) / tol
    max_y_error = np.amax(np.max(y_error, axis=1))
    max_z_error = np.amax(np.max(z_error, axis=1))
    max_residual = np.amax(residual)
    print('\terror: |y|: {}, |z|: {}'.format(max_y_error / tol, max_z_error / tol))
    return residual, residual_collocation, max_residual, max_y_error, max_z_error


def create_additional_accumulate_index(N, m_N):
    m_accumulate = np.zeros(N, dtype=int)  # accumulated collocation points used
    for i in range(1, N):
        # an additional collocation point when evaluating residual
        m_accumulate[i] = m_accumulate[i - 1] + (m_N[i - 1] + 1)
    return m_accumulate


def generate_nodal_intput_parallel(size_y, size_z, m_min, m_max, N, m_N, m_accumulate,
                                   c_m, t_span, y, y_dot, z_tilde, p):
    # grid dimension of the kernel of CUDA
    grid_dims_1d = ((N - 1) + TPB_N - 1) // TPB_N
    block_dims_1d = TPB_N
    # create the accumulative for evaluating the residual
    # as each interval need an additional collocation point
    m_accumulate_additional = create_additional_accumulate_index(N, m_N)
    # tranfer memory from CPU to GPU
    d_m_N = cuda.to_device(m_N)
    d_m_accumulate = cuda.to_device(m_accumulate)
    d_m_accumulate_additional = cuda.to_device(m_accumulate_additional)
    d_c_m = cuda.to_device(c_m)
    d_t_span = cuda.to_device(t_span)
    d_y = cuda.to_device(y)
    d_y_dot = cuda.to_device(y_dot)
    d_z_tilde = cuda.to_device(z_tilde)
    d_L = cuda.device_array((N - 1, m_max), dtype=np.float64)
    d_I = cuda.device_array((N - 1, m_max + 1), dtype=np.float64)
    # interpolated solution from the original solution
    d_y_interpolated = cuda.device_array((m_accumulate_additional[-1], size_y), dtype=np.float64)
    d_z_interpolated = cuda.device_array((m_accumulate_additional[-1], size_z), dtype=np.float64)
    d_y_dot_interpolated = cuda.device_array((m_accumulate_additional[-1], size_y), dtype=np.float64)
    generate_nodal_input_kernel[grid_dims_1d, block_dims_1d](
        size_y, size_z, m_min, N,
        d_m_N, d_m_accumulate, d_m_accumulate_additional, d_c_m, d_t_span,
        d_y, d_y_dot, d_z_tilde, d_L, d_I,
        d_y_interpolated, d_y_dot_interpolated, d_z_interpolated)
    y_tilde_interpolated = d_y_interpolated.copy_to_host()
    y_dot_interpolated = d_y_dot_interpolated.copy_to_host()
    z_tilde_interpolated = d_z_interpolated.copy_to_host()
    # initial guess for the problem with an additional collocation point
    # in each sub-interval,
    y_additional = np.copy(y)
    y_dot_additional = np.copy(y_dot_interpolated)
    z_tilde_additional = np.copy(z_tilde_interpolated)
    p_additional = np.copy(p)
    m_N_additional = m_N + 1
    return y_tilde_interpolated, z_tilde_interpolated, \
           y_additional, y_dot_additional, z_tilde_additional, p_additional, \
           m_N_additional, m_accumulate_additional


@cuda.jit()
def generate_nodal_input_kernel(
        size_y, size_z, m_min, N,
        d_m_N, d_m_accumulate, d_m_accumulate_additional, d_c_m, d_t_span,
        d_y, d_y_dot, d_z_tilde, d_L, d_I,
        d_y_additional, d_y_dot_additional, d_z_additional):
    j = cuda.grid(1)  # cuda thread index
    if j < (N - 1):
        m = d_m_N[j]  # number of collocation points in the jth interval
        delta_t_j = d_t_span[j + 1] - d_t_span[j]  # time span length
        # compute the residual at (m + 1) collocation points
        for i in range(m + 1):
            # compute initial y_dot approximation
            get_y_dot(size_y, m, d_c_m[m + 1 - m_min, i], d_L[j, 0: m],
                      d_y_dot[d_m_accumulate[j]: d_m_accumulate[j + 1], 0: size_y],
                      d_y_dot_additional[d_m_accumulate_additional[j] + i, 0: size_y])
            # compute initial y approximation
            get_y(size_y, m, d_c_m[m + 1 - m_min, i], delta_t_j, d_I[j, 0: m], d_y[j, 0: size_y],
                  d_y_dot[d_m_accumulate[j]: d_m_accumulate[j + 1], 0: size_y],
                  d_y_additional[d_m_accumulate_additional[j] + i, 0: size_y])
            # compute initial z approximation
            get_z(size_z, m, d_c_m[m + 1 - m_min, i], d_L[j, 0: m],
                  d_z_tilde[d_m_accumulate[j]: d_m_accumulate[j + 1], 0: size_z],
                  d_z_additional[d_m_accumulate_additional[j] + i, 0: size_z])
    return


if __name__ == '__main__':
    collocation_solver_parallel()

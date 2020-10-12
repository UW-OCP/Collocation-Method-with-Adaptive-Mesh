import numpy as np
from math import *
import sys
from lobatto_power_series_coeffcients import *


def normal_remesh_add_only(size_y, size_z, N, tspan, y0, z0, residual, thres_remove, thres_add):
    """
    Remesh the problem with only adding time nodes normally.
    :param size_y: number of y variables
    :param size_z: number of z variables
    :param N: number of time nodes in the current mesh
    :param tspan: time span of the current mesh
    :param y0: values of the differential variables in matrix form
    :param z0: values of the algebraic variables in matrix form
    :param residual: residual error evaluated for each time interval
    :param thres_remove: numerical threshold for removing nodes
    :param thres_add: numerical threshold for adding nodes
    :return:
        N_New : number of time nodes of the new mesh .
        tspan_New : new time span of the problem.
        y0_New : values of the differential variables in new mesh in matrix form
        z0_New : values of the algebraic variables in in new mesh in matrix form
    """
    N_temp = N
    tspan_temp = np.array(tspan)
    y0_temp = np.array(y0)
    z0_temp = np.array(z0)
    residual_temp = np.array(residual)
    # lists to hold the outputs
    N_new = 0
    tspan_new = []
    y0_new = []
    z0_new = []
    residual_new = []
    # Adding Nodes
    i = 0
    # Record the number of the added nodes
    k_a = 0

    while i < N_temp - 1:
        res_i = residual_temp[i]
        if res_i > thres_add:
            # add one node to the middle
            # add the time point of the new node
            delta_t = (tspan_temp[i + 1] - tspan_temp[i]) / 2
            t_i = tspan_temp[i]
            t_i_add = t_i + delta_t
            tspan_new.append(t_i)
            tspan_new.append(t_i_add)
            # add the y of the new node
            y0_i = y0_temp[i, 0: size_y]
            y0_i_next = y0_temp[i + 1, 0: size_y]
            delta_y0 = (y0_i_next - y0_i) / 2
            y0_i_add = y0_i + delta_y0
            y0_new.append(y0_i)
            y0_new.append(y0_i_add)
            # add the z of the new node
            z0_i = z0_temp[i, 0: size_z]
            z0_i_next = z0_temp[i + 1, 0: size_z]
            delta_z0 = (z0_i_next - z0_i) / 2
            z0_i_add = z0_i + delta_z0
            z0_new.append(z0_i)
            z0_new.append(z0_i_add)
            # update the index
            # 1 original node + 1 newly added node
            N_new += 2
            k_a += 1
            i += 1
        else:
            # add the current node only
            # add the time node of the current node
            t_i = tspan_temp[i]
            tspan_new.append(t_i)
            # add the residual of the current node
            residual_new.append(res_i)
            # add the y of the current node
            y0_i = y0_temp[i, 0: size_y]
            y0_new.append(y0_i)
            # add the z of the current node
            z0_i = z0_temp[i, 0: size_z]
            z0_new.append(z0_i)
            # update the index
            # 1 original node only
            N_new += 1
            i += 1
    # add the final node
    tspan_new.append(tspan_temp[N_temp - 1])
    y0_new.append(y0_temp[N_temp - 1, 0: size_y])
    z0_new.append(z0_temp[N_temp - 1, 0: size_z])
    N_new += 1
    # convert from list to numpy arrays for the convenience of indexing
    tspan_new = np.array(tspan_new)
    y0_new = np.array(y0_new)
    z0_new = np.array(z0_new)
    print("\tAdd nodes: {}; Number of nodes before mesh: {}; after mesh: {}".format(k_a, N, N_new))
    # return the output
    return N_new, tspan_new, y0_new, z0_new


def ph_remesh_increase_m_add_n(size_y, size_z, m_max, m_init, N,
              m_N, m_accumulate, tspan, y0, z0,
              residual, residual_collocation, thres_add):
    """
    Implementation for increasing collocation points or adding time nodes only.
    """
    # # convert from list to numpy arrays for the convenience of indexing
    # N_temp = N
    # tspan_temp = np.array(tspan)
    # y0_temp = np.array(y0)
    # z0_temp = np.array(z0)
    # residual_temp = np.array(residual)
    # residual_collocation_temp = np.array(residual_collocation)
    # m_N_temp = np.array(m_N)

    # lists to hold the outputs
    N_new = 0
    tspan_new = []
    y0_new = []
    z0_new = []
    m_N_new = []
    # Adding Nodes
    i = 0
    n_a = 0  # number of the added time nodes
    m_added = 0  # number of the added collocation points

    while i < N - 1:
        res_i = residual[i]
        m = m_N[i]
        t_cur = tspan[i]
        y0_cur = y0[i, 0: size_y]
        z0_cur = z0[i, 0: size_z]
        if res_i > thres_add:
            # values for current time node
            delta_t = tspan[i + 1] - tspan[i]
            if delta_t < sys.float_info.epsilon:
                print(f"{tspan[i]} to {tspan[i + 1]} interval too small")
                i += 1
                continue
            y0_next = y0[i + 1, 0: size_y]
            z0_next = z0[i + 1, 0: size_z]
            # compute the needed number of collocation points to be added
            m_additional = int(ceil(log(1 / res_i, delta_t)))
            if (m + m_additional) > m_max:
                # if exceeds the maximum allowed, divide the interval into multiple
                # determine the number of nodes to be added
                n_add = int(ceil((m + m_additional) / m_init))
                n_a += (n_add - 1)  # number of added time nodes
                for j in range(n_add):
                    # add the starting node for each sub-interval to be added
                    t_add = t_cur + j * delta_t / n_add
                    tspan_new.append(t_add)
                    y_add = (1 - j / n_add) * y0_cur + (j / n_add) * y0_next
                    y0_new.append(y_add)
                    z_add = (1 - j / n_add) * z0_cur + (j / n_add) * z0_next
                    z0_new.append(z_add)
                    m_N_new.append(m_init)
                    N_new += 1
                    m_added += m_init
                m_added -= m  # additional added time number of collocation points
            else:
                # increase the number of collocation points used
                tspan_new.append(t_cur)
                y0_new.append(y0_cur)
                z0_new.append(z0_cur)
                m_N_new.append(m + m_additional)
                N_new += 1
                m_added += m_additional
        else:
            # directly add the node
            tspan_new.append(t_cur)
            y0_new.append(y0_cur)
            z0_new.append(z0_cur)
            m_N_new.append(m)
            N_new += 1
        i += 1  # update the loop index
    # add the final node
    tspan_new.append(tspan[-1])
    y0_new.append(y0[-1, :])
    z0_new.append(z0[-1, :])
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
          "Add time nodes: {}; "
          "Number of nodes after mesh: {}".format(N, n_a, N_new))
    print("\tOriginal number of collocation points: {}; "
          "Add collocation points: {}; "
          "New total number of collocation points: {}.".format(
            m_accumulate[-1], m_added, m_accumulate_new[-1]))
    # return the output
    return N_new, tspan_new, y0_new, z0_new, m_N_new, m_accumulate_new


def generate_normalizer(size_y, size_z, N, m_accumulate, y_tilde, y_dot, z_tilde):
    y_normalizer = np.zeros((N, size_y))
    y_dot_normalizer = np.zeros((N, size_y))
    z_normalizer = np.zeros((N, size_z))
    for i in range(N - 1):
        m_start = m_accumulate[i]
        m_end = m_accumulate[i + 1]
        # collect the maximum for each element in y_dot
        y_normalizer[i, :] = 1 + np.max(np.abs(y_tilde[m_start: m_end, 0: size_y]), axis=0)
        # collect the maximum for each element in y_dot
        y_dot_normalizer[i, :] = 1 + np.max(np.abs(y_dot[m_start: m_end, 0: size_y]), axis=0)
        # collect the maximum for each element in z_tilde
        z_normalizer[i, :] = 1 + np.max(np.abs(z_tilde[m_start: m_end, 0: size_z]), axis=0)
    return y_normalizer, y_dot_normalizer, z_normalizer


def ph_remesh_decrease_m_remove_n(size_y, size_z, size_p, m_min, N, alpha,
              m_N, m_accumulate, tspan, y0, z0, p0, y_tilde, y_dot, z_tilde,
              residual, residual_collocation,
              thres_remove, tol):
    """
    Implementation for decreasing collocation points or removing time nodes only.
    """
    # necessary fields
    N_new = 0
    tspan_new = []
    y0_new = []
    z0_new = []
    residual_new = []
    residual_collocation_new = []
    m_N_new = []
    # create the normalizer
    # y_normalizer, y_dot_normalizer, z_normalizer = \
    #     generate_normalizer(size_y, size_z, N, m_accumulate, y_tilde, y_dot, z_tilde)
    # obtain the normalizer directly
    y_tilde_normalizer = np.max(np.abs(y_tilde), axis=0)
    y_dot_normalizer = np.max(np.abs(y_dot), axis=0)
    y_normalizer = 1 + np.maximum(y_tilde_normalizer, y_dot_normalizer)
    z_normalizer = 1 + np.max(np.abs(z_tilde), axis=0)

    # removing Nodes
    i = 0
    n_removed = 0  # number of removed nodes
    m_removed = 0  # number of removed collocation points

    # loop through all the sub-intervals
    while i < N - 1:
        res_i = residual[i]
        m = m_N[i]
        t_cur = tspan[i]
        y0_cur = y0[i, 0: size_y]
        z0_cur = z0[i, 0: size_z]
        residual_cur = residual[i]
        residual_collocation_cur = residual_collocation[i]
        if res_i < thres_remove:
            delta_t = tspan[i + 1] - tspan[i]
            if m > m_min:
                # decrease the number of collocation points first
                # by computing the bounded error for dropping from the highest order power series
                # lagrange polynomial power series coefficients with s^{k}
                # order: 0: m-1, totoal of m
                a_coef_0 = np.zeros((m, m))
                # integration of the lagrange polynomial power series coefficients with s^{k}
                # order: 0: m, totoal of m + 1
                a_integration_coef_0 = np.zeros((m, m + 1))
                get_power_series_coef_base_0(m, a_coef_0)
                get_power_series_integration_coef_base_0(m, a_integration_coef_0)
                m_start = m_accumulate[i]
                m_end = m_accumulate[i + 1]
                y_tilde_cur = y_tilde[m_start: m_end, 0: size_y]
                y_dot_cur = y_dot[m_start: m_end, 0: size_y]
                z_tilde_cur = z_tilde[m_start: m_end, 0: size_z]
                # compute the power series terms
                # b_k is the term for s^{k}
                # b_k_y = a_integration_{j, k} * y_{i, j}
                # order from 1 to m
                # b_k_y_dot = a_{j, k} * y_{i, j}
                # order from 0 to m - 1
                # b_k_z = a_{j, k} * z_{i, j}
                # order from 0 to m - 1
                b_y = np.zeros((m, size_y))
                b_y_dot = np.zeros((m, size_y))
                b_z = np.zeros((m, size_z))
                flag_continue = True
                for k in range(m):
                    # total of m terms in power series
                    # y: 1:m
                    # y_dot: 0:m-1
                    # z: 0:m-1
                    for j in range(m):
                        # loop through all the collocation points
                        # no constant terms in y, so b_y[k, :] corresponds to s^{k + 1} series
                        b_y[k, 0: size_y] += a_integration_coef_0[j, k + 1] * y_dot_cur[j, 0: size_y]
                        b_y_dot[k, 0: size_y] += a_coef_0[j, k] * y_dot_cur[j, 0: size_y]
                        b_z[k, 0: size_z] += a_coef_0[j, k] * z_tilde_cur[j, 0: size_z]
                # ---
                # # compute the gradient
                # D_g_max = np.zeros((size_z, size_y + size_z + size_p))
                # D_g = np.zeros((size_z, size_y + size_z + size_p))
                # for j in range(m):
                #     abvp_Dg(y_tilde_cur[j, :], z_tilde_cur[j, :], p0, alpha, D_g)
                #     D_g_max = np.maximum(D_g_max, np.abs(D_g))
                # # keep the necessary terms only dg/dy, dg/dz
                # D_g_max = D_g_max[0: size_z, 0: (size_y + size_z)]
                # ---
                # try to decrease the order from m to m_min
                # each time decrease by one degree
                m_new = m  # new collocation points
                for k in range(m - 1, m_min - 1, -1):
                    # try to decrease power series in y terms first
                    # check every y term
                    for j in range(size_y):
                        upper_bound = np.abs(b_y_dot[k, j]) / y_dot_normalizer[j]
                        if upper_bound > thres_remove * tol:
                            # stop checking now
                            flag_continue = False
                            break
                    if not flag_continue:
                        break
                    # +++
                    # try to decrease power series in z terms next
                    # check every z term
                    for j in range(size_z):
                        upper_bound = np.abs(b_z[k, j]) / z_normalizer[j]
                        if upper_bound > thres_remove * tol:
                            # stop checking now
                            flag_continue = False
                            break
                    if not flag_continue:
                        break
                    # +++
                    # ---
                    # # check every g term
                    # for j in range(size_z):
                    #     # check y terms corresponding to nonzero partial derivatives
                    #     D_g_y_max = 0
                    #     y_max = 0
                    #     for l in range(size_y):
                    #         if D_g_max[j, l] > 0:
                    #             # y_l = np.abs(b_y[k, l]) / y_normalizer[i, l]
                    #             y_l = np.abs(b_y[k, l]) / y_normalizer[l]
                    #             D_g_y_l = np.abs(D_g_max[j, l])
                    #             y_max = max(y_max, y_l)
                    #             D_g_y_max = max(D_g_y_max, D_g_y_l)
                    #     # check z terms corresponding to nonzero partial derivatives
                    #     D_g_z_max = 0
                    #     z_max = 0
                    #     for l in range(size_z):
                    #         if D_g_max[j, size_y + l] > 0:
                    #             # z_l = np.abs(b_z[k, l]) / z_normalizer[i, l]
                    #             z_l = np.abs(b_z[k, l]) / z_normalizer[l]
                    #             D_g_z_l = np.abs(D_g_max[j, size_y + l])
                    #             z_max = max(z_max, z_l)
                    #             D_g_z_max = max(D_g_z_max, D_g_z_l)
                    #     upper_bound = D_g_y_max * y_max + D_g_z_max * z_max
                    #     if upper_bound > thres_remove * tol:
                    #         # stop checking now
                    #         flag_continue = False
                    #         break
                    # if not flag_continue:
                    #     break
                    # ---
                    # current k order term can be removed
                    # update the collocation points number
                    m_new = k
                tspan_new.append(t_cur)
                y0_new.append(y0_cur)
                z0_new.append(z0_cur)
                residual_new.append(residual_cur)
                residual_collocation_new.append(residual_collocation_cur)
                m_N_new.append(m_new)
                N_new += 1
                m_removed += (m - m_new)
            else:
                # m == m_min
                # try to merge with the next interval
                merge = False  # flag for merging
                if i < N - 2:
                    res_i_next = residual[i + 1]
                    m_next = m_N[i + 1]
                    delta_t_next = tspan[i + 2] - tspan[i + 1]
                    # the residual from the next interval should be also below the threshold
                    # plus the number of collocation points is minimum
                    if res_i_next < thres_remove and m_next == m_min:
                        delta_t_small = min(delta_t, delta_t_next)
                        # m = m_next == m_min
                        # lagrange polynomial power series coefficients with s^{k}
                        # order: 0: m-1, totoal of m
                        a_coef_0 = np.zeros((m, m))
                        # lagrange polynomial power series coefficients with (s - 1)^{k}
                        # order: 0: m-1, totoal of m
                        a_coef_1 = np.zeros((m, m))
                        # integration of the lagrange polynomial power series coefficients with s^{k}
                        # order: 0: m, totoal of m + 1
                        a_integration_coef_0 = np.zeros((m, m + 1))
                        # integration of the lagrange polynomial power series coefficients with (s - 1)^{k}
                        # order: 0: m, totoal of m + 1
                        a_integration_coef_1 = np.zeros((m, m + 1))
                        get_power_series_coef_base_0(m, a_coef_0)
                        get_power_series_coef_base_1(m, a_coef_1)
                        get_power_series_integration_coef_base_0(m, a_integration_coef_0)
                        get_power_series_integration_coef_base_0(m, a_integration_coef_1)
                        m_start = m_accumulate[i]
                        m_end = m_accumulate[i + 1]
                        m_start_next = m_accumulate[i + 1]
                        m_end_next = m_accumulate[i + 2]
                        y_tilde_cur = y_tilde[m_start: m_end, 0: size_y]
                        y_dot_cur = y_dot[m_start: m_end, 0: size_y]
                        z_tilde_cur = z_tilde[m_start: m_end, 0: size_z]
                        y_tilde_next = y_tilde[m_start_next: m_end_next, 0: size_y]
                        y_dot_next = y_dot[m_start_next: m_end_next, 0: size_y]
                        z_tilde_next = z_tilde[m_start_next: m_end_next, 0: size_z]
                        # compute the power series terms
                        # b_cur uses the base 1 power seris as (s - 1)^{k}
                        # b_next uses the base 0 power seris as s^{k}
                        # b[k, :] is the computed coefficient term for s^{k}
                        b_y_cur = np.zeros((m, size_y))
                        b_y_dot_cur = np.zeros((m, size_y))
                        b_z_cur = np.zeros((m, size_z))
                        b_y_next = np.zeros((m, size_y))
                        b_y_dot_next = np.zeros((m, size_y))
                        b_z_next = np.zeros((m, size_z))
                        flag_continue = True
                        for k in range(m):
                            # total of m terms in power series
                            # y: 1:m
                            # y_dot: 0:m-1
                            # z: 0:m-1
                            for j in range(m):
                                # loop through all the collocation points
                                # no constant terms in y, so b_y[k, :] corresponds to s^{k + 1} series
                                b_y_cur[k, 0: size_y] += a_integration_coef_1[j, k + 1] * y_dot_cur[j, 0: size_y]
                                b_y_dot_cur[k, 0: size_y] += a_coef_1[j, k] * y_dot_cur[j, 0: size_y]
                                b_z_cur[k, 0: size_z] += a_coef_1[j, k] * z_tilde_cur[j, 0: size_z]
                                b_y_next[k, 0: size_y] += a_integration_coef_0[j, k + 1] * y_dot_next[j, 0: size_y]
                                b_y_dot_next[k, 0: size_y] += a_coef_0[j, k] * y_dot_next[j, 0: size_y]
                                b_z_next[k, 0: size_z] += a_coef_0[j, k] * z_tilde_next[j, 0: size_z]
                            # scale by the time
                            b_y_cur[k, 0: size_y] *= (delta_t_small / delta_t) ** (k + 1)
                            b_y_dot_cur[k, 0: size_y] *= (delta_t_small / delta_t) ** k
                            b_z_cur[k, 0: size_z] *= (delta_t_small / delta_t) ** k
                            b_y_next[k, 0: size_y] *= (delta_t_small / delta_t_next) ** (k + 1)
                            b_y_dot_next[k, 0: size_y] *= (delta_t_small / delta_t_next) ** k
                            b_z_next[k, 0: size_z] *= (delta_t_small / delta_t_next) ** k
                        # ---
                        # # compute the gradient
                        # D_g_max = np.zeros((size_z, size_y + size_z + size_p))
                        # D_g = np.zeros((size_z, size_y + size_z + size_p))
                        # for j in range(m):
                        #     if delta_t <= delta_t_next:
                        #         abvp_Dg(y_tilde_cur[j, :], z_tilde_cur[j, :], p0, alpha, D_g)
                        #     else:
                        #         abvp_Dg(y_tilde_next[j, :], z_tilde_next[j, :], p0, alpha, D_g)
                        #     D_g_max = np.maximum(D_g_max, np.abs(D_g))
                        # # keep the necessary terms only dg/dy, dg/dz
                        # D_g_max = D_g_max[0: size_z, 0: (size_y + size_z)]
                        # # check the bound for each component
                        # y_normalizer_i = np.max(y_normalizer[i: i+2, :], axis=0)
                        # y_dot_normalizer_i = np.max(y_dot_normalizer[i: i + 2, :], axis=0)
                        # z_normalizer_i = np.max(z_normalizer[i: i + 2, :], axis=0)
                        # ---
                        # check every y term
                        for j in range(size_y):
                            y_dot_diff_j = 0.0
                            for k in range(m):
                                y_dot_diff_j += np.abs(b_y_dot_cur[k, j] - b_y_dot_next[k, j]) / y_normalizer[j]
                            if y_dot_diff_j > thres_remove * tol:
                                # stop checking now
                                flag_continue = False
                                break
                        # continue only if y terms meet the condition
                        if flag_continue:
                            # ---
                            # # check every g term
                            # for j in range(size_z):
                            #     sum_j = 0.0
                            #     # check y terms corresponding to nonzero partial derivatives
                            #     D_g_y_max = 0
                            #     y_diff_max = 0
                            #     for l in range(size_y):
                            #         if D_g_max[j, l] > 0:
                            #             y_diff_l = 0.0
                            #             for k in range(m):
                            #                 # y_diff_l += np.abs(b_y_cur[k, l] - b_y_next[k, l]) / y_normalizer_i[l]
                            #                 y_diff_l += np.abs(b_y_cur[k, l] - b_y_next[k, l]) / y_normalizer[l]
                            #             D_g_y_l = np.abs(D_g_max[j, l])
                            #             y_diff_max = max(y_diff_max, y_diff_l)
                            #             D_g_y_max = max(D_g_y_max, D_g_y_l)
                            #     # check z terms corresponding to nonzero partial derivatives
                            #     D_g_z_max = 0
                            #     z_diff_max = 0
                            #     for l in range(size_z):
                            #         if D_g_max[j, size_y + l] > 0:
                            #             z_diff_l = 0.0
                            #             for k in range(m):
                            #                 # z_diff_l += np.abs(b_z_cur[k, l] - b_z_next[k, l]) / z_normalizer_i[l]
                            #                 z_diff_l += np.abs(b_z_cur[k, l] - b_z_next[k, l]) / z_normalizer[l]
                            #             D_g_z_l = np.abs(D_g_max[j, size_y + l])
                            #             z_diff_max = max(z_diff_max, z_diff_l)
                            #             D_g_z_max = max(D_g_z_max, D_g_z_l)
                            #     g_diff = D_g_y_max * y_diff_max + D_g_z_max * z_diff_max
                            #     if g_diff > thres_remove * tol:
                            #         # stop checking now
                            #         flag_continue = False
                            #         break
                            # ---
                            # +++
                            # check every z term
                            for j in range(size_z):
                                z_diff_j = 0.0
                                for k in range(m):
                                    z_diff_j += np.abs(b_z_cur[k, j] - b_z_next[k, j]) / z_normalizer[j]
                                if z_diff_j > thres_remove * tol:
                                    # stop checking now
                                    flag_continue = False
                                    break
                            # +++
                            # only when all g terms passed, merge the two intervals
                            if flag_continue:
                                merge = True
                tspan_new.append(t_cur)
                y0_new.append(y0_cur)
                z0_new.append(z0_cur)
                residual_new.append(residual_cur)
                residual_collocation_new.append(residual_collocation_cur)
                m_N_new.append(m)
                N_new += 1
                if merge:
                    # jump the next interval
                    n_removed += 1  # remove one time node
                    m_removed += m_min  # remove m_min collocation points
                    i += 1  # skip the next time node cause it's removed
        else:
            # directly add the time node
            tspan_new.append(t_cur)
            y0_new.append(y0_cur)
            z0_new.append(z0_cur)
            residual_new.append(residual_cur)
            residual_collocation_new.append(residual_collocation_cur)
            m_N_new.append(m)
            N_new += 1
        i += 1  # move to the next node

    # directly add the last time node
    tspan_new.append(tspan[-1])
    y0_new.append(y0[-1, 0: size_y])
    z0_new.append(z0[-1, 0: size_z])
    residual_new.append(residual[-1])
    N_new += 1

    # convert from list to numpy arrays for the convenience of indexing
    tspan_new = np.array(tspan_new)
    y0_new = np.array(y0_new)
    z0_new = np.array(z0_new)
    residual_new = np.array(residual_new)
    residual_collocation_new = np.array(residual_collocation_new)
    m_N_new = np.array(m_N_new)

    # generate the new accumulate collocation array
    m_accumulate_new = np.zeros(N_new, dtype=int)  # accumulated collocation points used
    for i in range(1, N_new):
        m_accumulate_new[i] = m_accumulate_new[i - 1] + m_N_new[i - 1]
    print("\tOriginal number of nodes: {}; "
          "Remove time nodes: {}; "
          "Number of nodes after mesh: {}".format(N, n_removed, N_new))
    print("\tOriginal number of collocation points: {}; "
          "Remove collocation points: {}; "
          "New total number of collocation points: {}.".format(
            m_accumulate[-1], m_removed, m_accumulate_new[-1]))
    # return the output
    return N_new, tspan_new, y0_new, z0_new, m_N_new, m_accumulate_new, residual_new, residual_collocation_new


def adaptive_remesh(size_y, size_z, size_p, m_min, m_max, m_init, N, alpha,
              m_N, m_accumulate, tspan, y0, z0, p0, y_tilde, y_dot, z_tilde,
              residual, residual_collocation,
              thres_remove, thres_add, tol):
    N_removed, tspan_removed, y0_removed, z0_removed, \
    m_N_removed, m_accumulate_removed, residual_removed, residual_collocation_removed = \
        ph_remesh_decrease_m_remove_n(size_y, size_z, size_p, m_min, N, alpha,
              m_N, m_accumulate, tspan, y0, z0, p0, y_tilde, y_dot, z_tilde,
              residual, residual_collocation,
              thres_remove, tol)
    N_new, tspan_new, y0_new, z0_new, m_N_new, m_accumulate_new = \
        ph_remesh_increase_m_add_n(size_y, size_z, m_max, m_init, N_removed,
              m_N_removed, m_accumulate_removed, tspan_removed, y0_removed, z0_removed,
              residual_removed, residual_collocation_removed, thres_add)
    return N_new, tspan_new, y0_new, z0_new, m_N_new, m_accumulate_new

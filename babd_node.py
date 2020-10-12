import numpy as np


class MultipleShootingNode:
    """
    Data structure used for multiple shooting.
    """
    def __init__(self, size_y, size_z, size_p):
        self.size_y = size_y
        self.size_z = size_z
        self.size_p = size_p
        self.A = np.zeros(0)
        self.C = np.zeros(0)
        self.H = np.zeros(0)
        self.b = np.zeros(0)
        self.B_0 = np.zeros(0)
        self.B_n = np.zeros(0)
        self.H_n = np.zeros(0)
        self.b_n = np.zeros(0)
        self.delta_s = np.zeros(0)

    def get_A(self):
        """
        :return: matrix A of the node
        """
        if self.A.shape[0] == 0:
            raise KeyError('The A element of the node is empty!')
        return np.copy(self.A)

    def get_C(self):
        """
        :return: matrix C of the node
        """
        if self.C.shape[0] == 0:
            raise KeyError('The C element of the node is empty!')
        return np.copy(self.C)

    def get_H(self):
        """
        :return: matrix H of the node
        """
        if self.H.shape[0] == 0:
            raise KeyError('The H element of the node is empty!')
        return np.copy(self.H)

    def get_b(self):
        """
        :return: vector b of the node
        """
        if self.b.shape[0] == 0:
            raise KeyError('The b element of the node is empty!')
        return np.copy(self.b)

    def get_B_0(self):
        """
        :return: matrix B_0 of the node
        """
        if self.B_0.shape[0] == 0:
            raise KeyError('The B_0 element of the node is empty!')
        return np.copy(self.B_0)

    def get_B_n(self):
        """
        :return: matrix B_n of the node
        """
        if self.B_n.shape[0] == 0:
            raise KeyError('The B_n element of the node is empty!')
        return np.copy(self.B_n)

    def get_H_n(self):
        """
        :return: matrix H_n of the node
        """
        if self.H_n.shape[0] == 0:
            raise KeyError('The H_n element of the node is empty!')
        return np.copy(self.H_n)

    def get_b_n(self):
        """
        :return: vector b_n of the node
        """
        if self.b_n.shape[0] == 0:
            raise KeyError('The r_bc element of the node is empty!')
        return np.copy(self.b_n)

    def get_delta_s(self):
        """
        :return: vector delta_s of the node
        """
        if self.delta_s.shape[0] == 0:
            raise KeyError('The delta_s element of the node is empty!')
        return np.copy(self.delta_s)

    def set_A(self, A):
        if A.shape[0] != (self.size_y + self.size_z) or A.shape[1] != (self.size_y + self.size_z):
            raise KeyError('The shape of the input matrix A is wrong!')
        self.A = np.copy(A)

    def set_C(self, C):
        if C.shape[0] != (self.size_y + self.size_z) or C.shape[1] != (self.size_y + self.size_z):
            raise KeyError('The shape of the input matrix C is wrong!')
        self.C = np.copy(C)

    def set_H(self, H):
        if H.shape[0] != (self.size_y + self.size_z) or H.shape[1] != self.size_p:
            raise KeyError('The shape of the input matrix H is wrong!')
        self.H = np.copy(H)

    def set_b(self, b):
        if b.shape[0] != (self.size_y + self.size_z):
            raise KeyError('The shape of the input vector b is wrong!')
        self.b = np.copy(b)

    def set_B_0(self, B_0):
        if B_0.shape[0] != (self.size_y + self.size_z + self.size_p) or B_0.shape[1] != (self.size_y + self.size_z):
            raise KeyError('The shape of the input matrix B_0 is wrong!')
        self.B_0 = np.copy(B_0)

    def set_B_n(self, B_n):
        if B_n.shape[0] != (self.size_y + self.size_z + self.size_p) or B_n.shape[1] != (self.size_y + self.size_z):
            raise KeyError('The shape of the input matrix B_n is wrong!')
        self.B_n = np.copy(B_n)

    def set_H_n(self, H_n):
        if H_n.shape[0] != (self.size_y + self.size_z + self.size_p) or H_n.shape[1] != self.size_p:
            raise KeyError('The shape of the input matrix H_n is wrong!')
        self.H_n = np.copy(H_n)

    def set_b_n(self, b_n):
        if b_n.shape[0] != (self.size_y + self.size_z + self.size_p):
            raise KeyError('The shape of the input vector b_n is wrong!')
        self.b_n = np.copy(b_n)

    def set_delta_s(self, delta_s):
        if delta_s.shape[0] != (self.size_y + self.size_z):
            raise KeyError('The shape of the input vector delta_s is wrong!')
        self.delta_s = np.copy(delta_s)

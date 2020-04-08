'''
The module contains functions for fast computation of
various values.
'''

import numpy as np
import quaternion


def build_f_tensor(coords, modes, weights):
    '''
    Computes tensor F.
    '''

    m = len(modes)
    n = len(modes[0])
    w = np.sum(weights)
    f_tensor = np.zeros((3, 3, m, m))
    rmodes = []
    for k in range(m):
        rmodes.append(modes[k].reshape(n // 3, 3))
    rmodes = np.array(rmodes)
    for i in range(3):
        for j in range(3):
            for k1 in range(m):
                for k2 in range(m):
                    f_tensor[i, j, k1, k2] =\
                        1 / w * np.dot(weights, rmodes[k1][:, i] * rmodes[k2][:, j])
    return f_tensor


def build_d_tensor(coords, modes, weights):
    '''
    Computes tensor D.
    '''

    m = len(modes)
    n = len(modes[0])
    w = np.sum(weights)
    d_tensor = np.zeros((3, 3, m))
    rmodes = []
    for k in range(m):
        rmodes.append(modes[k].reshape(n // 3, 3))
    rmodes = np.array(rmodes)
    for i in range(3):
        for j in range(3):
            for k in range(m):
                d_tensor[i, j, k] = 1 / w * np.dot(weights, coords[:, i] * rmodes[k][:, j])
    return d_tensor


def build_i_tensor(coords, c_tensor, weights):
    '''
    Computes tensor I.
    '''

    n = len(coords)
    coords = coords - np.tile(c_tensor, n).reshape((n, 3))
    i_tensor = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            i_tensor[i, j] = -np.dot(weights, coords[:, i] * coords[:, j])
    tr = np.trace(i_tensor)
    i_tensor[0, 0] -= tr
    i_tensor[1, 1] -= tr
    i_tensor[2, 2] -= tr
    return i_tensor


def build_c_tensor(coords, weights):
    '''
    Computes center of mass COM w.r.t weights.
    '''

    return np.apply_along_axis(lambda x: np.average(x, weights=weights), 0, coords)


def calc_torque(coords, force, c_tensor, weights):
    '''
    Computes torque.
    '''

    n = len(coords)
    torque = np.sum(np.array(
        [weights[i] * np.cross(coords[i] - c_tensor, force[i]) for i in range(n)]), 0)
    return torque


def calc_inertia_tensor(rotation_matrix, mode_pos, init_tensor, d_tensor, f_tensor, weights):
    '''
    Computes inertia tensor.
    '''

    inertia_tensor = np.zeros((3, 3))
    m = len(mode_pos)
    w = np.sum(weights)
    if m != 0:
        for i in range(3):
            for j in range(3):
                inertia_tensor[i, j] -= w * ((d_tensor[i, j] + d_tensor[j, i]).dot(mode_pos) + \
                        mode_pos.dot(f_tensor[i, j].dot(mode_pos)))
        tr = inertia_tensor.trace()
        inertia_tensor[0, 0] -= tr
        inertia_tensor[1, 1] -= tr
        inertia_tensor[2, 2] -= tr
    inertia_tensor += init_tensor
    inertia_tensor = rotation_matrix.dot(inertia_tensor).dot(rotation_matrix.T)
    return inertia_tensor


def rmsd(a1, a2, w):
    """
    Returns weighted rmsd

    @param a1: first atom group coordinates. Array of size n x 3, where n is number of atoms.
    @type: numpy.ndarray
    @param a2: first atom group coordinates. Array of size n x 3, where n is number of atoms.
    @type: numpy.ndarray
    @param w: weights. Array of size n.
    @type: numpy.ndarray
    @return: weighted rmsd
    @rtype: float
    """

    return np.sum(w * ((a1 - a2) ** 2).T / np.sum(w)) ** 0.5

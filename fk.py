import numpy as np

# --- Lie algebra helpers ---
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def so3_exp(omega):
    theta = np.linalg.norm(omega)
    if theta < 1e-8:
        return np.eye(3)
    k = omega / theta
    K = skew(k)
    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)

def se3_exp(xi):
    omega, v = xi[:3], xi[3:]
    theta = np.linalg.norm(omega)
    if theta < 1e-8:
        R = np.eye(3)
        p = v
    else:
        R = so3_exp(omega)
        K = skew(omega / theta)
        V = (np.eye(3) + (1 - np.cos(theta)) * K + (theta - np.sin(theta)) * (K @ K)) / theta
        p = V @ v
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = p
    return T

# --- Forward Kinematics using PoE ---
def fk_space(S, M, q):
    """
    Computes forward kinematics using space frame screw axes.
    S: 6×n matrix of screw axes
    M: 4×4 home configuration of end-effector
    q: n×1 joint angles
    """
    T = np.eye(4)
    for i in range(len(q)):
        T = T @ se3_exp(S[:,i] * q[i])
    return T @ M

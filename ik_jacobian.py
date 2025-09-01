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

def so3_log(R):
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    w_hat = (R - R.T) / (2 * np.sin(theta))
    return theta * np.array([w_hat[2,1], w_hat[0,2], w_hat[1,0]])

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

def adjoint(T):
    R, p = T[:3,:3], T[:3,3]
    Ad = np.zeros((6,6))
    Ad[:3,:3] = R
    Ad[3:,3:] = R
    Ad[3:,:3] = skew(p) @ R
    return Ad

# --- Kinematics: POE ---
def fk_space(S, M, q):
    T = np.eye(4)
    for i in range(len(q)):
        T = T @ se3_exp(S[:,i] * q[i])
    return T @ M

def jacobian_space(S, q):
    n = len(q)
    J = np.zeros((6, n))
    T = np.eye(4)
    for i in range(n):
        if i == 0:
            J[:,0] = S[:,0]
        else:
            T = T @ se3_exp(S[:,i-1] * q[i-1])
            J[:,i] = adjoint(T) @ S[:,i]
    return J

# --- Pose error: [position; orientation] ---
def pose_error(T_cur, T_des):
    R, p = T_cur[:3,:3], T_cur[:3,3]
    Rd, pd = T_des[:3,:3], T_des[:3,3]
    ep = pd - p
    eo = so3_log(Rd @ R.T)
    return np.hstack([ep, eo])

# --- DLS IK ---
def ik_dls_space(S, M, q0, T_des, lam=1e-3, alpha=1.0, tol=1e-5, max_iters=200):
    q = q0.copy().astype(float)
    for _ in range(max_iters):
        T = fk_space(S, M, q)
        e = pose_error(T, T_des)
        if np.linalg.norm(e) < tol:
            break
        J = jacobian_space(S, q)
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + (lam**2)*np.eye(6), e)
        q += alpha * dq
    return q

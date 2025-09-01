import numpy as np
from numpy.linalg import norm, inv
from urdf_to_poe import poe_from_urdf

# --- Math helpers ---
def skew(w): return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
def adjoint(T):
    R, p = T[:3, :3], T[:3, 3]
    return np.block([[R, np.zeros((3, 3))], [skew(p) @ R, R]])

def rpy_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy,  cy, 0], [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp], [  0, 1,  0], [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0], [0, cr, -sr], [0, sr,  cr]])
    return Rz @ Ry @ Rx

def create_target_pose(position_mm, euler_deg):
    position_m = np.array(position_mm) / 1000.0
    r, p, y = np.deg2rad(euler_deg)
    T_des = np.eye(4)
    T_des[:3, :3] = rpy_to_R(r, p, y)
    T_des[:3, 3] = position_m
    return T_des

def matrix_log6(T):
    """Numerically robust matrix logarithm."""
    R, p = T[:3, :3], T[:3, 3]
    if np.abs(np.trace(R) - 3) < 1e-9: # Pure translation
        return np.hstack([np.zeros(3), p])
    
    theta = np.arccos(np.clip(0.5 * (np.trace(R) - 1), -1, 1))
    
    if np.abs(theta - np.pi) < 1e-6: # 180-degree rotation
        w_hat_part = R + np.eye(3)
        # Find the column with the largest norm to find the axis
        norms = np.linalg.norm(w_hat_part, axis=0)
        col_idx = np.argmax(norms)
        w = w_hat_part[:, col_idx] / np.sqrt(2 * (1 + R[col_idx, col_idx]))
        w_theta = w * theta
        v = np.linalg.solve(np.eye(3) * theta + (1-np.cos(theta))*skew(w) + (theta-np.sin(theta))*skew(w)@skew(w), p*theta)
        return np.hstack([w_theta, v])

    w_hat = (theta / (2 * np.sin(theta))) * (R - R.T)
    w_theta = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]])
    
    G_inv = (np.eye(3) / theta) - 0.5 * w_hat + \
            (1/theta - 0.5 / np.tan(theta/2)) * (w_hat @ w_hat)
    
    v_theta = G_inv @ p
    return np.hstack([w_theta, v_theta])

def matrix_exp6(xi_theta):
    w_theta, v_theta = xi_theta[:3], xi_theta[3:]
    theta = norm(w_theta)
    T = np.eye(4)
    if theta < 1e-12:
        T[:3, 3] = v_theta
        return T
    w = w_theta / theta
    v = v_theta / theta
    w_hat = skew(w)
    w_hat2 = w_hat @ w_hat
    R = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat2
    G = np.eye(3) * theta + (1 - np.cos(theta)) * w_hat + (theta - np.sin(theta)) * w_hat2
    p = G @ v
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def fk_space(S, M, q):
    T = np.eye(4)
    for i in range(len(q)): T = T @ matrix_exp6(S[:, i] * q[i])
    return T @ M

def jacobian_space(S, q):
    J = np.zeros((6, len(q)))
    T = np.eye(4)
    for i in range(len(q)):
        J[:, i] = adjoint(T) @ S[:, i]
        T = T @ matrix_exp6(S[:, i] * q[i]) # This transform is for the *next* column
    return J

def jacobian_body(S, M, q):
    return adjoint(inv(fk_space(S, M, q))) @ jacobian_space(S, q)

def ik_dls(S, M, T_des, q0, joint_limits, pos_tol=1e-6, rot_tol=1e-6, max_iters=300, damping=1e-2, step_scale=0.5, dq_max=0.2):
    q = q0.copy()
    limits_lower, limits_upper = joint_limits[0], joint_limits[1]
    for _ in range(max_iters):
        T_cur = fk_space(S, M, q)
        Vb = matrix_log6(inv(T_cur) @ T_des)
        if norm(Vb[:3]) < rot_tol and norm(Vb[3:]) < pos_tol: return q, True
        Jb = jacobian_body(S, M, q)
        dq = Jb.T @ np.linalg.solve(Jb @ Jb.T + (damping ** 2) * np.eye(6), Vb)
        max_abs = np.max(np.abs(dq))
        if max_abs > dq_max: dq *= dq_max / max_abs
        q += step_scale * dq
        np.clip(q, limits_lower, limits_upper, out=q)
    return q, False

def ik_solve_robust(S, M, T_des, joint_limits, num_attempts=10):
    limits_lower, limits_upper = joint_limits[0].copy(), joint_limits[1].copy()
    n = S.shape[1]
    inf_mask = ~np.isfinite(limits_lower) | ~np.isfinite(limits_upper)
    limits_lower[inf_mask] = -np.pi
    limits_upper[inf_mask] = np.pi
    print(f"Attempting to solve IK with {num_attempts} randomized starts...")
    q0 = np.clip(np.zeros(n), limits_lower, limits_upper)
    q_sol, ok = ik_dls(S, M, T_des, q0, joint_limits)
    if ok:
        print("Converged on first attempt (from zero config).")
        return q_sol, True
    for i in range(num_attempts - 1):
        q0 = np.random.uniform(limits_lower, limits_upper)
        q_sol, ok = ik_dls(S, M, T_des, q0, joint_limits)
        if ok:
            print(f"Converged on attempt #{i + 2}.")
            return q_sol, True
    print("Failed to converge after all attempts.")
    return q_sol, False

# --- Main execution ---
if __name__ == "__main__":
    URDF_PATH = "rb3_730es_u.urdf"
    EE_LINK = "tcp"
    BASE_LINK = "link0"

    S, M, joint_limits, base, ee = poe_from_urdf(URDF_PATH, ee_link=EE_LINK, base_link=BASE_LINK)
    target_pos_mm = [131.9, 32.9, 854.5]
    target_rot_deg = [160.0, -15.0, 105.0]
    T_des = create_target_pose(target_pos_mm, target_rot_deg)
    print("Target FK Pose (T_des) from user input:\n", np.round(T_des, 4))
    q_sol_rad, converged = ik_solve_robust(S, M, T_des, joint_limits, num_attempts=15)
    print("\nConverged:", converged)
    if converged:
        q_sol_deg = np.rad2deg(q_sol_rad)
        print("Recovered q (degrees):\n", np.round(q_sol_deg, 4))
        T_check = fk_space(S, M, q_sol_rad)
        pos_err_mm = norm(T_check[:3, 3] - T_des[:3, 3]) * 1000.0
        R_err = inv(T_check[:3, :3]) @ T_des[:3, :3]
        angle_rad = np.arccos(np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0))
        angle_deg = np.rad2deg(angle_rad)
        print(f"\nPosition error: {pos_err_mm:.6e} mm")
        print(f"Rotation error: {angle_deg:.6e} degrees")
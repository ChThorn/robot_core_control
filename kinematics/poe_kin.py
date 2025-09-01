# kinematics/poe_kin.py
import numpy as np
from numpy.linalg import norm

# ------------------------- Math helpers -------------------------

def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2],  v[1]],
                     [v[2],  0.0, -v[0]],
                     [-v[1], v[0],  0.0]], dtype=float)

def adjoint(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6), dtype=float)
    Ad[:3, :3] = R
    Ad[3:, :3] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad

def matrix_exp6(xi_theta: np.ndarray) -> np.ndarray:
    """
    Exp map for SE(3) given xi_theta = [omega*theta, v*theta].
    Returns 4x4 T.
    """
    w_th, v_th = xi_theta[:3], xi_theta[3:]
    th = norm(w_th)
    T = np.eye(4, dtype=float)

    if th < 1e-12:
        # Pure translation
        T[:3, 3] = v_th
        return T

    w = w_th / th
    v = v_th / th
    w_hat = skew(w)
    w_hat2 = w_hat @ w_hat

    R = np.eye(3) + np.sin(th) * w_hat + (1.0 - np.cos(th)) * w_hat2
    G = np.eye(3) * th + (1.0 - np.cos(th)) * w_hat + (th - np.sin(th)) * w_hat2
    p = G @ v

    T[:3, :3] = R
    T[:3, 3] = p
    return T

def matrix_log6(T: np.ndarray) -> np.ndarray:
    """
    Log map for SE(3) returning xi_theta = [omega*theta, v*theta].
    MR-consistent with unit axis and V^{-1}.
    """
    R, p = T[:3, :3], T[:3, 3]
    cos_th = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    th = float(np.arccos(cos_th))

    if th < 1e-12:
        # Pure translation: omega*theta = 0, v*theta = p
        return np.hstack((np.zeros(3), p))

    s = np.sin(th)
    # unit-axis hat
    w_hat = (R - R.T) * (0.5 / s)
    w = np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]], dtype=float)

    I = np.eye(3)
    w_hat2 = w_hat @ w_hat
    V_inv = (I / th
             - 0.5 * w_hat
             + (1.0 / th - 0.5 / np.tan(th / 2.0)) * w_hat2)
    v = V_inv @ p
    return np.hstack((w * th, v * th))

# ------------------------- PoE kinematics -------------------------

def fk_space(S: np.ndarray, M: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Forward kinematics in space frame.
    S: 6xn screw axes (space frame), M: 4x4 home pose, q: n
    """
    T = np.eye(4)
    for i in range(q.size):
        T = T @ matrix_exp6(S[:, i] * q[i])
    return T @ M

def jacobian_space(S: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Space Jacobian J_s(q)
    """
    n = q.size
    J = np.zeros((6, n), dtype=float)
    T = np.eye(4)
    for i in range(n):
        J[:, i] = S[:, i] if i == 0 else adjoint(T) @ S[:, i]
        T = T @ matrix_exp6(S[:, i] * q[i])
    return J

def jacobian_body(S: np.ndarray, M: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Body Jacobian J_b(q) = Ad_{T^{-1}} J_s(q)
    """
    T = fk_space(S, M, q)
    return adjoint(np.linalg.inv(T)) @ jacobian_space(S, q)

# ------------------------- IK solvers -------------------------

class IKSolutionError(Exception):
    pass

def ik_dls(
    S: np.ndarray,
    M: np.ndarray,
    T_des: np.ndarray,
    q0: np.ndarray,
    joint_limits: np.ndarray,
    pos_tol: float = 1e-6,
    rot_tol: float = 1e-6,
    max_iters: int = 300,
    damping: float = 1e-2,
    step_scale: float = 0.5,
    dq_max: float = 0.2,
    limit_avoid_weight: float = 0.0,
    verbose: bool = False,
) -> tuple[np.ndarray, bool]:
    """
    Damped least squares IK in the body frame.
    joint_limits: 2xn array [lower; upper]
    limit_avoid_weight: optional soft joint-limit avoidance using gradient to mid-range.
    """
    q = q0.astype(float).copy()
    lower, upper = joint_limits[0], joint_limits[1]
    widths = upper - lower
    mids = (upper + lower) / 2.0

    for it in range(max_iters):
        T_cur = fk_space(S, M, q)
        T_err = np.linalg.inv(T_cur) @ T_des
        Vb = matrix_log6(T_err)

        rot_err = norm(Vb[:3])
        pos_err = norm(Vb[3:])
        if rot_err < rot_tol and pos_err < pos_tol:
            if verbose:
                print(f"[IK] Converged in {it} iterations. pos={pos_err:.2e}, rot={rot_err:.2e}")
            return q, True

        Jb = jacobian_body(S, M, q)
        JJt = Jb @ Jb.T
        # Right-damped pseudoinverse: dq = J^T (J J^T + λ^2 I)^{-1} V
        dq = Jb.T @ np.linalg.solve(JJt + (damping ** 2) * np.eye(6), Vb)

        # Soft limit avoidance: push toward mid, scaled by proximity
        if limit_avoid_weight > 0.0:
            # Proximity term: larger when near limits
            dist_to_lower = np.maximum(q - lower, 1e-6)
            dist_to_upper = np.maximum(upper - q, 1e-6)
            repel = (1.0 / dist_to_lower - 1.0 / dist_to_upper)
            dq += limit_avoid_weight * (-(q - mids) / np.maximum(widths, 1e-6) + 0.1 * repel)

        # Limit the step magnitude
        max_abs = np.max(np.abs(dq))
        if max_abs > dq_max:
            dq *= dq_max / max_abs

        q += step_scale * dq
        # Hard clamp
        np.clip(q, lower, upper, out=q)

    if verbose:
        print("[IK] Reached max iterations without convergence.")
    return q, False

def ik_solve_robust(
    S: np.ndarray,
    M: np.ndarray,
    T_des: np.ndarray,
    joint_limits: np.ndarray,
    num_attempts: int = 10,
    seed: int | None = None,
    **ik_kwargs,
) -> tuple[np.ndarray, bool]:
    """
    Multi-start IK with optional RNG seed.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    lower = joint_limits[0].copy()
    upper = joint_limits[1].copy()
    n = S.shape[1]

    # sample ranges for continuous/unbounded joints
    mask = ~np.isfinite(lower) | ~np.isfinite(upper)
    lower[mask] = -np.pi
    upper[mask] = np.pi

    # Attempt from zero (clipped to limits)
    q0 = np.clip(np.zeros(n), lower, upper)
    q_sol, ok = ik_dls(S, M, T_des, q0, joint_limits, **ik_kwargs)
    if ok:
        return q_sol, True

    # Random restarts
    for _ in range(max(num_attempts - 1, 0)):
        q0 = rng.uniform(lower, upper)
        q_sol, ok = ik_dls(S, M, T_des, q0, joint_limits, **ik_kwargs)
        if ok:
            return q_sol, True

    return q_sol, False

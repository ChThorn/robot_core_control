import numpy as np
from numpy.linalg import pinv  # Pseudo-inverse

# Adjusted DH parameters for Rainbow Robotics RB3-730ES to match path data at zero joints
d1 = 0.127
a2 = 0.400
a3 = 0.350
d4 = 0.0
d5 = 0.0
d6 = 0.0
alpha = [-np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]
a = [0, a2, a3, 0, 0, 0]
d = [d1, 0, 0, d4, d5, d6]
theta_offset = [0, -np.pi/2, 0, 0, 0, 0]

def dh_transform(theta, d, a, alpha):
    """Compute the DH transformation matrix."""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    return np.array([
        [ct, -st*ca, st*sa, a*ct],
        [st, ct*ca, -ct*sa, a*st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])

def forward_kinematics(q):
    """Compute forward kinematics: joint angles to end-effector pose.
    q: list or array of 6 joint angles in radians
    Returns: position (x,y,z) in meters, orientation as rotation matrix
    """
    T = np.eye(4)
    for i in range(6):
        theta = q[i] + theta_offset[i]
        Ti = dh_transform(theta, d[i], a[i], alpha[i])
        T = T @ Ti
    position = T[:3, 3]
    rotation = T[:3, :3]
    return position, rotation

def axis_angle_from_rot(R):
    """Compute axis-angle vector from rotation matrix."""
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    if np.isclose(angle, 0):
        return np.zeros(3)
    sin_a = np.sin(angle)
    axis = (1 / (2 * sin_a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis * angle

def jacobian(q):
    """Compute the Jacobian matrix at joint angles q."""
    J = np.zeros((6, 6))
    T = np.eye(4)
    zs = []  # z axes
    ps = []  # positions
    for i in range(6):
        theta = q[i] + theta_offset[i]
        Ti = dh_transform(theta, d[i], a[i], alpha[i])
        T = T @ Ti
        zs.append(T[:3, 2].copy())
        ps.append(T[:3, 3].copy())
    p_ee = ps[5]  # last is ee
    for i in range(6):
        z = zs[i]
        p = ps[i]
        J[:3, i] = np.cross(z, p_ee - p)
        J[3:, i] = z
    return J

def inverse_kinematics(target_pos, target_rot, q0, max_iter=100, tol=1e-6):
    """Iterative inverse kinematics using Jacobian pseudo-inverse.
    target_pos: desired position (3,)
    target_rot: desired rotation matrix (3x3)
    q0: initial joint angles (6,)
    Returns: joint angles or None if not converged
    """
    q = np.array(q0, dtype=np.float64)
    for _ in range(max_iter):
        curr_pos, curr_rot = forward_kinematics(q)
        delta_pos = target_pos - curr_pos
        R_err = target_rot @ curr_rot.T
        delta_orient = axis_angle_from_rot(R_err)
        e = np.hstack((delta_pos, delta_orient))
        if np.linalg.norm(e) < tol:
            return q
        J = jacobian(q)
        dq = pinv(J) @ e
        q += dq
        q = np.mod(q + np.pi, 2 * np.pi) - np.pi  # Normalize to [-pi, pi]
    print("IK did not converge")
    return None

# Example usage
if __name__ == "__main__":
    # Example joint angles from the first waypoint (approximately zero)
    q_example = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pos, rot = forward_kinematics(q_example)
    print("FK at zero joints:")
    print("Position (m):", pos)
    print("Rotation matrix:\n", rot)

    # Example IK
    target_pos = np.array([0.000, 0.0, 0.877])  # Adjusted to match approximate max height
    target_rot = np.eye(3)  # Identity rotation
    q0 = [0, 0, 0, 0, 0, 0]
    q_ik = inverse_kinematics(target_pos, target_rot, q0)
    if q_ik is not None:
        print("IK solution:", q_ik)
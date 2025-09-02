# tests/test_ik.py
import numpy as np
from robot_kinematics import RobotKinematics

def test_ik_recovers_fk_on_random_samples():
    robot = RobotKinematics("rb3_730es_u.urdf", ee_link="tcp", base_link="link0")
    lower, upper = robot.joint_limits
    # Replace inf with [-pi, pi] for sampling
    lower = np.where(np.isfinite(lower), lower, -np.pi)
    upper = np.where(np.isfinite(upper), upper,  np.pi)

    rng = np.random.default_rng(1234)
    for i in range(20):
        q = rng.uniform(lower, upper)
        T = robot.forward_kinematics(q)
        q_sol, ok = robot.inverse_kinematics(
            T,
            pos_tol=1e-6,
            rot_tol=1e-3,
            max_iters=400,
            num_attempts=10,
            seed=42 + i
        )
        assert ok, f"IK failed to converge on sample {i}"
        pos_err, rot_err = robot.check_pose_error(T, q_sol)
        assert pos_err < 1e-6, f"Position error too high: {pos_err}"
        assert rot_err < 1e-3, f"Rotation error too high: {rot_err}"

def test_se3_validation_and_normalization():
    robot = RobotKinematics("rb3_730es_u.urdf", ee_link="tcp", base_link="link0")
    T = robot.forward_kinematics(np.zeros(robot.n_joints))
    # Corrupt R slightly
    T_bad = T.copy()
    T_bad[0, 0] += 1e-4
    assert not robot.is_valid_SE3(T_bad)
    T_norm = robot.normalize_SE3(T_bad)
    assert robot.is_valid_SE3(T_norm)
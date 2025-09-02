# error_handling.py
from enum import Enum
import numpy as np

class IKErrorCode(Enum):
    SUCCESS = 0
    MAX_ITERATIONS_EXCEEDED = 1
    JOINT_LIMITS_VIOLATED = 2
    SINGULARITY_DETECTED = 3

def check_singularity(J, threshold=1e-3):
    """Check if the robot is near a singularity."""
    svals = np.linalg.svd(J, compute_uv=False)
    min_singular_value = np.min(svals)
    return min_singular_value < threshold, min_singular_value

def validate_joint_limits(q, limits, tol=1e-9):
    """Check if joint angles are within limits (with a tiny tolerance)."""
    lower, upper = limits
    violations = np.where((q < (lower - tol)) | (q > (upper + tol)))[0]
    return len(violations) == 0, violations

def robust_ik_solver(robot, T_des, ik_params=None, max_retries=3, singularity_threshold=1e-3):
    """
    Robust IK solver with error handling and recovery.
    Returns (q_sol, code, info) where info contains diagnostics.
    """
    ik_params = ik_params or {}
    last_info = {
        'attempts': 0,
        'iterations': 0,
        'pos_err': None,
        'rot_err': None,
        'min_sigma': None,
    }

    for attempt in range(max_retries):
        q_sol, converged = robot.inverse_kinematics(T_des, **ik_params)
        last_info['attempts'] += 1
        last_info['iterations'] += robot.last_solve_info.get('iterations', 0)

        if not converged:
            last_info['pos_err'] = robot.last_solve_info.get('final_pos_err')
            last_info['rot_err'] = robot.last_solve_info.get('final_rot_err')
            continue

        # Check for singularities
        J = robot.jacobian_body(q_sol)
        near_singularity, min_sigma = check_singularity(J, threshold=singularity_threshold)
        last_info['min_sigma'] = float(min_sigma)
        if near_singularity:
            # Try again with a different seed if available
            if 'seed' in ik_params and ik_params['seed'] is not None:
                ik_params = dict(ik_params)
                ik_params['seed'] += 1
            continue

        # Check joint limits
        within_limits, violations = validate_joint_limits(q_sol, robot.joint_limits)
        if not within_limits:
            last_info['violations'] = violations.tolist()
            # Try again with a different seed if possible
            if 'seed' in ik_params and ik_params['seed'] is not None:
                ik_params = dict(ik_params)
                ik_params['seed'] += 1
            continue

        # Compute final errors consistently
        pos_err, rot_err = robot.check_pose_error(T_des, q_sol)
        last_info['pos_err'] = float(pos_err)
        last_info['rot_err'] = float(rot_err)
        return q_sol, IKErrorCode.SUCCESS, last_info
    
    return None, IKErrorCode.MAX_ITERATIONS_EXCEEDED, last_info
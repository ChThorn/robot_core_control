# error_handling.py
from enum import Enum

class IKErrorCode(Enum):
    SUCCESS = 0
    MAX_ITERATIONS_EXCEEDED = 1
    JOINT_LIMITS_VIOLATED = 2
    SINGULARITY_DETECTED = 3

def check_singularity(J, threshold=1e-3):
    """Check if the robot is near a singularity."""
    min_singular_value = np.linalg.svd(J, compute_uv=False).min()
    return min_singular_value < threshold

def validate_joint_limits(q, limits):
    """Check if joint angles are within limits."""
    violations = np.where((q < limits[0]) | (q > limits[1]))[0]
    return len(violations) == 0, violations

def robust_ik_solver(robot, T_des, max_retries=3):
    """Robust IK solver with error handling and recovery."""
    for attempt in range(max_retries):
        q_sol, converged = robot.inverse_kinematics(T_des)
        
        if not converged:
            continue
            
        # Check for singularities
        J = robot.jacobian_body(q_sol)
        if check_singularity(J):
            print(f"Warning: Near singularity detected in attempt {attempt+1}")
            continue
            
        # Check joint limits
        within_limits, violations = validate_joint_limits(q_sol, robot.joint_limits)
        if not within_limits:
            print(f"Joint limit violations in attempt {attempt+1}: {violations}")
            continue
            
        return q_sol, IKErrorCode.SUCCESS
    
    return None, IKErrorCode.MAX_ITERATIONS_EXCEEDED

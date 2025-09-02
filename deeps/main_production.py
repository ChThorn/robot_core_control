# main_production.py
import numpy as np
import logging
import time
from robot_kinematics import RobotKinematics
from config import RobotConfig
from error_handling import robust_ik_solver, IKErrorCode
from monitoring import KinematicsMonitor, IKResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('robot_kinematics')

def main():
    # Load configuration
    config = RobotConfig('robot_config.yaml')
    ik_params = config.get_ik_params()
    
    # Initialize robot kinematics
    robot = RobotKinematics(
        config.urdf_path, 
        ee_link=config.ee_link, 
        base_link=config.base_link
    )
    
    # Example target pose (ensure it is valid SE(3))
    target_pose = np.array([
        [-0.2622, -0.9540,  0.1453, 0.1319],
        [ 0.9264, -0.2911, -0.2389, 0.0329],
        [ 0.2702,  0.0720,  0.9601, 0.8545],
        [ 0.0,     0.0,     0.0,    1.0   ]
    ], dtype=float)

    monitor = KinematicsMonitor()

    # Solve IK with robustness, monitoring, and timing
    start_time = time.time()
    q_sol, code, info = robust_ik_solver(robot, target_pose, ik_params=ik_params, max_retries=3)
    computation_time = time.time() - start_time

    if code == IKErrorCode.SUCCESS and q_sol is not None:
        pos_err, rot_err = robot.check_pose_error(target_pose, q_sol)
        iterations = int(info.get('iterations', robot.last_solve_info.get('iterations', 0)))
        result = IKResult(
            q=q_sol,
            converged=True,
            position_error=float(pos_err),
            rotation_error=float(rot_err),
            computation_time=float(computation_time),
            iterations=iterations
        )
        monitor.log_ik_attempt(target_pose, None, result, success=True)
        
        # Safety check before sending to controller
        within_limits, violations = validate_before_send(q_sol, robot.joint_limits)
        if not within_limits:
            logger.error(f"Refusing to send to controller. Joint limit violations at indices: {violations}")
            return

        send_to_controller(q_sol)
    else:
        iterations = int(info.get('iterations', robot.last_solve_info.get('iterations', 0)))
        pos_err = float(info.get('pos_err') or 0.0)
        rot_err = float(info.get('rot_err') or 0.0)
        result = IKResult(
            q=np.array([]),
            converged=False,
            position_error=pos_err,
            rotation_error=rot_err,
            computation_time=float(computation_time),
            iterations=iterations
        )
        monitor.log_ik_attempt(target_pose, None, result, success=False)
        handle_ik_failure(info)

def validate_before_send(q, limits):
    """Validate joint limits before commanding the controller."""
    lower, upper = limits
    violations = np.where((q < lower) | (q > upper))[0]
    return len(violations) == 0, violations

def send_to_controller(q):
    """Send joint angles to robot controller (placeholder)."""
    # Replace this with your actual controller communication (e.g., ROS2 action/service)
    print(f"Sending to controller: {np.round(q, 6)}")

def handle_ik_failure(info):
    """Handle IK failure appropriately."""
    # Could try alternative approaches, notify operator, etc.
    print(f"IK failed. Attempts: {info.get('attempts')}, "
          f"Iters: {info.get('iterations')}, "
          f"Best pos_err: {info.get('pos_err')}, rot_err: {info.get('rot_err')}, "
          f"min_sigma: {info.get('min_sigma')}")

if __name__ == "__main__":
    main()
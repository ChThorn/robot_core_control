# main_production.py
import numpy as np
import logging
import time
from robot_kinematics import RobotKinematics
from config import RobotConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('robot_kinematics')

def main():
    # Load configuration
    config = RobotConfig('robot_config.yaml')
    
    # Initialize robot kinematics
    robot = RobotKinematics(
        config.urdf_path, 
        ee_link=config.ee_link, 
        base_link=config.base_link
    )
    
    # Example target pose (same as before)
    target_pose = np.array([
        [-0.2622, -0.954, 0.1453, 0.1319],
        [0.9264, -0.2911, -0.2389, 0.0329],
        [0.2702, 0.072, 0.9601, 0.8545],
        [0, 0, 0, 1]
    ])
    
    # Solve IK with monitoring
    start_time = time.time()
    q_sol, converged = robot.inverse_kinematics(target_pose, **config.get_ik_params())
    computation_time = time.time() - start_time
    
    if converged:
        # Calculate errors
        pos_err, rot_err = robot.check_pose_error(target_pose, q_sol)
        
        logger.info(f"IK solved successfully in {computation_time:.4f}s")
        logger.info(f"Position error: {pos_err:.6e}, Rotation error: {rot_err:.6e}")
        
        # Send to robot controller
        send_to_controller(q_sol)
    else:
        # Handle failure
        logger.warning("IK failed to converge")
        handle_ik_failure(q_sol)

def send_to_controller(q):
    """Send joint angles to robot controller."""
    # Implementation depends on your specific robot controller
    print(f"Sending to controller: {q}")

def handle_ik_failure(q):
    """Handle IK failure appropriately."""
    # Could try alternative approaches, notify operator, etc.
    print(f"IK failed. Best solution: {q}")

if __name__ == "__main__":
    main()
# main.py
import numpy as np
from robot_kinematics import RobotKinematics

def main():
    # Initialize robot kinematics
    urdf_path = "rb3_730es_u.urdf"
    robot = RobotKinematics(urdf_path, ee_link="tcp", base_link="link0")
    
    # Test configuration
    q_known = np.array([0.5, 0.1, 0.2, 0.8, -0.4, 0.6])
    T_des = robot.forward_kinematics(q_known)
    
    print("Target FK Pose (T_des):")
    print(np.round(T_des, 4))
    
    # Solve IK
    q_sol, converged = robot.inverse_kinematics(
        T_des, 
        num_attempts=15,
        pos_tol=1e-6,
        rot_tol=1e-6
    )
    
    print(f"\nRecovered q from IK: {np.round(q_sol, 6)}")
    print(f"Converged: {converged}")
    
    if converged:
        pos_err, rot_err = robot.check_pose_error(T_des, q_sol)
        print(f"Position error: {pos_err:.6e} meters")
        print(f"Rotation error: {rot_err:.6e} radians")

if __name__ == "__main__":
    main()

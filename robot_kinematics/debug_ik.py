#!/usr/bin/env python3
import numpy as np
import json
from robot_controller import RobotController

def main():
    controller = RobotController("rb3_730es_u.urdf")
    
    with open("third_20250710_162459.json", "r") as f:
        data = json.load(f)
    
    wp = data["waypoints"][4]
    q_deg = np.array(wp['joint_positions'])
    tcp_recorded = np.array(wp['tcp_position'])
    
    print(f"Joint angles (deg): {q_deg}")
    print(f"TCP position (mm + deg): {tcp_recorded}")
    
    q_rad, T_recorded = controller.convert_from_robot_units(q_deg, tcp_recorded)
    T_fk = controller.forward_kinematics(q_rad)
    
    pos_err = np.linalg.norm(T_fk[:3, 3] - T_recorded[:3, 3])
    print(f"FK Position error: {pos_err*1000:.3f} mm")
    
    # Test IK with FK result
    print("Testing IK with FK result...")
    q_ik, converged = controller.inverse_kinematics(T_fk, q_init=q_rad)
    print(f"IK converged: {converged}")
    
    # Check limits and condition number
    limits = controller.robot.joint_limits
    print(f"Joint limits (deg): {np.rad2deg(limits)}")
    J = controller.robot.jacobian_body(q_rad)
    cond_num = np.linalg.cond(J)
    print(f"Jacobian condition number: {cond_num:.2e}")

if __name__ == "__main__":
    main()

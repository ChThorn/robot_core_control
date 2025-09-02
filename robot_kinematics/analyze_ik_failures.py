#!/usr/bin/env python3
import numpy as np
import json
from robot_controller import RobotController

def analyze_single_waypoint(controller, waypoint_idx, data):
    """Analyze why IK fails for a specific waypoint."""
    wp = data['waypoints'][waypoint_idx]
    q_deg_real = np.array(wp['joint_positions'])
    tcp_real = np.array(wp['tcp_position'])
    
    print(f"\n=== ANALYZING WAYPOINT {waypoint_idx} ===")
    print(f"Real robot joints (deg): {q_deg_real}")
    print(f"Real robot TCP (mm+deg): {tcp_real}")
    
    # Convert to our units
    q_rad_real, T_real = controller.convert_from_robot_units(q_deg_real, tcp_real)
    
    # What does our FK predict for these joints?
    T_our_fk = controller.forward_kinematics(q_rad_real)
    
    pos_diff = np.linalg.norm(T_our_fk[:3, 3] - T_real[:3, 3])
    R_diff = T_our_fk[:3, :3].T @ T_real[:3, :3]
    rot_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2.0, -1.0, 1.0))
    
    print(f"Model prediction error:")
    print(f"  Position error: {pos_diff*1000:.3f} mm")
    print(f"  Rotation error: {np.rad2deg(rot_diff):.3f}°")
    
    # Try IK on both targets
    print(f"\n1. IK targeting real robot TCP pose:")
    q_ik1, conv1 = controller.inverse_kinematics(T_real, q_init=q_rad_real)
    if conv1:
        print(f"   ✅ SUCCESS")
        T_check1 = controller.forward_kinematics(q_ik1)
        err1 = np.linalg.norm(T_check1[:3, 3] - T_real[:3, 3])
        print(f"   Final error: {err1*1000:.3f} mm")
    else:
        print(f"   ❌ FAILED")
        if q_ik1 is not None:
            T_check1 = controller.forward_kinematics(q_ik1)
            err1 = np.linalg.norm(T_check1[:3, 3] - T_real[:3, 3])
            print(f"   Best error achieved: {err1*1000:.3f} mm")
    
    print(f"2. IK targeting our FK prediction:")
    q_ik2, conv2 = controller.inverse_kinematics(T_our_fk, q_init=q_rad_real)
    if conv2:
        print(f"   ✅ SUCCESS (should always work)")
    else:
        print(f"   ❌ FAILED (this indicates IK algorithm issue)")
    
    # Check if the real pose is even reachable by our model
    print(f"\n3. Workspace reachability analysis:")
    target_pos = T_real[:3, 3]
    target_dist = np.linalg.norm(target_pos)
    
    # Rough workspace analysis - compute reach at home position
    T_home = controller.forward_kinematics(np.zeros(6))
    home_dist = np.linalg.norm(T_home[:3, 3])
    
    print(f"   Target distance from origin: {target_dist:.3f} m")
    print(f"   Home position distance: {home_dist:.3f} m")
    
    # Check if target is too different from what our model can achieve
    if pos_diff > 0.005:  # 5mm
        print(f"   ⚠️  Target pose may be unreachable due to model inaccuracy")
        print(f"      Model error ({pos_diff*1000:.1f}mm) exceeds typical IK tolerance")
        return "MODEL_ERROR"
    elif not conv1 and conv2:
        print(f"   ⚠️  IK algorithm has difficulty with this pose")
        return "ALGORITHM_ISSUE"
    elif not conv1 and not conv2:
        print(f"   ⚠️  Fundamental IK algorithm problem")
        return "SEVERE_ALGORITHM_ISSUE"
    else:
        print(f"   ✅ This pose should be solvable")
        return "OK"

def main():
    controller = RobotController("rb3_730es_u.urdf")
    
    with open("third_20250710_162459.json", "r") as f:
        data = json.load(f)
    
    print("=== IK FAILURE ANALYSIS ===")
    
    # Test several waypoints
    test_indices = [0, 4, 8, 12, 16, 20, 24, 28, 32, 37]  # Same as test_real_data.py
    
    results = {
        "MODEL_ERROR": 0,
        "ALGORITHM_ISSUE": 0, 
        "SEVERE_ALGORITHM_ISSUE": 0,
        "OK": 0
    }
    
    for idx in test_indices:
        if idx < len(data['waypoints']):
            result = analyze_single_waypoint(controller, idx, data)
            results[result] += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Total waypoints analyzed: {len(test_indices)}")
    for category, count in results.items():
        print(f"{category}: {count} ({count/len(test_indices)*100:.1f}%)")
    
    print(f"\n=== CONCLUSIONS ===")
    if results["MODEL_ERROR"] > 0:
        print(f"• {results['MODEL_ERROR']} poses fail due to model inaccuracy (>5mm error)")
        print("  → Solution: Accept larger tolerances or improve URDF model")
    
    if results["ALGORITHM_ISSUE"] > 0:
        print(f"• {results['ALGORITHM_ISSUE']} poses fail due to IK algorithm limitations") 
        print("  → Solution: Improve IK algorithm, more initial guesses, better optimization")
        
    if results["SEVERE_ALGORITHM_ISSUE"] > 0:
        print(f"• {results['SEVERE_ALGORITHM_ISSUE']} poses fail completely")
        print("  → Solution: Debug IK algorithm, check Jacobian, check matrix operations")

if __name__ == "__main__":
    main()

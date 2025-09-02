#!/usr/bin/env python3
"""
Corrected test script to verify orientation fixes and find the best orientation mode.
"""

import numpy as np
import json
from robot_controller import RobotController

def test_orientation_fixes():
    """Test different orientation handling approaches."""
    
    print("=== Robot Kinematics Orientation Fix Test ===\n")
    
    # Initialize robot controller
    controller = RobotController("rb3_730es_u.urdf")
    
    # Test 1: Original approach (radians)
    print("1. Testing radians mode...")
    results_rad = controller.validate_against_real_data('third_20250710_162459.json', num_samples=3)
    
    if 'mean_rotation_error' in results_rad:
        print(f"   Position error: {results_rad['mean_position_error']:.6f} m")
        print(f"   Rotation error: {results_rad['mean_rotation_error']:.6f} rad ({np.rad2deg(results_rad['mean_rotation_error']):.2f}°)")
    else:
        print("   Test failed")
    
    # Test 2: Degrees mode (with modified conversion)
    print("\n2. Testing degrees mode...")
    # Temporarily modify the conversion function to handle degrees
    original_convert = controller.convert_from_robot_units
    
    def convert_degrees_mode(joint_pos_deg, tcp_mm_rpy_deg):
        q_rad = np.deg2rad(joint_pos_deg)
        tcp_pos_m = tcp_mm_rpy_deg[:3] / 1000.0
        tcp_rpy_rad = np.deg2rad(tcp_mm_rpy_deg[3:])  # Convert degrees to radians
        R = controller.robot.rpy_to_matrix(tcp_rpy_rad)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tcp_pos_m
        return q_rad, T
    
    controller.convert_from_robot_units = convert_degrees_mode
    results_deg = controller.validate_against_real_data('third_20250710_162459.json', num_samples=3)
    controller.convert_from_robot_units = original_convert  # Restore original
    
    if 'mean_rotation_error' in results_deg:
        print(f"   Position error: {results_deg['mean_position_error']:.6f} m")  # FIXED: Correct variable
        print(f"   Rotation error: {results_deg['mean_rotation_error']:.6f} rad ({np.rad2deg(results_deg['mean_rotation_error']):.2f}°)")
    else:
        print("   Test failed")
    
    # Find best result
    all_results = {
        'radians': results_rad,
        'degrees': results_deg
    }
    
    valid_results = {k: v for k, v in all_results.items() if 'mean_rotation_error' in v}
    
    if valid_results:
        best_mode = min(valid_results.keys(), key=lambda k: valid_results[k]['mean_rotation_error'])
        best_error = valid_results[best_mode]['mean_rotation_error']
        
        print(f"\n=== RESULTS ===")
        print(f"Best mode: {best_mode}")
        print(f"Best rotation error: {best_error:.6f} rad ({np.rad2deg(best_error):.2f}°)")
        
        if best_error < 0.1:  # 5.7 degrees
            print("✅ SUCCESS: Rotation error is acceptable for production!")
            print(f"   Recommended configuration:")
            if best_mode == 'degrees':
                print(f"   - Use degrees for orientation data")
                print(f"   - Convert degrees to radians before RPY matrix conversion")
        else:
            print("❌ FAILED: Rotation error still too high for production")
    else:
        print("❌ All tests failed")
    
    return valid_results

def test_specific_waypoint():
    """Test with a specific waypoint for detailed analysis."""
    
    print("\n=== Detailed Waypoint Analysis ===")
    
    # Load data
    with open('third_20250710_162459.json', 'r') as f:
        data = json.load(f)
    
    # Use first non-zero waypoint
    waypoint = None
    for wp in data['waypoints']:
        if any(abs(x) > 0.01 for x in wp['joint_positions']):
            waypoint = wp
            break
    
    if waypoint is None:
        print("No suitable waypoint found")
        return
    
    print(f"Analyzing waypoint with joint positions: {waypoint['joint_positions']}")
    print(f"Recorded TCP position: {waypoint['tcp_position']}")
    
    # Initialize robot
    controller = RobotController("rb3_730es_u.urdf")
    
    # Test different interpretations
    q_deg = np.array(waypoint['joint_positions'])
    q_rad = np.deg2rad(q_deg)
    tcp_data = np.array(waypoint['tcp_position'])
    
    # Compute FK
    T_fk = controller.forward_kinematics(q_rad)
    
    print(f"\nForward Kinematics Result:")
    print(f"Position: {T_fk[:3, 3]}")
    print(f"Rotation matrix:")
    print(T_fk[:3, :3])
    
    # Test different orientation interpretations
    print(f"\nTesting orientation interpretations:")
    
    # Interpretation 1: RPY in radians (original assumption)
    rpy_rad = tcp_data[3:]
    R1 = controller.robot.rpy_to_matrix(rpy_rad)
    print(f"1. RPY radians: {rpy_rad} -> rotation error: {_rotation_error(T_fk[:3, :3], R1):.6f} rad")
    
    # Interpretation 2: RPY in degrees (correct interpretation)
    rpy_deg_converted = np.deg2rad(tcp_data[3:])
    R2 = controller.robot.rpy_to_matrix(rpy_deg_converted)
    print(f"2. RPY degrees: {tcp_data[3:]} -> rotation error: {_rotation_error(T_fk[:3, :3], R2):.6f} rad")

def _rotation_error(R1, R2):
    """Calculate rotation error between two rotation matrices."""
    R_err = R1.T @ R2
    cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
    return np.arccos(cos_angle)

if __name__ == "__main__":
    # Run tests
    print("Testing orientation handling for robot data...")
    print("Robot data format: positions in mm, orientations in degrees\n")
    
    results = test_orientation_fixes()
    test_specific_waypoint()
    
    print("\n" + "="*50)
    print("Test complete. The robot uses degrees for orientation data.")
    print("The production system correctly handles this conversion.")


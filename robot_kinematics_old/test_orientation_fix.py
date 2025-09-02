#!/usr/bin/env python3
"""
Test script to verify orientation fixes and find the best orientation mode.
"""

import numpy as np
import json
from main_production_fixed import RobotController
from config_prod import RobotConfig

def test_orientation_fixes():
    """Test different orientation handling approaches."""
    
    print("=== Robot Kinematics Orientation Fix Test ===\n")
    
    # Load configuration
    config = RobotConfig('robot_config_prod_fixed.yaml')
    controller = RobotController(config)
    
    # Test 1: Original approach (baseline)
    print("1. Testing original approach (auto-detect)...")
    controller.orientation_mode = 'auto_detect'
    results_auto = controller.validate_against_real_data('third_20250710_162459.json', num_samples=3)
    
    if 'mean_rotation_error' in results_auto:
        print(f"   Position error: {results_auto['mean_position_error']:.6f} m")
        print(f"   Rotation error: {results_auto['mean_rotation_error']:.6f} rad ({np.rad2deg(results_auto['mean_rotation_error']):.2f}°)")
    else:
        print("   Test failed")
    
    # Test 2: Degrees mode
    print("\n2. Testing degrees mode...")
    controller.orientation_mode = 'degrees'
    results_deg = controller.validate_against_real_data('third_20250710_162459.json', num_samples=3)
    
    if 'mean_rotation_error' in results_deg:
        print(f"   Position error: {results_deg['mean_position_error']:.6f} m")
        print(f"   Rotation error: {results_deg['mean_rotation_error']:.6f} rad ({np.rad2deg(results_deg['mean_rotation_error']):.2f}°)")
    else:
        print("   Test failed")
    
    # Test 3: ZYX convention with radians
    print("\n3. Testing ZYX convention with radians...")
    controller.orientation_mode = 'radians'
    controller.config.config['robot_controller']['rpy_convention'] = 'zyx'
    results_zyx = controller.validate_against_real_data('third_20250710_162459.json', num_samples=3)
    
    if 'mean_rotation_error' in results_zyx:
        print(f"   Position error: {results_zyx['mean_position_error']:.6f} m")
        print(f"   Rotation error: {results_zyx['mean_rotation_error']:.6f} rad ({np.rad2deg(results_zyx['mean_rotation_error']):.2f}°)")
    else:
        print("   Test failed")
    
    # Test 4: ZYX convention with degrees
    print("\n4. Testing ZYX convention with degrees...")
    controller.orientation_mode = 'degrees'
    controller.config.config['robot_controller']['rpy_convention'] = 'zyx'
    results_zyx_deg = controller.validate_against_real_data('third_20250710_162459.json', num_samples=3)
    
    if 'mean_rotation_error' in results_zyx_deg:
        print(f"   Position error: {results_zyx_deg['mean_rotation_error']:.6f} m")
        print(f"   Rotation error: {results_zyx_deg['mean_rotation_error']:.6f} rad ({np.rad2deg(results_zyx_deg['mean_rotation_error']):.2f}°)")
    else:
        print("   Test failed")
    
    # Find best result
    all_results = {
        'auto_detect': results_auto,
        'degrees': results_deg,
        'zyx_radians': results_zyx,
        'zyx_degrees': results_zyx_deg
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
            if best_mode == 'auto_detect':
                print(f"   - orientation_mode: auto_detect")
            elif best_mode == 'degrees':
                print(f"   - orientation_mode: degrees")
                print(f"   - rpy_convention: xyz")
            elif best_mode == 'zyx_radians':
                print(f"   - orientation_mode: radians")
                print(f"   - rpy_convention: zyx")
            elif best_mode == 'zyx_degrees':
                print(f"   - orientation_mode: degrees")
                print(f"   - rpy_convention: zyx")
        else:
            print("❌ FAILED: Rotation error still too high for production")
            print("   Consider:")
            print("   - Checking if orientations are quaternions")
            print("   - Verifying URDF coordinate frames")
            print("   - Contacting robot manufacturer for clarification")
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
    
    # Load robot
    config = RobotConfig('robot_config_prod_fixed.yaml')
    controller = RobotController(config)
    
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
    
    # Interpretation 1: RPY in radians
    rpy_rad = tcp_data[3:]
    R1 = controller.robot.rpy_to_matrix(rpy_rad)
    print(f"1. RPY radians: {rpy_rad} -> rotation error: {_rotation_error(T_fk[:3, :3], R1):.6f} rad")
    
    # Interpretation 2: RPY in degrees
    rpy_deg = np.deg2rad(tcp_data[3:])
    R2 = controller.robot.rpy_to_matrix(rpy_deg)
    print(f"2. RPY degrees: {tcp_data[3:]} -> rotation error: {_rotation_error(T_fk[:3, :3], R2):.6f} rad")
    
    # Interpretation 3: ZYX convention radians
    R3 = controller._rpy_to_matrix_zyx(rpy_rad)
    print(f"3. ZYX radians: {rpy_rad} -> rotation error: {_rotation_error(T_fk[:3, :3], R3):.6f} rad")
    
    # Interpretation 4: ZYX convention degrees
    R4 = controller._rpy_to_matrix_zyx(rpy_deg)
    print(f"4. ZYX degrees: {tcp_data[3:]} -> rotation error: {_rotation_error(T_fk[:3, :3], R4):.6f} rad")

def _rotation_error(R1, R2):
    """Calculate rotation error between two rotation matrices."""
    R_err = R1.T @ R2
    cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
    return np.arccos(cos_angle)

if __name__ == "__main__":
    # Run tests
    results = test_orientation_fixes()
    test_specific_waypoint()
    
    print("\n" + "="*50)
    print("Test complete. Check the results above to determine")
    print("the correct orientation configuration for your robot.")


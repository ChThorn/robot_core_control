#!/usr/bin/env python3
"""
Script to validate kinematics implementation against real robot data.
"""

import json
import numpy as np
from robot_kinematics import RobotKinematics
import matplotlib.pyplot as plt

def load_robot_data(json_path):
    """Load robot data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_waypoints(robot, data, num_samples=10):
    """Analyze waypoints and compare FK results with recorded TCP positions."""
    waypoints = data['waypoints']
    
    # Sample waypoints evenly
    indices = np.linspace(0, len(waypoints)-1, num_samples, dtype=int)
    
    results = []
    
    for i in indices:
        wp = waypoints[i]
        
        # Extract joint positions (convert to radians if needed)
        q = np.array(wp['joint_positions'])
        
        # Extract recorded TCP position and orientation
        tcp_recorded = wp['tcp_position']
        pos_recorded = np.array(tcp_recorded[:3]) / 1000.0  # Convert mm to m
        rpy_recorded = np.array(tcp_recorded[3:])  # Roll, pitch, yaw
        
        # Compute forward kinematics
        T_fk = robot.forward_kinematics(q)
        pos_fk = T_fk[:3, 3]
        
        # Convert rotation matrix to RPY for comparison
        R_fk = T_fk[:3, :3]
        rpy_fk = rotation_matrix_to_rpy(R_fk)
        
        # Calculate errors
        pos_error = np.linalg.norm(pos_fk - pos_recorded)
        rpy_error = np.linalg.norm(rpy_fk - rpy_recorded)
        
        result = {
            'waypoint_idx': i,
            'timestamp': wp['timestamp'],
            'joint_positions': q,
            'pos_recorded': pos_recorded,
            'pos_fk': pos_fk,
            'pos_error': pos_error,
            'rpy_recorded': rpy_recorded,
            'rpy_fk': rpy_fk,
            'rpy_error': rpy_error
        }
        
        results.append(result)
    
    return results

def rotation_matrix_to_rpy(R):
    """Convert rotation matrix to roll-pitch-yaw angles."""
    # Extract roll, pitch, yaw from rotation matrix
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])  # roll
        y = np.arctan2(-R[2,0], sy)     # pitch
        z = np.arctan2(R[1,0], R[0,0])  # yaw
    else:
        x = np.arctan2(-R[1,2], R[1,1])  # roll
        y = np.arctan2(-R[2,0], sy)      # pitch
        z = 0                            # yaw
    
    return np.array([x, y, z])

def test_specific_configuration():
    """Test the specific configuration from main.py."""
    robot = RobotKinematics("rb3_730es_u.urdf", ee_link="tcp", base_link="link0")
    
    # Test the same configuration as main.py
    q_test = np.array([0.5, 0.1, 0.2, 0.8, -0.4, 0.6])
    
    print("=== Validation of main.py test case ===")
    print(f"Joint angles: {q_test}")
    
    # Forward kinematics
    T_fk = robot.forward_kinematics(q_test)
    print(f"\nForward Kinematics Result:")
    print(np.round(T_fk, 4))
    
    # Inverse kinematics
    q_ik, converged = robot.inverse_kinematics(T_fk, num_attempts=15)
    print(f"\nInverse Kinematics:")
    print(f"Recovered joints: {np.round(q_ik, 6)}")
    print(f"Converged: {converged}")
    
    # Error analysis
    if converged:
        pos_err, rot_err = robot.check_pose_error(T_fk, q_ik)
        print(f"Position error: {pos_err:.6e} meters")
        print(f"Rotation error: {rot_err:.6e} radians")
        
        # Joint angle differences
        joint_errors = np.abs(q_test - q_ik)
        print(f"Joint angle errors: {joint_errors}")
        print(f"Max joint error: {np.max(joint_errors):.6e} radians")

def main():
    """Main validation function."""
    # Initialize robot
    robot = RobotKinematics("rb3_730es_u.urdf", ee_link="tcp", base_link="link0")
    
    # Test specific configuration
    test_specific_configuration()
    
    print("\n" + "="*60)
    
    # Load and analyze real robot data
    try:
        data = load_robot_data("third_20250710_162459.json")
        print(f"\n=== Real Robot Data Analysis ===")
        print(f"Recording duration: {data['metadata']['duration']:.2f} seconds")
        print(f"Number of waypoints: {data['metadata']['waypoint_count']}")
        
        # Analyze sample waypoints
        results = analyze_waypoints(robot, data, num_samples=5)
        
        print(f"\n=== Waypoint Analysis (5 samples) ===")
        for i, result in enumerate(results):
            print(f"\nWaypoint {result['waypoint_idx']}:")
            print(f"  Joint positions: {np.round(result['joint_positions'], 3)}")
            print(f"  Position error: {result['pos_error']:.6f} m")
            print(f"  RPY error: {result['rpy_error']:.6f} rad")
            
            if result['pos_error'] > 0.01:  # 1cm threshold
                print(f"  WARNING: Large position error detected!")
        
        # Summary statistics
        pos_errors = [r['pos_error'] for r in results]
        rpy_errors = [r['rpy_error'] for r in results]
        
        print(f"\n=== Error Statistics ===")
        print(f"Position errors - Mean: {np.mean(pos_errors):.6f} m, Max: {np.max(pos_errors):.6f} m")
        print(f"RPY errors - Mean: {np.mean(rpy_errors):.6f} rad, Max: {np.max(rpy_errors):.6f} rad")
        
    except Exception as e:
        print(f"Error analyzing robot data: {e}")

if __name__ == "__main__":
    main()


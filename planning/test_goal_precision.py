#!/usr/bin/env python3
"""
Test to verify if the robot reaches exact goal poses with precision analysis.
"""

import sys
import os
import numpy as np

# Add parent directories to path
current_dir = os.path.dirname(__file__)
kinematics_dir = os.path.join(current_dir, '..', 'robot_kinematics')
sys.path.append(kinematics_dir)

from robot_controller import RobotController
from motion_planning import MotionPlanner

def test_goal_precision():
    """Test if robot actually reaches exact goal poses."""
    print("�� Testing Goal Precision: Does Robot Reach Exact Poses?")
    print("=" * 60)
    
    # Initialize components
    robot_controller = RobotController()
    motion_planner = MotionPlanner(robot_controller)
    
    print("✅ Components initialized")
    
    # Test 1: Joint Space Goal Precision
    print("\n1️⃣  Joint Space Goal Precision Test")
    
    q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_goal = np.array([0.5, 0.3, -0.4, 0.2, 0.1, 0.0])
    
    print(f"  🎯 Target joint configuration (degrees):")
    print(f"     {np.degrees(q_goal).round(2)}")
    
    joint_path = motion_planner.plan_joint_path(q_start, q_goal)
    
    if joint_path and len(joint_path) > 0:
        q_final = joint_path[-1]
        joint_error = np.abs(q_final - q_goal)
        max_joint_error = np.max(joint_error)
        
        print(f"  📍 Achieved joint configuration (degrees):")
        print(f"     {np.degrees(q_final).round(2)}")
        print(f"  📏 Joint errors (degrees):")
        print(f"     {np.degrees(joint_error).round(4)}")
        print(f"  🎯 Maximum joint error: {np.degrees(max_joint_error):.4f} degrees")
        
        if max_joint_error < 1e-6:  # Very small tolerance
            print("  ✅ EXACT match - Joint space goal reached precisely!")
        elif max_joint_error < 1e-3:
            print("  ✅ EXCELLENT precision - Goal reached within 0.06 degrees")
        else:
            print("  ⚠️  Notable error - Goal not reached exactly")
    else:
        print("  ❌ Joint path planning failed")
        return
    
    # Test 2: Cartesian Space Goal Precision
    print("\n2️⃣  Cartesian Space Goal Precision Test")
    
    start_pose = np.array([0.4, 0.2, 0.4, 0, 0, 0])
    goal_pose = np.array([0.2, 0.4, 0.5, 0, 0, 0])
    
    print(f"  🎯 Target Cartesian pose:")
    print(f"     Position: [{goal_pose[0]:.3f}, {goal_pose[1]:.3f}, {goal_pose[2]:.3f}] m")
    print(f"     Rotation: [{goal_pose[3]:.3f}, {goal_pose[4]:.3f}, {goal_pose[5]:.3f}] rad")
    
    try:
        cartesian_result = motion_planner.plan_cartesian_path(start_pose, goal_pose)
        
        if cartesian_result:
            # Get the final joint configuration from planning
            joint_path = cartesian_result.get('joint_path', [])
            
            if joint_path and len(joint_path) > 0:
                q_final_cart = joint_path[-1]
                
                # Compute forward kinematics to see actual end-effector pose
                T_final = robot_controller.forward_kinematics(q_final_cart)
                final_position = T_final[:3, 3]
                
                # Extract rotation (simplified - just check orientation)
                final_rotation = T_final[:3, :3]
                
                # Compute position error
                position_error = np.linalg.norm(final_position - goal_pose[:3])
                
                print(f"  📍 Achieved Cartesian pose:")
                print(f"     Position: [{final_position[0]:.6f}, {final_position[1]:.6f}, {final_position[2]:.6f}] m")
                print(f"  📏 Position error: {position_error:.6f} m ({position_error*1000:.3f} mm)")
                
                if position_error < 1e-3:  # 1mm tolerance
                    print("  ✅ EXCELLENT precision - Cartesian goal reached within 1mm!")
                elif position_error < 5e-3:  # 5mm tolerance
                    print("  ✅ GOOD precision - Cartesian goal reached within 5mm")
                else:
                    print(f"  ⚠️  Notable error - Position error: {position_error*1000:.1f}mm")
                    
            else:
                print("  ⚠️  No joint path returned from Cartesian planning")
        else:
            print("  ❌ Cartesian path planning failed")
            
    except Exception as e:
        print(f"  ❌ Cartesian planning error: {e}")
    
    # Test 3: IK Solution Precision
    print("\n3️⃣  Inverse Kinematics Precision Test")
    
    # Test a specific pose
    target_T = np.eye(4)
    target_T[:3, 3] = [0.3, 0.2, 0.6]  # Target position
    
    print(f"  🎯 Target pose matrix:")
    print(f"     Position: {target_T[:3, 3]}")
    
    q_ik, ik_success = robot_controller.inverse_kinematics(target_T)
    
    if ik_success:
        # Verify by forward kinematics
        T_achieved = robot_controller.forward_kinematics(q_ik)
        
        position_error = np.linalg.norm(T_achieved[:3, 3] - target_T[:3, 3])
        rotation_error = np.linalg.norm(T_achieved[:3, :3] - target_T[:3, :3], 'fro')
        
        print(f"  📍 IK solution joint angles (degrees):")
        print(f"     {np.degrees(q_ik).round(3)}")
        print(f"  📍 Achieved position: {T_achieved[:3, 3].round(6)}")
        print(f"  📏 Position error: {position_error:.6f} m ({position_error*1000:.3f} mm)")
        print(f"  📏 Rotation error: {rotation_error:.6f}")
        
        if position_error < 1e-3 and rotation_error < 1e-2:
            print("  ✅ EXCELLENT IK precision - Target reached accurately!")
        else:
            print("  ⚠️  IK has some error")
    else:
        print("  ❌ IK solution failed")
    
    # Test 4: Path Following Precision
    print("\n4️⃣  Path Following Precision Test")
    
    # Create a simple straight-line path
    start_point = np.array([0.4, 0.1, 0.4])
    end_point = np.array([0.2, 0.3, 0.6])
    
    print(f"  🎯 Straight line path:")
    print(f"     Start: {start_point}")
    print(f"     End:   {end_point}")
    
    # Generate waypoints
    num_waypoints = 5
    waypoints = []
    for i in range(num_waypoints):
        alpha = i / (num_waypoints - 1)
        waypoint = start_point + alpha * (end_point - start_point)
        waypoints.append(waypoint)
    
    # Test each waypoint
    achieved_points = []
    for i, waypoint in enumerate(waypoints):
        T_target = np.eye(4)
        T_target[:3, 3] = waypoint
        
        q_wp, success = robot_controller.inverse_kinematics(T_target)
        
        if success:
            T_achieved = robot_controller.forward_kinematics(q_wp)
            achieved_points.append(T_achieved[:3, 3])
            
            error = np.linalg.norm(T_achieved[:3, 3] - waypoint)
            print(f"    Waypoint {i+1}: Target {waypoint.round(3)} → Achieved {T_achieved[:3, 3].round(3)} (Error: {error*1000:.2f}mm)")
        else:
            print(f"    Waypoint {i+1}: IK failed")
            achieved_points.append(None)
    
    # Calculate path deviation
    valid_points = [p for p in achieved_points if p is not None]
    if len(valid_points) >= 2:
        # Check linearity of achieved path
        path_errors = []
        for i in range(1, len(valid_points)-1):
            # Expected position on straight line
            alpha = i / (len(valid_points) - 1)
            expected = achieved_points[0] + alpha * (achieved_points[-1] - achieved_points[0])
            actual = valid_points[i]
            deviation = np.linalg.norm(actual - expected)
            path_errors.append(deviation)
        
        if path_errors:
            max_deviation = max(path_errors)
            print(f"  📏 Maximum path deviation: {max_deviation*1000:.2f}mm")
            
            if max_deviation < 2e-3:  # 2mm
                print("  ✅ EXCELLENT path following - Very linear path achieved!")
            else:
                print("  ⚠️  Some path deviation detected")
    
    print("\n" + "=" * 60)
    print("🎯 PRECISION ANALYSIS SUMMARY:")
    print("   Joint space planning: Tests exact joint angle achievement")
    print("   Cartesian planning: Tests end-effector position accuracy") 
    print("   IK precision: Tests inverse kinematics accuracy")
    print("   Path following: Tests trajectory linearity and waypoint precision")

if __name__ == "__main__":
    test_goal_precision()

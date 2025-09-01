#!/usr/bin/env python3
"""
Production Kinematics Usage Examples for RB3-730ES

Comprehensive examples showing how to use the production kinematics code
for real-world applications.
"""

import numpy as np
import time
import json
from rb3_kinematics import RB3Kinematics, create_robot, solve_ik_robust

def basic_usage_example():
    """Basic forward and inverse kinematics usage"""
    print("1. BASIC USAGE EXAMPLE")
    print("-" * 30)
    
    # Create robot instance
    robot = create_robot()
    
    # Forward kinematics example
    joint_angles = [0.5, -0.5, 1.0, 0.0, 0.5, 0.0]
    position, orientation = robot.forward_kinematics(joint_angles)
    
    print(f"Forward Kinematics:")
    print(f"  Input joints: {[f'{j:.3f}' for j in joint_angles]}")
    print(f"  TCP position: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}] m")
    print(f"  TCP orientation: [{orientation[0]:.4f}, {orientation[1]:.4f}, {orientation[2]:.4f}] rad")
    
    # Inverse kinematics example
    target_pos = [0.3, 0.2, 0.4]
    target_ori = [0.0, 0.0, 0.0]
    
    result_joints, success = robot.inverse_kinematics(target_pos, target_ori)
    
    print(f"\nInverse Kinematics:")
    print(f"  Target position: {target_pos}")
    print(f"  Target orientation: {target_ori}")
    print(f"  Success: {success}")
    
    if success:
        print(f"  Result joints: {[f'{j:.3f}' for j in result_joints]}")
        
        # Validate the solution
        validation = robot.validate_solution(result_joints, target_pos, target_ori)
        print(f"  Position error: {validation['position_error_mm']:.3f} mm")
        print(f"  Orientation error: {validation['orientation_error_deg']:.3f} deg")
        print(f"  Within tolerance: {validation['within_tolerance']}")

def trajectory_planning_example():
    """Example of trajectory planning with kinematics"""
    print("\n2. TRAJECTORY PLANNING EXAMPLE")
    print("-" * 35)
    
    robot = create_robot()
    
    # Define waypoints
    waypoints = [
        ([0.4, 0.0, 0.3], [0.0, 0.0, 0.0]),      # Point A
        ([0.3, 0.3, 0.4], [0.0, 0.0, np.pi/4]),  # Point B
        ([0.2, 0.0, 0.5], [0.0, np.pi/4, 0.0]),  # Point C
        ([0.4, 0.0, 0.3], [0.0, 0.0, 0.0]),      # Back to A
    ]
    
    print("Planning trajectory through waypoints...")
    
    trajectory = []
    current_joints = np.zeros(6)  # Start from zero position
    
    for i, (pos, ori) in enumerate(waypoints):
        print(f"\nWaypoint {i+1}: pos={pos}, ori={[f'{o:.3f}' for o in ori]}")
        
        # Use current position as initial guess for smoother motion
        result_joints, success = robot.inverse_kinematics(pos, ori, current_joints)
        
        if success:
            # Validate solution
            validation = robot.validate_solution(result_joints, pos, ori)
            
            print(f"  ✓ Solution found: {[f'{j:.3f}' for j in result_joints]}")
            print(f"  ✓ Error: {validation['position_error_mm']:.3f}mm, {validation['orientation_error_deg']:.3f}deg")
            
            # Check for large joint movements
            if len(trajectory) > 0:
                joint_diff = np.abs(result_joints - current_joints)
                max_diff = np.max(joint_diff)
                print(f"  ✓ Max joint change: {np.degrees(max_diff):.1f} degrees")
            
            trajectory.append(result_joints)
            current_joints = result_joints
        else:
            print(f"  ✗ No solution found")
    
    print(f"\nTrajectory planning complete: {len(trajectory)}/{len(waypoints)} waypoints solved")

def workspace_analysis_example():
    """Analyze robot workspace and capabilities"""
    print("\n3. WORKSPACE ANALYSIS EXAMPLE")
    print("-" * 35)
    
    robot = create_robot()
    
    # Get workspace limits
    limits = robot.get_workspace_limits()
    print(f"Workspace Limits:")
    print(f"  Max reach: {limits['max_reach']:.3f} m")
    print(f"  Min reach: {limits['min_reach']:.3f} m")
    print(f"  Height range: {limits['min_height']:.3f} to {limits['max_height']:.3f} m")
    
    # Test reachability of various points
    test_points = [
        ([0.6, 0.0, 0.2], "Max reach"),
        ([0.0, 0.0, 0.8], "High point"),
        ([0.2, 0.2, 0.2], "Diagonal"),
        ([0.8, 0.0, 0.2], "Beyond reach"),
        ([0.1, 0.0, -0.3], "Below base"),
    ]
    
    print(f"\nReachability Analysis:")
    for pos, description in test_points:
        result_joints, success = robot.inverse_kinematics(pos, [0, 0, 0])
        
        if success:
            # Check for singularities
            is_singular = robot.check_singularity(result_joints)
            status = "✓ Reachable" + (" (near singularity)" if is_singular else "")
        else:
            status = "✗ Unreachable"
        
        print(f"  {description:12s} {pos}: {status}")

def performance_benchmark_example():
    """Benchmark kinematics performance"""
    print("\n4. PERFORMANCE BENCHMARK EXAMPLE")
    print("-" * 37)
    
    robot = create_robot()
    
    # Forward kinematics benchmark
    print("Forward Kinematics Benchmark:")
    num_fk_tests = 1000
    
    # Generate random joint configurations
    joint_configs = []
    for _ in range(num_fk_tests):
        joints = np.random.uniform(
            robot.joint_limits[:, 0], 
            robot.joint_limits[:, 1]
        )
        joint_configs.append(joints)
    
    # Time forward kinematics
    start_time = time.time()
    for joints in joint_configs:
        robot.forward_kinematics(joints)
    fk_time = time.time() - start_time
    
    print(f"  {num_fk_tests} evaluations in {fk_time:.4f}s")
    print(f"  Average: {1000*fk_time/num_fk_tests:.4f} ms per evaluation")
    print(f"  Rate: {num_fk_tests/fk_time:.0f} Hz")
    
    # Inverse kinematics benchmark
    print(f"\nInverse Kinematics Benchmark:")
    num_ik_tests = 100
    
    # Generate target poses from FK
    target_poses = []
    for joints in joint_configs[:num_ik_tests]:
        pos, ori = robot.forward_kinematics(joints)
        target_poses.append((pos, ori))
    
    # Time inverse kinematics
    start_time = time.time()
    successes = 0
    
    for pos, ori in target_poses:
        result_joints, success = robot.inverse_kinematics(pos, ori)
        if success:
            successes += 1
    
    ik_time = time.time() - start_time
    
    print(f"  {num_ik_tests} evaluations in {ik_time:.4f}s")
    print(f"  Average: {1000*ik_time/num_ik_tests:.4f} ms per evaluation")
    print(f"  Rate: {num_ik_tests/ik_time:.0f} Hz")
    print(f"  Success rate: {100*successes/num_ik_tests:.1f}%")

def error_handling_example():
    """Demonstrate error handling and validation"""
    print("\n5. ERROR HANDLING EXAMPLE")
    print("-" * 30)
    
    robot = create_robot()
    
    # Test invalid inputs
    test_cases = [
        ("Invalid joint count", [1, 2, 3]),  # Wrong number of joints
        ("Joint limits exceeded", [4, 4, 4, 4, 4, 4]),  # Beyond limits
        ("Unreachable position", [2.0, 0.0, 0.0]),  # Too far
        ("Valid case", [0.3, 0.2, 0.4]),  # Should work
    ]
    
    for description, test_input in test_cases:
        print(f"\nTesting: {description}")
        
        try:
            if len(test_input) == 3:  # Position test
                result_joints, success = robot.inverse_kinematics(test_input, [0, 0, 0])
                if success:
                    validation = robot.validate_solution(result_joints, test_input, [0, 0, 0])
                    print(f"  ✓ Success: {validation['within_tolerance']}")
                else:
                    print(f"  ✗ IK failed")
            else:  # Joint angles test
                pos, ori = robot.forward_kinematics(test_input)
                print(f"  ✓ FK result: {pos}")
        
        except Exception as e:
            print(f"  ✗ Error caught: {e}")

def recorded_data_validation_example():
    """Validate against recorded robot data"""
    print("\n6. RECORDED DATA VALIDATION EXAMPLE")
    print("-" * 42)
    
    robot = create_robot()
    
    # Try to load recorded data
    data_file = 'third_20250710_162459.json'
    
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data['waypoints'])} recorded waypoints")
        
        # Test first few waypoints
        test_count = min(5, len(data['waypoints']))
        successes = 0
        
        print(f"\nValidating first {test_count} waypoints:")
        print("WP | Recorded Joints           | FK Position (m)      | IK Success")
        print("---|---------------------------|----------------------|-----------")
        
        for i in range(test_count):
            waypoint = data['waypoints'][i]
            recorded_joints = waypoint['joint_positions']
            
            # Test forward kinematics
            try:
                fk_pos, fk_ori = robot.forward_kinematics(recorded_joints)
                
                # Test inverse kinematics back to same position
                ik_joints, ik_success = robot.inverse_kinematics(fk_pos, fk_ori)
                
                if ik_success:
                    successes += 1
                    status = "✓"
                else:
                    status = "✗"
                
                print(f"{i+1:2d} | {[f'{j:5.2f}' for j in recorded_joints]} | "
                      f"[{fk_pos[0]:6.3f},{fk_pos[1]:6.3f},{fk_pos[2]:6.3f}] | {status}")
            
            except Exception as e:
                print(f"{i+1:2d} | Error: {str(e)[:40]:40s} | Error")
        
        print(f"\nValidation Results: {successes}/{test_count} successful round-trips")
    
    except FileNotFoundError:
        print("Recorded data file not found - skipping validation")
    except Exception as e:
        print(f"Error loading recorded data: {e}")

def production_integration_example():
    """Example of production integration patterns"""
    print("\n7. PRODUCTION INTEGRATION EXAMPLE")
    print("-" * 42)
    
    class RobotController:
        """Example robot controller using production kinematics"""
        
        def __init__(self):
            self.robot = create_robot()
            self.current_joints = np.zeros(6)
            self.max_joint_velocity = np.radians(90)  # 90 deg/s
        
        def move_to_pose(self, position, orientation, max_attempts=3):
            """Move robot to target pose with validation"""
            
            # Solve inverse kinematics
            target_joints, success = solve_ik_robust(
                self.robot, position, orientation, max_attempts
            )
            
            if not success:
                raise ValueError("Cannot reach target pose")
            
            # Validate solution
            validation = self.robot.validate_solution(target_joints, position, orientation)
            
            if not validation['within_tolerance']:
                raise ValueError(f"Solution accuracy insufficient: {validation}")
            
            if not validation['joint_limits_ok']:
                raise ValueError("Target violates joint limits")
            
            if validation['singularity_free'] is False:
                print("Warning: Target near singularity")
            
            # Check motion feasibility
            joint_diff = np.abs(target_joints - self.current_joints)
            max_diff = np.max(joint_diff)
            
            if max_diff > np.radians(180):
                print(f"Warning: Large joint motion required ({np.degrees(max_diff):.1f} deg)")
            
            # Simulate motion (in real application, send to robot)
            print(f"Moving to: pos={position}, ori={[f'{o:.3f}' for o in orientation]}")
            print(f"Joint target: {[f'{j:.3f}' for j in target_joints]}")
            print(f"Max joint change: {np.degrees(max_diff):.1f} degrees")
            
            self.current_joints = target_joints
            return target_joints
        
        def get_current_pose(self):
            """Get current TCP pose"""
            return self.robot.forward_kinematics(self.current_joints)
    
    # Demonstrate controller usage
    controller = RobotController()
    
    try:
        # Move to several poses
        poses = [
            ([0.4, 0.0, 0.3], [0.0, 0.0, 0.0]),
            ([0.3, 0.3, 0.4], [0.0, 0.0, np.pi/4]),
            ([0.2, 0.0, 0.5], [0.0, np.pi/4, 0.0]),
        ]
        
        for i, (pos, ori) in enumerate(poses):
            print(f"\nMove {i+1}:")
            controller.move_to_pose(pos, ori)
            
            # Verify current pose
            current_pos, current_ori = controller.get_current_pose()
            print(f"Achieved: pos=[{current_pos[0]:.3f},{current_pos[1]:.3f},{current_pos[2]:.3f}], "
                  f"ori=[{current_ori[0]:.3f},{current_ori[1]:.3f},{current_ori[2]:.3f}]")
    
    except Exception as e:
        print(f"Motion error: {e}")

def main():
    """Run all examples"""
    print("RB3-730ES Production Kinematics Examples")
    print("=" * 50)
    
    basic_usage_example()
    trajectory_planning_example()
    workspace_analysis_example()
    performance_benchmark_example()
    error_handling_example()
    recorded_data_validation_example()
    production_integration_example()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")

if __name__ == "__main__":
    main()


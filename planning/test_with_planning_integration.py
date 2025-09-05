#!/usr/bin/env python3
"""
Test visualization integration with the actual planning system results.
"""

import sys
import os
import numpy as np

# Add parent directories to path
current_dir = os.path.dirname(__file__)
kinematics_dir = os.path.join(current_dir, '..', 'robot_kinematics')
sys.path.append(kinematics_dir)

from robot_controller import RobotController
from visualize import RobotVisualizer
from motion_planning import MotionPlanner
from trajectory_planning import TrajectoryPlanner

def test_real_planning_visualization():
    """Test visualization with actual planning system results."""
    print("🔍 Testing Visualization with Real Planning Results")
    print("=" * 55)
    
    try:
        # Initialize all components
        robot_controller = RobotController()
        visualizer = RobotVisualizer(robot_controller)
        motion_planner = MotionPlanner(robot_controller)
        trajectory_planner = TrajectoryPlanner(robot_controller)
        
        print("✅ All planning components initialized")
        
        # Test 1: Simple joint path planning
        print("\n🎯 Test 1: Joint Space Planning")
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_goal = np.array([0.5, 0.3, -0.4, 0.2, 0.1, 0.0])
        
        joint_path = motion_planner.plan_joint_path(q_start, q_goal)
        
        if joint_path:
            print(f"  ✅ Joint path found with {len(joint_path)} waypoints")
            
            # Visualize the trajectory
            fig_comp, fig_3d = visualizer.plot_trajectory_comparison(
                joint_path,
                save_path="planning_integration_joint_trajectory.png"
            )
            print("  💾 Joint trajectory visualization saved")
            
            # Visualize robot at start and end
            fig_start = visualizer.plot_robot_configuration_matplotlib(
                q_start, save_path="planning_integration_start.png"
            )
            
            fig_end = visualizer.plot_robot_configuration_matplotlib(
                q_goal, save_path="planning_integration_end.png"
            )
            
            print("  💾 Start/end configurations saved")
            
        else:
            print("  ❌ Joint path planning failed")
            return False
        
        # Test 2: Cartesian pose planning
        print("\n🎯 Test 2: Cartesian Space Planning")
        
        # Define poses in [x, y, z, rx, ry, rz] format
        start_pose = np.array([0.4, 0.2, 0.3, 0, 0, 0])
        goal_pose = np.array([-0.3, 0.4, 0.5, 0, 0, 0])
        
        cartesian_result = motion_planner.plan_cartesian_path(start_pose, goal_pose)
        
        if cartesian_result:
            print(f"  ✅ Cartesian path planning successful")
            
            # Convert to joint configuration for visualization
            start_T = np.eye(4)
            start_T[:3, 3] = start_pose[:3]
            q_vis, success = robot_controller.inverse_kinematics(start_T)
            
            if success:
                # Create sample Cartesian path for visualization
                cartesian_path = []
                for i in range(10):
                    alpha = i / 9.0
                    waypoint = start_pose[:3] + alpha * (goal_pose[:3] - start_pose[:3])
                    cartesian_path.append(waypoint)
                
                fig_cart = visualizer.plot_robot_configuration_matplotlib(
                    q_vis,
                    cartesian_path=cartesian_path,
                    show_workspace=True,
                    save_path="planning_integration_cartesian.png"
                )
                
                fig_cart_interactive = visualizer.plot_robot_configuration_plotly(
                    q_vis,
                    cartesian_path=cartesian_path,
                    show_workspace=True,
                    save_path="planning_integration_cartesian_interactive.html"
                )
                
                print("  💾 Cartesian path visualization saved")
            else:
                print("  ⚠️  Could not visualize - IK failed for start pose")
        else:
            print("  ❌ Cartesian path planning failed")
        
        # Test 3: Trajectory generation
        print("\n🎯 Test 3: Trajectory Generation")
        
        if joint_path:
            trajectory_result = trajectory_planner.generate_trajectory(
                joint_path,
                max_velocity=1.0,
                max_acceleration=2.0
            )
            
            if trajectory_result['success']:
                print(f"  ✅ Trajectory generated: {trajectory_result['duration']:.2f}s duration")
                
                # Visualize trajectory with time information
                interpolated_path = trajectory_result['joint_trajectory']
                
                if len(interpolated_path) > 0:
                    fig_traj, fig_3d_traj = visualizer.plot_trajectory_comparison(
                        interpolated_path,
                        save_path="planning_integration_full_trajectory.png"
                    )
                    print("  💾 Full trajectory visualization saved")
                else:
                    print("  ⚠️  No interpolated trajectory data available")
            else:
                print(f"  ❌ Trajectory generation failed: {trajectory_result['message']}")
        
        print("\n" + "=" * 55)
        print("🎉 Planning Integration Test Complete!")
        print("\nGenerated files:")
        print("  📊 planning_integration_joint_trajectory_joints.png")
        print("  📊 planning_integration_joint_trajectory_3d.png") 
        print("  🤖 planning_integration_start.png")
        print("  🤖 planning_integration_end.png")
        print("  🛤️  planning_integration_cartesian.png")
        print("  🌐 planning_integration_cartesian_interactive.html")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_planning_visualization()
    exit(0 if success else 1)

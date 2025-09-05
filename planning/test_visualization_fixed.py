#!/usr/bin/env python3
"""
Test script for 3D visualization of robot planning results.
Demonstrates visualization of robot configurations, Cartesian paths, and joint trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add parent directories to path
current_dir = os.path.dirname(__file__)
kinematics_dir = os.path.join(current_dir, '..', 'robot_kinematics')
sys.path.append(kinematics_dir)

from robot_controller import RobotController
from visualize import RobotVisualizer
from path_planning import CartesianPathPlanner, Environment3D
from motion_planning import MotionPlanner
from trajectory_planning import TrajectoryPlanner


def test_robot_visualization():
    """Test robot configuration visualization."""
    print("🎨 Testing Robot Configuration Visualization")
    
    # Initialize robot
    robot_controller = RobotController()
    visualizer = RobotVisualizer(robot_controller)
    
    # Test different configurations
    configurations = [
        ("Home Position", np.zeros(6)),
        ("Extended Position", np.array([0.5, 0.3, -0.4, 0.2, 0.1, 0.0])),
        ("Folded Position", np.array([1.0, 1.2, -1.5, 0.0, 0.5, 0.0]))
    ]
    
    for name, q in configurations:
        print(f"  📍 Visualizing: {name}")
        
        # Create matplotlib plot
        fig = visualizer.plot_robot_configuration_matplotlib(
            q, save_path=f"robot_{name.lower().replace(' ', '_')}_mpl.png"
        )
        
        # Create interactive plotly plot
        fig_plotly = visualizer.plot_robot_configuration_plotly(
            q, save_path=f"robot_{name.lower().replace(' ', '_')}_plotly.html"
        )
        
        plt.close(fig)  # Close to save memory
        
    print("✅ Robot configuration visualization completed!")
    

def test_path_visualization():
    """Test Cartesian path planning and visualization."""
    print("\n🛤️  Testing Path Planning Visualization")
    
    # Initialize components
    robot_controller = RobotController()
    visualizer = RobotVisualizer(robot_controller)
    
    # Create environment
    environment = Environment3D(
        workspace_bounds=((-0.8, 0.8), (-0.8, 0.8), (0.0, 1.6))
    )
    environment.add_sphere_obstacle([0.2, 0.2, 0.5], 0.1)
    environment.add_box_obstacle([-0.3, -0.2, 0.2], [-0.1, 0.0, 0.4])
    
    # Create path planner
    path_planner = CartesianPathPlanner(environment)
    
    # Define start and goal poses
    start_pose = np.array([0.5, 0.3, 0.4])
    goal_pose = np.array([-0.4, 0.4, 0.6])
    
    print(f"  🎯 Planning path from {start_pose} to {goal_pose}")
    
    # Plan Cartesian path
    start_time = time.time()
    cartesian_path, path_length = path_planner.plan_path(start_pose, goal_pose)
    planning_time = time.time() - start_time
    
    if cartesian_path:
        print(f"  ✅ Path found! Length: {path_length:.3f}m, Time: {planning_time:.2f}s")
        print(f"  📊 Path has {len(cartesian_path)} waypoints")
        
        # Convert first waypoint to joint configuration for visualization
        start_T = np.eye(4)
        start_T[:3, 3] = start_pose
        q_start, success = robot_controller.inverse_kinematics(start_T)
        
        if success:
            # Visualize robot at start with planned path
            fig = visualizer.plot_robot_configuration_matplotlib(
                q_start, 
                cartesian_path=cartesian_path,
                show_workspace=True,
                save_path="path_planning_result_mpl.png"
            )
            
            fig_plotly = visualizer.plot_robot_configuration_plotly(
                q_start,
                cartesian_path=cartesian_path,
                show_workspace=True,
                save_path="path_planning_result_plotly.html"
            )
            
            plt.close(fig)
            print("  💾 Path visualization saved!")
            
        else:
            print("  ⚠️  Could not find IK solution for start pose")
            
    else:
        print("  ❌ No path found!")
        

def test_simple_visualization():
    """Test simple robot visualization with just the planning system results."""
    print("\n🚀 Creating Simple Robot Visualization with Planning Results")
    
    # Initialize components
    robot_controller = RobotController()
    visualizer = RobotVisualizer(robot_controller)
    motion_planner = MotionPlanner(robot_controller)
    
    # Use simple poses that we know work
    pick_pose = np.array([0.4, 0.2, 0.3, 0, 0, 0])  # [x, y, z, rx, ry, rz]
    place_pose = np.array([-0.3, 0.4, 0.5, 0, 0, 0])
    
    print(f"  🎯 Planning simple motion")
    print(f"    From: {pick_pose[:3]}")
    print(f"    To:   {place_pose[:3]}")
    
    try:
        # Plan motion
        result = motion_planner.plan_motion(pick_pose, place_pose)
        
        if result['success']:
            joint_path = result['joint_path']
            
            print(f"  ✅ Motion planned successfully!")
            print(f"    Joint waypoints: {len(joint_path)}")
            
            # Visualize robot at different configurations
            q_start = joint_path[0]
            q_middle = joint_path[len(joint_path)//2] if len(joint_path) > 1 else q_start
            q_end = joint_path[-1]
            
            # Create simple plots
            print("  📸 Creating static visualizations...")
            
            # Start configuration
            fig_start = visualizer.plot_robot_configuration_matplotlib(
                q_start, save_path="robot_start_config.png"
            )
            
            # End configuration  
            fig_end = visualizer.plot_robot_configuration_matplotlib(
                q_end, save_path="robot_end_config.png"
            )
            
            # Interactive plot
            fig_interactive = visualizer.plot_robot_configuration_plotly(
                q_middle, save_path="robot_interactive.html"
            )
            
            # Create simple trajectory plot
            print("  📊 Creating trajectory plots...")
            fig_traj, fig_3d = visualizer.plot_trajectory_comparison(
                joint_path, save_path="robot_trajectory.png"
            )
            
            plt.close('all')  # Close all figures
            
            print("  💾 Simple visualizations saved!")
            print("    - robot_start_config.png: Robot at start position")
            print("    - robot_end_config.png: Robot at end position") 
            print("    - robot_interactive.html: Interactive 3D view")
            print("    - robot_trajectory_joints.png: Joint angle plots")
            print("    - robot_trajectory_3d.png: 3D Cartesian path")
            
        else:
            print(f"  ❌ Motion planning failed: {result['message']}")
            
    except Exception as e:
        print(f"  💥 Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run visualization tests."""
    print("🎨 Robot Planning Visualization Test Suite")
    print("=" * 50)
    
    try:
        # Test robot configuration visualization (this worked before)
        test_robot_visualization()
        
        # Test simple visualization with planning results
        test_simple_visualization()
        
        print("\n" + "=" * 50)
        print("🎉 Visualization tests completed!")
        print("\nGenerated files:")
        print("  📸 Static plots: *.png files")
        print("  🌐 Interactive plots: *.html files")
        print("\nOpen the HTML files in a web browser for interactive 3D exploration!")
        
    except Exception as e:
        print(f"\n💥 Visualization test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

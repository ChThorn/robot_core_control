#!/usr/bin/env python3
"""
Simple demonstration of 3D robot visualization with planning results.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directories to path
current_dir = os.path.dirname(__file__)
kinematics_dir = os.path.join(current_dir, '..', 'robot_kinematics')
sys.path.append(kinematics_dir)

from robot_controller import RobotController
from visualize import RobotVisualizer


def create_sample_robot_configurations():
    """Create sample robot configurations for visualization."""
    print("🤖 Creating Sample Robot Configurations")
    
    # Initialize robot and visualizer
    robot_controller = RobotController()
    visualizer = RobotVisualizer(robot_controller)
    
    # Test configurations
    configurations = [
        ("Home", np.zeros(6)),
        ("Reach", np.array([0.5, 0.3, -0.4, 0.2, 0.1, 0.0])),
        ("Fold", np.array([1.0, 1.2, -1.5, 0.0, 0.5, 0.0]))
    ]
    
    for name, q in configurations:
        print(f"  📍 Creating {name} configuration visualization...")
        
        # Create static plot
        fig = visualizer.plot_robot_configuration_matplotlib(
            q, 
            show_workspace=True,
            save_path=f"robot_{name.lower()}_static.png"
        )
        
        # Create interactive plot
        fig_plotly = visualizer.plot_robot_configuration_plotly(
            q,
            show_workspace=True, 
            save_path=f"robot_{name.lower()}_interactive.html"
        )
        
        plt.close(fig)
        print(f"    💾 Saved: robot_{name.lower()}_static.png and robot_{name.lower()}_interactive.html")


def create_sample_cartesian_path():
    """Create and visualize a sample Cartesian path."""
    print("\n🛤️  Creating Sample Cartesian Path Visualization")
    
    # Initialize robot and visualizer
    robot_controller = RobotController()
    visualizer = RobotVisualizer(robot_controller)
    
    # Create a simple Cartesian path (straight line)
    start_point = np.array([0.5, 0.2, 0.4])
    end_point = np.array([-0.3, 0.4, 0.6])
    
    # Generate waypoints along the path
    num_waypoints = 10
    cartesian_path = []
    
    for i in range(num_waypoints):
        alpha = i / (num_waypoints - 1)
        waypoint = start_point + alpha * (end_point - start_point)
        cartesian_path.append(waypoint)
    
    print(f"  📊 Created path with {len(cartesian_path)} waypoints")
    print(f"    From: {start_point}")
    print(f"    To:   {end_point}")
    
    # Try to find robot configuration for start point
    start_T = np.eye(4)
    start_T[:3, 3] = start_point
    q_start, success = robot_controller.inverse_kinematics(start_T)
    
    if success:
        print("  ✅ Found valid robot configuration for path start")
        
        # Visualize robot with path
        fig = visualizer.plot_robot_configuration_matplotlib(
            q_start,
            cartesian_path=cartesian_path,
            show_workspace=True,
            save_path="robot_with_cartesian_path.png"
        )
        
        fig_plotly = visualizer.plot_robot_configuration_plotly(
            q_start,
            cartesian_path=cartesian_path,
            show_workspace=True,
            save_path="robot_with_cartesian_path_interactive.html"
        )
        
        plt.close(fig)
        print("  💾 Saved: robot_with_cartesian_path.png and robot_with_cartesian_path_interactive.html")
        
    else:
        print("  ⚠️  Could not find valid robot configuration for path start")


def create_joint_space_trajectory():
    """Create and visualize a joint space trajectory."""
    print("\n🦾 Creating Joint Space Trajectory Visualization")
    
    # Initialize robot and visualizer
    robot_controller = RobotController()
    visualizer = RobotVisualizer(robot_controller)
    
    # Create a trajectory between two configurations
    q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_end = np.array([0.8, 0.5, -0.6, 0.3, 0.2, 0.0])
    
    # Generate interpolated joint path
    num_steps = 20
    joint_path = []
    
    for i in range(num_steps):
        alpha = i / (num_steps - 1)
        q_interp = q_start + alpha * (q_end - q_start)
        joint_path.append(q_interp)
    
    print(f"  📊 Created joint trajectory with {num_steps} steps")
    print(f"    Start config: {np.degrees(q_start).round(1)} deg")
    print(f"    End config:   {np.degrees(q_end).round(1)} deg")
    
    # Visualize trajectory comparison
    fig_comp, fig_3d = visualizer.plot_trajectory_comparison(
        joint_path,
        save_path="joint_trajectory_analysis.png"
    )
    
    plt.close('all')
    print("  💾 Saved: joint_trajectory_analysis_joints.png and joint_trajectory_analysis_3d.png")
    
    # Create animation (sample every few frames for manageable size)
    animation_path = joint_path[::4]  # Every 4th frame
    
    print("  🎬 Creating animation...")
    anim = visualizer.animate_joint_path_matplotlib(
        animation_path,
        save_path="robot_motion_animation.gif"
    )
    
    plt.show()
    print("  💾 Saved: robot_motion_animation.gif")


def demo_all_visualizations():
    """Run all visualization demonstrations."""
    print("🎨 Robot 3D Visualization Demo")
    print("=" * 50)
    
    try:
        # Create robot configuration visualizations
        create_sample_robot_configurations()
        
        # Create Cartesian path visualization
        create_sample_cartesian_path()
        
        # Create joint trajectory visualization
        create_joint_space_trajectory()
        
        print("\n" + "=" * 50)
        print("🎉 All visualizations created successfully!")
        print("\nGenerated files:")
        print("  📸 Static robot configurations:")
        print("    - robot_home_static.png")
        print("    - robot_reach_static.png") 
        print("    - robot_fold_static.png")
        print("  🌐 Interactive robot configurations:")
        print("    - robot_home_interactive.html")
        print("    - robot_reach_interactive.html")
        print("    - robot_fold_interactive.html")
        print("  🛤️  Path visualization:")
        print("    - robot_with_cartesian_path.png")
        print("    - robot_with_cartesian_path_interactive.html")
        print("  📊 Trajectory analysis:")
        print("    - joint_trajectory_analysis_joints.png")
        print("    - joint_trajectory_analysis_3d.png")
        print("  🎬 Animation:")
        print("    - robot_motion_animation.gif")
        print("\n🔍 Open the .html files in a web browser for interactive 3D exploration!")
        
    except Exception as e:
        print(f"\n�� Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_all_visualizations()

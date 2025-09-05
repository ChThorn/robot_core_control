#!/usr/bin/env python3
"""
Quick demo showing 3D robot visualization results.
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


def main():
    print("🎨 Quick Robot Visualization Demo")
    print("=" * 40)
    
    # Initialize robot and visualizer
    robot_controller = RobotController()
    visualizer = RobotVisualizer(robot_controller)
    
    # Demo 1: Robot in different poses
    print("1️⃣  Robot Configuration Visualization")
    q_demo = np.array([0.5, 0.3, -0.4, 0.2, 0.1, 0.0])
    
    fig = visualizer.plot_robot_configuration_matplotlib(
        q_demo, 
        show_workspace=True,
        save_path="demo_robot_config.png"
    )
    
    fig_interactive = visualizer.plot_robot_configuration_plotly(
        q_demo,
        show_workspace=True,
        save_path="demo_robot_interactive.html"
    )
    
    plt.close(fig)
    print("   ✅ Created: demo_robot_config.png")
    print("   ✅ Created: demo_robot_interactive.html")
    
    # Demo 2: Robot with Cartesian path
    print("\n2️⃣  Robot with Cartesian Path")
    
    # Create a sample path
    start = np.array([0.4, 0.2, 0.5])
    end = np.array([-0.2, 0.4, 0.7])
    path = [start + t * (end - start) for t in np.linspace(0, 1, 8)]
    
    fig_path = visualizer.plot_robot_configuration_matplotlib(
        q_demo,
        cartesian_path=path,
        show_workspace=True,
        save_path="demo_robot_with_path.png"
    )
    
    fig_path_interactive = visualizer.plot_robot_configuration_plotly(
        q_demo,
        cartesian_path=path,
        show_workspace=True,
        save_path="demo_robot_with_path_interactive.html"
    )
    
    plt.close(fig_path)
    print("   ✅ Created: demo_robot_with_path.png")
    print("   ✅ Created: demo_robot_with_path_interactive.html")
    
    # Demo 3: Joint trajectory analysis
    print("\n3️⃣  Joint Trajectory Analysis")
    
    # Create trajectory
    q_start = np.zeros(6)
    q_end = np.array([0.8, 0.5, -0.6, 0.3, 0.2, 0.0])
    trajectory = [q_start + t * (q_end - q_start) for t in np.linspace(0, 1, 15)]
    
    fig_traj, fig_3d = visualizer.plot_trajectory_comparison(
        trajectory,
        save_path="demo_trajectory.png"
    )
    
    plt.close('all')
    print("   ✅ Created: demo_trajectory_joints.png")
    print("   ✅ Created: demo_trajectory_3d.png")
    
    print("\n" + "=" * 40)
    print("🎉 Demo complete! Generated files:")
    print("   📸 Static plots: demo_*.png")
    print("   🌐 Interactive: demo_*_interactive.html")
    print("\n🔍 Open .html files in browser for 3D interaction!")


if __name__ == "__main__":
    main()

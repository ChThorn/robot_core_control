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
        bounds=[(-0.8, 0.8), (-0.8, 0.8), (0.0, 1.6)],
        sphere_obstacles=[(np.array([0.2, 0.2, 0.5]), 0.1)],
        box_obstacles=[(np.array([-0.3, -0.2, 0.2]), np.array([-0.1, 0.0, 0.4]))]
    )
    
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
        

def test_motion_planning_visualization():
    """Test complete motion planning with joint space trajectory."""
    print("\n🤖 Testing Complete Motion Planning Visualization")
    
    # Initialize components
    robot_controller = RobotController()
    visualizer = RobotVisualizer(robot_controller)
    motion_planner = MotionPlanner(robot_controller)
    trajectory_planner = TrajectoryPlanner(robot_controller)
    
    # Define pick and place scenario
    pick_pose = np.array([0.4, 0.2, 0.3, 0, 0, 0])  # [x, y, z, rx, ry, rz]
    place_pose = np.array([-0.3, 0.4, 0.5, 0, 0, 0])
    
    print(f"  🎯 Planning pick-and-place motion")
    print(f"    Pick:  {pick_pose[:3]}")
    print(f"    Place: {place_pose[:3]}")
    
    try:
        # Plan motion
        result = motion_planner.plan_motion(pick_pose, place_pose)
        
        if result['success']:
            cartesian_path = result['cartesian_path']
            joint_path = result['joint_path']
            
            print(f"  ✅ Motion planned successfully!")
            print(f"    Cartesian waypoints: {len(cartesian_path)}")
            print(f"    Joint waypoints: {len(joint_path)}")
            
            # Generate trajectory with time parameterization
            trajectory_result = trajectory_planner.generate_trajectory(
                joint_path, 
                max_velocity=1.0,
                max_acceleration=2.0
            )
            
            if trajectory_result['success']:
                print(f"  ⏱️  Trajectory generated: {trajectory_result['duration']:.2f}s")
                
                # Visualize complete motion
                q_start = joint_path[0]
                
                # Static visualization with full path
                fig = visualizer.plot_robot_configuration_matplotlib(
                    q_start,
                    cartesian_path=[np.eye(4) for _ in cartesian_path],  # Convert to 4x4 matrices
                    show_workspace=True,
                    save_path="motion_planning_result_mpl.png"
                )
                
                # Interactive visualization
                fig_plotly = visualizer.plot_robot_configuration_plotly(
                    q_start,
                    cartesian_path=[np.eye(4) for _ in cartesian_path],
                    show_workspace=True,
                    save_path="motion_planning_result_plotly.html"
                )
                
                # Create animation of joint motion
                print("  🎬 Creating motion animation...")
                
                # Sample joint path for animation (every 5th waypoint)
                animation_path = joint_path[::max(1, len(joint_path)//20)]
                
                anim = visualizer.animate_joint_path_matplotlib(
                    animation_path,
                    save_path="motion_animation.gif"
                )
                
                # Plot trajectory comparison
                print("  📊 Creating trajectory analysis plots...")
                fig_comp, fig_3d = visualizer.plot_trajectory_comparison(
                    joint_path,
                    save_path="trajectory_comparison.png"
                )
                
                plt.close(fig)
                plt.close(fig_comp)
                plt.close(fig_3d)
                
                print("  💾 All visualizations saved!")
                
            else:
                print(f"  ⚠️  Trajectory generation failed: {trajectory_result['message']}")
                
        else:
            print(f"  ❌ Motion planning failed: {result['message']}")
            
    except Exception as e:
        print(f"  💥 Motion planning error: {e}")


def test_interactive_visualization():
    """Test interactive plotly visualizations."""
    print("\n🖱️  Testing Interactive Visualizations")
    
    # Initialize robot
    robot_controller = RobotController()
    visualizer = RobotVisualizer(robot_controller)
    
    # Create interesting robot configuration
    q_demo = np.array([0.7, 0.5, -0.8, 0.3, 0.4, 0.0])
    
    print("  🌐 Creating interactive 3D visualization...")
    
    # Create comprehensive interactive plot
    fig_interactive = visualizer.plot_robot_configuration_plotly(
        q_demo,
        show_workspace=True,
        save_path="interactive_robot_visualization.html"
    )
    
    print("  💾 Interactive visualization saved as 'interactive_robot_visualization.html'")
    print("  🔍 Open the HTML file in a web browser for interactive 3D exploration!")
    

def main():
    """Run all visualization tests."""
    print("🎨 Robot Planning Visualization Test Suite")
    print("=" * 50)
    
    try:
        # Test individual components
        test_robot_visualization()
        test_path_visualization()
        test_motion_planning_visualization()
        test_interactive_visualization()
        
        print("\n" + "=" * 50)
        print("🎉 All visualization tests completed successfully!")
        print("\nGenerated files:")
        print("  📸 Static plots: *.png files")
        print("  🌐 Interactive plots: *.html files")
        print("  🎬 Animations: *.gif files")
        print("\nOpen the HTML files in a web browser for interactive 3D exploration!")
        
    except Exception as e:
        print(f"\n💥 Visualization test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

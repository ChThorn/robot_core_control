#!/usr/bin/env python3
"""
3D Visualization module for robot planning system.
Provides comprehensive visualization of robot links, joint paths, and Cartesian trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple, Optional, Dict, Union
import sys
import os
import logging

# Add parent directories to path for imports
current_dir = os.path.dirname(__file__)
kinematics_dir = os.path.join(current_dir, '..', 'robot_kinematics')
sys.path.append(kinematics_dir)

try:
    from robot_controller import RobotController
    from robot_kinematics import RobotKinematics
except ImportError as e:
    logging.error(f"Failed to import robot kinematics: {e}")
    
from path_planning import Environment3D
from motion_planning import RobotEnvironment

logger = logging.getLogger(__name__)

# Type aliases
JointConfiguration = np.ndarray
CartesianPath = List[np.ndarray]
JointPath = List[JointConfiguration]


class RobotVisualizer:
    """3D visualization for robot planning system."""
    
    def __init__(self, robot_controller: RobotController):
        """
        Initialize robot visualizer.
        
        Args:
            robot_controller: Robot controller with kinematics
        """
        self.robot_controller = robot_controller
        self.robot = robot_controller.robot
        
        # Robot geometric parameters (for RB3-730ES-U)
        self.link_lengths = [0.1453, 0.286, 0.344, 0.1, 0.1]  # Approximate link lengths
        self.joint_positions = self._compute_joint_positions()
        
    def _compute_joint_positions(self) -> List[np.ndarray]:
        """Compute approximate joint positions for visualization."""
        # These are approximate positions based on the robot's DH parameters
        return [
            np.array([0.0, 0.0, 0.0]),          # Base
            np.array([0.0, 0.0, 0.1453]),       # Joint 1
            np.array([0.0, -0.00645, 0.1453]),  # Joint 2
            np.array([0.0, -0.00645, 0.4313]),  # Joint 3
            np.array([0.0, -0.00645, 0.4313]),  # Joint 4
            np.array([0.0, -0.00645, 0.7753]),  # Joint 5
            np.array([0.0, -0.00645, 0.8753])   # End-effector
        ]
        
    def compute_forward_kinematics_chain(self, q: JointConfiguration) -> List[np.ndarray]:
        """
        Compute forward kinematics for all links in the chain.
        
        Args:
            q: Joint configuration
            
        Returns:
            List of 4x4 transformation matrices for each link
        """
        transforms = []
        T_current = np.eye(4)
        
        # Add base transform
        transforms.append(T_current.copy())
        
        # Compute transforms for each joint
        for i in range(self.robot.n_joints):
            # Apply joint transformation (simplified)
            xi_theta = self.robot.S[:, i] * q[i]
            T_joint = self.robot._matrix_exp6(xi_theta)
            T_current = T_current @ T_joint
            transforms.append(T_current.copy())
            
        # Apply end-effector transform
        T_current = T_current @ self.robot.M
        transforms.append(T_current.copy())
        
        return transforms
        
    def extract_link_positions(self, transforms: List[np.ndarray]) -> List[np.ndarray]:
        """Extract link positions from transformation matrices."""
        positions = []
        for T in transforms:
            positions.append(T[:3, 3])
        return positions
        
    def plot_robot_configuration_matplotlib(self, q: JointConfiguration, 
                                           cartesian_path: Optional[CartesianPath] = None,
                                           joint_path: Optional[JointPath] = None,
                                           show_workspace: bool = True,
                                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot robot configuration using matplotlib.
        
        Args:
            q: Joint configuration to visualize
            cartesian_path: Optional Cartesian path to plot
            joint_path: Optional joint path to animate
            show_workspace: Whether to show workspace boundaries
            save_path: Optional path to save the plot
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Compute robot links
        transforms = self.compute_forward_kinematics_chain(q)
        link_positions = self.extract_link_positions(transforms)
        
        # Plot robot links
        x_coords = [pos[0] for pos in link_positions]
        y_coords = [pos[1] for pos in link_positions]
        z_coords = [pos[2] for pos in link_positions]
        
        # Plot links as lines
        ax.plot(x_coords, y_coords, z_coords, 'b-', linewidth=3, label='Robot Links')
        
        # Plot joints as spheres
        ax.scatter(x_coords[:-1], y_coords[:-1], z_coords[:-1], 
                  c='red', s=100, label='Joints')
        
        # Plot end-effector
        ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                  c='green', s=150, marker='s', label='End-Effector')
        
        # Plot Cartesian path if provided
        if cartesian_path is not None and len(cartesian_path) > 0:
            path_array = np.array([pos[:3, 3] if pos.shape == (4, 4) else pos[:3] 
                                 for pos in cartesian_path])
            ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                   'g--', linewidth=2, alpha=0.7, label='Cartesian Path')
            
            # Mark start and end points
            ax.scatter(path_array[0, 0], path_array[0, 1], path_array[0, 2], 
                      c='orange', s=200, marker='o', label='Start')
            ax.scatter(path_array[-1, 0], path_array[-1, 1], path_array[-1, 2], 
                      c='purple', s=200, marker='o', label='Goal')
        
        # Show workspace boundaries
        if show_workspace:
            try:
                robot_env = RobotEnvironment(self.robot_controller)
                bounds = robot_env.bounds
                
                # Draw workspace box
                for i, (min_bound, max_bound) in enumerate(bounds):
                    if i == 0:  # X bounds
                        ax.plot([min_bound, max_bound], [bounds[1][0], bounds[1][0]], 
                               [bounds[2][0], bounds[2][0]], 'k--', alpha=0.3)
                        ax.plot([min_bound, max_bound], [bounds[1][1], bounds[1][1]], 
                               [bounds[2][1], bounds[2][1]], 'k--', alpha=0.3)
                
                # Plot obstacles
                for center, radius in robot_env.sphere_obstacles:
                    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                    x = center[0] + radius * np.cos(u) * np.sin(v)
                    y = center[1] + radius * np.sin(u) * np.sin(v)
                    z = center[2] + radius * np.cos(v)
                    ax.plot_surface(x, y, z, alpha=0.3, color='red')
                    
                for min_corner, max_corner in robot_env.box_obstacles:
                    # Draw box outline
                    vertices = [
                        [min_corner[0], min_corner[1], min_corner[2]],
                        [max_corner[0], min_corner[1], min_corner[2]],
                        [max_corner[0], max_corner[1], min_corner[2]],
                        [min_corner[0], max_corner[1], min_corner[2]],
                        [min_corner[0], min_corner[1], max_corner[2]],
                        [max_corner[0], min_corner[1], max_corner[2]],
                        [max_corner[0], max_corner[1], max_corner[2]],
                        [min_corner[0], max_corner[1], max_corner[2]]
                    ]
                    vertices = np.array(vertices)
                    
                    # Draw box edges
                    edges = [
                        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
                    ]
                    
                    for edge in edges:
                        ax.plot3D(*vertices[edge].T, 'r-', alpha=0.6, linewidth=2)
                        
            except Exception as e:
                logger.warning(f"Could not plot workspace: {e}")
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Robot Configuration and Path Visualization')
        ax.legend()
        
        # Set equal aspect ratio
        max_range = 1.0
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range*2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
            
        return fig
        
    def plot_robot_configuration_plotly(self, q: JointConfiguration, 
                                       cartesian_path: Optional[CartesianPath] = None,
                                       joint_path: Optional[JointPath] = None,
                                       show_workspace: bool = True,
                                       save_path: Optional[str] = None) -> go.Figure:
        """
        Plot robot configuration using plotly (interactive).
        
        Args:
            q: Joint configuration to visualize
            cartesian_path: Optional Cartesian path to plot
            joint_path: Optional joint path to animate
            show_workspace: Whether to show workspace boundaries
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Compute robot links
        transforms = self.compute_forward_kinematics_chain(q)
        link_positions = self.extract_link_positions(transforms)
        
        # Plot robot links
        x_coords = [pos[0] for pos in link_positions]
        y_coords = [pos[1] for pos in link_positions]
        z_coords = [pos[2] for pos in link_positions]
        
        # Add robot links
        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='lines+markers',
            line=dict(color='blue', width=6),
            marker=dict(size=8, color='red'),
            name='Robot Links'
        ))
        
        # Add end-effector
        fig.add_trace(go.Scatter3d(
            x=[x_coords[-1]], y=[y_coords[-1]], z=[z_coords[-1]],
            mode='markers',
            marker=dict(size=12, color='green', symbol='square'),
            name='End-Effector'
        ))
        
        # Plot Cartesian path if provided
        if cartesian_path is not None and len(cartesian_path) > 0:
            path_array = np.array([pos[:3, 3] if pos.shape == (4, 4) else pos[:3] 
                                 for pos in cartesian_path])
            
            fig.add_trace(go.Scatter3d(
                x=path_array[:, 0], y=path_array[:, 1], z=path_array[:, 2],
                mode='lines+markers',
                line=dict(color='green', width=4, dash='dash'),
                marker=dict(size=4, color='green'),
                name='Cartesian Path'
            ))
            
            # Mark start and end points
            fig.add_trace(go.Scatter3d(
                x=[path_array[0, 0]], y=[path_array[0, 1]], z=[path_array[0, 2]],
                mode='markers',
                marker=dict(size=15, color='orange', symbol='circle'),
                name='Start'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[path_array[-1, 0]], y=[path_array[-1, 1]], z=[path_array[-1, 2]],
                mode='markers',
                marker=dict(size=15, color='purple', symbol='circle'),
                name='Goal'
            ))
        
        # Show workspace and obstacles
        if show_workspace:
            try:
                robot_env = RobotEnvironment(self.robot_controller)
                
                # Plot sphere obstacles
                for center, radius in robot_env.sphere_obstacles:
                    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                    x = center[0] + radius * np.cos(u) * np.sin(v)
                    y = center[1] + radius * np.sin(u) * np.sin(v)
                    z = center[2] + radius * np.cos(v)
                    
                    fig.add_trace(go.Surface(
                        x=x, y=y, z=z,
                        opacity=0.3,
                        colorscale='Reds',
                        showscale=False,
                        name=f'Sphere Obstacle'
                    ))
                
                # Plot box obstacles
                for i, (min_corner, max_corner) in enumerate(robot_env.box_obstacles):
                    # Create box mesh
                    vertices = np.array([
                        [min_corner[0], min_corner[1], min_corner[2]],
                        [max_corner[0], min_corner[1], min_corner[2]],
                        [max_corner[0], max_corner[1], min_corner[2]],
                        [min_corner[0], max_corner[1], min_corner[2]],
                        [min_corner[0], min_corner[1], max_corner[2]],
                        [max_corner[0], min_corner[1], max_corner[2]],
                        [max_corner[0], max_corner[1], max_corner[2]],
                        [min_corner[0], max_corner[1], max_corner[2]]
                    ])
                    
                    # Define box faces
                    faces = [
                        [0, 1, 2, 3], [4, 7, 6, 5],  # Bottom and top
                        [0, 4, 5, 1], [2, 6, 7, 3],  # Front and back
                        [0, 3, 7, 4], [1, 5, 6, 2]   # Left and right
                    ]
                    
                    # Add box outline
                    edges = [
                        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
                        [4, 5], [5, 6], [6, 7], [7, 4],  # Top
                        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical
                    ]
                    
                    for edge in edges:
                        fig.add_trace(go.Scatter3d(
                            x=vertices[edge, 0], y=vertices[edge, 1], z=vertices[edge, 2],
                            mode='lines',
                            line=dict(color='red', width=4),
                            showlegend=False
                        ))
                        
            except Exception as e:
                logger.warning(f"Could not plot workspace: {e}")
        
        # Update layout
        fig.update_layout(
            title='Interactive Robot Configuration and Path Visualization',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive plot saved to {save_path}")
            
        return fig
        
    def animate_joint_path_matplotlib(self, joint_path: JointPath, 
                                     cartesian_path: Optional[CartesianPath] = None,
                                     save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create animated visualization of joint path using matplotlib.
        
        Args:
            joint_path: List of joint configurations
            cartesian_path: Optional corresponding Cartesian path
            save_path: Optional path to save animation
            
        Returns:
            Animation object
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up the plot
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Robot Motion Animation')
        
        max_range = 1.0
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, max_range*2])
        
        # Initialize empty plots
        robot_line, = ax.plot([], [], [], 'b-', linewidth=3, label='Robot Links')
        joints_scatter = ax.scatter([], [], [], c='red', s=100, label='Joints')
        ee_scatter = ax.scatter([], [], [], c='green', s=150, marker='s', label='End-Effector')
        
        # Plot Cartesian path if provided
        if cartesian_path is not None:
            path_array = np.array([pos[:3, 3] if pos.shape == (4, 4) else pos[:3] 
                                 for pos in cartesian_path])
            ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 
                   'g--', linewidth=2, alpha=0.5, label='Planned Path')
        
        ax.legend()
        
        def animate(frame):
            if frame < len(joint_path):
                q = joint_path[frame]
                
                # Compute robot configuration
                transforms = self.compute_forward_kinematics_chain(q)
                link_positions = self.extract_link_positions(transforms)
                
                x_coords = [pos[0] for pos in link_positions]
                y_coords = [pos[1] for pos in link_positions]
                z_coords = [pos[2] for pos in link_positions]
                
                # Update robot links
                robot_line.set_data_3d(x_coords, y_coords, z_coords)
                
                # Update joints (remove old and add new)
                ax.collections.clear()
                ax.scatter(x_coords[:-1], y_coords[:-1], z_coords[:-1], 
                          c='red', s=100, label='Joints')
                ax.scatter(x_coords[-1], y_coords[-1], z_coords[-1], 
                          c='green', s=150, marker='s', label='End-Effector')
                
                # Add trajectory trace
                if frame > 0:
                    ee_trace_x = [self.extract_link_positions(
                        self.compute_forward_kinematics_chain(joint_path[i]))[-1][0] 
                        for i in range(frame+1)]
                    ee_trace_y = [self.extract_link_positions(
                        self.compute_forward_kinematics_chain(joint_path[i]))[-1][1] 
                        for i in range(frame+1)]
                    ee_trace_z = [self.extract_link_positions(
                        self.compute_forward_kinematics_chain(joint_path[i]))[-1][2] 
                        for i in range(frame+1)]
                    
                    ax.plot(ee_trace_x, ee_trace_y, ee_trace_z, 
                           'orange', linewidth=1, alpha=0.7)
                
                ax.set_title(f'Robot Motion Animation - Frame {frame+1}/{len(joint_path)}')
                
            return robot_line,
        
        # Create animation
        anim = FuncAnimation(fig, animate, frames=len(joint_path), 
                           interval=100, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
            logger.info(f"Animation saved to {save_path}")
            
        return anim
        
    def plot_trajectory_comparison(self, original_path: JointPath, 
                                  smoothed_path: Optional[JointPath] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison between original and smoothed trajectories.
        
        Args:
            original_path: Original joint path
            smoothed_path: Optional smoothed joint path
            save_path: Optional path to save the plot
            
        Returns:
            Figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Trajectory Comparison: Original vs Smoothed', fontsize=16)
        
        # Convert paths to Cartesian for comparison
        original_cartesian = []
        smoothed_cartesian = []
        
        for q in original_path:
            T = self.robot_controller.forward_kinematics(q)
            original_cartesian.append(T[:3, 3])
            
        if smoothed_path:
            for q in smoothed_path:
                T = self.robot_controller.forward_kinematics(q)
                smoothed_cartesian.append(T[:3, 3])
        
        original_cartesian = np.array(original_cartesian)
        if smoothed_path:
            smoothed_cartesian = np.array(smoothed_cartesian)
        
        # Plot joint angles
        for i in range(min(6, self.robot.n_joints)):
            row = i // 3
            col = i % 3
            
            original_joints = np.array([q[i] for q in original_path])
            axes[row, col].plot(np.degrees(original_joints), 'b-', 
                               label='Original', linewidth=2)
            
            if smoothed_path:
                smoothed_joints = np.array([q[i] for q in smoothed_path])
                axes[row, col].plot(np.degrees(smoothed_joints), 'r--', 
                                   label='Smoothed', linewidth=2)
            
            axes[row, col].set_title(f'Joint {i+1}')
            axes[row, col].set_xlabel('Waypoint')
            axes[row, col].set_ylabel('Angle (degrees)')
            axes[row, col].legend()
            axes[row, col].grid(True)
        
        plt.tight_layout()
        
        # Create 3D comparison plot
        fig_3d = plt.figure(figsize=(12, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        # Plot original path
        ax_3d.plot(original_cartesian[:, 0], original_cartesian[:, 1], 
                  original_cartesian[:, 2], 'b-', linewidth=3, label='Original Path')
        
        if smoothed_path:
            ax_3d.plot(smoothed_cartesian[:, 0], smoothed_cartesian[:, 1], 
                      smoothed_cartesian[:, 2], 'r--', linewidth=3, label='Smoothed Path')
        
        # Mark start and end points
        ax_3d.scatter(original_cartesian[0, 0], original_cartesian[0, 1], 
                     original_cartesian[0, 2], c='green', s=200, marker='o', label='Start')
        ax_3d.scatter(original_cartesian[-1, 0], original_cartesian[-1, 1], 
                     original_cartesian[-1, 2], c='red', s=200, marker='s', label='End')
        
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title('3D Cartesian Path Comparison')
        ax_3d.legend()
        
        if save_path:
            fig.savefig(save_path.replace('.png', '_joints.png'), dpi=300, bbox_inches='tight')
            fig_3d.savefig(save_path.replace('.png', '_3d.png'), dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plots saved to {save_path}")
            
        return fig, fig_3d


def demo_visualization():
    """Demonstrate the visualization capabilities."""
    print("🎨 Robot Visualization Demo")
    
    try:
        # Initialize robot controller
        from robot_controller import RobotController
        robot_controller = RobotController()
        
        # Create visualizer
        visualizer = RobotVisualizer(robot_controller)
        
        # Test configuration
        q_test = np.array([0.5, 0.3, -0.2, 0.0, 0.1, 0.0])
        
        print("Creating matplotlib visualization...")
        fig_mpl = visualizer.plot_robot_configuration_matplotlib(
            q_test, save_path='robot_config_matplotlib.png'
        )
        plt.show()
        
        print("Creating interactive plotly visualization...")
        fig_plotly = visualizer.plot_robot_configuration_plotly(
            q_test, save_path='robot_config_plotly.html'
        )
        fig_plotly.show()
        
        # Create sample joint path for animation
        print("Creating animation...")
        joint_path = []
        for i in range(20):
            factor = i / 19.0
            q_interp = q_test * factor
            joint_path.append(q_interp)
        
        anim = visualizer.animate_joint_path_matplotlib(
            joint_path, save_path='robot_animation.gif'
        )
        plt.show()
        
        print("✅ Visualization demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Visualization demo failed: {e}")


if __name__ == "__main__":
    demo_visualization()
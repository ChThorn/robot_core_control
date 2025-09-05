#!/usr/bin/env python3
"""
Verification script to ensure all visualization functionality is working correctly.
"""

import sys
import os
import numpy as np

# Add parent directories to path
current_dir = os.path.dirname(__file__)
kinematics_dir = os.path.join(current_dir, '..', 'robot_kinematics')
sys.path.append(kinematics_dir)

def test_imports():
    """Test all required imports."""
    print("🔍 Testing imports...")
    try:
        from robot_controller import RobotController
        from visualize import RobotVisualizer
        import matplotlib.pyplot as plt
        import plotly.graph_objects as go
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_initialization():
    """Test robot controller and visualizer initialization."""
    print("\n🔍 Testing initialization...")
    try:
        from robot_controller import RobotController
        from visualize import RobotVisualizer
        
        robot_controller = RobotController()
        visualizer = RobotVisualizer(robot_controller)
        print("✅ Robot controller and visualizer initialized successfully")
        return robot_controller, visualizer
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return None, None

def test_static_visualization(visualizer):
    """Test static matplotlib visualization."""
    print("\n🔍 Testing static visualization...")
    try:
        import matplotlib.pyplot as plt
        
        q_test = np.array([0.2, 0.1, -0.3, 0.1, 0.05, 0.0])
        fig = visualizer.plot_robot_configuration_matplotlib(
            q_test, 
            save_path='verification_static.png'
        )
        plt.close(fig)
        print("✅ Static visualization created successfully")
        return True
    except Exception as e:
        print(f"❌ Static visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interactive_visualization(visualizer):
    """Test interactive plotly visualization."""
    print("\n🔍 Testing interactive visualization...")
    try:
        q_test = np.array([0.2, 0.1, -0.3, 0.1, 0.05, 0.0])
        fig_plotly = visualizer.plot_robot_configuration_plotly(
            q_test, 
            save_path='verification_interactive.html'
        )
        print("✅ Interactive visualization created successfully")
        return True
    except Exception as e:
        print(f"❌ Interactive visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_path_visualization(visualizer):
    """Test path visualization."""
    print("\n🔍 Testing path visualization...")
    try:
        import matplotlib.pyplot as plt
        
        q_test = np.array([0.2, 0.1, -0.3, 0.1, 0.05, 0.0])
        
        # Create sample path
        path = [
            np.array([0.4, 0.2, 0.5]),
            np.array([0.2, 0.3, 0.6]),
            np.array([0.0, 0.4, 0.7]),
            np.array([-0.2, 0.3, 0.6])
        ]
        
        fig = visualizer.plot_robot_configuration_matplotlib(
            q_test,
            cartesian_path=path,
            save_path='verification_with_path.png'
        )
        plt.close(fig)
        print("✅ Path visualization created successfully")
        return True
    except Exception as e:
        print(f"❌ Path visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trajectory_analysis(visualizer):
    """Test trajectory analysis visualization."""
    print("\n🔍 Testing trajectory analysis...")
    try:
        import matplotlib.pyplot as plt
        
        # Create sample trajectory
        q_start = np.zeros(6)
        q_end = np.array([0.5, 0.3, -0.4, 0.2, 0.1, 0.0])
        trajectory = []
        
        for i in range(10):
            alpha = i / 9.0
            q_interp = q_start + alpha * (q_end - q_start)
            trajectory.append(q_interp)
        
        fig_comp, fig_3d = visualizer.plot_trajectory_comparison(
            trajectory,
            save_path='verification_trajectory.png'
        )
        plt.close('all')
        print("✅ Trajectory analysis created successfully")
        return True
    except Exception as e:
        print(f"❌ Trajectory analysis error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("🔍 Comprehensive Visualization System Verification")
    print("=" * 55)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        return
    
    # Test initialization
    robot_controller, visualizer = test_initialization()
    if robot_controller is None or visualizer is None:
        all_passed = False
        return
    
    # Test static visualization
    if not test_static_visualization(visualizer):
        all_passed = False
    
    # Test interactive visualization
    if not test_interactive_visualization(visualizer):
        all_passed = False
    
    # Test path visualization
    if not test_path_visualization(visualizer):
        all_passed = False
    
    # Test trajectory analysis
    if not test_trajectory_analysis(visualizer):
        all_passed = False
    
    print("\n" + "=" * 55)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Visualization system is working correctly.")
        print("\nGenerated verification files:")
        print("  📸 verification_static.png")
        print("  🌐 verification_interactive.html") 
        print("  🛤️  verification_with_path.png")
        print("  📊 verification_trajectory_joints.png")
        print("  📊 verification_trajectory_3d.png")
    else:
        print("❌ SOME TESTS FAILED! There are issues with the visualization system.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

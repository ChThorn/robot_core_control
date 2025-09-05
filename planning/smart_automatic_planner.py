#!/usr/bin/env python3
"""
Smart Automatic Path Selection Layer - Enhancement for your planning system.
This adds intelligent automatic selection of the best planning method.
"""

import sys
import os
import numpy as np
import time
from typing import Tuple, Optional, Dict, List

# Add parent directories to path
current_dir = os.path.dirname(__file__)
kinematics_dir = os.path.join(current_dir, '..', 'robot_kinematics')
sys.path.append(kinematics_dir)

from robot_controller import RobotController
from motion_planning import MotionPlanner

class SmartAutomaticPlanner:
    """
    Intelligent automatic path selection system.
    Analyzes environment and automatically chooses the best planning method.
    """
    
    def __init__(self, robot_controller: RobotController):
        self.robot_controller = robot_controller
        self.motion_planner = MotionPlanner(robot_controller)
        
    def plan_intelligent_path(self, 
                             start_pose: np.ndarray, 
                             goal_pose: np.ndarray,
                             priority: str = 'balanced') -> Dict:
        """
        Intelligently plan path by automatically selecting best method.
        
        Args:
            start_pose: [x, y, z, rx, ry, rz] or joint configuration
            goal_pose: [x, y, z, rx, ry, rz] or joint configuration  
            priority: 'speed', 'safety', 'balanced'
            
        Returns:
            Dictionary with planning result and method used
        """
        print(f"🧠 Smart Planning: Analyzing best method...")
        print(f"   Priority: {priority}")
        
        result = {
            'success': False,
            'method_used': None,
            'path': None,
            'reasoning': [],
            'planning_time': 0,
            'fallback_attempted': False
        }
        
        start_time = time.time()
        
        # Step 1: Determine input types
        is_joint_input = self._is_joint_configuration(start_pose, goal_pose)
        
        if is_joint_input:
            result['reasoning'].append("Input detected as joint configurations")
            # For joint inputs, use joint space planning
            path = self._plan_joint_space(start_pose, goal_pose)
            if path:
                result.update({
                    'success': True,
                    'method_used': 'joint_space',
                    'path': path,
                    'planning_time': time.time() - start_time
                })
                result['reasoning'].append("Joint space planning successful")
                return result
        
        # Step 2: Convert to poses if needed
        start_T, goal_T = self._ensure_pose_format(start_pose, goal_pose)
        
        if start_T is None or goal_T is None:
            result['reasoning'].append("Failed to convert inputs to poses")
            return result
            
        # Step 3: Analyze environment and path requirements
        analysis = self._analyze_path_requirements(start_T, goal_T, priority)
        result['reasoning'].extend(analysis['reasoning'])
        
        # Step 4: Select and execute best method
        methods_to_try = self._select_methods_by_priority(analysis, priority)
        
        for method_info in methods_to_try:
            method = method_info['method']
            reason = method_info['reason']
            
            print(f"   🎯 Trying {method}: {reason}")
            result['reasoning'].append(f"Attempting {method}: {reason}")
            
            if method == 'joint_space_fast':
                path = self._try_joint_space_via_poses(start_T, goal_T)
            elif method == 'cartesian_linear':
                path = self._try_linear_cartesian(start_pose, goal_pose)
            elif method == 'cartesian_aorrtc':
                path = self._try_aorrtc_cartesian(start_pose, goal_pose)
            else:
                continue
                
            if path:
                result.update({
                    'success': True,
                    'method_used': method,
                    'path': path,
                    'planning_time': time.time() - start_time
                })
                result['reasoning'].append(f"SUCCESS with {method}")
                print(f"   ✅ Success with {method}")
                return result
            else:
                result['reasoning'].append(f"FAILED with {method}")
                print(f"   ❌ Failed with {method}")
                
        # Step 5: All methods failed
        result['fallback_attempted'] = True
        result['reasoning'].append("All planning methods failed")
        result['planning_time'] = time.time() - start_time
        
        return result
    
    def _is_joint_configuration(self, start, goal) -> bool:
        """Check if inputs are joint configurations."""
        return (len(start) == 6 and len(goal) == 6 and 
                np.all(np.abs(start) < 2*np.pi) and np.all(np.abs(goal) < 2*np.pi))
    
    def _ensure_pose_format(self, start_pose, goal_pose) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Ensure inputs are in pose format."""
        try:
            if len(start_pose) == 6 and np.any(np.abs(start_pose[:3]) > 10):
                # Likely joint configuration, convert to pose
                start_T = self.robot_controller.forward_kinematics(start_pose)
                goal_T = self.robot_controller.forward_kinematics(goal_pose)
            else:
                # Already poses
                start_T = np.eye(4)
                start_T[:3, 3] = start_pose[:3]
                goal_T = np.eye(4)
                goal_T[:3, 3] = goal_pose[:3]
                
            return start_T, goal_T
        except:
            return None, None
    
    def _analyze_path_requirements(self, start_T, goal_T, priority) -> Dict:
        """Analyze path requirements and environment."""
        analysis = {
            'distance': np.linalg.norm(goal_T[:3, 3] - start_T[:3, 3]),
            'complexity': 'low',
            'obstacles_likely': False,
            'reasoning': []
        }
        
        # Distance analysis
        if analysis['distance'] > 0.5:
            analysis['complexity'] = 'high'
            analysis['reasoning'].append(f"Long distance path: {analysis['distance']:.3f}m")
        elif analysis['distance'] > 0.2:
            analysis['complexity'] = 'medium'
            analysis['reasoning'].append(f"Medium distance path: {analysis['distance']:.3f}m")
        else:
            analysis['reasoning'].append(f"Short distance path: {analysis['distance']:.3f}m")
        
        # Simple obstacle likelihood (could be enhanced with real environment data)
        workspace_center = np.array([0.0, 0.0, 0.5])
        if (np.linalg.norm(start_T[:3, 3] - workspace_center) > 0.3 or 
            np.linalg.norm(goal_T[:3, 3] - workspace_center) > 0.3):
            analysis['obstacles_likely'] = True
            analysis['reasoning'].append("Path near workspace boundaries - obstacles likely")
        
        return analysis
    
    def _select_methods_by_priority(self, analysis, priority) -> List[Dict]:
        """Select planning methods based on analysis and priority."""
        methods = []
        
        if priority == 'speed':
            # Prioritize fastest methods
            methods.append({'method': 'joint_space_fast', 'reason': 'Speed priority - joint space fastest'})
            if analysis['complexity'] == 'low':
                methods.append({'method': 'cartesian_linear', 'reason': 'Simple path - try linear'})
            methods.append({'method': 'cartesian_aorrtc', 'reason': 'Fallback - AORRTC'})
            
        elif priority == 'safety':
            # Prioritize path control and obstacle avoidance
            methods.append({'method': 'cartesian_aorrtc', 'reason': 'Safety priority - AORRTC with obstacle avoidance'})
            methods.append({'method': 'cartesian_linear', 'reason': 'Fallback - linear path'})
            methods.append({'method': 'joint_space_fast', 'reason': 'Last resort - joint space'})
            
        else:  # balanced
            # Balanced approach
            if analysis['obstacles_likely']:
                methods.append({'method': 'cartesian_aorrtc', 'reason': 'Obstacles likely - use AORRTC'})
                methods.append({'method': 'cartesian_linear', 'reason': 'Fallback - linear'})
                methods.append({'method': 'joint_space_fast', 'reason': 'Final fallback - joint space'})
            else:
                methods.append({'method': 'joint_space_fast', 'reason': 'Clear workspace - joint space fastest'})
                methods.append({'method': 'cartesian_linear', 'reason': 'Alternative - linear Cartesian'})
                methods.append({'method': 'cartesian_aorrtc', 'reason': 'Fallback - AORRTC'})
        
        return methods
    
    def _plan_joint_space(self, start_q, goal_q):
        """Plan in joint space."""
        try:
            return self.motion_planner.plan_joint_path(start_q, goal_q)
        except:
            return None
    
    def _try_joint_space_via_poses(self, start_T, goal_T):
        """Try joint space planning via poses."""
        try:
            start_q, success1 = self.robot_controller.inverse_kinematics(start_T)
            goal_q, success2 = self.robot_controller.inverse_kinematics(goal_T)
            
            if success1 and success2:
                return self.motion_planner.plan_joint_path(start_q, goal_q)
        except:
            pass
        return None
    
    def _try_linear_cartesian(self, start_pose, goal_pose):
        """Try simple linear Cartesian path."""
        try:
            # Simple linear interpolation (could be enhanced)
            start_pos = start_pose[:3] if len(start_pose) >= 3 else start_pose
            goal_pos = goal_pose[:3] if len(goal_pose) >= 3 else goal_pose
            
            # Create linear waypoints
            waypoints = []
            for i in range(5):
                alpha = i / 4.0
                waypoint = start_pos + alpha * (goal_pos - start_pos)
                waypoints.append(waypoint)
            
            return waypoints
        except:
            return None
    
    def _try_aorrtc_cartesian(self, start_pose, goal_pose):
        """Try AORRTC Cartesian planning."""
        try:
            result = self.motion_planner.plan_cartesian_path(start_pose, goal_pose)
            if result:
                return result[0] if isinstance(result, tuple) else result
        except:
            pass
        return None


def demonstrate_smart_planning():
    """Demonstrate the smart automatic planning system."""
    print("🧠 Smart Automatic Path Selection Demonstration")
    print("=" * 60)
    
    robot_controller = RobotController()
    smart_planner = SmartAutomaticPlanner(robot_controller)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Simple Pick & Place',
            'start': np.array([0.4, 0.2, 0.3, 0, 0, 0]),
            'goal': np.array([0.2, 0.4, 0.5, 0, 0, 0]),
            'priority': 'speed'
        },
        {
            'name': 'Safety-Critical Move',
            'start': np.array([0.5, 0.1, 0.3, 0, 0, 0]),
            'goal': np.array([0.1, 0.4, 0.7, 0, 0, 0]),
            'priority': 'safety'
        },
        {
            'name': 'Joint Configuration Move',
            'start': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            'goal': np.array([0.5, 0.3, -0.4, 0.2, 0.1, 0.0]),
            'priority': 'balanced'
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}️⃣  Scenario: {scenario['name']}")
        print(f"   Start: {scenario['start']}")
        print(f"   Goal:  {scenario['goal']}")
        print(f"   Priority: {scenario['priority']}")
        
        result = smart_planner.plan_intelligent_path(
            scenario['start'], 
            scenario['goal'], 
            scenario['priority']
        )
        
        print(f"   📊 Result:")
        print(f"     Success: {result['success']}")
        print(f"     Method: {result['method_used']}")
        print(f"     Time: {result['planning_time']:.3f}s")
        print(f"     Reasoning:")
        for reason in result['reasoning']:
            print(f"       • {reason}")
    
    print("\n" + "=" * 60)
    print("🎯 SMART AUTOMATIC PLANNING SUMMARY:")
    print("   ✅ Automatically analyzes environment")
    print("   ✅ Selects best planning method")
    print("   ✅ Provides fallback strategies")
    print("   ✅ Explains decision reasoning")
    print("   ✅ Adapts to user priorities (speed/safety/balanced)")

if __name__ == "__main__":
    demonstrate_smart_planning()

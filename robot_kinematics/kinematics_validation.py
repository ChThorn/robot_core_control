#!/usr/bin/env python3
"""
Enhanced kinematics validation and testing utilities.
"""

import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

class KinematicsValidator:
    """Enhanced validation utilities for robot kinematics."""
    
    def __init__(self, robot_kinematics):
        self.robot = robot_kinematics
        
    def verify_screw_axes(self) -> Dict[str, any]:
        """Verify screw axes computation against URDF data."""
        
        # URDF joint data (from rb3_730es_u.urdf)
        urdf_joints = [
            {'name': 'base', 'axis': [0,0,1], 'origin': [0,0,0.1453]},
            {'name': 'shoulder', 'axis': [0,1,0], 'origin': [0,0,0]},
            {'name': 'elbow', 'axis': [0,1,0], 'origin': [0,-0.00645,0.286]},
            {'name': 'wrist1', 'axis': [0,0,1], 'origin': [0,0,0]},
            {'name': 'wrist2', 'axis': [0,1,0], 'origin': [0,0,0.344]},
            {'name': 'wrist3', 'axis': [0,0,1], 'origin': [0,0,0]}
        ]
        
        # Compute expected screw axes
        expected_S = np.zeros((6, 6))
        T_cumulative = np.eye(4)
        
        for i, joint in enumerate(urdf_joints):
            # Transform to joint frame
            joint_origin = joint['origin']
            T_joint = np.eye(4)
            T_joint[:3, 3] = joint_origin
            T_cumulative = T_cumulative @ T_joint
            
            # Joint axis in base frame
            omega = np.array(joint['axis'])
            
            # Position of joint in base frame  
            p = T_cumulative[:3, 3]
            
            # Screw axis: [omega, p × omega]
            v = np.cross(p, omega)
            
            expected_S[:, i] = np.hstack([omega, v])
        
        # Compare with implemented screw axes
        S_diff = np.abs(self.robot.S - expected_S)
        max_diff = np.max(S_diff)
        
        results = {
            'expected_S': expected_S,
            'implemented_S': self.robot.S,
            'max_difference': max_diff,
            'is_valid': max_diff < 1e-6,
            'differences': S_diff
        }
        
        return results
    
    def test_fk_ik_consistency(self, num_tests: int = 100) -> Dict[str, any]:
        """Test FK-IK consistency with random configurations."""
        
        position_errors = []
        rotation_errors = []
        success_count = 0
        
        limits_lower, limits_upper = self.robot.joint_limits[0], self.robot.joint_limits[1]
        
        for _ in range(num_tests):
            # Generate random joint configuration
            q_test = np.random.uniform(limits_lower, limits_upper)
            
            # Forward kinematics
            T_target = self.robot.forward_kinematics(q_test)
            
            # Inverse kinematics
            q_solution, converged = self.robot.inverse_kinematics(T_target, q_init=q_test)
            
            if converged and q_solution is not None:
                success_count += 1
                
                # Verify solution
                T_check = self.robot.forward_kinematics(q_solution)
                pos_err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
                
                # Rotation error
                R_err = T_check[:3, :3].T @ T_target[:3, :3]
                rot_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0))
                
                position_errors.append(pos_err)
                rotation_errors.append(rot_err)
        
        return {
            'success_rate': success_count / num_tests,
            'mean_pos_error': np.mean(position_errors) if position_errors else float('inf'),
            'max_pos_error': np.max(position_errors) if position_errors else float('inf'),
            'mean_rot_error': np.mean(rotation_errors) if rotation_errors else float('inf'),
            'max_rot_error': np.max(rotation_errors) if rotation_errors else float('inf'),
            'position_errors': position_errors,
            'rotation_errors': rotation_errors
        }
    
    def analyze_workspace_coverage(self, num_samples: int = 1000) -> Dict[str, any]:
        """Analyze reachable workspace coverage."""
        
        reachable_positions = []
        limits_lower, limits_upper = self.robot.joint_limits[0], self.robot.joint_limits[1]
        
        for _ in range(num_samples):
            q = np.random.uniform(limits_lower, limits_upper)
            T = self.robot.forward_kinematics(q)
            pos = T[:3, 3]
            
            # Check workspace constraints
            if self.robot._check_workspace(pos):
                reachable_positions.append(pos)
        
        reachable_positions = np.array(reachable_positions)
        
        if len(reachable_positions) > 0:
            workspace_bounds = {
                'x_range': [np.min(reachable_positions[:, 0]), np.max(reachable_positions[:, 0])],
                'y_range': [np.min(reachable_positions[:, 1]), np.max(reachable_positions[:, 1])], 
                'z_range': [np.min(reachable_positions[:, 2]), np.max(reachable_positions[:, 2])],
                'volume_estimate': self._estimate_workspace_volume(reachable_positions)
            }
        else:
            workspace_bounds = None
            
        return {
            'reachable_positions': reachable_positions,
            'workspace_bounds': workspace_bounds,
            'coverage_percentage': len(reachable_positions) / num_samples * 100
        }
    
    def _estimate_workspace_volume(self, positions: np.ndarray) -> float:
        """Rough workspace volume estimate using bounding box."""
        if len(positions) == 0:
            return 0.0
            
        ranges = np.ptp(positions, axis=0)  # peak-to-peak range
        return np.prod(ranges)
    
    def plot_validation_results(self, fk_ik_results: Dict, save_path: str = None):
        """Create validation plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Position error histogram
        axes[0, 0].hist(fk_ik_results['position_errors'], bins=30, alpha=0.7)
        axes[0, 0].set_xlabel('Position Error (m)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Position Error Distribution')
        axes[0, 0].axvline(np.mean(fk_ik_results['position_errors']), 
                          color='red', linestyle='--', label='Mean')
        axes[0, 0].legend()
        
        # Rotation error histogram
        axes[0, 1].hist(np.rad2deg(fk_ik_results['rotation_errors']), bins=30, alpha=0.7)
        axes[0, 1].set_xlabel('Rotation Error (degrees)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Rotation Error Distribution')
        axes[0, 1].axvline(np.rad2deg(np.mean(fk_ik_results['rotation_errors'])), 
                          color='red', linestyle='--', label='Mean')
        axes[0, 1].legend()
        
        # Error correlation
        axes[1, 0].scatter(fk_ik_results['position_errors'], 
                          np.rad2deg(fk_ik_results['rotation_errors']), alpha=0.6)
        axes[1, 0].set_xlabel('Position Error (m)')
        axes[1, 0].set_ylabel('Rotation Error (degrees)')
        axes[1, 0].set_title('Position vs Rotation Error')
        
        # Summary statistics
        stats_text = f"""
        Success Rate: {fk_ik_results['success_rate']:.1%}
        Mean Pos Error: {fk_ik_results['mean_pos_error']*1000:.2f} mm
        Max Pos Error: {fk_ik_results['max_pos_error']*1000:.2f} mm
        Mean Rot Error: {np.rad2deg(fk_ik_results['mean_rot_error']):.3f}°
        Max Rot Error: {np.rad2deg(fk_ik_results['max_rot_error']):.3f}°
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center')
        axes[1, 1].set_title('Validation Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

# Usage example
def run_comprehensive_validation(robot_controller):
    """Run comprehensive validation tests."""
    
    validator = KinematicsValidator(robot_controller.robot)
    
    print("=== Comprehensive Kinematics Validation ===\n")
    
    # 1. Verify screw axes
    print("1. Screw Axes Verification:")
    screw_results = validator.verify_screw_axes()
    if screw_results['is_valid']:
        print("   ✅ Screw axes are mathematically correct")
    else:
        print(f"   ⚠️  Screw axes may have issues (max diff: {screw_results['max_difference']:.6f})")
        print("   Expected S matrix:")
        print(screw_results['expected_S'])
        print("   Implemented S matrix:")
        print(screw_results['implemented_S'])
    
    # 2. FK-IK consistency
    print("\n2. FK-IK Consistency Test:")
    fk_ik_results = validator.test_fk_ik_consistency(num_tests=200)
    print(f"   Success rate: {fk_ik_results['success_rate']:.1%}")
    print(f"   Mean position error: {fk_ik_results['mean_pos_error']*1000:.3f} mm")
    print(f"   Mean rotation error: {np.rad2deg(fk_ik_results['mean_rot_error']):.3f}°")
    
    # 3. Workspace analysis
    print("\n3. Workspace Coverage Analysis:")
    workspace_results = validator.analyze_workspace_coverage(num_samples=500)
    print(f"   Reachable positions: {len(workspace_results['reachable_positions'])}/500")
    print(f"   Coverage: {workspace_results['coverage_percentage']:.1f}%")
    
    if workspace_results['workspace_bounds']:
        bounds = workspace_results['workspace_bounds']
        print(f"   X range: [{bounds['x_range'][0]:.3f}, {bounds['x_range'][1]:.3f}] m")
        print(f"   Y range: [{bounds['y_range'][0]:.3f}, {bounds['y_range'][1]:.3f}] m") 
        print(f"   Z range: [{bounds['z_range'][0]:.3f}, {bounds['z_range'][1]:.3f}] m")
    
    # 4. Generate validation plots
    try:
        validator.plot_validation_results(fk_ik_results, 'kinematics_validation.png')
        print("\n   📊 Validation plots saved to 'kinematics_validation.png'")
    except ImportError:
        print("\n   Note: matplotlib not available for plotting")
    
    return {
        'screw_axes': screw_results,
        'fk_ik_consistency': fk_ik_results,
        'workspace_coverage': workspace_results
    }

#!/usr/bin/env python3
"""
Production robot controller interface with proper unit handling and orientation fixes.
"""

import numpy as np
import logging
import time
import json
from typing import Dict, Any, Optional, Tuple
from robot_kinematics import RobotKinematics, RobotKinematicsError

logger = logging.getLogger(__name__)

class RobotController:
    """Production robot controller interface."""
    
    def __init__(self, urdf_path: str, ee_link: str = "tcp", base_link: str = "link0"):
        """Initialize robot controller."""
        try:
            self.robot = RobotKinematics(urdf_path, ee_link, base_link)
        except RobotKinematicsError as e:
            logger.error(f"Failed to initialize robot kinematics: {e}")
            raise
        
        # Performance tracking
        self.performance_stats = {
            'fk_calls': 0, 'ik_calls': 0, 'ik_successes': 0,
            'avg_ik_time': 0.0, 'max_position_error': 0.0, 'max_rotation_error': 0.0
        }
        
        # IK parameters
        self.ik_params = {
            'pos_tol': 1e-6, 'rot_tol': 1e-6, 'max_iters': 300,
            'damping': 1e-2, 'step_scale': 0.5, 'dq_max': 0.2, 'num_attempts': 10
        }
    
    def set_ik_parameters(self, **params):
        """Update IK parameters."""
        self.ik_params.update(params)
        logger.info(f"Updated IK parameters: {params}")
    
    def convert_from_robot_units(self, joint_positions_deg: np.ndarray, 
                                tcp_position_mm_rpy_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert robot data units to standard SI units.
        
        Args:
            joint_positions_deg: Joint positions in degrees
            tcp_position_mm_rpy_deg: TCP position [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
            
        Returns:
            q_rad: Joint positions in radians
            T: Homogeneous transformation matrix
        """
        # Convert joint positions from degrees to radians
        q_rad = np.deg2rad(joint_positions_deg)
        
        # Convert TCP position from mm to m and orientation from degrees to radians
        tcp_pos_m = tcp_position_mm_rpy_deg[:3] / 1000.0
        tcp_rpy_rad = np.deg2rad(tcp_position_mm_rpy_deg[3:])
        
        # Create homogeneous transformation matrix
        R = self.robot.rpy_to_matrix(tcp_rpy_rad)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tcp_pos_m
        
        return q_rad, T
    
    def convert_to_robot_units(self, q_rad: np.ndarray) -> np.ndarray:
        """Convert joint angles from radians to degrees for robot controller."""
        return np.rad2deg(q_rad)
    
    def forward_kinematics(self, q_rad: np.ndarray) -> np.ndarray:
        """Compute forward kinematics with performance tracking."""
        try:
            start_time = time.time()
            T = self.robot.forward_kinematics(q_rad)
            computation_time = time.time() - start_time
            
            self.performance_stats['fk_calls'] += 1
            logger.debug(f"FK computed in {computation_time:.6f}s")
            
            return T
        except Exception as e:
            logger.error(f"Forward kinematics failed: {e}")
            raise
    
    def inverse_kinematics(self, T_des: np.ndarray, 
                          q_init: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], bool]:
        """Solve inverse kinematics with performance tracking."""
        try:
            start_time = time.time()
            
            q_sol, converged = self.robot.inverse_kinematics(
                T_des, q_init=q_init, **self.ik_params
            )
            
            computation_time = time.time() - start_time
            
            # Update performance statistics
            self.performance_stats['ik_calls'] += 1
            if converged:
                self.performance_stats['ik_successes'] += 1
                
                # Calculate errors for monitoring
                pos_err, rot_err = self.robot.check_pose_error(T_des, q_sol)
                self.performance_stats['max_position_error'] = max(
                    self.performance_stats['max_position_error'], pos_err
                )
                self.performance_stats['max_rotation_error'] = max(
                    self.performance_stats['max_rotation_error'], rot_err
                )
                
                logger.info(f"IK solved in {computation_time:.4f}s - "
                           f"Pos err: {pos_err:.6e}m, Rot err: {rot_err:.6e}rad")
            else:
                logger.warning(f"IK failed to converge after {computation_time:.4f}s")
            
            # Update average IK time
            total_calls = self.performance_stats['ik_calls']
            self.performance_stats['avg_ik_time'] = (
                (self.performance_stats['avg_ik_time'] * (total_calls - 1) + computation_time) / total_calls
            )
            
            return q_sol, converged
            
        except Exception as e:
            logger.error(f"Inverse kinematics failed: {e}")
            return None, False
    
    def validate_against_real_data(self, json_path: str, num_samples: int = 10) -> Dict[str, Any]:
        """Validate kinematics against real robot data."""
        logger.info(f"Validating kinematics against real data: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load robot data: {e}")
            return {'error': str(e)}
        
        waypoints = data['waypoints']
        indices = np.linspace(0, len(waypoints)-1, num_samples, dtype=int)
        
        results = {
            'num_samples': num_samples,
            'position_errors': [],
            'rotation_errors': [],
            'validation_success': True
        }
        
        for i in indices:
            wp = waypoints[i]
            
            try:
                # Convert units and compute FK
                q_deg = np.array(wp['joint_positions'])
                tcp_recorded = np.array(wp['tcp_position'])
                
                q_rad, T_recorded = self.convert_from_robot_units(q_deg, tcp_recorded)
                T_fk = self.forward_kinematics(q_rad)
                
                # Calculate errors
                pos_err = np.linalg.norm(T_fk[:3, 3] - T_recorded[:3, 3])
                
                # Calculate rotation error using rotation matrix comparison
                R_fk = T_fk[:3, :3]
                R_recorded = T_recorded[:3, :3]
                R_err = R_fk.T @ R_recorded
                cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
                rot_err = np.arccos(cos_angle)
                
                results['position_errors'].append(pos_err)
                results['rotation_errors'].append(rot_err)
                
            except Exception as e:
                logger.warning(f"Error processing waypoint {i}: {e}")
                results['validation_success'] = False
        
        if results['position_errors']:
            results['mean_position_error'] = np.mean(results['position_errors'])
            results['max_position_error'] = np.max(results['position_errors'])
            results['mean_rotation_error'] = np.mean(results['rotation_errors'])
            results['max_rotation_error'] = np.max(results['rotation_errors'])
            
            logger.info(f"Validation complete - Mean pos error: {results['mean_position_error']:.6f}m, "
                       f"Mean rot error: {results['mean_rotation_error']:.6f}rad")
        
        return results
    
    def send_to_robot(self, q_rad: np.ndarray) -> bool:
        """
        Send joint angles to robot controller.
        
        Note: This is a placeholder. Implement actual robot communication here.
        """
        try:
            q_deg = self.convert_to_robot_units(q_rad)
            
            # Validate joint limits
            limits_deg = np.rad2deg(self.robot.joint_limits)
            if np.any(q_deg < limits_deg[0]) or np.any(q_deg > limits_deg[1]):
                logger.error("Joint angles exceed limits")
                return False
            
            # TODO: Implement actual robot communication here
            # Example: send_command_to_robot(q_deg)
            
            logger.info(f"Would send to robot: {q_deg}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send command to robot: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        if stats['ik_calls'] > 0:
            stats['ik_success_rate'] = stats['ik_successes'] / stats['ik_calls']
        else:
            stats['ik_success_rate'] = 0.0
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'fk_calls': 0, 'ik_calls': 0, 'ik_successes': 0,
            'avg_ik_time': 0.0, 'max_position_error': 0.0, 'max_rotation_error': 0.0
        }


#!/usr/bin/env python3
"""
Robot controller with pure analytical inverse kinematics.
Provides fast, closed-form IK solutions for industrial robotics.
"""

import numpy as np
import logging
import time
import json
from typing import Dict, Any, Optional, Tuple, List
from analytical_ik import AnalyticalIK

logger = logging.getLogger(__name__)

class RobotController:
    """Robot controller with analytical IK capabilities."""
    
    def __init__(self, analytical_ik: AnalyticalIK):
        """Initialize robot controller with analytical IK."""
        self.analytical_ik = analytical_ik
        
        # Performance tracking
        self.performance_stats = {
            'fk_calls': 0,
            'ik_calls': 0, 
            'ik_successes': 0,
            'avg_ik_time': 0.0,
            'max_position_error': 0.0,
            'max_rotation_error': 0.0,
            'total_solutions_found': 0
        }
        
        logger.info("Analytical robot controller initialized")
    
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
        R = self.rpy_to_matrix(tcp_rpy_rad)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tcp_pos_m
        
        return q_rad, T
    
    def convert_to_robot_units(self, q_rad: np.ndarray) -> np.ndarray:
        """Convert joint angles from radians to degrees for robot controller."""
        return np.rad2deg(q_rad)
    
    def forward_kinematics(self, q_rad: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics using DH parameters.
        
        Args:
            q_rad: Joint angles in radians
            
        Returns:
            T: 4x4 homogeneous transformation matrix
        """
        start_time = time.time()
        
        try:
            q1, q2, q3, q4, q5, q6 = q_rad
            
            # DH parameters
            d1, a3, d3, d4, d6 = self.analytical_ik.d1, self.analytical_ik.a3, self.analytical_ik.d3, self.analytical_ik.d4, self.analytical_ik.d6
            
            # Individual transformation matrices
            c1, s1 = np.cos(q1), np.sin(q1)
            c2, s2 = np.cos(q2), np.sin(q2)
            c3, s3 = np.cos(q3), np.sin(q3)
            c4, s4 = np.cos(q4), np.sin(q4)
            c5, s5 = np.cos(q5), np.sin(q5)
            c6, s6 = np.cos(q6), np.sin(q6)
            
            # T01
            T01 = np.array([
                [c1, -s1, 0, 0],
                [s1,  c1, 0, 0],
                [0,   0,  1, d1],
                [0,   0,  0, 1]
            ])
            
            # T12
            T12 = np.array([
                [c2, 0,  s2, 0],
                [0,  1,  0,  0],
                [-s2, 0, c2, 0],
                [0,  0,  0,  1]
            ])
            
            # T23
            T23 = np.array([
                [c3, -s3, 0, a3],
                [s3,  c3, 0, 0],
                [0,   0,  1, d3],
                [0,   0,  0, 1]
            ])
            
            # T34
            T34 = np.array([
                [c4, 0,  s4, 0],
                [0,  1,  0,  0],
                [-s4, 0, c4, d4],
                [0,  0,  0,  1]
            ])
            
            # T45
            T45 = np.array([
                [c5, 0, -s5, 0],
                [0,  1,  0,  0],
                [s5, 0,  c5, 0],
                [0,  0,  0,  1]
            ])
            
            # T56
            T56 = np.array([
                [c6, -s6, 0, 0],
                [s6,  c6, 0, 0],
                [0,   0,  1, d6],
                [0,   0,  0, 1]
            ])
            
            # Complete transformation
            T = T01 @ T12 @ T23 @ T34 @ T45 @ T56
            
            computation_time = time.time() - start_time
            self.performance_stats['fk_calls'] += 1
            logger.debug(f"FK computed in {computation_time:.6f}s")
            
            return T
            
        except Exception as e:
            logger.error(f"Forward kinematics failed: {e}")
            raise
    
    def inverse_kinematics(self, T_des: np.ndarray, 
                          q_current: Optional[np.ndarray] = None,
                          return_all_solutions: bool = False) -> Tuple[Optional[np.ndarray], bool, List[np.ndarray]]:
        """
        Solve inverse kinematics using analytical method.
        
        Args:
            T_des: Desired 4x4 homogeneous transformation matrix
            q_current: Current joint configuration for solution selection
            return_all_solutions: Whether to return all solutions
            
        Returns:
            q_solution: Best joint configuration (None if failed)
            converged: Success flag
            all_solutions: List of all valid solutions (if requested)
        """
        try:
            start_time = time.time()
            
            # Solve analytical IK
            solutions, success = self.analytical_ik.inverse_kinematics(T_des)
            
            computation_time = time.time() - start_time
            
            # Update performance statistics
            self.performance_stats['ik_calls'] += 1
            
            if success:
                self.performance_stats['ik_successes'] += 1
                self.performance_stats['total_solutions_found'] += len(solutions)
                
                # Select best solution
                q_solution = self.analytical_ik.select_best_solution(solutions, q_current)
                
                # Calculate errors for monitoring
                pos_err, rot_err = self.check_pose_error(T_des, q_solution)
                self.performance_stats['max_position_error'] = max(
                    self.performance_stats['max_position_error'], pos_err
                )
                self.performance_stats['max_rotation_error'] = max(
                    self.performance_stats['max_rotation_error'], rot_err
                )
                
                logger.info(f"Analytical IK solved in {computation_time:.4f}s - "
                           f"Found {len(solutions)} solutions, "
                           f"Pos err: {pos_err:.6e}m, Rot err: {rot_err:.6e}rad")
            else:
                q_solution = None
                logger.warning(f"Analytical IK failed after {computation_time:.4f}s")
            
            # Update average IK time
            total_calls = self.performance_stats['ik_calls']
            self.performance_stats['avg_ik_time'] = (
                (self.performance_stats['avg_ik_time'] * (total_calls - 1) + computation_time) / total_calls
            )
            
            if return_all_solutions:
                return q_solution, success, solutions
            else:
                return q_solution, success, []
                
        except Exception as e:
            logger.error(f"Inverse kinematics failed: {e}")
            return None, False, []
    
    def get_all_solutions(self, T_des: np.ndarray, 
                         q_current: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Get all analytical IK solutions for a pose.
        
        Args:
            T_des: Desired pose
            q_current: Current configuration for sorting
            
        Returns:
            List of all valid joint configurations
        """
        solutions, success = self.analytical_ik.inverse_kinematics(T_des)
        
        if success and q_current is not None:
            # Sort by proximity to current configuration
            solutions.sort(key=lambda q: np.linalg.norm(q - q_current))
        
        return solutions if success else []
    
    def check_pose_error(self, T_des: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
        """Calculate position and orientation error."""
        T_actual = self.forward_kinematics(q)
        
        # Position error
        pos_err = np.linalg.norm(T_actual[:3, 3] - T_des[:3, 3])
        
        # Rotation error
        R_actual = T_actual[:3, :3]
        R_desired = T_des[:3, :3]
        R_err = R_actual.T @ R_desired
        cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
        rot_err = np.arccos(cos_angle)
        
        return pos_err, rot_err
    
    def validate_against_real_data(self, json_path: str, num_samples: int = 10) -> Dict[str, Any]:
        """Validate analytical IK against real robot data."""
        logger.info(f"Validating analytical IK against real data: {json_path}")
        
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
            'ik_successes': 0,
            'total_solutions': 0,
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
                
                # Test IK on the FK result
                q_ik, ik_success, all_solutions = self.inverse_kinematics(
                    T_fk, q_current=q_rad, return_all_solutions=True
                )
                
                if ik_success:
                    results['ik_successes'] += 1
                    results['total_solutions'] += len(all_solutions)
                    
                    # Check accuracy
                    T_check = self.forward_kinematics(q_ik)
                    pos_err = np.linalg.norm(T_check[:3, 3] - T_fk[:3, 3])
                    rot_err = self._rotation_error(T_check[:3, :3], T_fk[:3, :3])
                    
                    results['position_errors'].append(pos_err)
                    results['rotation_errors'].append(rot_err)
                
            except Exception as e:
                logger.warning(f"Error processing waypoint {i}: {e}")
                results['validation_success'] = False
        
        # Calculate statistics
        if results['position_errors']:
            results['mean_position_error'] = np.mean(results['position_errors'])
            results['max_position_error'] = np.max(results['position_errors'])
            results['mean_rotation_error'] = np.mean(results['rotation_errors'])
            results['max_rotation_error'] = np.max(results['rotation_errors'])
            results['ik_success_rate'] = results['ik_successes'] / num_samples
            results['avg_solutions_per_pose'] = results['total_solutions'] / max(results['ik_successes'], 1)
            
            logger.info(f"Validation complete - IK success rate: {results['ik_success_rate']:.1%}, "
                       f"Avg solutions: {results['avg_solutions_per_pose']:.1f}")
        
        return results
    
    def _rotation_error(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """Calculate rotation error between two rotation matrices."""
        R_err = R1.T @ R2
        cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
        return np.arccos(cos_angle)
    
    def send_to_robot(self, q_rad: np.ndarray) -> bool:
        """Send joint angles to robot controller."""
        try:
            q_deg = self.convert_to_robot_units(q_rad)
            
            # Validate joint limits
            if not self.analytical_ik._within_joint_limits(q_rad):
                logger.error("Joint angles exceed limits")
                return False
            
            # TODO: Implement actual robot communication here
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
            stats['avg_solutions_per_call'] = stats['total_solutions_found'] / stats['ik_calls']
        else:
            stats['ik_success_rate'] = 0.0
            stats['avg_solutions_per_call'] = 0.0
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'fk_calls': 0,
            'ik_calls': 0,
            'ik_successes': 0,
            'avg_ik_time': 0.0,
            'max_position_error': 0.0,
            'max_rotation_error': 0.0,
            'total_solutions_found': 0
        }
    
    @staticmethod
    def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
        """Convert RPY angles to rotation matrix (XYZ convention)."""
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        
        return Rz @ Ry @ Rx
    
    @staticmethod
    def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to RPY angles."""
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
            
        return np.array([x, y, z])


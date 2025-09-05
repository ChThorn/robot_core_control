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
from config import get_config

logger = logging.getLogger(__name__)

class RobotController:
    """Production robot controller interface."""
    
    def __init__(self, ee_link: str = "tcp", base_link: str = "link0", 
                 config_file: str = None):
        """Initialize robot controller with configuration management."""
        try:
            # Load configuration
            from config import get_config, set_config_file
            if config_file:
                set_config_file(config_file)
            
            self.config = get_config()
            
            # Set random seed for reproducible behavior if configured
            random_seed = self.config.get('performance', 'random_seed')
            if random_seed is not None:
                np.random.seed(random_seed)
                
        except ImportError:
            # Fallback to hardcoded values if config system not available
            logger.warning("Configuration system not available, using defaults")
            self.config = None
        
        try:
            # The urdf_path argument is no longer needed
            self.robot = RobotKinematics(ee_link, base_link)
        except RobotKinematicsError as e:
            logger.error(f"Failed to initialize robot kinematics: {e}")
            raise
        
        # Performance tracking
        self.performance_stats = {
            'fk_calls': 0, 'ik_calls': 0, 'ik_successes': 0,
            'avg_ik_time': 0.0, 'max_position_error': 0.0, 'max_rotation_error': 0.0
        }
        
        # Load IK parameters from configuration or use defaults
        if self.config:
            ik_config = self.config.get('ik')
            self.ik_params = {
                'pos_tol': ik_config.get('position_tolerance', 2e-3),
                'rot_tol': ik_config.get('rotation_tolerance', 5e-3),
                'max_iters': ik_config.get('max_iterations', 300),
                'damping': ik_config.get('damping', 5e-4),
                'step_scale': ik_config.get('step_scale', 0.3),
                'dq_max': ik_config.get('max_step_size', 0.3),
                'num_attempts': ik_config.get('max_attempts', 30)
            }
        else:
            # Fallback defaults
            self.ik_params = {
                'pos_tol': 2e-3, 'rot_tol': 5e-3, 'max_iters': 300,
                'damping': 5e-4, 'step_scale': 0.3, 'dq_max': 0.3, 'num_attempts': 30
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
        q_rad = np.deg2rad(joint_positions_deg)
        tcp_pos_m = tcp_position_mm_rpy_deg[:3] / 1000.0
        tcp_rpy_rad = np.deg2rad(tcp_position_mm_rpy_deg[3:])
        
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
        """Solve inverse kinematics with performance tracking and model error tolerance."""
        try:
            start_time = time.time()
            
            T_home = self.robot.forward_kinematics(np.zeros(self.robot.n_joints))
            home_pos_err = np.linalg.norm(T_des[:3, 3] - T_home[:3, 3])
            home_rot_err = np.arccos(np.clip((np.trace(T_des[:3, :3].T @ T_home[:3, :3]) - 1) / 2.0, -1.0, 1.0))
            
            if home_pos_err < self.ik_params['pos_tol'] and home_rot_err < self.ik_params['rot_tol']:
                logger.info("Target is home position, returning zero joints")
                return np.zeros(self.robot.n_joints), True
            
            limits_lower, limits_upper = self.robot.joint_limits[0], self.robot.joint_limits[1]
            
            best_q = None
            best_error = float('inf')
            best_converged = False
            
            attempts_configs = []
            
            if q_init is not None:
                attempts_configs.append(q_init)
            
            attempts_configs.extend([
                np.zeros(self.robot.n_joints),
                (limits_lower + limits_upper) / 2,
            ])
            
            for _ in range(self.ik_params['num_attempts'] - len(attempts_configs)):
                if len(attempts_configs) % 3 == 0:
                    q_rand = np.random.uniform(limits_lower, limits_upper)
                elif len(attempts_configs) % 3 == 1:
                    mid = (limits_lower + limits_upper) / 2
                    std = (limits_upper - limits_lower) / 6
                    q_rand = np.random.normal(mid, std)
                    q_rand = np.clip(q_rand, limits_lower, limits_upper)
                else:
                    base = q_init if q_init is not None else np.zeros(self.robot.n_joints)
                    q_rand = base + np.random.normal(0, 0.2, self.robot.n_joints)
                    q_rand = np.clip(q_rand, limits_lower, limits_upper)
                
                attempts_configs.append(q_rand)
            
            for i, q0 in enumerate(attempts_configs):
                q_sol, converged = self.robot._ik_dls(
                    T_des, q0, **{k: v for k, v in self.ik_params.items() if k != 'num_attempts'}
                )
                
                if q_sol is not None:
                    T_check = self.robot.forward_kinematics(q_sol)
                    pos_err, rot_err = self.robot.check_pose_error(T_des, q_sol)
                    total_err = pos_err + rot_err
                    
                    if total_err < best_error:
                        best_error = total_err
                        best_q = q_sol.copy()
                        best_converged = converged
                        
                        if converged:
                            logger.debug(f"IK converged on attempt {i+1}")
                            break
                            
                        combined_err = pos_err + rot_err * 0.1
                        if combined_err < 3e-3:
                            logger.debug(f"IK found acceptable solution on attempt {i+1} "
                                       f"(pos_err={pos_err*1000:.1f}mm, rot_err={np.rad2deg(rot_err):.2f}deg, "
                                       f"combined={combined_err*1000:.1f}mm)")
                            best_converged = True
                            break
                            
                        if combined_err < 4e-3:
                            logger.debug(f"IK found very good solution on attempt {i+1}")
                            best_converged = True
                            break
            
            computation_time = time.time() - start_time
            
            self.performance_stats['ik_calls'] += 1
            if best_converged:
                self.performance_stats['ik_successes'] += 1
                
                if best_q is not None:
                    pos_err, rot_err = self.robot.check_pose_error(T_des, best_q)
                    self.performance_stats['max_position_error'] = max(
                        self.performance_stats['max_position_error'], pos_err
                    )
                    self.performance_stats['max_rotation_error'] = max(
                        self.performance_stats['max_rotation_error'], rot_err
                    )
                    
                    logger.info(f"IK solved in {computation_time:.4f}s - "
                               f"Pos err: {pos_err:.6e}m, Rot err: {rot_err:.6e}rad")
            else:
                if best_q is not None:
                    pos_err, rot_err = self.robot.check_pose_error(T_des, best_q)
                    logger.warning(f"IK found best solution in {computation_time:.4f}s - "
                                 f"Pos err: {pos_err:.6e}m, Rot err: {rot_err:.6e}rad (not converged)")
                else:
                    logger.warning(f"IK completely failed after {computation_time:.4f}s")
            
            total_calls = self.performance_stats['ik_calls']
            self.performance_stats['avg_ik_time'] = (
                (self.performance_stats['avg_ik_time'] * (total_calls - 1) + computation_time) / total_calls
            )

            # If still not converged, try pose perturbation as a fallback
            if not best_converged and best_q is not None:
                logger.info("IK did not converge, attempting fallback with pose perturbation.")
                pos_err, rot_err = self.robot.check_pose_error(T_des, best_q)
                
                # Only try perturbation if we are somewhat close
                if pos_err < 0.05 and rot_err < 0.2: # 5cm and ~11.5 deg
                    perturbed_q, converged = self._ik_with_perturbation(T_des, best_q)
                    if converged:
                        logger.info("IK fallback with perturbation succeeded.")
                        return perturbed_q, True

            return best_q, best_converged
            
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
                q_deg = np.array(wp['joint_positions'])
                tcp_recorded = np.array(wp['tcp_position'])
                
                q_rad, T_recorded = self.convert_from_robot_units(q_deg, tcp_recorded)
                T_fk = self.forward_kinematics(q_rad)
                
                pos_err = np.linalg.norm(T_fk[:3, 3] - T_recorded[:3, 3])
                
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
    
    def send_to_robot(self, q_rad: np.ndarray, validate_limits: bool = True, 
                     dry_run: bool = True) -> bool:
        """
        Send joint angles to robot controller.
        """
        try:
            q_deg = self.convert_to_robot_units(q_rad)
            
            if validate_limits:
                limits_deg = np.rad2deg(self.robot.joint_limits)
                
                margin = 0.05
                if self.config:
                    margin = self.config.get('safety', 'joint_limit_margin')
                
                limit_range = limits_deg[1] - limits_deg[0]
                margin_abs = limit_range * margin
                
                effective_lower = limits_deg[0] + margin_abs
                effective_upper = limits_deg[1] - margin_abs
                
                if np.any(q_deg < effective_lower) or np.any(q_deg > effective_upper):
                    logger.error(f"Joint angles exceed safe limits (with {margin*100:.1f}% margin)")
                    logger.error(f"Commanded: {q_deg}")
                    logger.error(f"Safe limits: [{effective_lower}, {effective_upper}]")
                    return False
            
            if not np.all(np.isfinite(q_deg)):
                logger.error("Joint angles contain NaN or infinite values")
                return False
            
            if dry_run:
                logger.info(f"[DRY RUN] Would send to robot: {np.round(q_deg, 3)}")
                return True
            else:
                logger.warning("Production robot communication not implemented")
                logger.info(f"Command to send: {np.round(q_deg, 3)}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to send command to robot: {e}")
            return False

    def _ik_with_perturbation(self, T_des: np.ndarray, q_init: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """IK fallback that perturbs the target pose slightly to find a solution."""
        pos_relax_abs = self.ik_params.get('position_relaxation', 0.01) # 1 cm
        rot_relax_abs = self.ik_params.get('rotation_relaxation', 0.05) # ~2.8 deg

        for i in range(15): # 15 perturbation attempts
            # Create a small random perturbation
            pos_offset = np.random.uniform(-pos_relax_abs, pos_relax_abs, 3)
            
            # Create a small random rotation vector and convert to matrix
            rot_vec = np.random.uniform(-rot_relax_abs, rot_relax_abs, 3)
            angle = np.linalg.norm(rot_vec)
            if angle > 1e-6:
                axis = rot_vec / angle
                # Rodrigues' rotation formula
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                R_offset = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            else:
                R_offset = np.eye(3)

            # Apply perturbation
            T_perturbed = T_des.copy()
            T_perturbed[:3, 3] += pos_offset
            T_perturbed[:3, :3] = T_perturbed[:3, :3] @ R_offset

            # Attempt IK with the perturbed target
            q_sol, converged = self.robot._ik_dls(
                T_perturbed, q_init, **{k: v for k, v in self.ik_params.items() if k not in ['num_attempts', 'position_relaxation', 'rotation_relaxation']}
            )

            if converged:
                # Check if the solution for the perturbed pose is acceptable for the *original* pose
                final_pos_err, final_rot_err = self.robot.check_pose_error(T_des, q_sol)
                
                # Use slightly more generous tolerances for accepting a perturbed solution
                if final_pos_err < self.ik_params['pos_tol'] * 2.5 and final_rot_err < self.ik_params['rot_tol'] * 2.5:
                    logger.info(f"Perturbation attempt {i+1} found an acceptable solution.")
                    return q_sol, True
        
        return None, False

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
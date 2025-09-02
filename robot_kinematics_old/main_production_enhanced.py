#!/usr/bin/env python3
"""
Enhanced production-ready main application for robot kinematics.
Includes proper unit handling, error checking, and real robot data integration.
"""

import numpy as np
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from robot_kinematics_prod import RobotKinematics, RobotKinematicsError
from config_prod import RobotConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('robot_kinematics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('robot_kinematics_main')

class RobotController:
    """Production robot controller interface."""
    
    def __init__(self, config: RobotConfig):
        self.config = config
        try:
            self.robot = RobotKinematics(
                config.urdf_path, 
                ee_link=config.ee_link, 
                base_link=config.base_link
            )
        except RobotKinematicsError as e:
            logger.error(f"Failed to initialize robot kinematics: {e}")
            raise
        
        self.last_joint_positions = None
        self.performance_stats = {
            'fk_calls': 0,
            'ik_calls': 0,
            'ik_successes': 0,
            'avg_ik_time': 0.0,
            'max_position_error': 0.0,
            'max_rotation_error': 0.0
        }
    
    def convert_units_from_robot(self, joint_positions_deg: np.ndarray, 
                                tcp_position_mm_rpy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert robot data units to standard SI units."""
        # Convert joint positions from degrees to radians
        q_rad = np.deg2rad(joint_positions_deg)
        
        # Convert TCP position from mm to m, keep RPY in radians
        tcp_pos_m = tcp_position_mm_rpy[:3] / 1000.0
        tcp_rpy_rad = tcp_position_mm_rpy[3:]
        
        # Create homogeneous transformation matrix
        R = self.robot.rpy_to_matrix(tcp_rpy_rad)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tcp_pos_m
        
        return q_rad, T
    
    def convert_units_to_robot(self, q_rad: np.ndarray) -> np.ndarray:
        """Convert joint angles from radians to degrees for robot controller."""
        return np.rad2deg(q_rad)
    
    def forward_kinematics(self, q_rad: np.ndarray) -> np.ndarray:
        """Compute forward kinematics with error handling and logging."""
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
        """Solve inverse kinematics with enhanced error handling."""
        try:
            start_time = time.time()
            ik_params = self.config.get_ik_params()
            
            q_sol, converged = self.robot.inverse_kinematics(
                T_des, 
                q_init=q_init,
                **ik_params
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
            'mean_position_error': 0.0,
            'max_position_error': 0.0,
            'mean_rotation_error': 0.0,
            'max_rotation_error': 0.0,
            'validation_success': True
        }
        
        for i in indices:
            wp = waypoints[i]
            
            try:
                # Convert units and compute FK
                q_deg = np.array(wp['joint_positions'])
                tcp_recorded = np.array(wp['tcp_position'])
                
                q_rad, T_recorded = self.convert_units_from_robot(q_deg, tcp_recorded)
                T_fk = self.forward_kinematics(q_rad)
                
                # Calculate errors
                pos_err = np.linalg.norm(T_fk[:3, 3] - T_recorded[:3, 3])
                
                # Use rotation matrix comparison for more robust orientation error
                R_err = T_fk[:3, :3].T @ T_recorded[:3, :3]
                rot_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0))
                
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
                       f"Max pos error: {results['max_position_error']:.6f}m")
        
        return results
    
    def send_to_robot_controller(self, q_rad: np.ndarray) -> bool:
        """Send joint angles to robot controller."""
        try:
            q_deg = self.convert_units_to_robot(q_rad)
            
            # Validate joint limits
            limits_deg = np.rad2deg(self.robot.joint_limits)
            if np.any(q_deg < limits_deg[0]) or np.any(q_deg > limits_deg[1]):
                logger.error("Joint angles exceed limits")
                return False
            
            # Here you would implement the actual communication with the robot controller
            # For now, we'll just log the command
            logger.info(f"Sending to robot controller: {q_deg}")
            self.last_joint_positions = q_rad
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

def main():
    """Main application entry point."""
    try:
        # Load configuration
        config = RobotConfig('robot_config_prod.yaml')
        
        # Initialize robot controller
        controller = RobotController(config)
        
        # Example 1: Test with the original test case
        logger.info("=== Testing original test case ===")
        q_test = np.array([0.5, 0.1, 0.2, 0.8, -0.4, 0.6])
        T_target = controller.forward_kinematics(q_test)
        
        print("Target pose:")
        print(np.round(T_target, 4))
        
        # Solve IK
        q_solution, converged = controller.inverse_kinematics(T_target)
        
        if converged:
            print(f"IK Solution: {np.round(q_solution, 6)}")
            print(f"Original:    {np.round(q_test, 6)}")
            print(f"Difference:  {np.round(q_solution - q_test, 6)}")
            
            # Send to robot (simulation)
            success = controller.send_to_robot_controller(q_solution)
            if success:
                logger.info("Command sent successfully")
        else:
            logger.error("IK failed to converge")
        
        # Example 2: Validate against real robot data
        logger.info("\n=== Validating against real robot data ===")
        validation_results = controller.validate_against_real_data(
            'third_20250710_162459.json', 
            num_samples=5
        )
        
        if validation_results.get('validation_success', False):
            print(f"Validation Results:")
            print(f"  Mean position error: {validation_results['mean_position_error']:.6f} m")
            print(f"  Max position error:  {validation_results['max_position_error']:.6f} m")
            print(f"  Mean rotation error: {validation_results['mean_rotation_error']:.6f} rad")
            print(f"  Max rotation error:  {validation_results['max_rotation_error']:.6f} rad")
            
            # Check if errors are within acceptable limits
            if validation_results['mean_position_error'] < 0.005:  # 5mm
                logger.info("✓ Position accuracy is excellent")
            elif validation_results['mean_position_error'] < 0.01:  # 10mm
                logger.info("✓ Position accuracy is acceptable")
            else:
                logger.warning("⚠ Position errors may require calibration")
        
        # Display performance statistics
        logger.info("\n=== Performance Statistics ===")
        stats = controller.get_performance_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    main()


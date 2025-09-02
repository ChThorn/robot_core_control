#!/usr/bin/env python3
"""
Fixed production-ready main application for robot kinematics.
Includes fixes for orientation representation issues.
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
    """Production robot controller interface with orientation fixes."""
    
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
        
        # Orientation handling mode - can be configured
        self.orientation_mode = config.config.get('robot_controller', {}).get('orientation_mode', 'auto_detect')
    
    def convert_units_from_robot(self, joint_positions_deg: np.ndarray, 
                                tcp_position_mm_rpy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert robot data units to standard SI units with orientation fixes."""
        # Convert joint positions from degrees to radians
        q_rad = np.deg2rad(joint_positions_deg)
        
        # Convert TCP position from mm to m
        tcp_pos_m = tcp_position_mm_rpy[:3] / 1000.0
        tcp_orientation = tcp_position_mm_rpy[3:]
        
        # Handle different orientation formats
        if self.orientation_mode == 'auto_detect':
            # Auto-detect based on value ranges
            if np.max(np.abs(tcp_orientation)) > 10:
                # Likely degrees
                tcp_rpy_rad = np.deg2rad(tcp_orientation)
                logger.debug("Auto-detected orientation format: degrees")
            else:
                # Likely radians
                tcp_rpy_rad = tcp_orientation
                logger.debug("Auto-detected orientation format: radians")
        elif self.orientation_mode == 'degrees':
            tcp_rpy_rad = np.deg2rad(tcp_orientation)
        elif self.orientation_mode == 'radians':
            tcp_rpy_rad = tcp_orientation
        elif self.orientation_mode == 'quaternion':
            # Assume quaternion format [w, x, y, z] or [x, y, z, w]
            if len(tcp_orientation) == 4:
                R = self._quaternion_to_matrix(tcp_orientation)
            else:
                logger.warning("Quaternion mode selected but orientation data has wrong length")
                tcp_rpy_rad = tcp_orientation
                R = self.robot.rpy_to_matrix(tcp_rpy_rad)
        else:
            tcp_rpy_rad = tcp_orientation
        
        # Convert RPY to rotation matrix (try different conventions)
        if self.orientation_mode != 'quaternion':
            R = self._convert_rpy_to_matrix(tcp_rpy_rad)
        
        # Create homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tcp_pos_m
        
        return q_rad, T
    
    def _convert_rpy_to_matrix(self, rpy: np.ndarray) -> np.ndarray:
        """Convert RPY to rotation matrix trying different conventions."""
        # Get the convention from config
        rpy_convention = self.config.config.get('robot_controller', {}).get('rpy_convention', 'xyz')
        
        if rpy_convention.lower() == 'zyx':
            return self._rpy_to_matrix_zyx(rpy)
        elif rpy_convention.lower() == 'zxy':
            return self._rpy_to_matrix_zxy(rpy)
        else:
            # Default XYZ convention
            return self.robot.rpy_to_matrix(rpy)
    
    def _rpy_to_matrix_zyx(self, rpy: np.ndarray) -> np.ndarray:
        """Convert RPY to rotation matrix using ZYX convention (most common in robotics)."""
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
    
    def _rpy_to_matrix_zxy(self, rpy: np.ndarray) -> np.ndarray:
        """Convert RPY to rotation matrix using ZXY convention."""
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # ZXY convention: R = Rz(yaw) * Rx(roll) * Ry(pitch)
        return np.array([
            [cy*cp - sy*sr*sp, -sy*cr, cy*sp + sy*sr*cp],
            [sy*cp + cy*sr*sp, cy*cr, sy*sp - cy*sr*cp],
            [-cr*sp, sr, cr*cp]
        ])
    
    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        # Handle both [w,x,y,z] and [x,y,z,w] formats
        if len(q) == 4:
            # Try to detect format based on magnitude
            if abs(q[0]) > abs(q[3]):
                # Likely [w,x,y,z] format
                w, x, y, z = q
            else:
                # Likely [x,y,z,w] format
                x, y, z, w = q
        else:
            raise ValueError("Quaternion must have 4 elements")
        
        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
        # Convert to rotation matrix
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])
    
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
        """Validate kinematics against real robot data with improved orientation handling."""
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
            'validation_success': True,
            'orientation_mode_used': self.orientation_mode
        }
        
        for i in indices:
            wp = waypoints[i]
            
            try:
                # Convert units and compute FK
                q_deg = np.array(wp['joint_positions'])
                tcp_recorded = np.array(wp['tcp_position'])
                
                q_rad, T_recorded = self.convert_units_from_robot(q_deg, tcp_recorded)
                T_fk = self.forward_kinematics(q_rad)
                
                # Calculate position error
                pos_err = np.linalg.norm(T_fk[:3, 3] - T_recorded[:3, 3])
                
                # Calculate rotation error using rotation matrix comparison
                R_fk = T_fk[:3, :3]
                R_recorded = T_recorded[:3, :3]
                R_err = R_fk.T @ R_recorded
                
                # Calculate angle of rotation error
                trace_R = np.trace(R_err)
                cos_angle = np.clip((trace_R - 1) / 2.0, -1.0, 1.0)
                rot_err = np.arccos(cos_angle)
                
                results['position_errors'].append(pos_err)
                results['rotation_errors'].append(rot_err)
                
                logger.debug(f"Waypoint {i}: pos_err={pos_err:.6f}m, rot_err={rot_err:.6f}rad")
                
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
    
    def test_orientation_modes(self, json_path: str) -> Dict[str, Dict[str, float]]:
        """Test different orientation modes to find the best one."""
        logger.info("Testing different orientation modes...")
        
        modes_to_test = ['radians', 'degrees', 'zyx', 'zxy']
        results = {}
        
        original_mode = self.orientation_mode
        original_convention = self.config.config.get('robot_controller', {}).get('rpy_convention', 'xyz')
        
        for mode in modes_to_test:
            logger.info(f"Testing orientation mode: {mode}")
            
            # Set mode
            if mode in ['radians', 'degrees']:
                self.orientation_mode = mode
                self.config.config.setdefault('robot_controller', {})['rpy_convention'] = 'xyz'
            elif mode in ['zyx', 'zxy']:
                self.orientation_mode = 'radians'  # Assume radians for convention tests
                self.config.config.setdefault('robot_controller', {})['rpy_convention'] = mode
            
            # Test validation
            validation_results = self.validate_against_real_data(json_path, num_samples=5)
            
            if 'mean_rotation_error' in validation_results:
                results[mode] = {
                    'mean_rotation_error': validation_results['mean_rotation_error'],
                    'max_rotation_error': validation_results['max_rotation_error'],
                    'mean_position_error': validation_results['mean_position_error']
                }
                
                logger.info(f"Mode {mode}: rot_err={validation_results['mean_rotation_error']:.6f}rad")
            else:
                results[mode] = {'error': 'Validation failed'}
        
        # Restore original settings
        self.orientation_mode = original_mode
        self.config.config.setdefault('robot_controller', {})['rpy_convention'] = original_convention
        
        # Find best mode
        best_mode = min(results.keys(), 
                       key=lambda k: results[k].get('mean_rotation_error', float('inf')))
        
        logger.info(f"Best orientation mode: {best_mode} "
                   f"(rot_err={results[best_mode].get('mean_rotation_error', 'N/A'):.6f}rad)")
        
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
    """Main application entry point with orientation testing."""
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
        
        # Example 2: Test different orientation modes
        logger.info("\n=== Testing orientation modes ===")
        orientation_test_results = controller.test_orientation_modes('third_20250710_162459.json')
        
        print("Orientation Mode Test Results:")
        for mode, results in orientation_test_results.items():
            if 'mean_rotation_error' in results:
                print(f"  {mode:10s}: {results['mean_rotation_error']:.6f} rad "
                      f"({np.rad2deg(results['mean_rotation_error']):.2f} deg)")
            else:
                print(f"  {mode:10s}: {results.get('error', 'Failed')}")
        
        # Find and apply best mode
        best_mode = min(orientation_test_results.keys(), 
                       key=lambda k: orientation_test_results[k].get('mean_rotation_error', float('inf')))
        
        if orientation_test_results[best_mode].get('mean_rotation_error', float('inf')) < 0.1:
            logger.info(f"✅ Found good orientation mode: {best_mode}")
            
            # Apply best mode
            if best_mode in ['radians', 'degrees']:
                controller.orientation_mode = best_mode
            elif best_mode in ['zyx', 'zxy']:
                controller.orientation_mode = 'radians'
                controller.config.config.setdefault('robot_controller', {})['rpy_convention'] = best_mode
            
            # Re-run validation with best mode
            logger.info(f"\n=== Final validation with {best_mode} mode ===")
            final_results = controller.validate_against_real_data('third_20250710_162459.json', num_samples=5)
            
            print("Final Validation Results:")
            print(f"  Mean position error: {final_results['mean_position_error']:.6f} m")
            print(f"  Max position error:  {final_results['max_position_error']:.6f} m")
            print(f"  Mean rotation error: {final_results['mean_rotation_error']:.6f} rad "
                  f"({np.rad2deg(final_results['mean_rotation_error']):.2f} deg)")
            print(f"  Max rotation error:  {final_results['max_rotation_error']:.6f} rad "
                  f"({np.rad2deg(final_results['max_rotation_error']):.2f} deg)")
            
            if final_results['mean_rotation_error'] < 0.1:
                logger.info("✅ SYSTEM IS READY FOR PRODUCTION!")
            else:
                logger.warning("⚠️  Rotation errors still too high for production")
        else:
            logger.warning("❌ No orientation mode achieved acceptable accuracy")
        
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


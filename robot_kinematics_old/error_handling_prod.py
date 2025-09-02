#!/usr/bin/env python3
"""
Enhanced production error handling system for robot kinematics.
Provides comprehensive error recovery, logging, and safety mechanisms.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass
import traceback
import time

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    CONFIGURATION = "configuration"
    KINEMATICS = "kinematics"
    VALIDATION = "validation"
    COMMUNICATION = "communication"
    SAFETY = "safety"
    PERFORMANCE = "performance"

@dataclass
class ErrorInfo:
    """Comprehensive error information."""
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    traceback_info: Optional[str] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False

class SafetyChecker:
    """Safety validation for robot operations."""
    
    def __init__(self, config):
        self.config = config
        self.workspace_limits = config.get_workspace_limits()
        self.joint_limits = config.get_joint_limits_deg()
        
    def validate_joint_angles(self, q_rad: np.ndarray) -> Tuple[bool, str]:
        """Validate joint angles are within safe limits."""
        if self.joint_limits is not None:
            q_deg = np.rad2deg(q_rad)
            limits_deg = np.array(self.joint_limits)
            
            # Check lower limits
            if np.any(q_deg < limits_deg[:, 0]):
                violating_joints = np.where(q_deg < limits_deg[:, 0])[0]
                return False, f"Joint angles below lower limits: joints {violating_joints}"
            
            # Check upper limits
            if np.any(q_deg > limits_deg[:, 1]):
                violating_joints = np.where(q_deg > limits_deg[:, 1])[0]
                return False, f"Joint angles above upper limits: joints {violating_joints}"
        
        return True, "Joint angles within limits"
    
    def validate_workspace_position(self, position: np.ndarray) -> Tuple[bool, str]:
        """Validate position is within workspace limits."""
        if self.workspace_limits is None:
            return True, "No workspace limits defined"
        
        x, y, z = position
        
        # Check X limits
        x_range = self.workspace_limits.get('x_range', [-np.inf, np.inf])
        if not (x_range[0] <= x <= x_range[1]):
            return False, f"X position {x:.3f} outside range {x_range}"
        
        # Check Y limits
        y_range = self.workspace_limits.get('y_range', [-np.inf, np.inf])
        if not (y_range[0] <= y <= y_range[1]):
            return False, f"Y position {y:.3f} outside range {y_range}"
        
        # Check Z limits
        z_range = self.workspace_limits.get('z_range', [-np.inf, np.inf])
        if not (z_range[0] <= z <= z_range[1]):
            return False, f"Z position {z:.3f} outside range {z_range}"
        
        return True, "Position within workspace"
    
    def validate_pose(self, T: np.ndarray) -> Tuple[bool, str]:
        """Validate complete pose (position and orientation)."""
        # Check position
        position_valid, position_msg = self.validate_workspace_position(T[:3, 3])
        if not position_valid:
            return False, position_msg
        
        # Check rotation matrix validity
        R = T[:3, :3]
        if not self._is_valid_rotation_matrix(R):
            return False, "Invalid rotation matrix in pose"
        
        return True, "Pose is valid"
    
    def _is_valid_rotation_matrix(self, R: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if matrix is a valid rotation matrix."""
        # Check if R^T * R = I
        should_be_identity = R.T @ R
        identity = np.eye(3)
        if not np.allclose(should_be_identity, identity, atol=tolerance):
            return False
        
        # Check if det(R) = 1
        if not np.isclose(np.linalg.det(R), 1.0, atol=tolerance):
            return False
        
        return True

class ErrorRecoveryManager:
    """Manages error recovery strategies."""
    
    def __init__(self):
        self.recovery_strategies = {}
        self.error_history = []
        self.max_recovery_attempts = 3
        
    def register_recovery_strategy(self, error_type: type, strategy: Callable):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = strategy
        logger.info(f"Registered recovery strategy for {error_type.__name__}")
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle error with appropriate recovery strategy."""
        error_info = ErrorInfo(
            timestamp=time.time(),
            category=self._classify_error(error),
            severity=self._assess_severity(error),
            message=str(error),
            details=context,
            traceback_info=traceback.format_exc()
        )
        
        self.error_history.append(error_info)
        
        # Log error
        logger.error(f"Error occurred: {error_info.category.value} - {error_info.message}")
        
        # Attempt recovery
        recovery_result = self._attempt_recovery(error, error_info, context)
        
        return recovery_result
    
    def _classify_error(self, error: Exception) -> ErrorCategory:
        """Classify error into appropriate category."""
        error_type = type(error).__name__
        
        if 'Config' in error_type or 'YAML' in error_type:
            return ErrorCategory.CONFIGURATION
        elif 'Kinematics' in error_type or 'Singular' in error_type:
            return ErrorCategory.KINEMATICS
        elif 'Validation' in error_type or 'Assert' in error_type:
            return ErrorCategory.VALIDATION
        elif 'Connection' in error_type or 'Timeout' in error_type:
            return ErrorCategory.COMMUNICATION
        elif 'Safety' in error_type or 'Limit' in error_type:
            return ErrorCategory.SAFETY
        else:
            return ErrorCategory.PERFORMANCE
    
    def _assess_severity(self, error: Exception) -> ErrorSeverity:
        """Assess error severity."""
        error_type = type(error).__name__
        
        if any(keyword in error_type.lower() for keyword in ['critical', 'fatal', 'safety']):
            return ErrorSeverity.CRITICAL
        elif any(keyword in error_type.lower() for keyword in ['timeout', 'connection', 'limit']):
            return ErrorSeverity.HIGH
        elif any(keyword in error_type.lower() for keyword in ['validation', 'convergence']):
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _attempt_recovery(self, error: Exception, error_info: ErrorInfo, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Attempt to recover from error."""
        error_type = type(error)
        
        if error_type in self.recovery_strategies:
            try:
                logger.info(f"Attempting recovery for {error_type.__name__}")
                error_info.recovery_attempted = True
                
                result = self.recovery_strategies[error_type](error, context)
                error_info.recovery_successful = True
                
                logger.info("Recovery successful")
                return True, result
                
            except Exception as recovery_error:
                logger.error(f"Recovery failed: {recovery_error}")
                error_info.recovery_successful = False
                return False, None
        else:
            logger.warning(f"No recovery strategy available for {error_type.__name__}")
            return False, None

class RobustKinematicsWrapper:
    """Robust wrapper for kinematics operations with error handling."""
    
    def __init__(self, robot_kinematics, config):
        self.robot = robot_kinematics
        self.config = config
        self.safety_checker = SafetyChecker(config)
        self.error_manager = ErrorRecoveryManager()
        
        # Register recovery strategies
        self._register_recovery_strategies()
        
        # Performance tracking
        self.operation_count = 0
        self.error_count = 0
        
    def _register_recovery_strategies(self):
        """Register default recovery strategies."""
        
        def ik_convergence_recovery(error, context):
            """Recovery strategy for IK convergence failures."""
            logger.info("Attempting IK recovery with relaxed tolerances")
            
            T_des = context.get('target_pose')
            if T_des is None:
                raise ValueError("No target pose in context for IK recovery")
            
            # Try with relaxed tolerances
            relaxed_params = self.config.get_ik_params()
            relaxed_params['pos_tol'] *= 10
            relaxed_params['rot_tol'] *= 10
            relaxed_params['num_attempts'] *= 2
            
            q_sol, converged = self.robot.inverse_kinematics(T_des, **relaxed_params)
            
            if converged:
                return q_sol
            else:
                raise Exception("IK recovery failed even with relaxed tolerances")
        
        def validation_error_recovery(error, context):
            """Recovery strategy for validation errors."""
            logger.info("Attempting validation recovery")
            
            # Return safe default configuration
            return np.zeros(self.robot.n_joints)
        
        # Register strategies
        self.error_manager.register_recovery_strategy(ValueError, ik_convergence_recovery)
        self.error_manager.register_recovery_strategy(AssertionError, validation_error_recovery)
    
    def safe_forward_kinematics(self, q: np.ndarray) -> Tuple[Optional[np.ndarray], bool, str]:
        """Safe forward kinematics with error handling."""
        self.operation_count += 1
        
        try:
            # Safety check
            joint_valid, joint_msg = self.safety_checker.validate_joint_angles(q)
            if not joint_valid:
                raise ValueError(f"Unsafe joint configuration: {joint_msg}")
            
            # Compute FK
            T = self.robot.forward_kinematics(q)
            
            # Validate result
            pose_valid, pose_msg = self.safety_checker.validate_pose(T)
            if not pose_valid:
                raise ValueError(f"Invalid pose result: {pose_msg}")
            
            return T, True, "Success"
            
        except Exception as e:
            self.error_count += 1
            context = {'joint_angles': q, 'operation': 'forward_kinematics'}
            
            recovered, result = self.error_manager.handle_error(e, context)
            
            if recovered:
                return result, True, "Recovered"
            else:
                return None, False, str(e)
    
    def safe_inverse_kinematics(self, T_des: np.ndarray, **kwargs) -> Tuple[Optional[np.ndarray], bool, str]:
        """Safe inverse kinematics with error handling."""
        self.operation_count += 1
        
        try:
            # Safety check on target pose
            pose_valid, pose_msg = self.safety_checker.validate_pose(T_des)
            if not pose_valid:
                raise ValueError(f"Unsafe target pose: {pose_msg}")
            
            # Compute IK
            ik_params = {**self.config.get_ik_params(), **kwargs}
            q_sol, converged = self.robot.inverse_kinematics(T_des, **ik_params)
            
            if not converged:
                raise ValueError("IK failed to converge")
            
            # Safety check on solution
            joint_valid, joint_msg = self.safety_checker.validate_joint_angles(q_sol)
            if not joint_valid:
                raise ValueError(f"Unsafe IK solution: {joint_msg}")
            
            return q_sol, True, "Success"
            
        except Exception as e:
            self.error_count += 1
            context = {'target_pose': T_des, 'operation': 'inverse_kinematics', 'ik_params': kwargs}
            
            recovered, result = self.error_manager.handle_error(e, context)
            
            if recovered:
                # Validate recovered result
                if result is not None:
                    joint_valid, joint_msg = self.safety_checker.validate_joint_angles(result)
                    if joint_valid:
                        return result, True, "Recovered"
                
                return None, False, "Recovery validation failed"
            else:
                return None, False, str(e)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if self.operation_count == 0:
            return {'error_rate': 0.0, 'total_operations': 0, 'total_errors': 0}
        
        error_rate = self.error_count / self.operation_count
        
        # Categorize errors
        error_categories = {}
        recovery_stats = {'attempted': 0, 'successful': 0}
        
        for error_info in self.error_manager.error_history:
            category = error_info.category.value
            error_categories[category] = error_categories.get(category, 0) + 1
            
            if error_info.recovery_attempted:
                recovery_stats['attempted'] += 1
                if error_info.recovery_successful:
                    recovery_stats['successful'] += 1
        
        return {
            'error_rate': error_rate,
            'total_operations': self.operation_count,
            'total_errors': self.error_count,
            'error_categories': error_categories,
            'recovery_stats': recovery_stats,
            'recovery_success_rate': (recovery_stats['successful'] / recovery_stats['attempted'] 
                                    if recovery_stats['attempted'] > 0 else 0.0)
        }

def setup_error_logging(log_file: str = "robot_errors.log"):
    """Setup comprehensive error logging."""
    error_logger = logging.getLogger('robot_errors')
    error_logger.setLevel(logging.ERROR)
    
    # File handler for errors
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    error_logger.addHandler(file_handler)
    
    return error_logger

if __name__ == "__main__":
    # Example usage
    from config_prod import RobotConfig
    from robot_kinematics_prod import RobotKinematics
    
    # Setup
    config = RobotConfig('robot_config_prod.yaml')
    robot = RobotKinematics(config.urdf_path, config.ee_link, config.base_link)
    
    # Create robust wrapper
    robust_robot = RobustKinematicsWrapper(robot, config)
    
    # Test with valid configuration
    q_test = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    T, success, msg = robust_robot.safe_forward_kinematics(q_test)
    print(f"FK Test: Success={success}, Message={msg}")
    
    if success:
        q_recovered, ik_success, ik_msg = robust_robot.safe_inverse_kinematics(T)
        print(f"IK Test: Success={ik_success}, Message={ik_msg}")
    
    # Print error statistics
    stats = robust_robot.get_error_statistics()
    print(f"Error Statistics: {stats}")


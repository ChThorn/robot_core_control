#!/usr/bin/env python3
"""
Inverse Kinematics for Rainbow Robotics RB3-730ES-U
Improved numerical approach using optimized DH parameters
"""

import numpy as np
from typing import List, Optional
from forward_kinematics import ForwardKinematics

class InverseKinematics:
    """Inverse kinematics implementation using numerical optimization"""
    
    def __init__(self):
        """Initialize inverse kinematics with forward kinematics instance"""
        self.fk = ForwardKinematics()
        # Extended joint limits based on recorded data analysis
        self.joint_limits = [(-15.0, 15.0)] * 6  # Wider limits based on recorded data
        
    def euler_to_rotation_matrix(self, euler_angles: np.ndarray, order: str = 'xyz') -> np.ndarray:
        """
        Convert Euler angles to rotation matrix
        
        Args:
            euler_angles: [rx, ry, rz] in radians
            order: Euler angle order
            
        Returns:
            3x3 rotation matrix
        """
        rx, ry, rz = euler_angles
        
        if order.lower() == 'xyz':
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(rx), -np.sin(rx)],
                          [0, np.sin(rx), np.cos(rx)]])
            
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                          [0, 1, 0],
                          [-np.sin(ry), 0, np.cos(ry)]])
            
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                          [np.sin(rz), np.cos(rz), 0],
                          [0, 0, 1]])
            
            R = Rz @ Ry @ Rx
            
        elif order.lower() == 'zyx':
            Rx = np.array([[1, 0, 0],
                          [0, np.cos(rx), -np.sin(rx)],
                          [0, np.sin(rx), np.cos(rx)]])
            
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                          [0, 1, 0],
                          [-np.sin(ry), 0, np.cos(ry)]])
            
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                          [np.sin(rz), np.cos(rz), 0],
                          [0, 0, 1]])
            
            R = Rz @ Ry @ Rx
            
        else:
            raise ValueError(f"Unsupported Euler angle order: {order}")
        
        return R
    
    def inverse_kinematics(self, target_pose: np.ndarray, 
                          current_joints: Optional[List[float]] = None,
                          euler_order: str = 'xyz',
                          max_iterations: int = 2000,
                          tolerance: float = 1e-3) -> List[List[float]]:
        """
        Calculate inverse kinematics for target pose using improved numerical optimization
        
        Args:
            target_pose: 6D pose [x, y, z, rx, ry, rz] in meters and radians
            current_joints: Current joint configuration for solution selection
            euler_order: Euler angle order for orientation
            max_iterations: Maximum iterations for numerical solver
            tolerance: Convergence tolerance
            
        Returns:
            List of possible joint angle solutions
        """
        try:
            from scipy.optimize import minimize, differential_evolution
        except ImportError:
            raise ImportError("scipy is required for inverse kinematics. Install with: pip install scipy")
        
        x, y, z, rx, ry, rz = target_pose
        target_position = np.array([x, y, z])
        target_rotation = self.euler_to_rotation_matrix(np.array([rx, ry, rz]), euler_order)
        
        def objective_function(joints):
            """Objective function for optimization"""
            try:
                calc_position, calc_rotation = self.fk.get_tcp_pose(joints.tolist())
                
                # Position error (weighted more heavily)
                pos_error = np.linalg.norm(calc_position - target_position)
                
                # Orientation error (Frobenius norm of rotation difference)
                rot_error = np.linalg.norm(calc_rotation - target_rotation, 'fro')
                
                # Combined error with position weighted more heavily
                return 10.0 * pos_error + 0.1 * rot_error
            except:
                return 1000  # Large penalty for invalid configurations
        
        solutions = []
        
        # Method 1: Try differential evolution (global optimization)
        try:
            result = differential_evolution(
                objective_function,
                bounds=self.joint_limits,
                maxiter=500,
                seed=42,
                atol=tolerance,
                tol=tolerance
            )
            
            if result.success and result.fun < tolerance:
                solution = result.x.tolist()
                solutions.append(self._normalize_angles(solution))
        except:
            pass
        
        # Method 2: Multiple local optimizations with strategic starting points
        starting_points = []
        
        # Add current joints if provided
        if current_joints is not None:
            starting_points.append(current_joints)
        
        # Add zero configuration
        starting_points.append([0.0] * 6)
        
        # Add some strategic configurations based on workspace analysis
        strategic_configs = [
            [0.0, 0.0, 1.57, 0.0, 0.0, 0.0],    # Elbow up
            [0.0, 0.0, -1.57, 0.0, 0.0, 0.0],   # Elbow down
            [1.57, 0.0, 0.0, 0.0, 0.0, 0.0],    # Base rotated
            [-1.57, 0.0, 0.0, 0.0, 0.0, 0.0],   # Base rotated opposite
            [0.0, 1.57, 0.0, 0.0, 0.0, 0.0],    # Shoulder up
            [0.0, -1.57, 0.0, 0.0, 0.0, 0.0],   # Shoulder down
        ]
        starting_points.extend(strategic_configs)
        
        # Add random starting points
        np.random.seed(42)
        for _ in range(20):
            random_joints = [np.random.uniform(low, high) for low, high in self.joint_limits]
            starting_points.append(random_joints)
        
        # Try optimization from each starting point
        for start_joints in starting_points:
            try:
                # Try multiple optimization methods
                methods = ['L-BFGS-B', 'SLSQP', 'TNC']
                
                for method in methods:
                    try:
                        result = minimize(
                            objective_function,
                            start_joints,
                            method=method,
                            bounds=self.joint_limits,
                            options={'maxiter': max_iterations, 'ftol': tolerance}
                        )
                        
                        if result.success and result.fun < tolerance:
                            solution = result.x.tolist()
                            normalized_solution = self._normalize_angles(solution)
                            
                            # Check if this solution is already found
                            if not self._is_duplicate_solution(normalized_solution, solutions):
                                solutions.append(normalized_solution)
                                break  # Found a good solution, move to next starting point
                                
                    except:
                        continue
                        
            except:
                continue
        
        # Sort solutions by distance to current configuration if provided
        if current_joints is not None and solutions:
            solutions.sort(key=lambda sol: np.linalg.norm(np.array(sol) - np.array(current_joints)))
        
        return solutions
    
    def inverse_kinematics_position_only(self, target_position: np.ndarray,
                                       current_joints: Optional[List[float]] = None,
                                       max_iterations: int = 2000,
                                       tolerance: float = 1e-3) -> List[List[float]]:
        """
        Calculate inverse kinematics for target position only (ignoring orientation)
        
        Args:
            target_position: 3D position [x, y, z] in meters
            current_joints: Current joint configuration for solution selection
            max_iterations: Maximum iterations for numerical solver
            tolerance: Convergence tolerance
            
        Returns:
            List of possible joint angle solutions
        """
        try:
            from scipy.optimize import minimize, differential_evolution
        except ImportError:
            raise ImportError("scipy is required for inverse kinematics. Install with: pip install scipy")
        
        def objective_function(joints):
            """Objective function for position-only optimization"""
            try:
                calc_position = self.fk.get_tcp_position(joints.tolist())
                pos_error = np.linalg.norm(calc_position - target_position)
                return pos_error
            except:
                return 1000  # Large penalty for invalid configurations
        
        solutions = []
        
        # Method 1: Differential evolution
        try:
            result = differential_evolution(
                objective_function,
                bounds=self.joint_limits,
                maxiter=500,
                seed=42,
                atol=tolerance,
                tol=tolerance
            )
            
            if result.success and result.fun < tolerance:
                solution = result.x.tolist()
                solutions.append(self._normalize_angles(solution))
        except:
            pass
        
        # Method 2: Multiple local optimizations
        starting_points = []
        
        # Add current joints if provided
        if current_joints is not None:
            starting_points.append(current_joints)
        
        # Add strategic starting points
        starting_points.extend([
            [0.0] * 6,
            [0.0, 0.0, 1.57, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.57, 0.0, 0.0, 0.0],
            [1.57, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.57, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        
        # Add random starting points
        np.random.seed(42)
        for _ in range(30):
            random_joints = [np.random.uniform(low, high) for low, high in self.joint_limits]
            starting_points.append(random_joints)
        
        # Try optimization from each starting point
        for start_joints in starting_points:
            try:
                methods = ['L-BFGS-B', 'SLSQP', 'TNC']
                
                for method in methods:
                    try:
                        result = minimize(
                            objective_function,
                            start_joints,
                            method=method,
                            bounds=self.joint_limits,
                            options={'maxiter': max_iterations, 'ftol': tolerance}
                        )
                        
                        if result.success and result.fun < tolerance:
                            solution = result.x.tolist()
                            normalized_solution = self._normalize_angles(solution)
                            
                            if not self._is_duplicate_solution(normalized_solution, solutions):
                                solutions.append(normalized_solution)
                                break
                                
                    except:
                        continue
                        
            except:
                continue
        
        # Sort solutions by distance to current configuration if provided
        if current_joints is not None and solutions:
            solutions.sort(key=lambda sol: np.linalg.norm(np.array(sol) - np.array(current_joints)))
        
        return solutions
    
    def _normalize_angles(self, angles: List[float]) -> List[float]:
        """Normalize angles to [-π, π] range"""
        normalized = []
        for angle in angles:
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi
            normalized.append(angle)
        return normalized
    
    def _is_duplicate_solution(self, solution: List[float], existing_solutions: List[List[float]], 
                              tolerance: float = 1e-2) -> bool:
        """Check if solution is duplicate of existing solutions"""
        for existing in existing_solutions:
            if np.allclose(solution, existing, atol=tolerance):
                return True
        return False


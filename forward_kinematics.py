#!/usr/bin/env python3
"""
Production-Ready Forward Kinematics for Rainbow Robotics RB3-730ES Robot Manipulator

This module implements improved forward kinematics with:
- Refined DH parameters based on data analysis
- Multiple coordinate frame support
- Calibration capabilities
- Enhanced accuracy and robustness

Author: Production version for RB3-730ES kinematics
"""

import numpy as np
import json
from typing import List, Tuple, Union, Optional

class ForwardKinematicsV2:
    """Production-ready forward kinematics implementation for RB3-730ES robot"""
    
    def __init__(self, dh_variant: str = 'refined'):
        """
        Initialize with DH parameters for RB3-730ES
        
        Args:
            dh_variant: 'original', 'refined', or 'calibrated'
        """
        self.dh_variant = dh_variant
        self.load_dh_parameters(dh_variant)
        self.num_joints = len(self.dh_params)
        
        # Coordinate frame transformation (if needed)
        self.base_transform = np.eye(4)  # Identity by default
        self.tcp_transform = np.eye(4)   # Identity by default
        
        # Calibration parameters
        self.joint_offsets = np.zeros(6)
        self.link_corrections = np.zeros(6)
        
    def load_dh_parameters(self, variant: str):
        """Load DH parameters based on variant"""
        
        if variant == 'original':
            # Original DH parameters from URDF
            self.dh_params = [
                {'a': 0.0,    'alpha': np.pi/2,  'd': 0.1453,  'theta_offset': 0.0},
                {'a': 0.0,    'alpha': 0.0,     'd': 0.0,     'theta_offset': 0.0},
                {'a': 0.286,  'alpha': 0.0,     'd': -0.00645, 'theta_offset': 0.0},
                {'a': 0.0,    'alpha': np.pi/2,  'd': 0.0,     'theta_offset': 0.0},
                {'a': 0.0,    'alpha': -np.pi/2, 'd': 0.344,   'theta_offset': 0.0},
                {'a': 0.0,    'alpha': 0.0,     'd': 0.1,     'theta_offset': 0.0},
            ]
        
        elif variant == 'refined':
            # Refined DH parameters based on analysis
            self.dh_params = [
                {'a': 0.0,    'alpha': np.pi/2,  'd': 0.1453,  'theta_offset': 0.0},
                {'a': 0.0,    'alpha': 0.0,     'd': 0.0,     'theta_offset': -np.pi/2},
                {'a': 0.286,  'alpha': 0.0,     'd': 0.0,     'theta_offset': 0.0},
                {'a': 0.0,    'alpha': np.pi/2,  'd': 0.344,   'theta_offset': 0.0},
                {'a': 0.0,    'alpha': -np.pi/2, 'd': 0.0,     'theta_offset': 0.0},
                {'a': 0.0,    'alpha': 0.0,     'd': 0.1,     'theta_offset': 0.0},
            ]
        
        elif variant == 'calibrated':
            # Load calibrated parameters if available
            try:
                with open('calibrated_dh_parameters.json', 'r') as f:
                    self.dh_params = json.load(f)
            except:
                # Fall back to refined parameters
                self.load_dh_parameters('refined')
        
        else:
            raise ValueError(f"Unknown DH variant: {variant}")
    
    def set_base_transform(self, transform: np.ndarray):
        """Set base coordinate frame transformation"""
        if transform.shape != (4, 4):
            raise ValueError("Base transform must be 4x4 matrix")
        self.base_transform = transform.copy()
    
    def set_tcp_transform(self, transform: np.ndarray):
        """Set TCP coordinate frame transformation"""
        if transform.shape != (4, 4):
            raise ValueError("TCP transform must be 4x4 matrix")
        self.tcp_transform = transform.copy()
    
    def set_joint_offsets(self, offsets: Union[List[float], np.ndarray]):
        """Set joint angle offsets for calibration"""
        if len(offsets) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} offsets, got {len(offsets)}")
        self.joint_offsets = np.array(offsets)
    
    def dh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """
        Create DH transformation matrix using Craig's convention
        
        Args:
            a: Link length
            alpha: Link twist
            d: Link offset
            theta: Joint angle
            
        Returns:
            4x4 transformation matrix
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        T = np.array([
            [cos_theta, -sin_theta * cos_alpha,  sin_theta * sin_alpha, a * cos_theta],
            [sin_theta,  cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0,          sin_alpha,              cos_alpha,             d],
            [0,          0,                      0,                     1]
        ])
        
        return T
    
    def forward_kinematics(self, joint_angles: Union[List[float], np.ndarray], 
                          include_base: bool = True, 
                          include_tcp: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Compute forward kinematics for given joint angles
        
        Args:
            joint_angles: List or array of 6 joint angles in radians
            include_base: Include base transformation
            include_tcp: Include TCP transformation
            
        Returns:
            Tuple of:
            - End effector transformation matrix (4x4)
            - List of transformation matrices for each joint
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} joint angles, got {len(joint_angles)}")
        
        # Apply joint offsets
        joint_angles = np.array(joint_angles) + self.joint_offsets
        
        # Start with base transformation
        if include_base:
            T_total = self.base_transform.copy()
        else:
            T_total = np.eye(4)
        
        T_matrices = []
        
        # Compute transformation for each joint
        for i, angle in enumerate(joint_angles):
            params = self.dh_params[i]
            theta = angle + params['theta_offset']
            
            # Create DH transformation matrix
            T_i = self.dh_transform(
                params['a'], 
                params['alpha'], 
                params['d'], 
                theta
            )
            
            # Accumulate transformation
            T_total = T_total @ T_i
            T_matrices.append(T_total.copy())
        
        # Apply TCP transformation
        if include_tcp:
            T_total = T_total @ self.tcp_transform
        
        return T_total, T_matrices
    
    def get_tcp_pose(self, joint_angles: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get TCP (Tool Center Point) position and orientation
        
        Args:
            joint_angles: List or array of 6 joint angles in radians
            
        Returns:
            Tuple of:
            - Position vector [x, y, z] in meters
            - Rotation matrix (3x3)
        """
        T_end, _ = self.forward_kinematics(joint_angles)
        
        position = T_end[:3, 3]
        rotation = T_end[:3, :3]
        
        return position, rotation
    
    def get_tcp_pose_euler(self, joint_angles: Union[List[float], np.ndarray], 
                          convention: str = 'xyz') -> Tuple[np.ndarray, np.ndarray]:
        """
        Get TCP position and orientation as Euler angles
        
        Args:
            joint_angles: List or array of 6 joint angles in radians
            convention: Euler angle convention ('xyz', 'zyx', 'zyz')
            
        Returns:
            Tuple of:
            - Position vector [x, y, z] in meters
            - Euler angles [rx, ry, rz] in radians
        """
        position, rotation = self.get_tcp_pose(joint_angles)
        
        if convention == 'xyz':
            euler_angles = self.rotation_matrix_to_euler_xyz(rotation)
        elif convention == 'zyx':
            euler_angles = self.rotation_matrix_to_euler_zyx(rotation)
        elif convention == 'zyz':
            euler_angles = self.rotation_matrix_to_euler_zyz(rotation)
        else:
            raise ValueError(f"Unknown Euler convention: {convention}")
        
        return position, euler_angles
    
    def rotation_matrix_to_euler_xyz(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (XYZ convention)"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([x, y, z])
    
    def rotation_matrix_to_euler_zyx(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (ZYX convention)"""
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.array([z, y, x])  # Return in ZYX order
    
    def rotation_matrix_to_euler_zyz(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (ZYZ convention)"""
        sy = np.sqrt(R[0, 2] * R[0, 2] + R[1, 2] * R[1, 2])
        
        singular = sy < 1e-6
        
        if not singular:
            z1 = np.arctan2(R[1, 2], R[0, 2])
            y = np.arctan2(sy, R[2, 2])
            z2 = np.arctan2(R[2, 1], -R[2, 0])
        else:
            z1 = np.arctan2(-R[0, 1], R[0, 0])
            y = np.arctan2(sy, R[2, 2])
            z2 = 0
        
        return np.array([z1, y, z2])
    
    def get_joint_positions(self, joint_angles: Union[List[float], np.ndarray]) -> List[np.ndarray]:
        """
        Get positions of all joints in the kinematic chain
        
        Args:
            joint_angles: List or array of 6 joint angles in radians
            
        Returns:
            List of position vectors for each joint
        """
        _, T_matrices = self.forward_kinematics(joint_angles)
        
        # Base position
        if np.allclose(self.base_transform, np.eye(4)):
            positions = [np.array([0, 0, 0])]
        else:
            positions = [self.base_transform[:3, 3]]
        
        # Joint positions
        for T in T_matrices:
            positions.append(T[:3, 3])
        
        return positions
    
    def get_jacobian(self, joint_angles: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Compute geometric Jacobian matrix
        
        Args:
            joint_angles: List or array of 6 joint angles in radians
            
        Returns:
            6x6 Jacobian matrix [linear_velocity; angular_velocity]
        """
        joint_positions = self.get_joint_positions(joint_angles)
        tcp_position = joint_positions[-1]
        
        _, T_matrices = self.forward_kinematics(joint_angles, include_tcp=False)
        
        J = np.zeros((6, self.num_joints))
        
        for i in range(self.num_joints):
            if i == 0:
                # Base joint
                z_axis = np.array([0, 0, 1])
                joint_pos = joint_positions[0]
            else:
                # Extract z-axis from transformation matrix
                z_axis = T_matrices[i-1][:3, 2]
                joint_pos = joint_positions[i]
            
            # Linear velocity component
            J[:3, i] = np.cross(z_axis, tcp_position - joint_pos)
            
            # Angular velocity component
            J[3:, i] = z_axis
        
        return J
    
    def calibrate_from_data(self, joint_angles_list: List[np.ndarray], 
                           tcp_positions_list: List[np.ndarray],
                           method: str = 'least_squares') -> dict:
        """
        Calibrate DH parameters from recorded data
        
        Args:
            joint_angles_list: List of joint angle configurations
            tcp_positions_list: List of corresponding TCP positions
            method: Calibration method ('least_squares', 'weighted')
            
        Returns:
            Dictionary with calibration results
        """
        from scipy.optimize import least_squares
        
        def objective_function(params):
            """Objective function for calibration"""
            # Update DH parameters
            for i, param_set in enumerate(self.dh_params):
                param_set['a'] = params[i*4 + 0]
                param_set['alpha'] = params[i*4 + 1]
                param_set['d'] = params[i*4 + 2]
                param_set['theta_offset'] = params[i*4 + 3]
            
            errors = []
            for joint_angles, target_pos in zip(joint_angles_list, tcp_positions_list):
                computed_pos, _ = self.get_tcp_pose(joint_angles)
                error = computed_pos - target_pos
                errors.extend(error)
            
            return np.array(errors)
        
        # Initial parameters
        initial_params = []
        for params in self.dh_params:
            initial_params.extend([params['a'], params['alpha'], params['d'], params['theta_offset']])
        
        # Calibration
        result = least_squares(objective_function, initial_params)
        
        # Extract calibrated parameters
        calibrated_params = []
        for i in range(self.num_joints):
            calibrated_params.append({
                'a': result.x[i*4 + 0],
                'alpha': result.x[i*4 + 1],
                'd': result.x[i*4 + 2],
                'theta_offset': result.x[i*4 + 3]
            })
        
        # Save calibrated parameters
        with open('calibrated_dh_parameters.json', 'w') as f:
            json.dump(calibrated_params, f, indent=2)
        
        return {
            'success': result.success,
            'cost': result.cost,
            'parameters': calibrated_params,
            'residual_norm': np.linalg.norm(result.fun)
        }
    
    def print_dh_table(self):
        """Print the DH parameters table"""
        print(f"DH Parameters Table for RB3-730ES ({self.dh_variant}):")
        print("Joint |    a    |  alpha  |    d    | theta_offset")
        print("------|---------|---------|---------|-------------")
        for i, params in enumerate(self.dh_params):
            print(f"  {i+1}   | {params['a']:7.4f} | {params['alpha']:7.4f} | {params['d']:7.4f} | {params['theta_offset']:7.4f}")

def test_forward_kinematics_v2():
    """Test the improved forward kinematics implementation"""
    print("Testing Production-Ready Forward Kinematics for RB3-730ES")
    print("=" * 60)
    
    # Test different variants
    variants = ['original', 'refined']
    
    for variant in variants:
        print(f"\nTesting {variant} DH parameters:")
        print("-" * 40)
        
        fk = ForwardKinematicsV2(dh_variant=variant)
        fk.print_dh_table()
        
        # Test with zero configuration
        joint_angles_zero = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        position, euler = fk.get_tcp_pose_euler(joint_angles_zero)
        
        print(f"\nZero Configuration:")
        print(f"TCP Position: [{position[0]:8.4f}, {position[1]:8.4f}, {position[2]:8.4f}] m")
        print(f"TCP Orientation: [{euler[0]:8.4f}, {euler[1]:8.4f}, {euler[2]:8.4f}] rad")
        
        # Test Jacobian
        J = fk.get_jacobian(joint_angles_zero)
        print(f"Jacobian condition number: {np.linalg.cond(J):.2f}")

if __name__ == "__main__":
    test_forward_kinematics_v2()


#!/usr/bin/env python3
"""
Pure analytical inverse kinematics for RB3-730ES-U robot.
Implements closed-form geometric solutions for 6-DOF industrial robot.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AnalyticalIK:
    """
    Pure analytical inverse kinematics for RB3-730ES-U robot.
    
    Robot DH Parameters (extracted from URDF):
    Link    a_i     α_i     d_i         θ_i
    1       0       0       0.1453      θ1*
    2       0       π/2     0           θ2*
    3       0.286   0       -0.00645    θ3*
    4       0       π/2     0.344       θ4*
    5       0       -π/2    0           θ5*
    6       0       π/2     0.1         θ6*
    
    * indicates variable joint
    """
    
    def __init__(self):
        # DH parameters (from URDF analysis)
        self.d1 = 0.1453    # Base height
        self.a3 = 0.286     # Upper arm length  
        self.d3 = -0.00645  # Elbow offset
        self.d4 = 0.344     # Forearm length
        self.d6 = 0.1       # Tool length
        
        # Joint limits (±180°)
        self.joint_limits = np.array([
            [-np.pi, np.pi],
            [-np.pi, np.pi], 
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi],
            [-np.pi, np.pi]
        ])
        
        logger.info("Analytical IK initialized for RB3-730ES-U")
    
    def inverse_kinematics(self, T_des: np.ndarray) -> Tuple[List[np.ndarray], bool]:
        """
        Compute all analytical IK solutions.
        
        Args:
            T_des: 4x4 desired homogeneous transformation matrix
            
        Returns:
            solutions: List of valid 6-DOF joint configurations
            success: True if at least one solution found
        """
        try:
            # Extract position and orientation
            px, py, pz = T_des[:3, 3]
            R_des = T_des[:3, :3]
            
            # Calculate wrist center position
            # Wrist center = TCP position - d6 * approach vector
            approach_vector = R_des[:, 2]  # Z-axis of end-effector
            wrist_center = np.array([px, py, pz]) - self.d6 * approach_vector
            
            # Solve for first 3 joints (position)
            position_solutions = self._solve_position_ik(wrist_center)
            
            if not position_solutions:
                logger.warning("No position solutions found")
                return [], False
            
            # For each position solution, solve orientation
            all_solutions = []
            
            for q123 in position_solutions:
                orientation_solutions = self._solve_orientation_ik(R_des, q123)
                
                for q456 in orientation_solutions:
                    q_full = np.concatenate([q123, q456])
                    
                    # Check joint limits
                    if self._within_joint_limits(q_full):
                        all_solutions.append(q_full)
            
            success = len(all_solutions) > 0
            if success:
                logger.info(f"Found {len(all_solutions)} analytical IK solutions")
            else:
                logger.warning("No solutions within joint limits")
                
            return all_solutions, success
            
        except Exception as e:
            logger.error(f"Analytical IK failed: {e}")
            return [], False
    
    def _solve_position_ik(self, wrist_center: np.ndarray) -> List[np.ndarray]:
        """
        Solve for first 3 joints using geometric approach.
        
        Args:
            wrist_center: 3D position of wrist center
            
        Returns:
            List of [q1, q2, q3] solutions
        """
        solutions = []
        
        wx, wy, wz = wrist_center
        
        try:
            # Joint 1: Base rotation (around Z-axis)
            # Two solutions: atan2(wy, wx) and atan2(wy, wx) + π
            r_xy = np.sqrt(wx**2 + wy**2)
            
            if r_xy < 1e-6:
                # Singularity: wrist on Z-axis
                q1_candidates = [0.0]
            else:
                q1_base = np.arctan2(wy, wx)
                q1_candidates = [q1_base, self._normalize_angle(q1_base + np.pi)]
            
            for q1 in q1_candidates:
                # Transform wrist center to joint 1 frame
                c1, s1 = np.cos(q1), np.sin(q1)
                
                # Project onto X1-Z1 plane
                x1 = wx * c1 + wy * s1  # Should equal r_xy
                z1 = wz - self.d1
                
                # Account for elbow offset d3
                z1_adj = z1 - self.d3
                
                # Distance from shoulder to wrist center in X1-Z1 plane
                r = np.sqrt(x1**2 + z1_adj**2)
                
                # Check reachability
                L1 = self.a3  # Upper arm
                L2 = self.d4  # Forearm
                
                if r > (L1 + L2) + 1e-6:  # Unreachable (too far)
                    continue
                if r < abs(L1 - L2) - 1e-6:  # Unreachable (too close)
                    continue
                
                # Joint 3: Elbow angle (law of cosines)
                cos_q3 = (r**2 - L1**2 - L2**2) / (2 * L1 * L2)
                cos_q3 = np.clip(cos_q3, -1.0, 1.0)
                
                # Two elbow configurations
                q3_candidates = [np.arccos(cos_q3), -np.arccos(cos_q3)]
                
                for q3 in q3_candidates:
                    # Joint 2: Shoulder angle
                    # Angle from X1-axis to wrist center
                    alpha = np.arctan2(z1_adj, x1)
                    
                    # Angle from upper arm to line connecting shoulder to wrist
                    beta = np.arctan2(L2 * np.sin(q3), L1 + L2 * np.cos(q3))
                    
                    q2 = alpha - beta
                    
                    # Store solution
                    q123 = np.array([q1, q2, q3])
                    solutions.append(q123)
                    
        except Exception as e:
            logger.warning(f"Position IK error: {e}")
        
        return solutions
    
    def _solve_orientation_ik(self, R_des: np.ndarray, q123: np.ndarray) -> List[np.ndarray]:
        """
        Solve for last 3 joints using spherical wrist assumption.
        
        Args:
            R_des: Desired 3x3 rotation matrix
            q123: First 3 joint angles [q1, q2, q3]
            
        Returns:
            List of [q4, q5, q6] solutions
        """
        solutions = []
        
        try:
            q1, q2, q3 = q123
            
            # Calculate transformation from base to joint 3
            T03 = self._forward_kinematics_03(q1, q2, q3)
            R03 = T03[:3, :3]
            
            # Required rotation from joint 3 to end-effector
            R36 = R03.T @ R_des
            
            # Extract spherical wrist angles (ZYZ Euler angles)
            # R36 = Rz(q4) * Ry(q5) * Rz(q6)
            
            r11, r12, r13 = R36[0, :]
            r21, r22, r23 = R36[1, :]
            r31, r32, r33 = R36[2, :]
            
            # Joint 5: Wrist bend angle
            cos_q5 = r33
            cos_q5 = np.clip(cos_q5, -1.0, 1.0)
            
            # Two solutions for q5
            q5_candidates = [np.arccos(cos_q5), -np.arccos(cos_q5)]
            
            for q5 in q5_candidates:
                sin_q5 = np.sin(q5)
                
                if abs(sin_q5) < 1e-6:  # Singularity: q5 ≈ 0 or π
                    # Wrist singularity: infinite solutions for q4 + q6
                    # Choose q4 = 0 and solve for q6
                    q4 = 0.0
                    
                    if abs(cos_q5 - 1.0) < 1e-6:  # q5 ≈ 0
                        q6 = np.arctan2(r21, r11)
                    else:  # q5 ≈ π  
                        q6 = np.arctan2(-r21, r11)
                        
                    q456 = np.array([q4, q5, q6])
                    solutions.append(q456)
                    
                else:  # Normal case
                    # Solve for q4 and q6
                    q4 = np.arctan2(r23 / sin_q5, r13 / sin_q5)
                    q6 = np.arctan2(r32 / sin_q5, -r31 / sin_q5)
                    
                    # Normalize angles
                    q4 = self._normalize_angle(q4)
                    q6 = self._normalize_angle(q6)
                    
                    q456 = np.array([q4, q5, q6])
                    solutions.append(q456)
                    
        except Exception as e:
            logger.warning(f"Orientation IK error: {e}")
        
        return solutions
    
    def _forward_kinematics_03(self, q1: float, q2: float, q3: float) -> np.ndarray:
        """Calculate forward kinematics from base to joint 3."""
        
        # Individual transformation matrices using DH parameters
        c1, s1 = np.cos(q1), np.sin(q1)
        c2, s2 = np.cos(q2), np.sin(q2)  
        c3, s3 = np.cos(q3), np.sin(q3)
        
        # T01: Base to joint 1
        T01 = np.array([
            [c1, -s1, 0, 0],
            [s1,  c1, 0, 0],
            [0,   0,  1, self.d1],
            [0,   0,  0, 1]
        ])
        
        # T12: Joint 1 to joint 2 (90° rotation about X)
        T12 = np.array([
            [c2, 0,  s2, 0],
            [0,  1,  0,  0],
            [-s2, 0, c2, 0],
            [0,  0,  0,  1]
        ])
        
        # T23: Joint 2 to joint 3
        T23 = np.array([
            [c3, -s3, 0, self.a3],
            [s3,  c3, 0, 0],
            [0,   0,  1, self.d3],
            [0,   0,  0, 1]
        ])
        
        return T01 @ T12 @ T23
    
    def _within_joint_limits(self, q: np.ndarray) -> bool:
        """Check if joint configuration is within limits."""
        for i, angle in enumerate(q):
            if angle < self.joint_limits[i, 0] or angle > self.joint_limits[i, 1]:
                return False
        return True
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-π, π]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def select_best_solution(self, solutions: List[np.ndarray], 
                           q_current: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Select the best solution from multiple options.
        
        Args:
            solutions: List of valid joint configurations
            q_current: Current joint configuration for proximity selection
            
        Returns:
            Best joint configuration
        """
        if not solutions:
            raise ValueError("No solutions provided")
        
        if len(solutions) == 1:
            return solutions[0]
        
        # Selection criteria
        best_solution = solutions[0]
        best_score = float('-inf')
        
        for q in solutions:
            score = 0.0
            
            # 1. Proximity to current configuration (highest priority)
            if q_current is not None:
                # Consider joint angle wrapping
                diff = np.abs(q - q_current)
                diff = np.minimum(diff, 2*np.pi - diff)  # Handle wrapping
                distance = np.sum(diff)
                score += 100.0 / (1.0 + distance)
            
            # 2. Prefer elbow-up configuration
            if q[2] > 0:  # q3 > 0 (elbow up)
                score += 10.0
            
            # 3. Avoid joint limits
            joint_usage = np.mean(np.abs(q) / np.pi)
            score += 5.0 * (1.0 - joint_usage)
            
            # 4. Avoid wrist singularity (q5 ≈ 0)
            if abs(q[4]) > 0.1:
                score += 2.0
            
            if score > best_score:
                best_score = score
                best_solution = q
        
        return best_solution
    
    def get_solution_info(self, q: np.ndarray) -> dict:
        """Get information about a joint configuration."""
        return {
            'joint_angles_deg': np.rad2deg(q),
            'joint_angles_rad': q,
            'elbow_configuration': 'up' if q[2] > 0 else 'down',
            'wrist_configuration': 'non-singular' if abs(q[4]) > 0.1 else 'near-singular',
            'joint_utilization': np.mean(np.abs(q) / np.pi)
        }


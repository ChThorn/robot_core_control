#!/usr/bin/env python3
"""
Camera Pose Processor for Robot Kinematics
==========================================

Production-grade system for processing camera-detected TCP poses and computing
robot kinematics (FK/IK) with round-trip validation.

Input: Camera-detected poses (x, y, z, rx, ry, rz in mm and degrees)
Output: Joint angles, validation results, and motion commands

Author: Production Kinematics System
Date: September 2, 2025
"""

import numpy as np
import json
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from robot_controller import RobotController
from config import KinematicsConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CameraPose:
    """Camera-detected pose data structure."""
    x: float          # Position X (mm)
    y: float          # Position Y (mm) 
    z: float          # Position Z (mm)
    rx: float         # Rotation X (degrees)
    ry: float         # Rotation Y (degrees)
    rz: float         # Rotation Z (degrees)
    timestamp: float  # Detection timestamp
    confidence: float = 1.0  # Detection confidence (0-1)
    
    def to_transformation_matrix(self) -> np.ndarray:
        """Convert camera pose to 4x4 transformation matrix."""
        # Convert to standard units (meters and radians)
        pos = np.array([self.x, self.y, self.z]) / 1000.0  # mm to m
        rot_rad = np.deg2rad([self.rx, self.ry, self.rz])
        
        # Create rotation matrix from Euler angles (XYZ convention)
        cx, cy, cz = np.cos(rot_rad)
        sx, sy, sz = np.sin(rot_rad)
        
        # Rotation matrix (XYZ Euler angles)
        R = np.array([
            [cy*cz, -cy*sz, sy],
            [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy],
            [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy]
        ])
        
        # Build transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        
        return T
    
    @classmethod
    def from_transformation_matrix(cls, T: np.ndarray, timestamp: float = None) -> 'CameraPose':
        """Create CameraPose from 4x4 transformation matrix."""
        if timestamp is None:
            timestamp = time.time()
            
        # Extract position (m to mm)
        pos_mm = T[:3, 3] * 1000.0
        
        # Extract rotation matrix and convert to Euler angles
        R = T[:3, :3]
        
        # Convert rotation matrix to XYZ Euler angles
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0
            
        # Convert to degrees
        rot_deg = np.rad2deg([rx, ry, rz])
        
        return cls(
            x=pos_mm[0], y=pos_mm[1], z=pos_mm[2],
            rx=rot_deg[0], ry=rot_deg[1], rz=rot_deg[2],
            timestamp=timestamp
        )

@dataclass
class KinematicsSolution:
    """Results from kinematics computation."""
    success: bool
    joint_angles_deg: np.ndarray
    joint_angles_rad: np.ndarray
    target_pose: CameraPose
    achieved_pose: CameraPose
    position_error_mm: float
    rotation_error_deg: float
    computation_time_ms: float
    ik_iterations: int = 0
    
class CameraPoseProcessor:
    """
    Production-grade processor for camera-detected poses.
    
    Handles:
    - Camera pose input processing
    - Forward/inverse kinematics computation  
    - Round-trip validation
    - Motion command generation
    - Safety validation
    """
    
    def __init__(self, urdf_path: str, config_path: str = None):
        """
        Initialize camera pose processor.
        
        Args:
            urdf_path: Path to robot URDF file
            config_path: Path to configuration file
        """
        self.controller = RobotController(urdf_path)
        self.config = KinematicsConfig(config_path) if config_path else None
        
        # Performance tracking
        self.stats = {
            'total_poses_processed': 0,
            'ik_success_rate': 0.0,
            'mean_computation_time_ms': 0.0,
            'mean_position_error_mm': 0.0,
            'mean_rotation_error_deg': 0.0
        }
        
        self.processing_history = []
        
        logger.info(f"Camera pose processor initialized with robot: {urdf_path}")
    
    def process_camera_pose(self, camera_pose: CameraPose, 
                          validate_reachability: bool = True,
                          perform_round_trip: bool = True) -> KinematicsSolution:
        """
        Process a single camera-detected pose.
        
        Args:
            camera_pose: Camera-detected pose
            validate_reachability: Check if pose is reachable
            perform_round_trip: Perform FK-IK-FK round-trip validation
            
        Returns:
            KinematicsSolution with results
        """
        start_time = time.time()
        
        try:
            # Convert camera pose to transformation matrix
            T_target = camera_pose.to_transformation_matrix()
            
            # Validate pose if requested
            if validate_reachability:
                if not self._validate_pose_reachability(T_target):
                    return KinematicsSolution(
                        success=False,
                        joint_angles_deg=np.zeros(6),
                        joint_angles_rad=np.zeros(6),
                        target_pose=camera_pose,
                        achieved_pose=camera_pose,
                        position_error_mm=float('inf'),
                        rotation_error_deg=float('inf'),
                        computation_time_ms=(time.time() - start_time) * 1000
                    )
            
            # Solve inverse kinematics
            q_solution, converged = self.controller.inverse_kinematics(T_target)
            
            if not converged:
                logger.warning(f"IK failed to converge for pose: {camera_pose}")
                return KinematicsSolution(
                    success=False,
                    joint_angles_deg=np.zeros(6),
                    joint_angles_rad=np.zeros(6), 
                    target_pose=camera_pose,
                    achieved_pose=camera_pose,
                    position_error_mm=float('inf'),
                    rotation_error_deg=float('inf'),
                    computation_time_ms=(time.time() - start_time) * 1000
                )
            
            # Perform round-trip validation if requested
            if perform_round_trip:
                T_achieved = self.controller.forward_kinematics(q_solution)
                achieved_pose = CameraPose.from_transformation_matrix(T_achieved, camera_pose.timestamp)
                
                # Calculate errors
                pos_error_mm = np.linalg.norm([
                    achieved_pose.x - camera_pose.x,
                    achieved_pose.y - camera_pose.y, 
                    achieved_pose.z - camera_pose.z
                ])
                
                rot_error_deg = np.linalg.norm([
                    achieved_pose.rx - camera_pose.rx,
                    achieved_pose.ry - camera_pose.ry,
                    achieved_pose.rz - camera_pose.rz
                ])
            else:
                achieved_pose = camera_pose
                pos_error_mm = 0.0
                rot_error_deg = 0.0
            
            computation_time_ms = (time.time() - start_time) * 1000
            
            # Create solution
            solution = KinematicsSolution(
                success=True,
                joint_angles_deg=np.rad2deg(q_solution),
                joint_angles_rad=q_solution,
                target_pose=camera_pose,
                achieved_pose=achieved_pose,
                position_error_mm=pos_error_mm,
                rotation_error_deg=rot_error_deg,
                computation_time_ms=computation_time_ms
            )
            
            # Update statistics
            self._update_statistics(solution)
            
            return solution
            
        except Exception as e:
            logger.error(f"Error processing camera pose: {e}")
            return KinematicsSolution(
                success=False,
                joint_angles_deg=np.zeros(6),
                joint_angles_rad=np.zeros(6),
                target_pose=camera_pose,
                achieved_pose=camera_pose,
                position_error_mm=float('inf'),
                rotation_error_deg=float('inf'),
                computation_time_ms=(time.time() - start_time) * 1000
            )
    
    def process_pose_batch(self, camera_poses: List[CameraPose]) -> List[KinematicsSolution]:
        """
        Process a batch of camera poses.
        
        Args:
            camera_poses: List of camera-detected poses
            
        Returns:
            List of kinematic solutions
        """
        logger.info(f"Processing batch of {len(camera_poses)} camera poses...")
        
        solutions = []
        for i, pose in enumerate(camera_poses):
            solution = self.process_camera_pose(pose)
            solutions.append(solution)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(camera_poses)} poses")
        
        # Log batch statistics
        successful = sum(1 for s in solutions if s.success)
        success_rate = successful / len(solutions) * 100
        
        logger.info(f"Batch processing complete: {successful}/{len(solutions)} successful ({success_rate:.1f}%)")
        
        return solutions
    
    def execute_robot_motion(self, solution: KinematicsSolution, 
                           dry_run: bool = True,
                           validate_safety: bool = True) -> bool:
        """
        Execute robot motion based on kinematics solution.
        
        Args:
            solution: Kinematics solution to execute
            dry_run: If True, simulate without actual robot motion
            validate_safety: Perform safety validation
            
        Returns:
            Success status
        """
        if not solution.success:
            logger.error("Cannot execute motion: kinematics solution failed")
            return False
        
        try:
            # Send command to robot
            success = self.controller.send_to_robot(
                solution.joint_angles_rad,
                validate_limits=validate_safety,
                dry_run=dry_run
            )
            
            if success:
                logger.info(f"Motion command sent successfully (dry_run={dry_run})")
                logger.info(f"Target: ({solution.target_pose.x:.1f}, {solution.target_pose.y:.1f}, {solution.target_pose.z:.1f}) mm")
                logger.info(f"Joints: {np.round(solution.joint_angles_deg, 2)} deg")
                logger.info(f"Error: {solution.position_error_mm:.2f} mm")
            else:
                logger.error("Failed to send motion command")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing robot motion: {e}")
            return False
    
    def _validate_pose_reachability(self, T_target: np.ndarray) -> bool:
        """Validate if pose is within robot workspace."""
        try:
            # Quick reachability check based on position
            pos = T_target[:3, 3]
            distance_from_base = np.linalg.norm(pos[:2])  # XY distance
            height = pos[2]
            
            # Simple workspace bounds (adjust based on robot specs)
            max_reach = 1.5  # meters
            min_height = -0.5  # meters  
            max_height = 2.0   # meters
            
            if distance_from_base > max_reach:
                logger.warning(f"Pose too far from base: {distance_from_base:.3f}m > {max_reach}m")
                return False
                
            if height < min_height or height > max_height:
                logger.warning(f"Pose height out of bounds: {height:.3f}m not in [{min_height}, {max_height}]")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating pose reachability: {e}")
            return False
    
    def _update_statistics(self, solution: KinematicsSolution):
        """Update processing statistics."""
        self.processing_history.append(solution)
        self.stats['total_poses_processed'] += 1
        
        # Keep only recent history (last 1000 poses)
        if len(self.processing_history) > 1000:
            self.processing_history = self.processing_history[-1000:]
        
        # Calculate running statistics
        recent_solutions = self.processing_history[-100:]  # Last 100 poses
        successful_solutions = [s for s in recent_solutions if s.success]
        
        if successful_solutions:
            self.stats['ik_success_rate'] = len(successful_solutions) / len(recent_solutions) * 100
            self.stats['mean_computation_time_ms'] = np.mean([s.computation_time_ms for s in successful_solutions])
            self.stats['mean_position_error_mm'] = np.mean([s.position_error_mm for s in successful_solutions])
            self.stats['mean_rotation_error_deg'] = np.mean([s.rotation_error_deg for s in successful_solutions])
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        return self.stats.copy()
    
    def generate_test_poses(self, num_poses: int = 10) -> List[CameraPose]:
        """
        Generate example camera poses for testing.
        
        Args:
            num_poses: Number of test poses to generate
            
        Returns:
            List of example camera poses
        """
        test_poses = []
        
        # Define workspace bounds for realistic poses
        x_range = (400, 700)    # mm
        y_range = (-200, 200)   # mm  
        z_range = (100, 500)    # mm
        rx_range = (-20, 20)    # degrees
        ry_range = (-20, 20)    # degrees
        rz_range = (-90, 90)  # degrees
        
        for i in range(num_poses):
            pose = CameraPose(
                x=np.random.uniform(*x_range),
                y=np.random.uniform(*y_range),
                z=np.random.uniform(*z_range),
                rx=np.random.uniform(*rx_range),
                ry=np.random.uniform(*ry_range), 
                rz=np.random.uniform(*rz_range),
                timestamp=time.time() + i,
                confidence=np.random.uniform(0.8, 1.0)
            )
            test_poses.append(pose)
        
        logger.info(f"Generated {num_poses} test camera poses")
        return test_poses

def main():
    """Example usage of the camera pose processor."""
    print("CAMERA POSE PROCESSOR - PRODUCTION SYSTEM")
    print("=" * 60)
    
    # Initialize processor
    processor = CameraPoseProcessor("rb3_730es_u.urdf")
    
    # Example 1: Single pose processing
    print("\n=== SINGLE POSE PROCESSING ===")
    example_pose = CameraPose(
        x=500.0,      # mm
        y=100.0,      # mm
        z=300.0,      # mm
        rx=10.0,      # degrees
        ry=5.0,       # degrees
        rz=45.0,      # degrees
        timestamp=time.time()
    )
    
    print(f"Input pose: ({example_pose.x}, {example_pose.y}, {example_pose.z}) mm, "
          f"({example_pose.rx}, {example_pose.ry}, {example_pose.rz})°")
    
    solution = processor.process_camera_pose(example_pose)
    
    if solution.success:
        print(f"✅ IK Solution: {np.round(solution.joint_angles_deg, 2)}°")
        print(f"✅ Position error: {solution.position_error_mm:.2f} mm")
        print(f"✅ Rotation error: {solution.rotation_error_deg:.2f}°")
        print(f"✅ Computation time: {solution.computation_time_ms:.1f} ms")
        
        # Execute motion (dry run)
        processor.execute_robot_motion(solution, dry_run=True)
    else:
        print("❌ IK failed to find solution")
    
    # Example 2: Batch processing
    print("\n=== BATCH PROCESSING ===")
    test_poses = processor.generate_test_poses(10)
    solutions = processor.process_pose_batch(test_poses)
    
    # Show batch results
    successful = [s for s in solutions if s.success]
    print(f"Batch results: {len(successful)}/{len(solutions)} successful")
    
    if successful:
        mean_error = np.mean([s.position_error_mm for s in successful])
        mean_time = np.mean([s.computation_time_ms for s in successful])
        print(f"Mean position error: {mean_error:.2f} mm")
        print(f"Mean computation time: {mean_time:.1f} ms")
    
    # Show performance statistics
    print("\n=== PERFORMANCE STATISTICS ===")
    stats = processor.get_performance_stats()
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    print("\n" + "=" * 60)
    print("Camera pose processing complete!")

if __name__ == "__main__":
    main()

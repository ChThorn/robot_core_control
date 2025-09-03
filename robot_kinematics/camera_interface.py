#!/usr/bin/env python3
"""
Simple Camera Interface for Robot Control
=========================================

Easy-to-use interface for processing camera-detected poses.
Just provide x, y, z, rx, ry, rz and get joint angles back!

Usage:
    camera = CameraInterface()
    joints = camera.pose_to_joints(x=500, y=100, z=300, rx=10, ry=5, rz=45)
"""

import numpy as np
import time
from camera_pose_processor import CameraPoseProcessor, CameraPose, KinematicsSolution
from typing import Optional, Tuple, List

class CameraInterface:
    """Simple interface for camera-based robot control."""
    
    def __init__(self, urdf_path: str = "rb3_730es_u.urdf"):
        """Initialize the camera interface."""
        self.processor = CameraPoseProcessor(urdf_path)
        print(f"Camera interface ready! Using robot model: {urdf_path}")
    
    def pose_to_joints(self, x: float, y: float, z: float, 
                      rx: float, ry: float, rz: float,
                      validate_reachability: bool = True) -> Optional[np.ndarray]:
        """
        Convert camera pose to joint angles.
        
        Args:
            x, y, z: Position in mm
            rx, ry, rz: Rotation in degrees
            validate_reachability: Check if pose is reachable
            
        Returns:
            Joint angles in degrees, or None if failed
        """
        # Create camera pose
        pose = CameraPose(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz, timestamp=time.time())
        
        # Process pose
        solution = self.processor.process_camera_pose(pose, validate_reachability=validate_reachability)
        
        if solution.success:
            print(f"✅ Success! Joints: {np.round(solution.joint_angles_deg, 2)}°")
            print(f"   Error: {solution.position_error_mm:.2f}mm, {solution.rotation_error_deg:.2f}°")
            print(f"   Time: {solution.computation_time_ms:.1f}ms")
            return solution.joint_angles_deg
        else:
            print(f"❌ Failed to find solution for pose ({x}, {y}, {z})mm")
            return None
    
    def joints_to_pose(self, joint_angles_deg: np.ndarray) -> CameraPose:
        """
        Convert joint angles to camera pose.
        
        Args:
            joint_angles_deg: Joint angles in degrees
            
        Returns:
            Camera pose
        """
        # Convert to radians
        q_rad = np.deg2rad(joint_angles_deg)
        
        # Forward kinematics
        T = self.processor.controller.forward_kinematics(q_rad)
        
        # Convert to camera pose
        pose = CameraPose.from_transformation_matrix(T)
        
        print(f"✅ FK Result: ({pose.x:.1f}, {pose.y:.1f}, {pose.z:.1f})mm, "
              f"({pose.rx:.1f}, {pose.ry:.1f}, {pose.rz:.1f})°")
        
        return pose
    
    def validate_pose(self, x: float, y: float, z: float, 
                     rx: float, ry: float, rz: float) -> bool:
        """
        Check if a pose is reachable without solving IK.
        
        Args:
            x, y, z: Position in mm
            rx, ry, rz: Rotation in degrees
            
        Returns:
            True if pose is likely reachable
        """
        pose = CameraPose(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz, timestamp=time.time())
        T = pose.to_transformation_matrix()
        
        reachable = self.processor._validate_pose_reachability(T)
        
        if reachable:
            print(f"✅ Pose ({x}, {y}, {z})mm is reachable")
        else:
            print(f"❌ Pose ({x}, {y}, {z})mm is NOT reachable")
        
        return reachable
    
    def round_trip_test(self, x: float, y: float, z: float, 
                       rx: float, ry: float, rz: float) -> bool:
        """
        Test pose → joints → pose consistency.
        
        Args:
            x, y, z: Position in mm
            rx, ry, rz: Rotation in degrees
            
        Returns:
            True if round-trip is successful
        """
        print(f"Round-trip test for ({x}, {y}, {z})mm, ({rx}, {ry}, {rz})°")
        
        # Step 1: Pose → Joints
        joints = self.pose_to_joints(x, y, z, rx, ry, rz)
        if joints is None:
            print("❌ Round-trip failed: IK failed")
            return False
        
        # Step 2: Joints → Pose
        achieved_pose = self.joints_to_pose(joints)
        
        # Step 3: Compare
        pos_error = np.sqrt((achieved_pose.x - x)**2 + (achieved_pose.y - y)**2 + (achieved_pose.z - z)**2)
        rot_error = np.sqrt((achieved_pose.rx - rx)**2 + (achieved_pose.ry - ry)**2 + (achieved_pose.rz - rz)**2)
        
        print(f"Round-trip errors: {pos_error:.2f}mm position, {rot_error:.2f}° rotation")
        
        # Check if errors are acceptable
        success = pos_error < 5.0 and rot_error < 10.0  # 5mm, 10° tolerance
        
        if success:
            print("✅ Round-trip test PASSED")
        else:
            print("❌ Round-trip test FAILED")
        
        return success
    
    def batch_process_poses(self, poses: List[Tuple[float, float, float, float, float, float]]) -> List[Optional[np.ndarray]]:
        """
        Process multiple poses at once.
        
        Args:
            poses: List of (x, y, z, rx, ry, rz) tuples
            
        Returns:
            List of joint angle arrays (None for failed poses)
        """
        print(f"Batch processing {len(poses)} poses...")
        
        # Convert to CameraPose objects
        camera_poses = []
        for i, (x, y, z, rx, ry, rz) in enumerate(poses):
            pose = CameraPose(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz, timestamp=time.time() + i)
            camera_poses.append(pose)
        
        # Process batch
        solutions = self.processor.process_pose_batch(camera_poses)
        
        # Extract joint angles
        results = []
        for solution in solutions:
            if solution.success:
                results.append(solution.joint_angles_deg)
            else:
                results.append(None)
        
        successful = sum(1 for r in results if r is not None)
        print(f"Batch complete: {successful}/{len(poses)} successful")
        
        return results
    
    def send_to_robot(self, joint_angles_deg: np.ndarray, dry_run: bool = True) -> bool:
        """
        Send joint angles to robot.
        
        Args:
            joint_angles_deg: Joint angles in degrees
            dry_run: If True, simulate without actual robot motion
            
        Returns:
            Success status
        """
        q_rad = np.deg2rad(joint_angles_deg)
        return self.processor.controller.send_to_robot(q_rad, dry_run=dry_run)

# Simple usage examples
def demo_usage():
    """Demonstrate simple usage patterns."""
    print("CAMERA INTERFACE DEMO")
    print("=" * 40)
    
    # Initialize interface
    camera = CameraInterface()
    
    # Example 1: Simple pose to joints conversion
    print("\n1. Convert camera pose to joint angles:")
    joints = camera.pose_to_joints(x=500, y=100, z=300, rx=10, ry=5, rz=45)
    
    if joints is not None:
        # Example 2: Convert back to pose
        print("\n2. Convert joint angles back to pose:")
        pose = camera.joints_to_pose(joints)
        
        # Example 3: Round-trip validation
        print("\n3. Round-trip validation test:")
        camera.round_trip_test(x=500, y=100, z=300, rx=10, ry=5, rz=45)
    
    # Example 4: Batch processing
    print("\n4. Batch processing multiple poses:")
    poses = [
        (500, 100, 300, 10, 5, 45),
        (600, 0, 400, 0, 0, 90),
        (450, -50, 250, -10, 15, -30)
    ]
    
    joint_results = camera.batch_process_poses(poses)
    
    for i, joints in enumerate(joint_results):
        if joints is not None:
            print(f"  Pose {i+1}: {np.round(joints, 1)}°")
        else:
            print(f"  Pose {i+1}: FAILED")

if __name__ == "__main__":
    demo_usage()

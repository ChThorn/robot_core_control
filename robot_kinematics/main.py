#!/usr/bin/env python3
"""
Production-ready main application for robot kinematics.
"""

import numpy as np
import logging
from robot_controller import RobotController

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('robot_main')

def main():
    """Main application entry point."""
    try:
        # Initialize robot controller without the URDF path
        controller = RobotController(
            ee_link="tcp",
            base_link="link0"
        )
        
        logger.info("Robot kinematics system initialized successfully")
        
        # Example 1: Test forward and inverse kinematics
        logger.info("=== Testing Forward and Inverse Kinematics ===")
        
        # Test configuration (radians)
        q_test = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Compute forward kinematics
        T_target = controller.forward_kinematics(q_test)
        
        print("Test joint angles (rad):", np.round(q_test, 4))
        print("Test joint angles (deg):", np.round(np.rad2deg(q_test), 2))
        print("\nForward kinematics result:")
        print("Position (m):", np.round(T_target[:3, 3], 4))
        print("Rotation matrix:")
        print(np.round(T_target[:3, :3], 4))
        
        # Solve inverse kinematics
        q_solution, converged = controller.inverse_kinematics(T_target)
        
        if converged:
            print(f"\nIK Solution (rad): {np.round(q_solution, 6)}")
            print(f"IK Solution (deg): {np.round(np.rad2deg(q_solution), 2)}")
            print(f"Original    (rad): {np.round(q_test, 6)}")
            print(f"Difference  (rad): {np.round(q_solution - q_test, 6)}")
            
            # Check accuracy
            pos_err, rot_err = controller.robot.check_pose_error(T_target, q_solution)
            print(f"\nAccuracy:")
            print(f"Position error: {pos_err:.2e} m")
            print(f"Rotation error: {rot_err:.2e} rad ({np.rad2deg(rot_err):.4f} deg)")
            
            # Send to robot (placeholder)
            success = controller.send_to_robot(q_solution)
            if success:
                logger.info("Command would be sent to robot successfully")
        else:
            logger.error("IK failed to converge")
        
        # Example 2: Validate against real robot data (if available)
        try:
            logger.info("\n=== Validating Against Real Robot Data ===")
            validation_results = controller.validate_against_real_data(
                'third_20250710_162459.json', 
                num_samples=5
            )
            
            if 'error' not in validation_results:
                print("Validation Results:")
                print(f"  Mean position error: {validation_results['mean_position_error']:.6f} m")
                print(f"  Max position error:  {validation_results['max_position_error']:.6f} m")
                print(f"  Mean rotation error: {validation_results['mean_rotation_error']:.6f} rad "
                      f"({np.rad2deg(validation_results['mean_rotation_error']):.2f} deg)")
                print(f"  Max rotation error:  {validation_results['max_rotation_error']:.6f} rad "
                      f"({np.rad2deg(validation_results['max_rotation_error']):.2f} deg)")
                
                # Assessment
                if validation_results['mean_position_error'] < 0.005:
                    logger.info("✅ Position accuracy is excellent")
                else:
                    logger.warning("⚠️  Position accuracy may need improvement")
                
                if validation_results['mean_rotation_error'] < 0.1:
                    logger.info("✅ Rotation accuracy is excellent")
                else:
                    logger.warning("⚠️  Rotation accuracy may need improvement")
            else:
                logger.warning(f"Validation failed: {validation_results['error']}")
                
        except FileNotFoundError:
            logger.info("Real robot data file not found - skipping validation")
        
        # Display performance statistics
        logger.info("\n=== Performance Statistics ===")
        stats = controller.get_performance_stats()
        print(f"Forward kinematics calls: {stats['fk_calls']}")
        print(f"Inverse kinematics calls: {stats['ik_calls']}")
        print(f"IK success rate: {stats['ik_success_rate']:.1%}")
        print(f"Average IK time: {stats['avg_ik_time']:.4f} seconds")
        print(f"Max position error: {stats['max_position_error']:.2e} m")
        print(f"Max rotation error: {stats['max_rotation_error']:.2e} rad")
        
        logger.info("Application completed successfully")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    main()
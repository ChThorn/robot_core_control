#!/usr/bin/env python3
"""
Production-ready main application for robot kinematics with enhanced validation.
"""

import numpy as np
import logging
from robot_controller import RobotController

# Import the validation suite
from kinematics_validation import KinematicsValidator, run_comprehensive_validation

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
        
        # Example 1: Test forward and inverse kinematics (your existing test)
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
        
        # NEW: Comprehensive validation suite
        logger.info("\n" + "="*60)
        logger.info("=== COMPREHENSIVE KINEMATICS VALIDATION ===")
        logger.info("="*60)
        
        try:
            validation_results = run_comprehensive_validation(controller)
            
            # Additional analysis of results
            screw_results = validation_results['screw_axes']
            fk_ik_results = validation_results['fk_ik_consistency']
            workspace_results = validation_results['workspace_coverage']
            
            # Summary assessment
            logger.info("\n=== VALIDATION SUMMARY ===")
            
            # Screw axes assessment
            if screw_results['is_valid']:
                logger.info("✅ Screw axes: PASSED")
            else:
                logger.warning(f"⚠️  Screw axes: Check needed (max diff: {screw_results['max_difference']:.6f})")
            
            # FK-IK consistency assessment
            success_rate = fk_ik_results['success_rate']
            if success_rate > 0.95:
                logger.info(f"✅ IK success rate: EXCELLENT ({success_rate:.1%})")
            elif success_rate > 0.80:
                logger.info(f"⚠️  IK success rate: GOOD ({success_rate:.1%})")
            else:
                logger.warning(f"❌ IK success rate: NEEDS IMPROVEMENT ({success_rate:.1%})")
            
            # Accuracy assessment  
            mean_pos_mm = fk_ik_results['mean_pos_error'] * 1000
            mean_rot_deg = np.rad2deg(fk_ik_results['mean_rot_error'])
            
            if mean_pos_mm < 0.1:
                logger.info(f"✅ Position accuracy: EXCELLENT ({mean_pos_mm:.3f} mm)")
            elif mean_pos_mm < 1.0:
                logger.info(f"⚠️  Position accuracy: GOOD ({mean_pos_mm:.3f} mm)")
            else:
                logger.warning(f"❌ Position accuracy: NEEDS IMPROVEMENT ({mean_pos_mm:.3f} mm)")
                
            if mean_rot_deg < 0.1:
                logger.info(f"✅ Rotation accuracy: EXCELLENT ({mean_rot_deg:.3f}°)")
            elif mean_rot_deg < 1.0:
                logger.info(f"⚠️  Rotation accuracy: GOOD ({mean_rot_deg:.3f}°)")
            else:
                logger.warning(f"❌ Rotation accuracy: NEEDS IMPROVEMENT ({mean_rot_deg:.3f}°)")
            
            # Workspace coverage assessment
            coverage = workspace_results['coverage_percentage']
            if coverage > 80:
                logger.info(f"✅ Workspace coverage: EXCELLENT ({coverage:.1f}%)")
            elif coverage > 50:
                logger.info(f"⚠️  Workspace coverage: ADEQUATE ({coverage:.1f}%)")
            else:
                logger.warning(f"❌ Workspace coverage: LOW ({coverage:.1f}%)")
                
        except Exception as e:
            logger.error(f"Enhanced validation failed: {e}")
            logger.info("Continuing with basic validation...")
        
        # Example 2: Validate against real robot data (your existing validation)
        try:
            logger.info("\n=== Validating Against Real Robot Data ===")
            validation_results_real = controller.validate_against_real_data(
                'third_20250710_162459.json', 
                num_samples=5
            )
            
            if 'error' not in validation_results_real:
                print("Real Robot Validation Results:")
                print(f"  Mean position error: {validation_results_real['mean_position_error']:.6f} m")
                print(f"  Max position error:  {validation_results_real['max_position_error']:.6f} m")
                print(f"  Mean rotation error: {validation_results_real['mean_rotation_error']:.6f} rad "
                      f"({np.rad2deg(validation_results_real['mean_rotation_error']):.2f} deg)")
                print(f"  Max rotation error:  {validation_results_real['max_rotation_error']:.6f} rad "
                      f"({np.rad2deg(validation_results_real['max_rotation_error']):.2f} deg)")
                
                # Assessment
                if validation_results_real['mean_position_error'] < 0.005:
                    logger.info("✅ Real robot position accuracy is excellent")
                else:
                    logger.warning("⚠️  Real robot position accuracy may need improvement")
                
                if validation_results_real['mean_rotation_error'] < 0.1:
                    logger.info("✅ Real robot rotation accuracy is excellent")
                else:
                    logger.warning("⚠️  Real robot rotation accuracy may need improvement")
            else:
                logger.warning(f"Real robot validation failed: {validation_results_real['error']}")
                
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
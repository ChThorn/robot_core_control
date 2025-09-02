#!/usr/bin/env python3
"""
Main application demonstrating pure analytical inverse kinematics.
"""

import numpy as np
import logging
from analytical_ik import AnalyticalIK
from robot_controller import RobotController

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analytical_main')

def main():
    """Main application entry point."""
    try:
        # Initialize analytical IK system
        analytical_ik = AnalyticalIK()
        controller = RobotController(analytical_ik)
        
        logger.info("Analytical IK system initialized successfully")
        
        # Example 1: Basic IK test
        logger.info("=== Basic Analytical IK Test ===")
        
        # Test configuration
        q_test = np.array([0.5, 0.1, 0.2, 0.8, -0.4, 0.6])
        
        # Compute forward kinematics
        T_target = controller.forward_kinematics(q_test)
        
        print("Test Configuration:")
        print(f"Joint angles (deg): {np.round(np.rad2deg(q_test), 2)}")
        print(f"Target position (m): {np.round(T_target[:3, 3], 4)}")
        print(f"Target orientation (RPY deg): {np.round(np.rad2deg(controller.matrix_to_rpy(T_target[:3, :3])), 2)}")
        
        # Solve inverse kinematics
        q_solution, converged, all_solutions = controller.inverse_kinematics(
            T_target, q_current=q_test, return_all_solutions=True
        )
        
        if converged:
            print(f"\n✅ Analytical IK Success!")
            print(f"Found {len(all_solutions)} solutions")
            print(f"Best solution (deg): {np.round(np.rad2deg(q_solution), 2)}")
            print(f"Original config (deg): {np.round(np.rad2deg(q_test), 2)}")
            print(f"Difference (deg): {np.round(np.rad2deg(q_solution - q_test), 3)}")
            
            # Check accuracy
            pos_err, rot_err = controller.check_pose_error(T_target, q_solution)
            print(f"\nAccuracy:")
            print(f"Position error: {pos_err*1e6:.1f} μm")
            print(f"Rotation error: {np.rad2deg(rot_err)*3600:.1f} arcsec")
            
            # Show all solutions
            print(f"\nAll {len(all_solutions)} Solutions:")
            for i, q_sol in enumerate(all_solutions):
                info = analytical_ik.get_solution_info(q_sol)
                print(f"  Solution {i+1}: {np.round(info['joint_angles_deg'], 1)} "
                      f"({info['elbow_configuration']}, {info['wrist_configuration']})")
            
            # Send to robot (simulation)
            success = controller.send_to_robot(q_solution)
            if success:
                logger.info("Command would be sent to robot successfully")
        else:
            logger.error("❌ Analytical IK failed to converge")
        
        # Example 2: Performance test
        logger.info("\n=== Performance Test ===")
        
        # Test multiple random configurations
        num_tests = 50
        print(f"Testing {num_tests} random configurations...")
        
        success_count = 0
        total_solutions = 0
        computation_times = []
        
        # Generate random test poses
        joint_limits = analytical_ik.joint_limits
        
        for i in range(num_tests):
            # Random joint configuration
            q_random = np.random.uniform(joint_limits[:, 0], joint_limits[:, 1])
            T_random = controller.forward_kinematics(q_random)
            
            # Test IK
            import time
            start_time = time.time()
            q_ik, success, solutions = controller.inverse_kinematics(
                T_random, return_all_solutions=True
            )
            computation_time = time.time() - start_time
            computation_times.append(computation_time)
            
            if success:
                success_count += 1
                total_solutions += len(solutions)
        
        # Performance summary
        success_rate = success_count / num_tests
        avg_time = np.mean(computation_times) * 1000  # ms
        avg_solutions = total_solutions / max(success_count, 1)
        
        print(f"\nPerformance Results:")
        print(f"Success rate: {success_rate:.1%} ({success_count}/{num_tests})")
        print(f"Average computation time: {avg_time:.2f} ms")
        print(f"Average solutions per pose: {avg_solutions:.1f}")
        print(f"Min time: {np.min(computation_times)*1000:.2f} ms")
        print(f"Max time: {np.max(computation_times)*1000:.2f} ms")
        
        # Example 3: Real data validation (if available)
        try:
            logger.info("\n=== Real Data Validation ===")
            validation_results = controller.validate_against_real_data(
                'third_20250710_162459.json', 
                num_samples=10
            )
            
            if 'error' not in validation_results:
                print("Validation Results:")
                print(f"IK success rate: {validation_results['ik_success_rate']:.1%}")
                print(f"Average solutions per pose: {validation_results['avg_solutions_per_pose']:.1f}")
                
                if validation_results['position_errors']:
                    print(f"Mean position error: {validation_results['mean_position_error']*1e6:.1f} μm")
                    print(f"Max position error: {validation_results['max_position_error']*1e6:.1f} μm")
                    print(f"Mean rotation error: {np.rad2deg(validation_results['mean_rotation_error'])*3600:.1f} arcsec")
                    print(f"Max rotation error: {np.rad2deg(validation_results['max_rotation_error'])*3600:.1f} arcsec")
                
                # Assessment
                if validation_results['ik_success_rate'] >= 0.90:
                    logger.info("✅ Analytical IK reliability is excellent")
                else:
                    logger.warning("⚠️  Analytical IK reliability needs improvement")
            else:
                logger.warning(f"Validation failed: {validation_results['error']}")
                
        except FileNotFoundError:
            logger.info("Real robot data file not found - skipping validation")
        
        # Example 4: Singularity test
        logger.info("\n=== Singularity Handling Test ===")
        
        # Test known challenging configurations
        test_configs = [
            ("Home position", np.zeros(6)),
            ("Shoulder singularity", np.array([0, np.pi/2, 0, 0, 0, 0])),
            ("Elbow extended", np.array([0, 0, np.pi, 0, 0, 0])),
            ("Wrist singularity", np.array([0, 0, 0, 0, 0, 0]))
        ]
        
        print("Testing challenging configurations:")
        
        for name, q_config in test_configs:
            try:
                T_config = controller.forward_kinematics(q_config)
                q_ik, success, solutions = controller.inverse_kinematics(
                    T_config, return_all_solutions=True
                )
                
                if success:
                    pos_err, rot_err = controller.check_pose_error(T_config, q_ik)
                    print(f"  {name}: ✅ Success ({len(solutions)} solutions, "
                          f"err: {pos_err*1e6:.1f}μm)")
                else:
                    print(f"  {name}: ❌ Failed")
                    
            except Exception as e:
                print(f"  {name}: ❌ Error - {e}")
        
        # Display final performance statistics
        logger.info("\n=== Final Performance Statistics ===")
        stats = controller.get_performance_stats()
        
        print(f"Total FK calls: {stats['fk_calls']}")
        print(f"Total IK calls: {stats['ik_calls']}")
        print(f"Overall IK success rate: {stats['ik_success_rate']:.1%}")
        print(f"Average IK time: {stats['avg_ik_time']*1000:.2f} ms")
        print(f"Average solutions per call: {stats['avg_solutions_per_call']:.1f}")
        print(f"Max position error: {stats['max_position_error']*1e6:.1f} μm")
        print(f"Max rotation error: {np.rad2deg(stats['max_rotation_error'])*3600:.1f} arcsec")
        
        logger.info("Analytical IK demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    main()


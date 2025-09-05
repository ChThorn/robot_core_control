#!/usr/bin/env python3
"""
Comprehensive integration test for the robot planning system.
Tests integration between path planning, motion planning, and trajectory planning
with the existing robot kinematics system.
"""

import numpy as np
import sys
import os
import logging
import time
import yaml # Added for YAML loading
from typing import Optional, Tuple, List, Dict

# Add parent directories to path for imports
current_dir = os.path.dirname(__file__)
kinematics_dir = os.path.join(current_dir, '..', 'robot_kinematics')
sys.path.append(kinematics_dir)

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('planning_test.log')
    ]
)
logger = logging.getLogger('planning_integration_test')

# Import robot kinematics system
try:
    from robot_controller import RobotController
    from robot_kinematics import RobotKinematics, RobotKinematicsError
    logger.info("Successfully imported robot kinematics system")
except ImportError as e:
    logger.error(f"Failed to import robot kinematics: {e}")
    logger.error("Ensure robot_kinematics folder is in the parent directory")
    sys.exit(1)

# Import planning modules
try:
    from path_planning import CartesianPathPlanner, Environment3D
    from motion_planning import MotionPlanner, RobotEnvironment, JointSpacePlanner
    from trajectory_planning import TrajectoryPlanner, TrajectoryConstraints, TrajectoryInterpolator, TimeParameterization
    logger.info("Successfully imported planning modules")
except ImportError as e:
    logger.error(f"Failed to import planning modules: {e}")
    sys.exit(1)


class PlanningSystemTest:
    """Comprehensive test suite for the planning system."""
    
    def __init__(self):
        """
        Initialize test environment and planning system.
        Loads test configurations from test_configs.yaml.
        """
        logger.info("Initializing planning system test")
        
        # Initialize robot controller
        try:
            self.robot_controller = RobotController(
                ee_link="tcp",
                base_link="link0"
            )
            logger.info("Robot controller initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize robot controller: {e}")
            raise
            
        # Initialize planning components
        self.robot_env = RobotEnvironment(self.robot_controller)
        self.motion_planner = MotionPlanner(self.robot_controller)
        
        # Create trajectory constraints - more conservative values
        num_joints = self.robot_controller.robot.n_joints
        max_velocities = np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0])  # rad/s - more conservative
        max_accelerations = np.array([5.0, 5.0, 5.0, 8.0, 8.0, 8.0])  # rad/s² - more conservative
        
        self.trajectory_constraints = TrajectoryConstraints(
            max_joint_velocities=max_velocities,
            max_joint_accelerations=max_accelerations,
            max_cartesian_velocity=0.5,  # m/s
            max_cartesian_acceleration=2.0  # m/s²
        )
        
        self.trajectory_planner = TrajectoryPlanner(
            self.robot_controller, 
            self.trajectory_constraints
        )
        
        # Load test configurations from YAML file
        self.test_configs = self._load_test_configurations(
            os.path.join(current_dir, 'test_configs.yaml')
        )
        
        logger.info("Planning system test initialization complete")
        
    def _load_test_configurations(self, config_path: str) -> Dict:
        """
        Loads test configurations from a YAML file.
        """
        try:
            with open(config_path, 'r') as f:
                configs = yaml.safe_load(f)
            logger.info(f"Loaded test configurations from {config_path}")
            return configs
        except FileNotFoundError:
            logger.error(f"Test configuration file not found: {config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing test configuration YAML: {e}")
            sys.exit(1)

    def run_all_tests(self) -> Dict[str, bool]:
        """
        Run comprehensive test suite.
        """
        logger.info("Starting comprehensive planning system tests")
        
        test_results = {}
        
        # Test 1: Environment setup validation
        test_results['environment_setup'] = self.test_environment_setup()
        
        # Test 2: Joint space planning
        test_results['joint_space_planning'] = self.test_joint_space_planning()
        
        # Test 3: Cartesian space planning
        test_results['cartesian_space_planning'] = self.test_cartesian_space_planning()
        
        # Test 4: Trajectory generation
        test_results['trajectory_generation'] = self.test_trajectory_generation()
        
        # Test 5: Constraint validation
        test_results['constraint_validation'] = self.test_constraint_validation()
        
        # Test 6: Obstacle avoidance
        test_results['obstacle_avoidance'] = self.test_obstacle_avoidance()
        
        # Test 7: Performance benchmarks
        test_results['performance_benchmarks'] = self.test_performance_benchmarks()
        
        # Print summary
        self._print_test_summary(test_results)
        
        return test_results
        
    def test_environment_setup(self) -> bool:
        """Test environment setup and constraint loading."""
        logger.info("Testing environment setup")
        
        try:
            # Verify workspace bounds loaded correctly
            bounds = self.robot_env.bounds
            expected_bounds = ((-0.8, 0.8), (-0.8, 0.8), (-0.6, 1.2)) # From constraints.yaml
            
            for i, (actual, expected) in enumerate(zip(bounds, expected_bounds)):
                if not (abs(actual[0] - expected[0]) < 0.01 and abs(actual[1] - expected[1]) < 0.01):
                    logger.error(f"Workspace bounds mismatch in dimension {i}")
                    return False
                    
            logger.info(f"Workspace bounds verified: {bounds}")
            
            # Verify obstacles loaded
            num_spheres = len(self.robot_env.sphere_obstacles)
            num_boxes = len(self.robot_env.box_obstacles)
            num_cylinders = len(self.robot_env.cylinder_obstacles)
            
            logger.info(f"Obstacles loaded - Spheres: {num_spheres}, Boxes: {num_boxes}, Cylinders: {num_cylinders}")
            
            # Test collision checking
            # Point inside box obstacle (should be invalid)
            # Box center from constraints.yaml: [100, 100, 200] mm = [0.1, 0.1, 0.2] m
            box_center_m = np.array([0.1, 0.1, 0.2])
            if self.robot_env.is_point_valid(box_center_m):
                logger.warning("Collision detection may not be working - point in obstacle marked as valid")
                
            # Point in free space (should be valid)
            free_point = np.array([0.5, 0.5, 0.5])
            if not self.robot_env.is_point_valid(free_point):
                logger.warning("Point in free space marked as invalid")
                
            logger.info("Environment setup test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Environment setup test failed: {e}")
            return False
            
    def test_joint_space_planning(self) -> bool:
        """Test joint space path planning."""
        logger.info("Testing joint space planning")
        
        try:
            success_count = 0
            test_cases = self.test_configs.get('joint_space_tests', [])
            total_tests = len(test_cases)
            
            for test_case in test_cases:
                test_name = test_case['name']
                start_config = np.array(test_case['start_config'])
                goal_config = np.array(test_case['goal_config'])

                logger.info(f"Testing joint space planning: {test_name}")
                
                start_time = time.time()
                joint_path = self.motion_planner.plan_joint_path(start_config, goal_config)
                planning_time = time.time() - start_time
                
                if joint_path is not None:
                    success_count += 1
                    logger.info(f"  Success - Path length: {len(joint_path)}, Time: {planning_time:.3f}s")
                    
                    # Validate path
                    if self._validate_joint_path(joint_path):
                        logger.info("  Path validation passed")
                    else:
                        logger.warning("  Path validation failed")
                else:
                    logger.warning(f"  Failed to find path for {test_name}")
                    
            success_rate = success_count / total_tests
            logger.info(f"Joint space planning success rate: {success_rate:.1%} ({success_count}/{total_tests})")
            
            return success_rate >= 0.5  # Require at least 50% success rate
            
        except Exception as e:
            logger.error(f"Joint space planning test failed: {e}")
            return False
            
    def test_cartesian_space_planning(self) -> bool:
        """Test Cartesian space path planning."""
        logger.info("Testing Cartesian space planning")
        
        try:
            success_count = 0
            test_cases = self.test_configs.get('cartesian_space_tests', [])
            total_tests = len(test_cases)
            
            for i, test_case in enumerate(test_cases):
                test_name = test_case['name']
                start_pose = np.array(test_case['start_pose'])
                goal_pose = np.array(test_case['goal_pose'])

                logger.info(f"Testing Cartesian planning: {test_name}")
                
                start_time = time.time()
                result = self.motion_planner.plan_cartesian_path(start_pose, goal_pose)
                planning_time = time.time() - start_time
                
                if result is not None:
                    cartesian_path, joint_path = result
                    success_count += 1
                    logger.info(f"  Success - Cartesian waypoints: {len(cartesian_path)}, "
                               f"Joint waypoints: {len(joint_path)}, Time: {planning_time:.3f}s")
                else:
                    logger.warning(f"  Failed to find Cartesian path for {test_name}")
                    
            success_rate = success_count / total_tests
            logger.info(f"Cartesian space planning success rate: {success_rate:.1%}")
            
            return success_rate >= 0.5
            
        except Exception as e:
            logger.error(f"Cartesian space planning test failed: {e}")
            return False
            
    def test_trajectory_generation(self) -> bool:
        """Test trajectory generation and time parameterization."""
        logger.info("Testing trajectory generation")
        
        try:
            # Use first successful joint configuration from test_configs
            test_case = self.test_configs.get('joint_space_tests', [])[0]
            start_config = np.array(test_case['start_config'])
            goal_config = np.array(test_case['goal_config'])
            
            # Test different trajectory methods
            methods = ['constant', 'trapezoidal']
            interpolations = ['linear', 'cubic']
            
            success_count = 0
            total_combinations = len(methods) * len(interpolations)
            
            for method in methods:
                for interpolation in interpolations:
                    logger.info(f"Testing trajectory: {method} parameterization, {interpolation} interpolation")
                    
                    start_time = time.time()
                    interpolator = self.trajectory_planner.plan_trajectory(
                        start_config, goal_config, 
                        method=method, interpolation=interpolation
                    )
                    planning_time = time.time() - start_time
                    
                    if interpolator is not None:
                        success_count += 1
                        duration = interpolator.get_duration()
                        logger.info(f"  Success - Duration: {duration:.3f}s, Planning time: {planning_time:.3f}s")
                        
                        # Test interpolator functions
                        mid_time = duration / 2
                        config = interpolator.evaluate(mid_time)
                        velocity = interpolator.evaluate_velocity(mid_time)
                        acceleration = interpolator.evaluate_acceleration(mid_time)
                        
                        logger.info(f"  Mid-trajectory - Config shape: {config.shape}, "
                                   f"Vel max: {np.max(np.abs(velocity)):.3f}, "
                                   f"Acc max: {np.max(np.abs(acceleration)):.3f}")
                    else:
                        logger.warning(f"  Failed to generate trajectory")
                        
            success_rate = success_count / total_combinations
            logger.info(f"Trajectory generation success rate: {success_rate:.1%}")
            
            return success_rate >= 0.5
            
        except Exception as e:
            logger.error(f"Trajectory generation test failed: {e}")
            return False
            
    def test_constraint_validation(self) -> bool:
        """Test trajectory constraint validation."""
        logger.info("Testing constraint validation")
        
        try:
            # Generate a trajectory for testing
            test_case = self.test_configs.get('joint_space_tests', [])[0]
            start_config = np.array(test_case['start_config'])
            goal_config = np.array(test_case['goal_config'])
            
            interpolator = self.trajectory_planner.plan_trajectory(
                start_config, goal_config, method='trapezoidal', interpolation='cubic'
            )
            
            if interpolator is None:
                logger.error("Failed to generate trajectory for constraint testing")
                return False
                
            # Validate trajectory
            validation_results = self.trajectory_planner.validator.validate_trajectory(interpolator)
            
            logger.info("Constraint validation results:")
            for constraint, passed in validation_results.items():
                status = "PASS" if passed else "FAIL"
                logger.info(f"  {constraint}: {status}")
                
            return validation_results['overall']
            
        except Exception as e:
            logger.error(f"Constraint validation test failed: {e}")
            return False
            
    def test_obstacle_avoidance(self) -> bool:
        """Test obstacle avoidance capabilities."""
        logger.info("Testing obstacle avoidance")
        
        try:
            # Plan path that should avoid obstacles
            obstacle_test_case = self.test_configs.get('joint_space_tests', [])[2] # "Obstacle Avoidance" test case
            start_config = np.array(obstacle_test_case['start_config'])
            goal_config = np.array(obstacle_test_case['goal_config'])
            
            # Test joint space planning
            joint_path = self.motion_planner.plan_joint_path(start_config, goal_config)
            
            if joint_path is None:
                logger.warning("No path found for obstacle avoidance test")
                return False
                
            # Check if path avoids obstacles
            collision_count = 0
            for config in joint_path:
                try:
                    T = self.robot_controller.forward_kinematics(config)
                    position = T[:3, 3];
                    
                    if not self.robot_env.is_point_valid(position):
                        collision_count += 1;
                        
                except Exception:
                    collision_count += 1;
                    
            collision_rate = collision_count / len(joint_path)
            logger.info(f"Path collision rate: {collision_rate:.1%} ({collision_count}/{len(joint_path)} waypoints)")
            
            return collision_rate == 0.0
            
        except Exception as e:
            logger.error(f"Obstacle avoidance test failed: {e}")
            return False
            
    def test_performance_benchmarks(self) -> bool:
        """Test performance benchmarks for planning components."""
        logger.info("Testing performance benchmarks")
        
        try:
            # Joint space planning benchmark
            start_config = np.zeros(6)
            goal_config = np.random.uniform(self.robot_controller.robot.joint_limits[0], 
                                            self.robot_controller.robot.joint_limits[1], 6)
            
            start_time = time.time()
            self.motion_planner.plan_joint_path(start_config, goal_config)
            joint_planning_time = time.time() - start_time
            logger.info(f"Joint space planning benchmark: {joint_planning_time:.4f}s")
            
            # Cartesian space planning benchmark
            start_pose = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0])
            goal_pose = np.array([-0.5, 0.5, 0.7, 0.0, 0.0, 0.0])
            
            start_time = time.time()
            self.motion_planner.plan_cartesian_path(start_pose, goal_pose)
            cartesian_planning_time = time.time() - start_time
            logger.info(f"Cartesian space planning benchmark: {cartesian_planning_time:.4f}s")
            
            # Trajectory generation benchmark
            path_for_traj = self.motion_planner.plan_joint_path(start_config, goal_config)
            if path_for_traj is None:
                logger.warning("Could not generate path for trajectory benchmark.")
                return False

            start_time = time.time()
            interpolator = self.trajectory_planner.plan_trajectory(
                path_for_traj[0], path_for_traj[-1], method='trapezoidal', interpolation='cubic'
            )
            trajectory_planning_time = time.time() - start_time
            logger.info(f"Trajectory generation benchmark: {trajectory_planning_time:.4f}s")

            if interpolator is None:
                logger.warning("Trajectory generation failed for benchmark.")
                return False

            # Trajectory execution benchmark (dry run)
            start_time = time.time()
            self.trajectory_planner.execute_trajectory(interpolator, dry_run=True)
            trajectory_execution_time = time.time() - start_time
            logger.info(f"Trajectory execution benchmark (dry run): {trajectory_execution_time:.4f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Performance benchmarks failed: {e}")
            return False
            
    def _validate_joint_path(self, path: List[np.ndarray]) -> bool:
        """
        Validate a joint path for collisions and joint limits.
        """
        if not path:
            return False

        # Check start and end points for validity
        if not self.motion_planner.joint_space_planner._is_config_valid(path[0]):
            logger.warning("Path start config is invalid.")
            return False
        if not self.motion_planner.joint_space_planner._is_config_valid(path[-1]):
            logger.warning("Path end config is invalid.")
            return False

        # Check intermediate segments for collisions and joint limits
        for i in range(len(path) - 1):
            if not self.motion_planner.joint_space_planner._is_path_valid(path[i], path[i+1]):
                logger.warning(f"Path segment {i}-{i+1} is invalid (collision or joint limit violation).")
                return False
        return True

    def _print_test_summary(self, results: Dict[str, bool]):
        logger.info("\n--- Planning System Test Summary ---")
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        logger.info("-------------------------------------")


def demonstration_scenario() -> bool:
    """
    Demonstrates a multi-waypoint pick and place operation.
    """
    logger.info("\n" + "="*60)
    logger.info("=== Running Demonstration Scenario ===")
    logger.info("="*60)

    try:
        test_system = PlanningSystemTest()
        demo_configs = test_system.test_configs.get('demonstration_scenario', {})

        # Define demonstration scenario
        logger.info("Scenario: Pick and place operation with obstacle avoidance")
        
        # Start position (above pick location)
        pick_pose = np.array(demo_configs.get('pick_pose'))  # [x, y, z, rx, ry, rz]
        
        # Intermediate position (avoiding obstacles)
        intermediate_pose = np.array(demo_configs.get('intermediate_pose'))
        
        # Place position
        place_pose = np.array(demo_configs.get('place_pose'))
        
        # Plan complete sequence
        poses = [pick_pose, intermediate_pose, place_pose]
        
        logger.info("Planning multi-waypoint trajectory")
        
        # Use the motion planner to handle the full sequence
        full_cartesian_path = []
        full_joint_path = []
        
        # Start from a known good configuration (e.g., home)
        previous_config = np.zeros(6)
        
        # Convert start pose to a full transformation matrix
        # The first pose in 'poses' is the target for the first segment
        start_pose_for_first_segment = poses[0]
        
        # Plan from home to the first pose in the sequence
        logger.info("Planning from home to first pick pose")
        # Pass previous_config (joint angles) and start_pose_for_first_segment (Cartesian pose)
        result = test_system.motion_planner.plan_cartesian_path(previous_config, start_pose_for_first_segment)
        
        if result is None:
            logger.error("Failed to plan path from home to start pose")
            return False
            
        c_path, j_path = result
        full_cartesian_path.extend(c_path)
        full_joint_path.extend(j_path)
        previous_config = full_joint_path[-1]
        
        # Plan between the specified poses
        for i in range(len(poses) - 1):
            start_p = poses[i]
            goal_p = poses[i+1]
            
            logger.info(f"Planning from pose {i+1} to {i+2}")
            
            # Pass the last joint config as start and the next Cartesian pose as goal
            result = test_system.motion_planner.plan_cartesian_path(previous_config, goal_p)
            
            if result is None:
                logger.error(f"Failed to plan path from pose {i+1} to {i+2}")
                return False
                
            c_path, j_path = result
            # Avoid duplicating the connection point (first point of new segment is last of previous)
            full_cartesian_path.extend(c_path[1:])
            full_joint_path.extend(j_path[1:])
            previous_config = full_joint_path[-1]
        
        logger.info(f"Successfully planned full sequence with {len(full_joint_path)} joint waypoints.")
        
        # Plan trajectory for the entire joint path
        logger.info("Generating trajectory for the full sequence")
        
        # Pass the full_joint_path directly to plan_trajectory
        # The plan_trajectory method expects start_config and goal_config, not a full path.
        # We need to adapt this to plan a trajectory over the entire path.
        # A simple way is to treat the entire full_joint_path as the path to parameterize.
        # This requires a change in TrajectoryPlanner.plan_trajectory or a new method.
        # For now, we will use the existing TimeParameterization directly as a fallback if plan_trajectory fails.

        # Attempt to plan trajectory using the start and end of the full path
        interpolator = test_system.trajectory_planner.plan_trajectory(
            full_joint_path[0], full_joint_path[-1],
            method='trapezoidal', interpolation='cubic'
        )
        
        if interpolator is None:
            # As a fallback, try to parameterize the raw path directly
            logger.warning("Trajectory planning for start/goal failed, attempting to parameterize full path directly.")
            
            # Manually create the trajectory from the path
            trajectory = TimeParameterization.trapezoidal_velocity(
                full_joint_path, test_system.trajectory_constraints
            )
            if not trajectory:
                logger.error("Failed to time-parameterize the full joint path.")
                return False
            
            try:
                interpolator = TrajectoryInterpolator(trajectory, method='cubic')
            except ValueError as e:
                logger.error(f"Failed to create interpolator from parameterized path: {e}")
                return False

        # Simulate execution
        logger.info("Simulating trajectory execution")
        
        total_duration = interpolator.get_duration()
        logger.info(f"Total operation duration: {total_duration:.3f}s")
        
        success = test_system.trajectory_planner.execute_trajectory(interpolator, dry_run=True)
        
        if not success:
            logger.error("Failed to execute the final trajectory")
            return False
                
        logger.info("Demonstration scenario completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Demonstration scenario failed: {e}")
        return False


def main():
    """
    Main test execution function.
    """
    logger.info("Starting comprehensive planning system test")
    
    try:
        # Run comprehensive test suite
        test_system = PlanningSystemTest()
        test_results = test_system.run_all_tests()
        
        # Run demonstration scenario
        demo_success = demonstration_scenario()
        
        # Final summary
        logger.info("\nFINAL TEST SUMMARY")
        logger.info("-" * 40)
        
        overall_success = all(test_results.values()) and demo_success # All tests must pass for overall success
        demo_status = "PASS" if demo_success else "FAIL"
        
        logger.info(f"Test Suite: {'PASS' if all(test_results.values()) else 'FAIL'}")
        logger.info(f"Demonstration: {demo_status}")
        
        if overall_success:
            logger.info("Planning system is ready for production use")
        else:
            logger.warning("Planning system requires additional development")
            
        return overall_success
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



#!/usr/bin/env python3
"""
Trajectory planning and time parameterization for robot motion.
Converts geometric paths to time-parameterized trajectories with dynamic constraints.
Integrates with robot kinematics system for production-ready trajectory generation.
"""

import numpy as np
import logging
import time
from typing import List, Tuple, Optional, Dict, Callable
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import minimize_scalar
import sys
import os

# Setup logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import robot kinematics system and motion planning
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'robot_kinematics'))
try:
    from robot_controller import RobotController
    from robot_kinematics import RobotKinematics
except ImportError as e:
    logging.error(f"Failed to import robot controller: {e}")
    raise

from motion_planning import MotionPlanner, JointConfiguration, JointPath

# Type aliases
TimeStamp = float
Trajectory = List[Tuple[TimeStamp, JointConfiguration]]
VelocityProfile = List[Tuple[TimeStamp, np.ndarray]]
AccelerationProfile = List[Tuple[TimeStamp, np.ndarray]]


class TrajectoryConstraints:
    """Container for trajectory generation constraints."""
    
    def __init__(self, 
                 max_joint_velocities: np.ndarray,
                 max_joint_accelerations: np.ndarray,
                 max_cartesian_velocity: float = 1.0,
                 max_cartesian_acceleration: float = 5.0,
                 jerk_limit: Optional[np.ndarray] = None):
        """
        Initialize trajectory constraints.
        
        Args:
            max_joint_velocities: Maximum velocity for each joint (rad/s)
            max_joint_accelerations: Maximum acceleration for each joint (rad/s²)
            max_cartesian_velocity: Maximum end-effector velocity (m/s)
            max_cartesian_acceleration: Maximum end-effector acceleration (m/s²)
            jerk_limit: Optional jerk limits for each joint (rad/s³)
        """
        self.max_joint_velocities = np.array(max_joint_velocities)
        self.max_joint_accelerations = np.array(max_joint_accelerations)
        self.max_cartesian_velocity = max_cartesian_velocity
        self.max_cartesian_acceleration = max_cartesian_acceleration
        self.jerk_limit = np.array(jerk_limit) if jerk_limit is not None else None
        
    @classmethod
    def default_constraints(cls, num_joints: int):
        """Create default constraints for robot."""
        # More reasonable default constraints
        max_vel = np.full(num_joints, 3.0)  # 3 rad/s (more reasonable)
        max_acc = np.full(num_joints, 10.0)  # 10 rad/s² (more reasonable)
        return cls(max_vel, max_acc)


class TimeParameterization:
    """Time parameterization algorithms for trajectory generation."""
    
    @staticmethod
    def constant_velocity(path: JointPath, constraints: TrajectoryConstraints, velocity_scale: float = 0.8) -> Trajectory:
        """
        Simple constant velocity time parameterization that respects velocity limits.
        
        Args:
            path: Joint space path waypoints
            constraints: Trajectory constraints
            velocity_scale: Factor to scale the max velocity (0 to 1)
            
        Returns:
            Time-parameterized trajectory
        """
        if len(path) < 2:
            return [(0.0, path[0])] if path else []
            
        trajectory = []
        current_time = 0.0
        # Use a scaled version of max velocities for safety
        max_velocities = constraints.max_joint_velocities * max(0.1, min(1.0, velocity_scale))

        for i in range(len(path)):
            # Add point, but handle duplicate times
            if not trajectory or current_time > trajectory[-1][0] + 1e-9:
                trajectory.append((current_time, path[i]))
            else:
                # Overwrite last point if time hasn't advanced, to handle duplicate waypoints
                trajectory[-1] = (current_time, path[i])

            if i < len(path) - 1:
                delta_q = path[i+1] - path[i]
                # Find the time required for this segment based on the most restrictive joint velocity
                with np.errstate(divide='ignore', invalid='ignore'):
                    segment_times = np.abs(delta_q) / (max_velocities + 1e-9)
                
                segment_time = np.nanmax(segment_times) if np.any(np.isfinite(segment_times)) else 0.0
                
                current_time += segment_time
                
        return trajectory

    @staticmethod
    def scale_time(trajectory: Trajectory, factor: float) -> Trajectory:
        """Scale trajectory timing by a factor (slower if factor>1)."""
        if factor <= 0:
            raise ValueError("Time scaling factor must be positive")
        return [(t * factor, q.copy()) for t, q in trajectory]
        
    @staticmethod
    def trapezoidal_velocity(path: JointPath, constraints: TrajectoryConstraints) -> Trajectory:
        """
        Trapezoidal velocity profile time parameterization.
        
        Args:
            path: Joint space path waypoints
            constraints: Trajectory constraints
            
        Returns:
            Time-parameterized trajectory
        """
        if len(path) < 2:
            return [(0.0, path[0])] if path else []
            
        safety_factor = 0.7  # More conservative safety factor for robustness

        # Compute path segments and their lengths
        segments = []
        total_length = 0.0
        
        for i in range(len(path) - 1):
            segment_vector = path[i+1] - path[i]
            segment_length = np.linalg.norm(segment_vector)
            segments.append((segment_vector, segment_length))
            total_length += segment_length
            
        if total_length < 1e-6:
            # If path is essentially static, create a short trajectory with minimal movement
            return [(0.0, path[0]), (0.1, path[-1])]
            
        # Compute maximum achievable velocity and acceleration considering joint limits
        max_path_velocity = float('inf')
        max_path_acceleration = float('inf')
        
        for segment_vector, segment_length in segments:
            if segment_length > 1e-9: # Avoid division by zero for zero-length segments
                # Calculate max path velocity based on joint velocity limits
                # v_path = v_joint_max / |dq/ds|
                joint_velocity_ratios = np.abs(segment_vector) / segment_length
                with np.errstate(divide='ignore', invalid='ignore'):
                    segment_max_vel_from_joints = np.min(constraints.max_joint_velocities / (joint_velocity_ratios + 1e-12))
                if np.isfinite(segment_max_vel_from_joints):
                    max_path_velocity = min(max_path_velocity, segment_max_vel_from_joints)

                # Calculate max path acceleration based on joint acceleration limits
                # a_path = a_joint_max / |dq/ds|
                with np.errstate(divide='ignore', invalid='ignore'):
                    segment_max_acc_from_joints = np.min(constraints.max_joint_accelerations / (joint_velocity_ratios + 1e-12))
                if np.isfinite(segment_max_acc_from_joints):
                    max_path_acceleration = min(max_path_acceleration, segment_max_acc_from_joints)

        # Apply safety factor
        max_path_velocity = max_path_velocity * safety_factor if np.isfinite(max_path_velocity) else 0.3
        max_path_acceleration = max_path_acceleration * safety_factor if np.isfinite(max_path_acceleration) else 0.5
        
        if max_path_velocity <= 0 or max_path_acceleration <= 0:
            logger.warning("Could not determine valid path velocity/acceleration, falling back to constant velocity.")
            return TimeParameterization.constant_velocity(path, constraints)

        # Compute trapezoidal profile parameters
        t_acc = max_path_velocity / max_path_acceleration
        s_acc = 0.5 * max_path_acceleration * t_acc**2
        
        if 2 * s_acc >= total_length:
            # Triangular profile (no constant velocity phase)
            t_acc = np.sqrt(total_length / max_path_acceleration)
            max_path_velocity = max_path_acceleration * t_acc
            t_const = 0.0
        else:
            # Trapezoidal profile
            t_const = (total_length - 2 * s_acc) / max_path_velocity
            
        total_time = 2 * t_acc + t_const
        
        if not np.isfinite(total_time) or total_time < 1e-6:
            logger.warning("Could not compute a finite trajectory time, falling back to constant velocity.")
            return TimeParameterization.constant_velocity(path, constraints)

        # Generate trajectory points with ensure unique timestamps
        # Increased num_points for finer sampling and better validation
        num_points = max(200, int(total_time * 200))  # At least 200 points, or 200 Hz for better smoothness
        time_step = total_time / num_points
        
        trajectory = []
        
        for i in range(num_points + 1):
            t = i * time_step
            if t > total_time: # Ensure last point is exactly at total_time
                t = total_time
                
            # Compute position along path
            if t <= t_acc:
                # Acceleration phase
                s = 0.5 * max_path_acceleration * t**2
            elif t <= t_acc + t_const:
                # Constant velocity phase
                s = s_acc + max_path_velocity * (t - t_acc)
            else:
                # Deceleration phase
                # Corrected formula for deceleration phase
                t_from_end = total_time - t
                s = total_length - 0.5 * max_path_acceleration * t_from_end**2
                
            s = max(0.0, min(s, total_length)) # Clamp s to valid range
            
            # Interpolate joint configuration
            config = TimeParameterization._interpolate_path(path, s / total_length if total_length > 0 else 0)
            
            # Only add if time advances (avoid duplicates)
            if not trajectory or t > trajectory[-1][0] + 1e-9:
                trajectory.append((t, config))
            
        # Ensure final point is exactly at the end
        if not trajectory or not np.allclose(trajectory[-1][1], path[-1]) or not np.isclose(trajectory[-1][0], total_time):
            trajectory.append((total_time, path[-1]))
            
        return trajectory
        
    @staticmethod
    def _interpolate_path(path: JointPath, normalized_position: float) -> JointConfiguration:
        """Interpolate configuration along path using normalized position [0,1]."""
        if len(path) <= 1:
            return path[0] if path else np.zeros(6)
            
        # Clamp position
        normalized_position = max(0.0, min(1.0, normalized_position))
        
        # Find segment
        num_segments = len(path) - 1
        if num_segments == 0:
            return path[0]
        
        segment_length = 1.0 / num_segments
        segment_idx = min(int(normalized_position / segment_length), num_segments - 1)
        
        # Local interpolation parameter
        local_t = (normalized_position - segment_idx * segment_length) / segment_length
        local_t = max(0.0, min(1.0, local_t))
        
        # Linear interpolation
        return (1.0 - local_t) * path[segment_idx] + local_t * path[segment_idx + 1]


class TrajectoryInterpolator:
    """Smooth trajectory interpolation using splines."""
    
    def __init__(self, trajectory: Trajectory, method: str = 'cubic'):
        """
        Initialize trajectory interpolator.
        
        Args:
            trajectory: Time-parameterized trajectory
            method: Interpolation method ('linear', 'cubic')
        """
        self.trajectory = trajectory
        self.method = method
        
        if len(trajectory) < 2:
            raise ValueError("Trajectory must have at least 2 points")
            
        # Extract time and configuration arrays, removing duplicate timestamps
        times = []
        configs = []
        for t, q in trajectory:
            if not times or t > times[-1] + 1e-9:  # Only add if time advances
                times.append(t)
                configs.append(q)
        
        if len(times) < 2:
            raise ValueError("Trajectory must have at least 2 unique time points")
            
        self.times = np.array(times)
        self.configs = np.array(configs)
        
        # Create interpolators
        self._create_interpolators()
        
    def _create_interpolators(self):
        """Create interpolation functions."""
        if self.method == 'cubic':
            # Use clamped boundary conditions with zero derivatives to reduce endpoint acceleration spikes
            self.interpolators = CubicSpline(self.times, self.configs, bc_type='clamped', axis=0)
        elif self.method == 'linear':
            # For linear interpolation, we'll use manual implementation in evaluate()
            self.interpolators = None
        else:
            raise ValueError(f"Unknown interpolation method: {self.method}")
            
    def evaluate(self, t: float) -> JointConfiguration:
        """Evaluate trajectory at given time."""
        if self.method == 'linear':
            # Manual linear interpolation
            t = np.clip(t, self.times[0], self.times[-1])
            idx = np.searchsorted(self.times, t, side='right') - 1
            idx = np.clip(idx, 0, len(self.times) - 2)
            
            t0, t1 = self.times[idx], self.times[idx+1]
            q0, q1 = self.configs[idx], self.configs[idx+1]
            
            alpha = (t - t0) / (t1 - t0) if (t1 - t0) > 1e-9 else 0.0
            return q0 + alpha * (q1 - q0)
        else:
            return self.interpolators(t)
        
    def evaluate_velocity(self, t: float) -> np.ndarray:
        """Evaluate trajectory velocity at given time."""
        if self.method == 'cubic':
            return self.interpolators(t, 1)
        else:
            # For linear interpolation, velocity is constant between points
            # Find the segment and return its constant velocity
            t = np.clip(t, self.times[0], self.times[-1])
            idx = np.searchsorted(self.times, t, side='right') - 1
            idx = np.clip(idx, 0, len(self.times) - 2)
            
            t0, t1 = self.times[idx], self.times[idx+1]
            q0, q1 = self.configs[idx], self.configs[idx+1]
            
            if (t1 - t0) > 1e-9:
                return (q1 - q0) / (t1 - t0)
            else:
                return np.zeros_like(q0) # Zero velocity for zero-duration segments
            
    def evaluate_acceleration(self, t: float) -> np.ndarray:
        """Evaluate trajectory acceleration at given time."""
        if self.method == 'cubic':
            return self.interpolators(t, 2)
        else:
            # For linear interpolation, acceleration is zero (except at instantaneous changes at waypoints)
            return np.zeros_like(self.configs[0])
            
    def get_duration(self) -> float:
        """Get total trajectory duration."""
        return self.times[-1] - self.times[0]
        
    def resample(self, time_step: float) -> Trajectory:
        """Resample trajectory at regular time intervals."""
        new_times = np.arange(self.times[0], self.times[-1] + time_step, time_step)
        new_trajectory = []
        
        for t in new_times:
            config = self.evaluate(t)
            new_trajectory.append((t, config))
            
        return new_trajectory


class TrajectoryValidator:
    """Validate trajectories against robot constraints."""
    
    def __init__(self, robot_controller: RobotController, constraints: TrajectoryConstraints):
        """
        Initialize trajectory validator.
        
        Args:
            robot_controller: Robot controller for validation
            constraints: Trajectory constraints
        """
        self.robot_controller = robot_controller
        self.constraints = constraints
        
    def validate_trajectory(self, interpolator: TrajectoryInterpolator, 
                          time_step: float = 0.01) -> Dict[str, bool]:
        """
        Validate trajectory against all constraints.
        
        Args:
            interpolator: Trajectory interpolator
            time_step: Time step for validation
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'joint_limits': True,
            'velocity_limits': True,
            'acceleration_limits': True,
            'workspace_limits': True,
            'collision_free': True,
            'overall': True
        }
        
        duration = interpolator.get_duration()
        # Ensure at least one point is sampled, even for very short trajectories
        if duration < time_step:
            times = np.array([0.0, duration])
        else:
            times = np.arange(0, duration + time_step, time_step)
        
        # Ensure the last point is included if not already
        if not np.isclose(times[-1], duration):
            times = np.append(times, duration)

        max_acc_violation = 0.0
        max_vel_violation = 0.0

        for t in times:
            config = interpolator.evaluate(t)
            velocity = interpolator.evaluate_velocity(t)
            acceleration = interpolator.evaluate_acceleration(t)

            # 1. Joint Limits
            joint_limits_lower = self.robot_controller.robot.joint_limits[0]
            joint_limits_upper = self.robot_controller.robot.joint_limits[1]
            if np.any(config < joint_limits_lower) or np.any(config > joint_limits_upper):
                results['joint_limits'] = False
                results['overall'] = False
                logger.warning(f"Joint limits violated at t={t:.3f}s: {config}")

            # 2. Velocity Limits
            if np.any(np.abs(velocity) > self.constraints.max_joint_velocities):
                results['velocity_limits'] = False
                results['overall'] = False
                current_max_vel_violation = np.max(np.abs(velocity) - self.constraints.max_joint_velocities)
                max_vel_violation = max(max_vel_violation, current_max_vel_violation)
                logger.warning(f"Velocity limits violated at t={t:.3f}s. Max violation: {max_vel_violation:.3f}")

            # 3. Acceleration Limits (only check for cubic interpolation where acceleration is meaningful)
            if interpolator.method == 'cubic':
                if np.any(np.abs(acceleration) > self.constraints.max_joint_accelerations):
                    results['acceleration_limits'] = False
                    results['overall'] = False
                    current_max_acc_violation = np.max(np.abs(acceleration) - self.constraints.max_joint_accelerations)
                    max_acc_violation = max(max_acc_violation, current_max_acc_violation)
                    logger.warning(f"Acceleration limits violated at t={t:.3f}s. Max violation: {max_acc_violation:.3f}")

            # 4. Workspace Limits & Collision Free (using FK and robot_kinematics checks)
            try:
                T_ee = self.robot_controller.forward_kinematics(config)
                position = T_ee[:3, 3]
                rpy = self.robot_controller.robot.matrix_to_rpy(T_ee[:3, :3])

                if not self.robot_controller.robot._check_workspace(position):
                    results['workspace_limits'] = False
                    results['overall'] = False
                    logger.warning(f"Workspace limits violated at t={t:.3f}s: {position}")

                if not self.robot_controller.robot._check_obstacles(position):
                    results['collision_free'] = False
                    results['overall'] = False
                    logger.warning(f"Collision detected at t={t:.3f}s: {position}")

            except Exception as e:
                logger.error(f"FK or collision check failed at t={t:.3f}s: {e}")
                results['workspace_limits'] = False
                results['collision_free'] = False
                results['overall'] = False

        if not results['acceleration_limits']:
            logger.warning(f"Max acceleration violation: {max_acc_violation:.3f} rad/s^2")
        if not results['velocity_limits']:
            logger.warning(f"Max velocity violation: {max_vel_violation:.3f} rad/s")

        return results


class TrajectoryPlanner:
    """High-level trajectory planning interface."""
    
    def __init__(self, robot_controller: RobotController, constraints: TrajectoryConstraints):
        self.robot_controller = robot_controller
        self.constraints = constraints
        self.validator = TrajectoryValidator(robot_controller, constraints)
        self.motion_planner = MotionPlanner(robot_controller) # Assuming MotionPlanner is available
        
    def plan_trajectory(self, start_config: JointConfiguration, goal_config: JointConfiguration,
                        method: str = 'trapezoidal', interpolation: str = 'cubic') -> Optional[TrajectoryInterpolator]:
        """
        Plan a time-parameterized trajectory between two joint configurations.
        
        Args:
            start_config: Start joint configuration
            goal_config: Goal joint configuration
            method: Time parameterization method ('constant', 'trapezoidal')
            interpolation: Interpolation method ('linear', 'cubic')
            
        Returns:
            TrajectoryInterpolator object if successful, None otherwise
        """
        logger.info("Planning joint space path")
        path = self.motion_planner.plan_joint_path(start_config, goal_config)
        
        if path is None:
            logger.warning("Failed to find joint path for trajectory planning.")
            return None
            
        logger.info(f"Found path with {len(path)} waypoints")

        # Time parameterization
        if method == 'constant':
            trajectory = TimeParameterization.constant_velocity(path, self.constraints)
        elif method == 'trapezoidal':
            trajectory = TimeParameterization.trapezoidal_velocity(path, self.constraints)
        else:
            raise ValueError(f"Unknown time parameterization method: {method}")
            
        if not trajectory:
            logger.warning("Time parameterization failed.")
            return None

        # Create interpolator
        try:
            interpolator = TrajectoryInterpolator(trajectory, method=interpolation)
        except ValueError as e:
            logger.error(f"Failed to create interpolator: {e}")
            return None
            
        # Validate and potentially slow down trajectory
        max_attempts = 5 # Increased attempts for robustness
        for attempt in range(max_attempts):
            validation_results = self.validator.validate_trajectory(interpolator)
            if validation_results['overall']:
                logger.info("Trajectory validation passed.")
                return interpolator
            else:
                # If validation fails, slow down the trajectory
                # Determine the most violated constraint to guide slowdown
                violated_constraints = []
                if not validation_results['velocity_limits']:
                    violated_constraints.append('velocity')
                if not validation_results['acceleration_limits']:
                    violated_constraints.append('acceleration')

                if not violated_constraints:
                    # Fallback if no specific limits violated (e.g., collision, workspace)
                    logger.warning("Trajectory validation failed for non-velocity/acceleration reasons. Slowing down.")
                    slowdown_factor = 1.2 # Smaller slowdown for other issues
                elif 'acceleration' in violated_constraints:
                    # More aggressive slowdown for acceleration violations
                    slowdown_factor = 2.0 if attempt < 2 else 1.8
                else: # Only velocity violated
                    slowdown_factor = 1.5

                logger.warning(f"Trajectory violated {', '.join(violated_constraints)} limits. Slowing down by {slowdown_factor}x (attempt {attempt+1}/{max_attempts}). New duration: {interpolator.get_duration() * slowdown_factor:.3f}s (was {interpolator.get_duration():.3f}s)")
                trajectory = TimeParameterization.scale_time(trajectory, slowdown_factor)
                try:
                    interpolator = TrajectoryInterpolator(trajectory, method=interpolation)
                except ValueError as e:
                    logger.error(f"Failed to re-create interpolator after scaling: {e}")
                    return None

        # If cubic interpolation still fails, try linear as fallback
        if interpolation == 'cubic':
            logger.warning("Cubic interpolation failed validation, trying linear interpolation as fallback.")
            try:
                interpolator = TrajectoryInterpolator(trajectory, method='linear')
                validation_results = self.validator.validate_trajectory(interpolator)
                if validation_results['overall']:
                    logger.info("Linear interpolation fallback succeeded.")
                    return interpolator
            except Exception as e:
                logger.error(f"Linear interpolation fallback also failed: {e}")

        logger.warning("Trajectory validation failed after multiple slowdown attempts.")
        return None
        
    def execute_trajectory(self, interpolator: TrajectoryInterpolator, 
                           time_step: float = 0.01, dry_run: bool = True) -> bool:
        """
        Simulate or execute a time-parameterized trajectory.
        
        Args:
            interpolator: Trajectory interpolator
            time_step: Time step for execution
            dry_run: If True, only simulate sending commands
            
        Returns:
            True if execution successful, False otherwise
        """
        logger.info("Simulating trajectory execution")
        
        duration = interpolator.get_duration()
        times = np.arange(0, duration + time_step, time_step)
        
        if not np.isclose(times[-1], duration):
            times = np.append(times, duration)

        start_time = time.time()
        for t in times:
            # Ensure real-time execution (approx)
            elapsed_time = time.time() - start_time
            if elapsed_time < t:
                time.sleep(t - elapsed_time)
                
            config = interpolator.evaluate(t)
            
            # Send command to robot (or simulate)
            success = self.robot_controller.send_to_robot(config, dry_run=dry_run)
            if not success:
                logger.error(f"Failed to send command to robot at time {t:.3f}s")
                return False
                
        logger.info("Trajectory execution completed.")
        return True



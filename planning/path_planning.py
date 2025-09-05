#!/usr/bin/env python3
"""
Geometric path planning using AORRTC for robot manipulators.
Handles Cartesian space path planning with obstacle avoidance.
Integrates with the robot kinematics system for production-ready path planning.
"""

import numpy as np
import sys
import os
import logging
import time
import math
import random
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from scipy.spatial import cKDTree

# Import robot kinematics system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'robot_kinematics'))

if TYPE_CHECKING:
    from robot_controller import RobotController

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases
Point3D = np.ndarray
Path3D = List[Point3D]
Tree = Dict[str, List]


class Environment3D:
    """3D environment representation with geometric obstacles."""
    
    def __init__(self, workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]):
        """
        Initialize environment with workspace bounds.
        
        Args:
            workspace_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        """
        self.bounds = workspace_bounds
        self.sphere_obstacles: List[Tuple[Point3D, float]] = []
        self.box_obstacles: List[Tuple[Point3D, Point3D]] = []
        self.cylinder_obstacles: List[Tuple[Point3D, float, float]] = []
        
    def add_sphere_obstacle(self, center: List[float], radius: float):
        """Add spherical obstacle."""
        self.sphere_obstacles.append((np.array(center), radius))
        
    def add_box_obstacle(self, min_corner: List[float], max_corner: List[float]):
        """Add box obstacle defined by min and max corners."""
        self.box_obstacles.append((np.array(min_corner), np.array(max_corner)))
        
    def add_cylinder_obstacle(self, center: List[float], radius: float, height: float):
        """Add cylindrical obstacle (axis aligned with Z)."""
        self.cylinder_obstacles.append((np.array(center), radius, height))
        
    def is_point_valid(self, point: Point3D, safety_margin: float = 0.0, robot_controller: 'Optional[RobotController]' = None) -> bool:
        """
        Check if point is within bounds and collision-free.
        
        Args:
            point: 3D point to check
            safety_margin: Additional safety distance from obstacles
            robot_controller: Optional robot controller for more advanced checks
        """
        # Check workspace bounds
        for i, (min_bound, max_bound) in enumerate(self.bounds):
            if not (min_bound <= point[i] <= max_bound):
                return False
                
        # Check sphere obstacles
        for obs_center, obs_radius in self.sphere_obstacles:
            if np.linalg.norm(point - obs_center) <= obs_radius + safety_margin:
                return False
                
        # Check box obstacles
        for min_corner, max_corner in self.box_obstacles:
            if np.all(point >= min_corner - safety_margin) and np.all(point <= max_corner + safety_margin):
                return False
                
        # Check cylinder obstacles
        for obs_center, obs_radius, obs_height in self.cylinder_obstacles:
            # Check radial distance (XY plane)
            if np.linalg.norm(point[:2] - obs_center[:2]) <= obs_radius + safety_margin:
                # Check height
                if abs(point[2] - obs_center[2]) <= obs_height / 2.0 + safety_margin:
                    return False
                    
        # If a robot controller is provided, use its more accurate checks
        if robot_controller:
            # These checks are already performed by RobotEnvironment, but kept for redundancy/clarity
            if not robot_controller.robot._check_workspace(point):
                return False
            if not robot_controller.robot._check_obstacles(point):
                return False

        return True
        
    def is_path_valid(self, point1: Point3D, point2: Point3D, num_steps: int = 20, safety_margin: float = 0.0, robot_controller: 'Optional[RobotController]' = None) -> bool:
        """Check if straight-line path between points is collision-free."""
        if np.allclose(point1, point2):
            return True
            
        path_points = np.linspace(point1, point2, num_steps)
        for point in path_points:
            if not self.is_point_valid(point, safety_margin=safety_margin, robot_controller=robot_controller):
                return False
        return True


class CartesianPathPlanner:
    """AORRTC-based path planner for Cartesian space."""
    
    def __init__(self, environment: Environment3D, 
                 robot_controller: 'Optional[RobotController]' = None,
                 max_iterations: int = 5000,
                 step_size: float = 0.05,
                 goal_bias: float = 0.1,
                 connect_threshold: float = 0.1,
                 rewire_radius: float = 0.2):
        """
        Initialize AORRTC path planner.
        
        Args:
            environment: 3D environment with obstacles
            robot_controller: Optional robot controller for kinematic validation
            max_iterations: Maximum planning iterations
            step_size: Step size for tree extension
            goal_bias: Probability of sampling goal
            connect_threshold: Distance threshold for connecting trees
            rewire_radius: Radius for rewiring optimization
        """
        self.env = environment
        self.robot_controller = robot_controller
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.connect_threshold = connect_threshold
        self.rewire_radius = rewire_radius
        
        # Planning state
        self.tree_start: Optional[Tree] = None
        self.tree_goal: Optional[Tree] = None
        self.best_path: Optional[Path3D] = None
        self.best_cost = float('inf')
        self.iteration = 0
        self.last_improvement = 0
        self.use_informed_sampling = False
        self.c_min = 0.0
        self.informed_rotation_matrix = np.identity(3)
        
    def plan(self, start: Point3D, goal: Point3D) -> Optional[Path3D]:
        """
        Plan path from start to goal using AORRTC.
        
        Args:
            start: Start position in Cartesian space
            goal: Goal position in Cartesian space
            
        Returns:
            List of waypoints if path found, None otherwise
        """
        start_array = np.array(start)
        goal_array = np.array(goal)
        
        # Validate start and goal
        if not self.env.is_point_valid(start_array, robot_controller=self.robot_controller):
            logger.error("Start point is invalid")
            return None
        if not self.env.is_point_valid(goal_array, robot_controller=self.robot_controller):
            logger.error("Goal point is invalid")
            return None
            
        # Initialize planning
        self._initialize_planning(start_array, goal_array)
        
        logger.info(f"Starting AORRTC path planning (max_iter={self.max_iterations})")
        start_time = time.time()
        
        for i in range(self.max_iterations):
            self.iteration = i
            
            # Alternate between trees
            tree_from, tree_to = (self.tree_start, self.tree_goal) if i % 2 == 0 else (self.tree_goal, self.tree_start)
            
            # Sample and extend
            rand_point = self._sample(tree_to)
            new_idx = self._extend_tree(tree_from, rand_point)
            
            if new_idx is not None:
                # Try to connect trees
                if self._try_connect(tree_from, tree_to, new_idx):
                    # Rewire for optimization
                    self._rewire_tree(tree_from, new_idx)
                    
        planning_time = time.time() - start_time
        logger.info(f"Planning completed in {planning_time:.2f}s")
        
        if self.best_path:
            logger.info(f"Path found with cost: {self.best_cost:.4f}")
            # Smooth the path
            smoothed_path = self.smooth_path(self.best_path)
            return smoothed_path
        else:
            logger.warning("No path found")
            
        return self.best_path
        
    def _initialize_planning(self, start: Point3D, goal: Point3D):
        """Initialize planning data structures."""
        self.tree_start = {
            'points': [start],
            'parents': [-1],
            'costs': [0.0]
        }
        self.tree_goal = {
            'points': [goal],
            'parents': [-1],
            'costs': [0.0]
        }
        self.best_path = None
        self.best_cost = float('inf')
        self.iteration = 0
        self.last_improvement = 0
        self.use_informed_sampling = False
        self.c_min = np.linalg.norm(goal - start)
        self.informed_rotation_matrix = self._compute_rotation_matrix(start, goal)
        
    def _sample(self, tree_to: Tree) -> Point3D:
        """Sample point for tree extension."""
        if self.use_informed_sampling and random.random() > self.goal_bias:
            return self._sample_informed()
        elif random.random() < self.goal_bias:
            return tree_to['points'][0]
        else:
            return self._sample_random()
            
    def _sample_random(self) -> Point3D:
        """Sample random point in workspace."""
        return np.array([random.uniform(bound[0], bound[1]) for bound in self.env.bounds])
        
    def _sample_informed(self) -> Point3D:
        """Sample from informed subset (ellipsoid)."""
        start = self.tree_start['points'][0]
        goal = self.tree_goal['points'][0]
        center = (start + goal) / 2.0
        
        # Ellipsoid parameters
        r1 = self.best_cost / 2.0
        r2 = math.sqrt(max(0, self.best_cost**2 - self.c_min**2)) / 2.0
        
        # Sample from unit ball and transform
        x = np.random.randn(3)
        x /= np.linalg.norm(x) * (random.random() ** (1.0/3.0))
        
        # Scale and rotate
        scaled_x = np.diag([r1, r2, r2]) @ x
        rotated_x = self.informed_rotation_matrix @ scaled_x
        sample = center + rotated_x
        
        # Clamp to workspace bounds
        return np.clip(sample, 
                      [bound[0] for bound in self.env.bounds],
                      [bound[1] for bound in self.env.bounds])
                      
    def _compute_rotation_matrix(self, start: Point3D, goal: Point3D) -> np.ndarray:
        """Compute rotation matrix for informed sampling ellipsoid."""
        direction = (goal - start) / self.c_min
        x_axis = np.array([1.0, 0.0, 0.0])
        
        if np.allclose(direction, x_axis):
            return np.identity(3)
        
        v = np.cross(x_axis, direction)
        s = np.linalg.norm(v)
        c = np.dot(x_axis, direction)
        
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        
        rotation_matrix = np.identity(3) + vx + vx @ vx * ((1 - c) / (s**2))
        return rotation_matrix
        
    def _extend_tree(self, tree: Tree, target: Point3D) -> Optional[int]:
        """Extend tree towards target point."""
        # Find nearest neighbor using k-d tree
        kdtree = cKDTree(tree['points'])
        _, nearest_idx = kdtree.query(target)
        nearest_point = tree['points'][nearest_idx]
        
        # Compute new point
        direction = target - nearest_point
        dist = np.linalg.norm(direction)
        
        if dist == 0:
            return None
            
        new_point = nearest_point + direction / dist * min(self.step_size, dist)
        
        # Check path validity
        if not self.env.is_path_valid(nearest_point, new_point, safety_margin=0.01, robot_controller=self.robot_controller):
            return None
            
        # Add to tree
        new_cost = tree['costs'][nearest_idx] + np.linalg.norm(new_point - nearest_point)
        tree['points'].append(new_point)
        tree['parents'].append(nearest_idx)
        tree['costs'].append(new_cost)
        
        return len(tree['points']) - 1
        
    def _try_connect(self, tree_from: Tree, tree_to: Tree, new_idx: int) -> bool:
        """Try to connect the two trees."""
        new_point = tree_from['points'][new_idx]
        
        # Find nearest neighbor in the other tree
        kdtree_to = cKDTree(tree_to['points'])
        dist, nearest_idx_to = kdtree_to.query(new_point)
        
        if dist < self.connect_threshold:
            nearest_point_to = tree_to['points'][nearest_idx_to]
            if self.env.is_path_valid(new_point, nearest_point_to, robot_controller=self.robot_controller):
                self._update_best_path(tree_from, new_idx, tree_to, nearest_idx_to)
                return True
                
        return False
        
    def _update_best_path(self, tree1: Tree, idx1: int, tree2: Tree, idx2: int):
        """Update best path if a better one is found."""
        path1 = self._extract_path(tree1, idx1)
        path2 = self._extract_path(tree2, idx2)
        
        # Combine paths
        if tree1 is self.tree_start:
            full_path = path1[::-1] + path2
        else:
            full_path = path2[::-1] + path1
            
        cost = self._path_cost(full_path)
        
        if cost < self.best_cost:
            self.best_path = full_path
            self.best_cost = cost
            self.use_informed_sampling = True
            self.last_improvement = self.iteration
            logger.debug(f"New best path found with cost: {cost:.4f}")
                
    def _rewire_tree(self, tree: Tree, new_idx: int):
        """Rewire tree to optimize paths."""
        new_point = tree['points'][new_idx]
        new_cost = tree['costs'][new_idx]
        
        kdtree = cKDTree(tree['points'])
        nearby_indices = kdtree.query_ball_point(new_point, self.rewire_radius)
        
        for idx in nearby_indices:
            if idx == new_idx:
                continue
                
            neighbor_point = tree['points'][idx]
            potential_cost = new_cost + np.linalg.norm(neighbor_point - new_point)
            
            if potential_cost < tree['costs'][idx] and self.env.is_path_valid(new_point, neighbor_point, robot_controller=self.robot_controller):
                tree['parents'][idx] = new_idx
                tree['costs'][idx] = potential_cost
                
    def _extract_path(self, tree: Tree, node_idx: int) -> Path3D:
        """Extract path from tree by backtracking."""
        path = []
        current = node_idx
        while current != -1:
            path.append(tree['points'][current])
            current = tree['parents'][current]
        return path
        
    @staticmethod
    def _path_cost(path: Path3D) -> float:
        """Compute total cost of a path."""
        return sum(np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path)))
        
    def smooth_path(self, path: Path3D, iterations: int = 100, 
                   smoothness_weight: float = 0.4, 
                   obstacle_weight: float = 0.2,
                   learning_rate: float = 0.1) -> Path3D:
        """
        Smooth path using gradient-based optimization.
        
        Args:
            path: The path to smooth
            iterations: Number of smoothing iterations
            smoothness_weight: Weight for path smoothness
            obstacle_weight: Weight for obstacle avoidance
            learning_rate: Step size for optimization
            
        Returns:
            The smoothed path
        """
        smoothed_path = [np.copy(p) for p in path]
        num_points = len(smoothed_path)
        
        if num_points < 3:
            return smoothed_path
            
        for _ in range(iterations):
            for i in range(1, num_points - 1):
                # Smoothness term (pulls point towards midpoint of neighbors)
                smoothness_grad = smoothed_path[i-1] + smoothed_path[i+1] - 2 * smoothed_path[i]
                
                # Obstacle avoidance term (pushes point away from nearest obstacle)
                obstacle_grad = np.zeros(3)
                min_dist_to_obs = float('inf')
                
                # Iterate through all obstacles (spheres, boxes, cylinders)
                # Spheres
                for obs_center, obs_radius in self.env.sphere_obstacles:
                    dist_vec = smoothed_path[i] - obs_center
                    dist = np.linalg.norm(dist_vec)
                    if dist < min_dist_to_obs:
                        min_dist_to_obs = dist
                        influence_radius = 2.0 * obs_radius # Example influence radius
                        if dist < influence_radius and dist > 1e-9: # Avoid division by zero
                            obstacle_grad = (1.0 / dist - 1.0 / influence_radius) * (dist_vec / dist)
                
                # Boxes (simplified gradient - push away from closest face/edge)
                for min_corner, max_corner in self.env.box_obstacles:
                    # Calculate closest point on box to smoothed_path[i]
                    closest_point_on_box = np.maximum(min_corner, np.minimum(smoothed_path[i], max_corner))
                    dist_vec = smoothed_path[i] - closest_point_on_box
                    dist = np.linalg.norm(dist_vec)
                    if dist < min_dist_to_obs:
                        min_dist_to_obs = dist
                        influence_radius = np.linalg.norm(max_corner - min_corner) / 2.0 # Half diagonal as influence
                        if dist < influence_radius and dist > 1e-9:
                            obstacle_grad = (1.0 / dist - 1.0 / influence_radius) * (dist_vec / dist)

                # Cylinders (simplified gradient - push away from axis/surface)
                for obs_center, obs_radius, obs_height in self.env.cylinder_obstacles:
                    # Project point onto cylinder axis plane
                    projected_point = np.array([smoothed_path[i][0], smoothed_path[i][1], obs_center[2]])
                    dist_vec_xy = projected_point[:2] - obs_center[:2]
                    dist_xy = np.linalg.norm(dist_vec_xy)
                    
                    # Check if within cylinder height
                    if abs(smoothed_path[i][2] - obs_center[2]) <= obs_height / 2.0:
                        if dist_xy < min_dist_to_obs:
                            min_dist_to_obs = dist_xy
                            influence_radius = 2.0 * obs_radius
                            if dist_xy < influence_radius and dist_xy > 1e-9:
                                # Push away radially
                                obstacle_grad_xy = (1.0 / dist_xy - 1.0 / influence_radius) * (dist_vec_xy / dist_xy)
                                obstacle_grad = np.array([obstacle_grad_xy[0], obstacle_grad_xy[1], 0.0])

                # Combine gradients
                update = learning_rate * (smoothness_weight * smoothness_grad + obstacle_weight * obstacle_grad)
                smoothed_path[i] += update
                
                # Ensure the updated point is still valid (within bounds and not in collision)
                if not self.env.is_point_valid(smoothed_path[i], robot_controller=self.robot_controller):
                    # If invalid, revert the update or apply a smaller step
                    smoothed_path[i] -= update # Simple revert
                    # Or try to project back to valid space, or reduce learning_rate

        return smoothed_path



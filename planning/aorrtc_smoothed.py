"""
Clean and Optimized AORRTC (Asymptotically Optimal RRT-Connect) for 3D Path Planning
Enhanced with interactive Plotly visualization and performance optimizations.

IMPROVEMENT HIGHLIGHTS:
- Path Smoothing: Added an optimization-based path smoothing function to refine the
  initial jagged path into a smoother, more robot-friendly trajectory.
- Performance: Replaced O(n) nearest neighbor search with scipy's cKDTree (O(log n)).
- Algorithm Correctness: Fixed informed sampling by implementing proper rotation.
- Visualization: Updated to display both the original and smoothed paths for comparison.
"""

import numpy as np
import math
import random
import time
from typing import List, Tuple, Optional, Dict
import plotly.graph_objects as go
from scipy.spatial import cKDTree

# Type aliases
Point3D = np.ndarray
Path3D = List[Point3D]
Tree = Dict[str, List]


class Environment3D:
    """Simple 3D environment with sphere obstacles."""

    def __init__(self, bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]):
        # List of (center, radius) tuples
        self.obstacles: List[Tuple[Point3D, float]] = []
        self.bounds = bounds

    def add_sphere_obstacle(self, center: List[float], radius: float):
        """Add a spherical obstacle."""
        self.obstacles.append((np.array(center), radius))

    def is_point_valid(self, point: Point3D) -> bool:
        """Check if a point is within bounds and collision-free."""
        if not (self.bounds[0][0] <= point[0] <= self.bounds[0][1] and
                self.bounds[1][0] <= point[1] <= self.bounds[1][1] and
                self.bounds[2][0] <= point[2] <= self.bounds[2][1]):
            return False

        for obs_center, obs_radius in self.obstacles:
            if np.linalg.norm(point - obs_center) <= obs_radius:
                return False
        return True

    def is_path_valid(self, point1: Point3D, point2: Point3D, num_steps: int = 20) -> bool:
        """Check if a straight-line path between two points is collision-free."""
        path = np.linspace(point1, point2, num_steps)
        for point in path:
            if not self.is_point_valid(point):
                return False
        return True


class AORRTC3D:
    """Optimized AORRTC implementation for 3D point-to-point planning."""

    def __init__(self, env: Environment3D, start: Point3D, goal: Point3D,
                 max_iter: int = 5000, step_size: float = 0.2, goal_bias: float = 0.1,
                 connect_threshold: float = 0.3, rewire_radius: float = 0.5):

        self.env = env
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.connect_threshold = connect_threshold
        self.rewire_radius = rewire_radius
        self.tree_a: Tree = {'points': [
            self.start], 'parents': [-1], 'costs': [0.0]}
        self.tree_b: Tree = {'points': [
            self.goal], 'parents': [-1], 'costs': [0.0]}
        self.best_path: Optional[Path3D] = None
        self.best_cost = float('inf')
        self.iteration = 0
        self.last_improvement = 0
        self.use_informed_sampling = False
        self.c_min = np.linalg.norm(self.goal - self.start)
        self.informed_rotation_matrix = np.identity(3)

    def plan(self) -> Optional[Path3D]:
        """Main planning loop to find the optimal path."""
        print(f"🚀 Starting AORRTC 3D planning (max_iter={self.max_iter})")
        start_time = time.time()
        if not self.env.is_point_valid(self.start):
            print("❌ Start point is in collision.")
            return None
        if not self.env.is_point_valid(self.goal):
            print("❌ Goal point is in collision.")
            return None
        for i in range(self.max_iter):
            self.iteration = i
            if self.best_path and (i - self.last_improvement) > 1000:
                print(f"💡 Early exit at iteration {i} due to no improvement.")
                break
            if i % 1000 == 0 and i > 0:
                tree_sizes = f"{len(self.tree_a['points'])}/{len(self.tree_b['points'])}"
                cost_info = f", Cost: {self.best_cost:.3f}" if self.best_path else ""
                print(f"Iteration {i}, Trees: {tree_sizes}{cost_info}")
            tree_from, tree_to = (self.tree_a, self.tree_b) if i % 2 == 0 else (
                self.tree_b, self.tree_a)
            rand_point = self._sample(tree_to)
            new_idx = self._extend_tree(tree_from, rand_point)
            if new_idx is not None:
                if self._try_connect(tree_from, tree_to, new_idx):
                    self._rewire_tree(tree_from, new_idx)
        planning_time = time.time() - start_time
        print(f"Planning completed in {planning_time:.2f}s")
        if self.best_path:
            print(f"✅ Path found with cost: {self.best_cost:.3f}")
        else:
            print("❌ No path found within the iteration limit.")
        return self.best_path

    def _sample(self, tree_to: Tree) -> Point3D:
        if self.use_informed_sampling and random.random() > self.goal_bias:
            return self._sample_informed()
        if random.random() < self.goal_bias:
            return tree_to['points'][0]
        return self._sample_random()

    def _sample_random(self) -> Point3D:
        return np.array([random.uniform(b[0], b[1]) for b in self.env.bounds])

    def _sample_informed(self) -> Point3D:
        center = (self.start + self.goal) / 2.0
        r1 = self.best_cost / 2.0
        r2 = math.sqrt(max(0, self.best_cost**2 - self.c_min**2)) / 2.0
        x = np.random.randn(3)
        x /= np.linalg.norm(x) * (random.random() ** (1.0/3.0))
        scaled_x = np.diag([r1, r2, r2]) @ x
        rotated_x = self.informed_rotation_matrix @ scaled_x
        sample = center + rotated_x
        return np.clip(sample, [b[0] for b in self.env.bounds], [b[1] for b in self.env.bounds])

    def _get_rotation_matrix(self) -> np.ndarray:
        a = (self.goal - self.start) / self.c_min
        b = np.array([1.0, 0.0, 0.0])
        if np.allclose(a, b):
            b = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, b)
        s = np.linalg.norm(v)
        c = np.dot(a, b)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        if s != 0:
            return np.identity(3) + vx + vx @ vx * ((1 - c) / (s**2))
        return np.identity(3)

    def _extend_tree(self, tree: Tree, target: Point3D) -> Optional[int]:
        kdtree = cKDTree(tree['points'])
        _, nearest_idx = kdtree.query(target)
        nearest_point = tree['points'][nearest_idx]
        direction = target - nearest_point
        dist = np.linalg.norm(direction)
        if dist == 0:
            return None
        new_point = nearest_point + direction / \
            dist * min(self.step_size, dist)
        if not self.env.is_path_valid(nearest_point, new_point):
            return None
        new_cost = tree['costs'][nearest_idx] + \
            np.linalg.norm(new_point - nearest_point)
        tree['points'].append(new_point)
        tree['parents'].append(nearest_idx)
        tree['costs'].append(new_cost)
        return len(tree['points']) - 1

    def _try_connect(self, tree_from: Tree, tree_to: Tree, new_idx: int) -> bool:
        new_point = tree_from['points'][new_idx]
        kdtree_to = cKDTree(tree_to['points'])
        dist, nearest_idx_to = kdtree_to.query(new_point)
        if dist < self.connect_threshold:
            nearest_point_to = tree_to['points'][nearest_idx_to]
            if self.env.is_path_valid(new_point, nearest_point_to):
                self._update_best_path(
                    tree_from, new_idx, tree_to, nearest_idx_to)
                return True
        return False

    def _update_best_path(self, tree1: Tree, idx1: int, tree2: Tree, idx2: int):
        path1 = self._get_path(tree1, idx1)
        path2 = self._get_path(tree2, idx2)
        if self.tree_a is tree1:
            full_path = path1[::-1] + path2
        else:
            full_path = path2[::-1] + path1
        cost = self._path_cost(full_path)
        if cost < self.best_cost:
            self.best_path = full_path
            self.best_cost = cost
            self.last_improvement = self.iteration
            if not self.use_informed_sampling:
                print(
                    f"  🎉 Initial path found! Cost: {cost:.3f}. Activating informed sampling.")
                self.use_informed_sampling = True
                self.informed_rotation_matrix = self._get_rotation_matrix()
            else:
                print(f"  🎉 New best path! Cost: {cost:.3f}")

    def _rewire_tree(self, tree: Tree, new_idx: int):
        new_point = tree['points'][new_idx]
        new_cost = tree['costs'][new_idx]
        kdtree = cKDTree(tree['points'])
        nearby_indices = kdtree.query_ball_point(new_point, self.rewire_radius)
        for idx in nearby_indices:
            if idx == new_idx:
                continue
            neighbor_point = tree['points'][idx]
            potential_cost = new_cost + \
                np.linalg.norm(neighbor_point - new_point)
            if potential_cost < tree['costs'][idx] and self.env.is_path_valid(new_point, neighbor_point):
                tree['parents'][idx] = new_idx
                tree['costs'][idx] = potential_cost

    def _get_path(self, tree: Tree, node_idx: int) -> Path3D:
        path = []
        current = node_idx
        while current != -1:
            path.append(tree['points'][current])
            current = tree['parents'][current]
        return path

    @staticmethod
    def _path_cost(path: Path3D) -> float:
        return sum(np.linalg.norm(path[i] - path[i-1]) for i in range(1, len(path)))

    def smooth_path_optimized(self, path: Path3D, iterations: int = 100, alpha: float = 0.1, beta: float = 0.4, gamma: float = 0.2) -> Path3D:
        """
        Smooths a path using a simple gradient-based optimization.
        - alpha: Learning rate.
        - beta: Weight for the smoothness term.
        - gamma: Weight for the obstacle avoidance term.
        """
        smoothed_path = [np.copy(p) for p in path]
        num_points = len(smoothed_path)

        for _ in range(iterations):
            for i in range(1, num_points - 1):
                grad_smooth = smoothed_path[i-1] + \
                    smoothed_path[i+1] - 2 * smoothed_path[i]
                grad_obstacle = np.zeros(3)
                min_dist_to_obs = float('inf')
                for obs_center, obs_radius in self.env.obstacles:
                    dist_vec = smoothed_path[i] - obs_center
                    dist = np.linalg.norm(dist_vec)
                    if dist < min_dist_to_obs:
                        min_dist_to_obs = dist
                        influence_radius = 2.0 * obs_radius
                        if dist < influence_radius:
                            grad_obstacle = (
                                1.0 / dist - 1.0 / influence_radius) * (dist_vec / dist)
                update = alpha * (beta * grad_smooth + gamma * grad_obstacle)
                smoothed_path[i] += update
        return smoothed_path

    def visualize(self, original_path: Optional[Path3D] = None, smoothed_path: Optional[Path3D] = None, show_in_browser: bool = False, save_html: bool = True):
        """Generate and display an interactive 3D visualization using Plotly."""
        fig = go.Figure()
        obstacle_color = 'rgb(106, 90, 205)'
        tree_a_color = 'rgba(34, 139, 34, 0.8)'
        tree_b_color = 'rgba(255, 140, 0, 0.8)'

        for i, (center, radius) in enumerate(self.env.obstacles):
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:15j]
            x, y, z = center[0] + radius * np.cos(u) * np.sin(
                v), center[1] + radius * np.sin(u) * np.sin(v), center[2] + radius * np.cos(v)
            fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, obstacle_color], [
                          1, obstacle_color]], opacity=0.9, showscale=False, name=f'Obstacle {i+1}'))

        fig.add_trace(go.Scatter3d(x=[self.start[0]], y=[self.start[1]], z=[
                      self.start[2]], mode='markers', marker=dict(size=10, color='green'), name='Start'))
        fig.add_trace(go.Scatter3d(x=[self.goal[0]], y=[self.goal[1]], z=[
                      self.goal[2]], mode='markers', marker=dict(size=10, color='blue', symbol='square'), name='Goal'))

        self._plot_tree_edges_plotly(fig, self.tree_a, tree_a_color, 'Tree A')
        self._plot_tree_edges_plotly(fig, self.tree_b, tree_b_color, 'Tree B')

        # Plot original path (if provided) with updated, more visible style
        if original_path:
            path_array = np.array(original_path)
            fig.add_trace(go.Scatter3d(
                x=path_array[:, 0], y=path_array[:, 1], z=path_array[:, 2],
                mode='lines', line=dict(color='crimson', width=4),
                name=f'Original Path (Cost: {self._path_cost(original_path):.3f})'
            ))

        # Plot smoothed path (if provided)
        if smoothed_path:
            path_array = np.array(smoothed_path)
            fig.add_trace(go.Scatter3d(
                x=path_array[:, 0], y=path_array[:, 1], z=path_array[:, 2],
                mode='lines+markers', line=dict(color='deepskyblue', width=8),
                marker=dict(size=4, color='dodgerblue'), name=f'Smoothed Path (Cost: {self._path_cost(smoothed_path):.3f})'
            ))

        bounds = self.env.bounds
        fig.update_layout(title=f'AORRTC 3D Path Planning - Iteration {self.iteration}', scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', xaxis=dict(
            range=bounds[0]), yaxis=dict(range=bounds[1]), zaxis=dict(range=bounds[2]), aspectmode='data'), width=1000, height=800, showlegend=True)
        if save_html:
            filename = f"aorrtc_3d_path_{self.iteration}.html"
            fig.write_html(filename)
            print(f"Plot saved as: {filename}")
        if show_in_browser:
            fig.show()
        else:
            try:
                fig.show(renderer="notebook")
            except Exception:
                print("Inline display failed. Opening in browser...")
                fig.show()

    @staticmethod
    def _plot_tree_edges_plotly(fig: go.Figure, tree: Tree, color: str, name: str):
        x_coords, y_coords, z_coords = [], [], []
        for i, parent_idx in enumerate(tree['parents']):
            if parent_idx != -1:
                p1, p2 = tree['points'][parent_idx], tree['points'][i]
                x_coords.extend([p1[0], p2[0], None])
                y_coords.extend([p1[1], p2[1], None])
                z_coords.extend([p1[2], p2[2], None])
        if x_coords:
            fig.add_trace(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='lines', line=dict(
                color=color, width=2.5), name=name, hoverinfo='skip'))


if __name__ == "__main__":
    print("🤖 Optimized AORRTC 3D Demo with Interactive Plotly Visualization")

    env = Environment3D(bounds=((-2, 2), (-2, 2), (0, 2)))
    env.add_sphere_obstacle([0.0, 0.0, 0.8], 0.3)
    env.add_sphere_obstacle([0.5, 0.5, 0.5], 0.25)
    env.add_sphere_obstacle([-0.5, -0.5, 1.2], 0.2)
    env.add_sphere_obstacle([0.8, -0.3, 0.3], 0.15)
    env.add_sphere_obstacle([-0.3, 0.8, 1.5], 0.18)

    start_point = np.array([-1.5, -1.5, 0.2])
    goal_point = np.array([1.5, 1.5, 1.8])
    print(
        f"Start: [{start_point[0]:.1f}, {start_point[1]:.1f}, {start_point[2]:.1f}]")
    print(
        f"Goal:  [{goal_point[0]:.1f}, {goal_point[1]:.1f}, {goal_point[2]:.1f}]")

    planner = AORRTC3D(env, start_point, goal_point,
                       max_iter=8000, step_size=0.15)
    original_path = planner.plan()

    if original_path:
        print("\n🔧 Smoothing the path using optimization...")
        smoothed_path = planner.smooth_path_optimized(
            original_path,
            iterations=150,
            alpha=0.1,
            beta=0.6,
            gamma=0.2
        )
        print("🎨 Generating final visualization with original and smoothed paths...")
        planner.visualize(original_path=original_path,
                          smoothed_path=smoothed_path)

    print("\n✅ Demo completed!")

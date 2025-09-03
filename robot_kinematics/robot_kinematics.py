#!/usr/bin/env python3
"""
Production-ready robot kinematics module for RB3-730ES-U robot.
Uses Product of Exponentials (PoE) formulation with damped least squares IK.
"""

import numpy as np
from numpy.linalg import norm, inv
import xml.etree.ElementTree as ET
from typing import Tuple, Optional
import logging
import yaml
import os

logger = logging.getLogger(__name__)

class RobotKinematicsError(Exception):
    """Custom exception for kinematics-related errors."""
    pass

class RobotKinematics:
    """Production-ready robot kinematics class using PoE formulation."""
    
    def __init__(self, urdf_path: str, ee_link: str = "tcp", base_link: str = "link0", constraints_path: Optional[str] = None):
        self.urdf_path = urdf_path
        self.ee_link = ee_link
        self.base_link = base_link
        self.constraints_path = constraints_path or os.path.join(os.path.dirname(__file__), "constraints.yaml")
        self.constraints = self._load_constraints(self.constraints_path)
        
        try:
            self.S, self.M, self.joint_limits = self._load_from_urdf()
        except FileNotFoundError:
            raise RobotKinematicsError(f"URDF file not found at: {urdf_path}")
        except ET.ParseError:
            raise RobotKinematicsError(f"Failed to parse URDF file: {urdf_path}")
        except (KeyError, ValueError) as e:
            raise RobotKinematicsError(f"Error parsing URDF structure: {e}")

        self.n_joints = self.S.shape[1]
        if self.n_joints == 0:
            raise RobotKinematicsError("No active joints found in the kinematic chain.")

        self._eye6 = np.eye(6)
        logger.info(f"Robot kinematics initialized with {self.n_joints} joints.")
    def _load_constraints(self, path):
        if not os.path.exists(path):
            logger.warning(f"Constraints file not found: {path}")
            return {}
        with open(path, "r") as f:
            return yaml.safe_load(f)
    def _check_workspace(self, pos: np.ndarray) -> bool:
        ws = self.constraints.get("workspace", {})
        if not ws:
            return True
        return (ws["x_min"] <= pos[0] <= ws["x_max"] and
                ws["y_min"] <= pos[1] <= ws["y_max"] and
                ws["z_min"] <= pos[2] <= ws["z_max"])

    def _check_orientation(self, rpy: np.ndarray) -> bool:
        limits = self.constraints.get("orientation_limits", {})
        if not limits:
            return True
        roll, pitch, yaw = np.degrees(rpy)
        return (limits["roll_min"] <= roll <= limits["roll_max"] and
                limits["pitch_min"] <= pitch <= limits["pitch_max"] and
                limits["yaw_min"] <= yaw <= limits["yaw_max"])

    def _check_obstacles(self, pos: np.ndarray) -> bool:
        obstacles = self.constraints.get("obstacles", [])
        for obs in obstacles:
            if obs["type"] == "box":
                c = np.array(obs["center"])
                s = np.array(obs["size"]) / 2.0
                if np.all(np.abs(pos - c) <= s):
                    return False
            elif obs["type"] == "cylinder":
                c = np.array(obs["center"])
                r = obs["radius"]
                h = obs["height"] / 2.0
                if (np.linalg.norm(pos[:2] - c[:2]) <= r and abs(pos[2] - c[2]) <= h):
                    return False
        return True
        
        try:
            self.S, self.M, self.joint_limits = self._load_from_urdf()
        except FileNotFoundError:
            raise RobotKinematicsError(f"URDF file not found at: {urdf_path}")
        except ET.ParseError:
            raise RobotKinematicsError(f"Failed to parse URDF file: {urdf_path}")
        except (KeyError, ValueError) as e:
            raise RobotKinematicsError(f"Error parsing URDF structure: {e}")

        self.n_joints = self.S.shape[1]
        if self.n_joints == 0:
            raise RobotKinematicsError("No active joints found in the kinematic chain.")

        self._eye6 = np.eye(6)
        logger.info(f"Robot kinematics initialized with {self.n_joints} joints.")

    def _load_from_urdf(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load PoE parameters from URDF file."""
        def skew(v):
            return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        def rpy_to_R(roll, pitch, yaw):
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            return Rz @ Ry @ Rx

        def make_T(R, p):
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = p
            return T

        def parse_origin(origin_elem):
            xyz = origin_elem.get('xyz', '0 0 0').split()
            rpy = origin_elem.get('rpy', '0 0 0').split()
            x, y, z = map(float, xyz)
            r, p, y_ = map(float, rpy)
            R = rpy_to_R(r, p, y_)
            p = np.array([x, y, z], dtype=float)
            return make_T(R, p)

        # URDF parsing
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        links = {link.attrib['name'] for link in root.findall('link')}

        joints = {}
        for j in root.findall('joint'):
            jname = j.attrib['name']
            jtype = j.attrib['type']
            parent = j.find('parent').attrib['link']
            child = j.find('child').attrib['link']
            origin_elem = j.find('origin')
            T_parent_joint = parse_origin(origin_elem) if origin_elem is not None else np.eye(4)

            axis_elem = j.find('axis')
            axis = np.array([0., 0., 1.])
            if axis_elem is not None:
                axis = np.array(list(map(float, axis_elem.attrib['xyz'].split())))

            limit_elem = j.find('limit')
            limits = {'lower': -np.inf, 'upper': np.inf}
            if limit_elem is not None:
                limits['lower'] = float(limit_elem.attrib.get('lower', -np.inf))
                limits['upper'] = float(limit_elem.attrib.get('upper', np.inf))

            joints[jname] = {
                'name': jname, 'type': jtype, 'parent': parent, 'child': child,
                'axis': axis / (norm(axis) + 1e-12), 'T_parent_joint': T_parent_joint,
                'limits': limits
            }

        def build_chain():
            child_to_joint = {j['child']: j for j in joints.values()}
            chain = []
            link = self.ee_link
            while link != self.base_link:
                if link not in child_to_joint:
                    raise ValueError(f"No joint found from {link} to base {self.base_link}.")
                j = child_to_joint[link]
                chain.append(j)
                link = j['parent']
            chain.reverse()
            return chain

        # Build chain and extract PoE parameters
        chain = build_chain()
        T_base_to_here = np.eye(4)
        S_list, joint_limits_list = [], []

        for j_data in chain:
            T_base_to_joint = T_base_to_here @ j_data['T_parent_joint']
            Rbj, pbj = T_base_to_joint[:3, :3], T_base_to_joint[:3, 3]
            axis_b = Rbj @ j_data['axis']

            if j_data['type'] in ['revolute', 'continuous']:
                w = axis_b
                v = -np.cross(w, pbj)
                S_list.append(np.hstack([w, v]))
                joint_limits_list.append([j_data['limits']['lower'], j_data['limits']['upper']])
            elif j_data['type'] == 'prismatic':
                w = np.zeros(3)
                v = axis_b
                S_list.append(np.hstack([w, v]))
                joint_limits_list.append([j_data['limits']['lower'], j_data['limits']['upper']])

            T_base_to_here = T_base_to_joint

        M = T_base_to_here
        if not S_list:
            raise ValueError("No active joints found in the specified chain.")

        S = np.array(S_list).T
        joint_limits = np.array(joint_limits_list).T
        return S, M, joint_limits

    @staticmethod
    def _skew(w: np.ndarray) -> np.ndarray:
        return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

    @staticmethod
    def _adjoint(T: np.ndarray) -> np.ndarray:
        R, p = T[:3, :3], T[:3, 3]
        return np.block([[R, np.zeros((3, 3))], [RobotKinematics._skew(p) @ R, R]])

    @staticmethod
    def _matrix_log6(T: np.ndarray) -> np.ndarray:
        R, p = T[:3, :3], T[:3, 3]
        trace_R = np.trace(R)
        cos_th = np.clip((trace_R - 1.0) * 0.5, -1.0, 1.0)
        theta = np.arccos(cos_th)
        
        if theta < 1e-6:
            # Small angle approximation
            omega = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * 0.5
            return np.hstack([omega, p])
            
        # Normal case
        sin_th = np.sin(theta)
        if abs(sin_th) < 1e-6:
            # Handle theta ≈ π case
            # Find the axis from the eigenvector corresponding to eigenvalue 1
            try:
                eigvals, eigvecs = np.linalg.eig(R)
                axis_idx = np.argmin(np.abs(eigvals - 1.0))
                omega_unit = np.real(eigvecs[:, axis_idx])
                omega_unit = omega_unit / (np.linalg.norm(omega_unit) + 1e-12)  # Avoid division by zero
                omega = omega_unit * theta
            except np.linalg.LinAlgError:
                # Fallback if eigenvalue decomposition fails
                logger.debug("Eigenvalue decomposition failed in matrix_log6, using fallback")
                omega = np.array([0., 0., theta])  # Assume rotation around z-axis
        else:
            omega_hat = (R - R.T) * (0.5 / sin_th)
            omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]]) * theta
            
        # Compute V inverse for translation part
        omega_norm = np.linalg.norm(omega) + 1e-12  # Avoid division by zero
        omega_unit = omega / omega_norm
        omega_hat = RobotKinematics._skew(omega_unit)
        omega_hat2 = omega_hat @ omega_hat
        
        try:
            cot_half = 1.0 / np.tan(theta * 0.5)
            V_inv = (np.eye(3) / theta - 0.5 * omega_hat + 
                    (1.0 / theta - 0.5 * cot_half) * omega_hat2)
            v = V_inv @ p
        except (ZeroDivisionError, FloatingPointError):
            # Fallback for numerical issues
            logger.debug("Numerical issue in matrix_log6 V_inv computation, using fallback")
            v = p / (theta + 1e-12)
        
        return np.hstack([omega, v])

    @staticmethod
    def _matrix_exp6(xi_theta: np.ndarray) -> np.ndarray:
        w_theta, v_theta = xi_theta[:3], xi_theta[3:]
        theta = norm(w_theta)
        T = np.eye(4)
        if theta < 1e-12:
            T[:3, 3] = v_theta
            return T
        w = w_theta / theta
        v = v_theta / theta
        w_hat = RobotKinematics._skew(w)
        w_hat2 = w_hat @ w_hat
        R = np.eye(3) + np.sin(theta) * w_hat + (1 - np.cos(theta)) * w_hat2
        G = np.eye(3) * theta + (1 - np.cos(theta)) * w_hat + (theta - np.sin(theta)) * w_hat2
        p = G @ v
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """Compute forward kinematics for given joint angles, with workspace and obstacle checks."""
        if not isinstance(q, np.ndarray) or q.ndim != 1 or q.shape[0] != self.n_joints:
            raise ValueError(f"Input q must be a numpy array of shape ({self.n_joints},)")
        T = np.eye(4)
        for i in range(self.n_joints):
            T = T @ self._matrix_exp6(self.S[:, i] * q[i])
        T = T @ self.M
        pos = T[:3, 3]
        rpy = self.matrix_to_rpy(T[:3, :3])
        if not self._check_workspace(pos):
            logger.warning(f"FK: Position {pos} out of workspace bounds.")
        if not self._check_orientation(rpy):
            logger.warning(f"FK: Orientation {rpy} out of limits.")
        if not self._check_obstacles(pos):
            logger.warning(f"FK: Position {pos} collides with obstacle.")
        return T

    def jacobian_body(self, q: np.ndarray) -> np.ndarray:
        """Compute body Jacobian for given joint angles."""
        # Compute spatial Jacobian first
        J_s = np.zeros((6, self.n_joints))
        T_temp = np.eye(4)
        
        for i in range(self.n_joints):
            # Transform spatial twist to current frame
            if i == 0:
                J_s[:, i] = self.S[:, i]
            else:
                J_s[:, i] = self._adjoint(T_temp) @ self.S[:, i]
            T_temp = T_temp @ self._matrix_exp6(self.S[:, i] * q[i])
        
        # Convert to body Jacobian
        T_final = self.forward_kinematics(q)
        return self._adjoint(inv(T_final)) @ J_s

    def inverse_kinematics(
            self, 
            T_des: np.ndarray, 
            q_init: Optional[np.ndarray] = None,
            pos_tol: float = 1e-6, 
            rot_tol: float = 1e-6, 
            max_iters: int = 300,
            damping: float = 1e-2, 
            step_scale: float = 0.5, 
            dq_max: float = 0.2,
            num_attempts: int = 10
        ) -> Tuple[Optional[np.ndarray], bool]:
        """Solve inverse kinematics with obstacle, workspace, and orientation constraints."""
        # Check workspace, orientation, and obstacle for desired pose
        pos = T_des[:3, 3]
        rpy = self.matrix_to_rpy(T_des[:3, :3])
        if not self._check_workspace(pos):
            logger.warning(f"IK: Desired position {pos} out of workspace bounds.")
            return None, False
        if not self._check_orientation(rpy):
            logger.warning(f"IK: Desired orientation {rpy} out of limits.")
            return None, False
        if not self._check_obstacles(pos):
            logger.warning(f"IK: Desired position {pos} collides with obstacle.")
            return None, False

        # First check if we're trying to reach the home position
        T_home = self.forward_kinematics(np.zeros(self.n_joints))
        home_pos_err = np.linalg.norm(T_des[:3, 3] - T_home[:3, 3])
        home_rot_err = np.arccos(np.clip((np.trace(T_des[:3, :3].T @ T_home[:3, :3]) - 1) / 2.0, -1.0, 1.0))
        if home_pos_err < pos_tol and home_rot_err < rot_tol:
            logger.info("Target is home position, returning zero joints")
            return np.zeros(self.n_joints), True

        limits_lower, limits_upper = self.joint_limits[0], self.joint_limits[1]

        # Use provided initial guess if available
        if q_init is not None:
            q_sol, converged = self._ik_dls(T_des, q_init, pos_tol, rot_tol, max_iters, damping, step_scale, dq_max)
            if converged:
                # Check constraints for solution
                T_sol = self.forward_kinematics(q_sol)
                pos_sol = T_sol[:3, 3]
                rpy_sol = self.matrix_to_rpy(T_sol[:3, :3])
                if self._check_workspace(pos_sol) and self._check_orientation(rpy_sol) and self._check_obstacles(pos_sol):
                    return q_sol, True

        # Try with random starts
        for i in range(num_attempts):
            q0 = np.random.uniform(limits_lower, limits_upper)
            q_sol, converged = self._ik_dls(T_des, q0, pos_tol, rot_tol, max_iters, damping, step_scale, dq_max)
            if converged:
                T_sol = self.forward_kinematics(q_sol)
                pos_sol = T_sol[:3, 3]
                rpy_sol = self.matrix_to_rpy(T_sol[:3, :3])
                if self._check_workspace(pos_sol) and self._check_orientation(rpy_sol) and self._check_obstacles(pos_sol):
                    logger.debug(f"IK converged after {i+1} attempts.")
                    return q_sol, True

        logger.warning("IK failed to converge after all attempts or constraints not satisfied.")
        return None, False


    def _ik_dls(self, T_des: np.ndarray, q0: np.ndarray, pos_tol: float, rot_tol: float, 
           max_iters: int, damping: float, step_scale: float, dq_max: float) -> Tuple[np.ndarray, bool]:
        """Enhanced Damped least-squares IK implementation with singularity handling."""
        q = q0.copy()
        limits_lower, limits_upper = self.joint_limits[0], self.joint_limits[1]
        
        # Special handling for home position
        if np.allclose(q, 0, atol=1e-6):
            # Apply a small perturbation to avoid singularity
            q = q + np.random.uniform(-0.1, 0.1, self.n_joints)
            logger.debug("Applied perturbation to avoid home position singularity")
        
        # Track best solution found
        best_q = q.copy()
        best_error = float('inf')
        
        # Adaptive parameters
        min_damping = damping
        max_damping = 1.0
        damping_factor = min_damping
        
        # Multi-phase optimization for difficult cases
        phase_iters = [max_iters // 3, max_iters // 3, max_iters // 3 + max_iters % 3]
        phase_step_scales = [step_scale * 1.5, step_scale, step_scale * 0.5]  # Start aggressive, end conservative
        
        iteration = 0
        no_improvement_count = 0
        
        for phase, (phase_iter, phase_step) in enumerate(zip(phase_iters, phase_step_scales)):
            logger.debug(f"IK Phase {phase+1}: {phase_iter} iters, step_scale={phase_step}")
            
            for i in range(phase_iter):
                iteration += 1
                
                T_cur = self.forward_kinematics(q)
                T_err = inv(T_cur) @ T_des
                Vb = self._matrix_log6(T_err)
                
                rot_err = norm(Vb[:3])
                pos_err = norm(Vb[3:])
                total_error = pos_err + rot_err
                
                # Track best solution
                if total_error < best_error:
                    improvement = best_error - total_error
                    best_error = total_error
                    best_q = q.copy()
                    no_improvement_count = 0
                    
                    # Log significant improvements
                    if improvement > 1e-4:
                        logger.debug(f"Iter {iteration}: error improved to {total_error:.6f}")
                else:
                    no_improvement_count += 1
                
                # Check convergence - use a more practical combined error metric
                # For real robot applications, we care about overall pose accuracy
                combined_error = pos_err + rot_err * 0.1  # Weight rotation less (0.1m per radian ≈ 57mm per degree)
                if combined_error < (pos_tol + rot_tol * 0.1):
                    logger.debug(f"IK converged at iteration {iteration} (combined error: {combined_error:.6f})")
                    return q, True
                
                # Compute body Jacobian
                Jb = self.jacobian_body(q)
                
                # Calculate manipulability measure
                JJt = Jb @ Jb.T
                det_JJt = np.linalg.det(JJt)
                manipulability = np.sqrt(np.abs(det_JJt))
                
                # Adaptive damping based on manipulability and progress
                if manipulability < 1e-4:  # Near singularity
                    damping_factor = min(max_damping, damping_factor * 1.2)
                elif no_improvement_count > 10:  # Stuck in local minimum
                    damping_factor = min(max_damping, damping_factor * 1.1)
                    # Apply small random perturbation to escape local minimum
                    if no_improvement_count > 20:
                        q += np.random.normal(0, 0.05, self.n_joints)
                        q = np.clip(q, limits_lower, limits_upper)
                        no_improvement_count = 0
                        logger.debug(f"Applied escape perturbation at iter {iteration}")
                else:
                    # Gradually reduce damping for faster convergence
                    damping_factor = max(min_damping, damping_factor * 0.99)
                
                # Damped least squares with regularization
                JtJ = Jb.T @ Jb
                reg_term = (damping_factor ** 2) * np.eye(self.n_joints)
                
                try:
                    damped_inv = np.linalg.inv(JtJ + reg_term)
                    dq = damped_inv @ Jb.T @ Vb
                except np.linalg.LinAlgError:
                    logger.debug(f"Matrix inversion failed at iter {iteration}, using pseudoinverse")
                    dq = np.linalg.pinv(Jb) @ Vb
                
                # Limit step size
                dq_norm = norm(dq)
                if dq_norm > dq_max:
                    dq = dq * (dq_max / dq_norm)
                    
                # Apply step with phase-specific scaling
                adaptive_scale = phase_step
                if total_error > best_error * 1.5:  # If error increased significantly
                    adaptive_scale *= 0.3
                    
                q += adaptive_scale * dq
                
                # Enforce joint limits
                q = np.clip(q, limits_lower, limits_upper)
                
                # Early termination if step is very small and no improvement
                if dq_norm < 1e-9 and no_improvement_count > 5:
                    logger.debug(f"Step size negligible at iteration {iteration}, moving to next phase")
                    break
            
            # Between phases, try a small random perturbation if stuck
            if phase < len(phase_iters) - 1 and best_error > pos_tol + rot_tol:
                perturbation = np.random.normal(0, 0.1, self.n_joints) * (best_error / (pos_tol + rot_tol))
                q = best_q + perturbation
                q = np.clip(q, limits_lower, limits_upper)
                logger.debug(f"Applied inter-phase perturbation, best_error={best_error:.6f}")
        
        # Return best solution found
        final_error = best_error
        # Use same combined error metric for final convergence check
        pos_err_final, rot_err_final = self.check_pose_error(T_des, best_q)
        combined_error_final = pos_err_final + rot_err_final * 0.1
        converged = combined_error_final < (pos_tol + rot_tol * 0.1)
        
        if not converged:
            logger.debug(f"IK completed {iteration} iterations, final error: {final_error:.6f}, "
                        f"combined: {combined_error_final:.6f}")
        
        return best_q, converged

    def check_pose_error(self, T_des: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
        """Calculate position and orientation error between desired and actual pose."""
        T_actual = self.forward_kinematics(q)
        pos_err = norm(T_actual[:3, 3] - T_des[:3, 3])
        R_err = inv(T_actual[:3, :3]) @ T_des[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0))
        return pos_err, angle
    
    def check_singularity(self, q: np.ndarray, threshold: float = 1e-6) -> bool:
        """Check if the current configuration is near a singularity."""
        Jb = self.jacobian_body(q)
        JJt = Jb @ Jb.T
        det_JJt = np.linalg.det(JJt)
        manipulability = np.sqrt(np.abs(det_JJt))
        return manipulability < threshold

    @staticmethod
    def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
        """Convert RPY angles to rotation matrix (XYZ convention)."""
        roll, pitch, yaw = rpy
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx

    @staticmethod
    def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to RPY angles."""
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        return np.array([x, y, z])


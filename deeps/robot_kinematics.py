# robot_kinematics.py
import numpy as np
from numpy.linalg import norm, inv
import xml.etree.ElementTree as ET
from typing import Tuple, Optional, List

class RobotKinematics:
    """Production-ready robot kinematics class using PoE formulation."""
    
    def __init__(self, urdf_path: str, ee_link: str = "tcp", base_link: str = "link0"):
        """
        Initialize robot kinematics from URDF.
        
        Args:
            urdf_path: Path to URDF file
            ee_link: End effector link name
            base_link: Base link name
        """
        self.urdf_path = urdf_path
        self.ee_link = ee_link
        self.base_link = base_link
        
        # Load robot parameters from URDF
        self.S, self.M, self.joint_limits = self._load_from_urdf()
        self.n_joints = self.S.shape[1]
        
        # Precompute for performance
        self._eye6 = np.eye(6)
        
    def _load_from_urdf(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load PoE parameters from URDF file."""
        # Math helpers
        def skew(v):
            return np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])

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
                'name': jname,
                'type': jtype,
                'parent': parent,
                'child': child,
                'axis': axis / (norm(axis) + 1e-12),
                'T_parent_joint': T_parent_joint,
                'limits': limits
            }

        def find_base_link():
            children = {j['child'] for j in joints.values()}
            base_candidates = list(links - children)
            if not base_candidates:
                raise ValueError("Could not determine base link automatically.")
            return base_candidates[0]

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
        """Skew-symmetric matrix from vector."""
        return np.array([[0, -w[2], w[1]],
                        [w[2], 0, -w[0]],
                        [-w[1], w[0], 0]])

    @staticmethod
    def _adjoint(T: np.ndarray) -> np.ndarray:
        """Adjoint transformation of homogeneous matrix."""
        R, p = T[:3, :3], T[:3, 3]
        return np.block([[R, np.zeros((3, 3))],
                         [RobotKinematics._skew(p) @ R, R]])

    @staticmethod
    def _matrix_log6(T: np.ndarray) -> np.ndarray:
        """Matrix logarithm of homogeneous transformation."""
        R, p = T[:3, :3], T[:3, 3]
        trace_R = np.trace(R)
        cos_th = np.clip((trace_R - 1.0) * 0.5, -1.0, 1.0)
        theta = np.arccos(cos_th)

        if theta < 1e-12:
            return np.hstack([np.zeros(3), p])

        s = np.sin(theta)
        omega_hat = (R - R.T) * (0.5 / s)
        omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])

        I = np.eye(3)
        omega_hat2 = omega_hat @ omega_hat
        V_inv = (I / theta - 0.5 * omega_hat + 
                (1.0 / theta - 0.5 / np.tan(theta / 2.0)) * omega_hat2)

        v = V_inv @ p
        return np.hstack([omega * theta, v * theta])

    @staticmethod
    def _matrix_exp6(xi_theta: np.ndarray) -> np.ndarray:
        """Exponential map of twist."""
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
        """Compute forward kinematics for given joint angles."""
        T = np.eye(4)
        for i in range(self.n_joints):
            T = T @ self._matrix_exp6(self.S[:, i] * q[i])
        return T @ self.M

    def jacobian_space(self, q: np.ndarray) -> np.ndarray:
        """Compute spatial Jacobian for given joint angles."""
        J = np.zeros((6, self.n_joints))
        T = np.eye(4)
        for i in range(self.n_joints):
            J[:, i] = self.S[:, i] if i == 0 else self._adjoint(T) @ self.S[:, i]
            T = T @ self._matrix_exp6(self.S[:, i] * q[i])
        return J

    def jacobian_body(self, q: np.ndarray) -> np.ndarray:
        """Compute body Jacobian for given joint angles."""
        T = self.forward_kinematics(q)
        return self._adjoint(inv(T)) @ self.jacobian_space(q)

    def inverse_kinematics(
        self, 
        T_des: np.ndarray, 
        q0: Optional[np.ndarray] = None,
        pos_tol: float = 1e-6, 
        rot_tol: float = 1e-6, 
        max_iters: int = 300,
        damping: float = 1e-2, 
        step_scale: float = 0.5, 
        dq_max: float = 0.2,
        num_attempts: int = 10
    ) -> Tuple[np.ndarray, bool]:
        # Ensure parameters are the correct type
        pos_tol = float(pos_tol)
        rot_tol = float(rot_tol)
        max_iters = int(max_iters)
        damping = float(damping)
        step_scale = float(step_scale)
        dq_max = float(dq_max)
        num_attempts = int(num_attempts)
        """
        Solve inverse kinematics using damped least squares.
        
        Args:
            T_des: Desired end effector pose (4x4)
            q0: Initial guess (if None, uses zero config)
            pos_tol: Position tolerance
            rot_tol: Rotation tolerance
            max_iters: Maximum iterations per attempt
            damping: Damping factor for DLS
            step_scale: Step scaling factor
            dq_max: Maximum joint angle change per step
            num_attempts: Number of random attempts
            
        Returns:
            q_sol: Solution joint angles
            converged: Whether IK converged
        """
        limits_lower, limits_upper = self.joint_limits[0].copy(), self.joint_limits[1].copy()
        
        # Replace infinite limits with reasonable values for sampling
        inf_mask = ~np.isfinite(limits_lower) | ~np.isfinite(limits_upper)
        limits_lower[inf_mask] = -np.pi
        limits_upper[inf_mask] = np.pi
        
        # Default to zero configuration
        if q0 is None:
            q0 = np.clip(np.zeros(self.n_joints), limits_lower, limits_upper)
        
        # Try with initial guess first
        q_sol, converged = self._ik_dls(
            T_des, q0, pos_tol, rot_tol, max_iters, damping, step_scale, dq_max
        )
        if converged:
            return q_sol, True
        
        # Try with random starts
        for _ in range(num_attempts - 1):
            q0 = np.random.uniform(limits_lower, limits_upper)
            q_sol, converged = self._ik_dls(
                T_des, q0, pos_tol, rot_tol, max_iters, damping, step_scale, dq_max
            )
            if converged:
                return q_sol, True
        
        return q_sol, False

    def _ik_dls(
        self, 
        T_des: np.ndarray, 
        q0: np.ndarray,
        pos_tol: float, 
        rot_tol: float, 
        max_iters: int,
        damping: float, 
        step_scale: float, 
        dq_max: float
    ) -> Tuple[np.ndarray, bool]:
        """Damped least-squares IK implementation."""
        q = q0.copy()
        limits_lower, limits_upper = self.joint_limits[0], self.joint_limits[1]
        
        for _ in range(max_iters):
            T_cur = self.forward_kinematics(q)
            T_err = inv(T_cur) @ T_des
            Vb = self._matrix_log6(T_err)
            
            rot_err = norm(Vb[:3])
            pos_err = norm(Vb[3:])
            if rot_err < rot_tol and pos_err < pos_tol:
                return q, True
            
            Jb = self.jacobian_body(q)
            JJt = Jb @ Jb.T
            dq = Jb.T @ np.linalg.solve(JJt + (damping ** 2) * self._eye6, Vb)
            
            # Step limiting
            max_abs = np.max(np.abs(dq))
            if max_abs > dq_max:
                dq *= dq_max / max_abs
                
            q += step_scale * dq
            np.clip(q, limits_lower, limits_upper, out=q)
            
        return q, False

    def check_pose_error(self, T_des: np.ndarray, q: np.ndarray) -> Tuple[float, float]:
        """Calculate position and orientation error between desired and actual pose."""
        T_actual = self.forward_kinematics(q)
        pos_err = norm(T_actual[:3, 3] - T_des[:3, 3])
        
        R_err = inv(T_actual[:3, :3]) @ T_des[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0))
        
        return pos_err, angle

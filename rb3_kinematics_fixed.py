#!/usr/bin/env python3
"""
rb3_kinematics_full.py

Features:
 - Load URDF (or expanded xacro -> URDF) and build kinematic chain
 - forward_kinematics_chain(chain, q) -> T_end, transforms_list
 - inverse_kinematics_dls(chain, T_target, q0) -> numeric DLS IK
 - map joint name -> index and convenience wrappers (dict input/output)
 - round-trip tests with random joint vectors
"""
import numpy as np
import xml.etree.ElementTree as ET
import subprocess
import math
from typing import List, Tuple, Dict, Optional

# -------------------------
# Utils
# -------------------------
def wrap_to_pi(angle):
    a = np.array(angle, dtype=float)
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def axis_angle_to_rotmat(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c = math.cos(theta); s = math.sin(theta); C = 1 - c
    R = np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]
    ], dtype=float)
    return R

def rpy_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll); sr = math.sin(roll)
    cp = math.cos(pitch); sp = math.sin(pitch)
    cy = math.cos(yaw); sy = math.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx

def transform_from_xyz_rpy(xyz, rpy) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_to_rotmat(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = np.array(xyz, dtype=float)
    return T

def adjust_angle_into_limits(angle, lower, upper):
    if lower is None or upper is None:
        return wrap_to_pi(angle)
    for k in range(-4, 5):
        cand = angle + 2.0 * math.pi * k
        if lower - 1e-12 <= cand <= upper + 1e-12:
            return cand
    # if no equivalent found, clamp to nearest bound
    mid = 0.5 * (lower + upper)
    k = round((mid - angle) / (2.0 * math.pi))
    cand = angle + 2.0 * math.pi * k
    if cand < lower:
        return lower
    if cand > upper:
        return upper
    return cand

def clamp_prismatic(val, lower, upper):
    if lower is None and upper is None:
        return val
    if lower is None:
        return min(val, upper)
    if upper is None:
        return max(val, lower)
    return max(lower, min(upper, val))

# -------------------------
# URDF parser & chain
# -------------------------
class JointInfo:
    def __init__(self, name, jtype, parent, child, axis, origin_xyz, origin_rpy, limit):
        self.name = name
        self.type = jtype
        self.parent = parent
        self.child = child
        self.axis = np.array(axis, dtype=float) if axis is not None else np.array([0.,0.,1.])
        self.origin_xyz = np.array(origin_xyz, dtype=float)
        self.origin_rpy = tuple(origin_rpy)
        self.limit = limit

    def __repr__(self):
        return f"Joint(name={self.name}, type={self.type}, parent={self.parent}, child={self.child}, axis={self.axis.tolist()}, origin_xyz={self.origin_xyz.tolist()}, origin_rpy={self.origin_rpy}, limit={self.limit})"

def load_urdf(urdf_path: str, expand_xacro: bool=False) -> Tuple[Dict[str, JointInfo], str]:
    if expand_xacro and urdf_path.endswith('.xacro'):
        try:
            out = subprocess.run(['xacro', urdf_path], capture_output=True, text=True, check=True)
            xml_text = out.stdout
            root = ET.fromstring(xml_text)
        except Exception as e:
            raise RuntimeError("xacro expand failed: " + str(e))
    else:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

    joints = {}
    children = set()
    parents = set()
    for j in root.findall('joint'):
        name = j.attrib['name']
        jtype = j.attrib.get('type', 'revolute')
        parent = j.find('parent').attrib['link']
        child = j.find('child').attrib['link']
        children.add(child); parents.add(parent)
        origin_elem = j.find('origin')
        if origin_elem is not None:
            xyz_attr = origin_elem.attrib.get('xyz', '0 0 0')
            rpy_attr = origin_elem.attrib.get('rpy', '0 0 0')
            xyz = tuple(float(x) for x in xyz_attr.split())
            rpy = tuple(float(x) for x in rpy_attr.split())
        else:
            xyz = (0.0, 0.0, 0.0); rpy = (0.0, 0.0, 0.0)
        axis_elem = j.find('axis')
        axis = (1.0, 0.0, 0.0) if axis_elem is None else tuple(float(x) for x in axis_elem.attrib.get('xyz','1 0 0').split())
        limit_elem = j.find('limit')
        limit = None
        if limit_elem is not None:
            lower = limit_elem.attrib.get('lower')
            upper = limit_elem.attrib.get('upper')
            try:
                limit = {'lower': float(lower) if lower is not None else None,
                         'upper': float(upper) if upper is not None else None}
            except:
                limit = None
        joints[name] = JointInfo(name, jtype, parent, child, axis, xyz, rpy, limit)

    all_links = set(l.attrib['name'] for l in root.findall('link'))
    candidate_roots = list((all_links | parents) - children)
    root_link = candidate_roots[0] if candidate_roots else list(all_links)[0]
    return joints, root_link

def build_chain(joints: Dict[str, JointInfo], root_link: str, tip_link: Optional[str]=None) -> List[JointInfo]:
    parent_to_joints = {}
    child_to_joint = {}
    for j in joints.values():
        parent_to_joints.setdefault(j.parent, []).append(j)
        child_to_joint[j.child] = j

    if tip_link is None:
        cur = root_link
        chain = []
        while True:
            outs = parent_to_joints.get(cur, [])
            if not outs:
                break
            j = outs[0]
            chain.append(j)
            cur = j.child
        return chain

    chain = []
    cur_link = tip_link
    while cur_link != root_link:
        if cur_link not in child_to_joint:
            raise ValueError(f"Cannot reach root_link {root_link} from tip_link {tip_link}")
        j = child_to_joint[cur_link]
        chain.append(j)
        cur_link = j.parent
    chain.reverse()
    return chain

def movable_joints(chain: List[JointInfo]) -> List[JointInfo]:
    return [j for j in chain if j.type != 'fixed']

# -------------------------
# Kinematics core
# -------------------------
def forward_kinematics_chain(chain: List[JointInfo], q: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    T = np.eye(4)
    Ts = []
    qi = 0
    for j in chain:
        T_origin = transform_from_xyz_rpy(j.origin_xyz, j.origin_rpy)
        if j.type == 'fixed':
            T = T @ T_origin
        elif j.type == 'revolute':
            theta = float(q[qi])
            axis = j.axis / (np.linalg.norm(j.axis) + 1e-12)
            R = axis_angle_to_rotmat(axis, theta)
            T_joint = np.eye(4); T_joint[:3,:3] = R
            T = T @ T_origin @ T_joint
            qi += 1
        elif j.type == 'prismatic':
            d = float(q[qi])
            T_joint = np.eye(4); T_joint[:3,3] = j.axis * d
            T = T @ T_origin @ T_joint
            qi += 1
        else:
            T = T @ T_origin
        Ts.append(T.copy())
    return T, Ts

def analytic_jacobian(chain: List[JointInfo], q: np.ndarray) -> np.ndarray:
    movs = movable_joints(chain)
    n = len(movs)
    T = np.eye(4)
    joint_positions = []
    joint_axes_world = []
    qi = 0
    for j in chain:
        T_origin = transform_from_xyz_rpy(j.origin_xyz, j.origin_rpy)
        T = T @ T_origin
        if j.type == 'fixed':
            continue
        pos_world = T[:3,3].copy()
        axis_world = T[:3,:3] @ j.axis
        joint_positions.append(pos_world)
        joint_axes_world.append(axis_world)
        if j.type == 'revolute':
            Rj = axis_angle_to_rotmat(j.axis / (np.linalg.norm(j.axis)+1e-12), q[qi])
            Tj = np.eye(4); Tj[:3,:3] = Rj
            T = T @ Tj
            qi += 1
        elif j.type == 'prismatic':
            Tj = np.eye(4); Tj[:3,3] = j.axis * q[qi]
            T = T @ Tj
            qi += 1

    T_end, _ = forward_kinematics_chain(chain, q)
    p_end = T_end[:3,3]

    J = np.zeros((6, n), dtype=float)
    for i in range(n):
        axis = joint_axes_world[i]
        pi = joint_positions[i]
        if movs[i].type == 'revolute':
            J_v = np.cross(axis, (p_end - pi))
            J_w = axis
        else:
            J_v = axis
            J_w = np.zeros(3)
        J[:3, i] = J_v
        J[3:, i] = J_w
    return J

def pose_to_vec(T: np.ndarray) -> np.ndarray:
    p = T[:3,3]
    R = T[:3,:3]
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)
    if abs(theta) < 1e-8:
        rvec = np.zeros(3)
    else:
        denom = 2.0 * math.sin(theta)
        rx = (R[2,1] - R[1,2]) / denom
        ry = (R[0,2] - R[2,0]) / denom
        rz = (R[1,0] - R[0,1]) / denom
        rvec = theta * np.array([rx, ry, rz])
    return np.concatenate([p, rvec])

def inverse_kinematics_dls(chain: List[JointInfo], T_target: np.ndarray, q0: Optional[np.ndarray]=None,
                           max_iters=400, tol_pos=1e-4, tol_ori=1e-3, alpha=0.7, damp=1e-6):
    movs = movable_joints(chain)
    n = len(movs)
    if q0 is None:
        q = np.zeros(n)
    else:
        q = np.array(q0, dtype=float)

    target_vec = pose_to_vec(T_target)
    for it in range(max_iters):
        Tcur, _ = forward_kinematics_chain(chain, q)
        cur_vec = pose_to_vec(Tcur)
        err = target_vec - cur_vec
        pos_err = np.linalg.norm(err[:3])
        ori_err = np.linalg.norm(err[3:])
        if pos_err < tol_pos and ori_err < tol_ori:
            # adjust into limits and return
            q_adj = np.zeros_like(q)
            for i,j in enumerate(movs):
                if j.type == 'revolute':
                    low = j.limit['lower'] if j.limit else None
                    up  = j.limit['upper'] if j.limit else None
                    q_adj[i] = adjust_angle_into_limits(q[i], low, up)
                else:
                    low = j.limit['lower'] if j.limit else None
                    up  = j.limit['upper'] if j.limit else None
                    q_adj[i] = clamp_prismatic(q[i], low, up)
            return q_adj, {'success': True, 'iters': it, 'pos_err': pos_err, 'ori_err': ori_err}

        J = analytic_jacobian(chain, q)
        JTJ = J.T @ J
        lambdaI = damp * np.eye(JTJ.shape[0])
        try:
            dq = np.linalg.solve(JTJ + lambdaI, J.T @ err)
        except np.linalg.LinAlgError:
            dq = np.linalg.pinv(J) @ err

        q = q + alpha * dq
        # wrap revolute joints to canonical range during iteration
        for i,j in enumerate(movs):
            if j.type == 'revolute':
                q[i] = wrap_to_pi(q[i])

    # didn't converge: adjust to limits best-effort
    q_adj = np.zeros_like(q)
    for i,j in enumerate(movs):
        if j.type == 'revolute':
            low = j.limit['lower'] if j.limit else None
            up  = j.limit['upper'] if j.limit else None
            q_adj[i] = adjust_angle_into_limits(q[i], low, up)
        else:
            low = j.limit['lower'] if j.limit else None
            up  = j.limit['upper'] if j.limit else None
            q_adj[i] = clamp_prismatic(q[i], low, up)
    return q_adj, {'success': False, 'iters': max_iters, 'pos_err': pos_err, 'ori_err': ori_err}

# -------------------------
# Convenience wrappers: name <-> index, dict in/out
# -------------------------
def joint_name_index_map(chain: List[JointInfo]) -> Tuple[List[str], Dict[str,int]]:
    movs = movable_joints(chain)
    names = [j.name for j in movs]
    idx = {name:i for i,name in enumerate(names)}
    return names, idx

def fk_from_joint_dict(chain: List[JointInfo], joint_dict: Dict[str,float]) -> np.ndarray:
    names, idx = joint_name_index_map(chain)
    n = len(names)
    q = np.zeros(n)
    for name,val in joint_dict.items():
        if name not in idx:
            raise KeyError(f"Joint name {name} not in chain movables")
        q[idx[name]] = val
    T, _ = forward_kinematics_chain(chain, q)
    return T

def ik_to_joint_dict(chain: List[JointInfo], T_target: np.ndarray, q0: Optional[np.ndarray]=None):
    q_sol, info = inverse_kinematics_dls(chain, T_target, q0=q0)
    names, idx = joint_name_index_map(chain)
    jd = {names[i]: float(q_sol[i]) for i in range(len(names))}
    return jd, info

# -------------------------
# Main demo + round-trip tests
# -------------------------
if __name__ == "__main__":
    URDF_PATH = "rb3_730es_u.urdf"
    EXPAND_XACRO = False
    TIP_LINK = None

    joints_map, root_link = load_urdf(URDF_PATH, expand_xacro=EXPAND_XACRO)
    chain = build_chain(joints_map, root_link, TIP_LINK)
    movs = movable_joints(chain)
    names, name_idx = joint_name_index_map(chain)
    n = len(movs)

    print("Root link:", root_link)
    print("Movable joints (order):", names)
    print("Movable joints count:", n)
    for j in chain:
        print("  ", j)

    # FK at zero
    q0 = np.zeros(n)
    T_end, Ts = forward_kinematics_chain(chain, q0)
    print("\nFK at zero pose:\n", T_end)

    # Basic IK round-trip: recover q0
    q_sol, info = inverse_kinematics_dls(chain, T_end, q0=np.zeros(n))
    print("\nIK info for zero pose:", info)
    print("IK solution (deg) normalized:", np.degrees(wrap_to_pi(q_sol)))
    # Now do randomized round-trip tests
    print("\nRandom round-trip tests (5 samples):")
    rng = np.random.default_rng(12345)
    for t in range(5):
        # sample within joint limits if available, else within [-pi,pi]
        q_rand = np.zeros(n)
        for i,j in enumerate(movs):
            if j.type == 'revolute':
                if j.limit:
                    low, up = j.limit['lower'], j.limit['upper']
                    q_rand[i] = rng.uniform(low, up)
                else:
                    q_rand[i] = rng.uniform(-np.pi, np.pi)
            else:
                if j.limit:
                    low, up = j.limit['lower'], j.limit['upper']
                    q_rand[i] = rng.uniform(low, up)
                else:
                    q_rand[i] = rng.uniform(-0.1, 0.1)
        Tt, _ = forward_kinematics_chain(chain, q_rand)
        q_rec, info = inverse_kinematics_dls(chain, Tt, q0=np.zeros(n))
        # compare minimal angle differences for revolute
        diff = np.zeros(n)
        for i,j in enumerate(movs):
            if j.type == 'revolute':
                diff[i] = wrap_to_pi(q_rand[i] - q_rec[i])
            else:
                diff[i] = q_rand[i] - q_rec[i]
        pos_err = np.linalg.norm((pose_to_vec(Tt) - pose_to_vec(forward_kinematics_chain(chain, q_rec)[0]))[:3])
        ori_err = np.linalg.norm((pose_to_vec(Tt) - pose_to_vec(forward_kinematics_chain(chain, q_rec)[0]))[3:])
        print(f" sample {t+1}: pos_err={pos_err:.6e}, ori_err={ori_err:.6e}, max_joint_deg_err={np.max(np.abs(np.degrees(diff))):.6f}")

    print("\nYou can call fk_from_joint_dict(chain, {'base': 0.1, ...}) or ik_to_joint_dict(chain, T_target).")

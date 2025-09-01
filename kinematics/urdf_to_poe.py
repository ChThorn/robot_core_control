# kinematics/urdf_to_poe.py
from __future__ import annotations
import xml.etree.ElementTree as ET
import numpy as np

def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]])
    Ry = np.array([[ cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cr, -sr],
                   [0.0,  sr,  cr]])
    return Rz @ Ry @ Rx

def make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = p
    return T

def parse_origin(origin_elem: ET.Element | None) -> np.ndarray:
    if origin_elem is None:
        return np.eye(4, dtype=float)
    xyz = origin_elem.get('xyz', '0 0 0').split()
    rpy = origin_elem.get('rpy', '0 0 0').split()
    x, y, z = map(float, xyz)
    r, p, y_ = map(float, rpy)
    return make_T(rpy_to_R(r, p, y_), np.array([x, y, z], dtype=float))

def load_urdf(urdf_path: str) -> tuple[set[str], dict[str, dict]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = {link.attrib['name'] for link in root.findall('link')}
    joints: dict[str, dict] = {}
    for j in root.findall('joint'):
        jname = j.attrib['name']
        jtype = j.attrib['type']
        parent = j.find('parent').attrib['link']
        child = j.find('child').attrib['link']
        T_parent_joint = parse_origin(j.find('origin'))

        axis = np.array([0.0, 0.0, 1.0], dtype=float)
        axis_elem = j.find('axis')
        if axis_elem is not None and 'xyz' in axis_elem.attrib:
            axis = np.array(list(map(float, axis_elem.attrib['xyz'].split())), dtype=float)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 0:
            axis = axis / axis_norm

        limit_elem = j.find('limit')
        limits = {'lower': -np.inf, 'upper': np.inf}
        if limit_elem is not None:
            if 'lower' in limit_elem.attrib:
                limits['lower'] = float(limit_elem.attrib['lower'])
            if 'upper' in limit_elem.attrib:
                limits['upper'] = float(limit_elem.attrib['upper'])

        joints[jname] = {
            'name': jname,
            'type': jtype,
            'parent': parent,
            'child': child,
            'axis': axis,
            'T_parent_joint': T_parent_joint,
            'limits': limits,
        }
    return links, joints

def find_base_link(links: set[str], joints: dict[str, dict]) -> str:
    children = {j['child'] for j in joints.values()}
    base_candidates = list(links - children)
    if not base_candidates:
        raise ValueError("Could not determine base link automatically.")
    if len(base_candidates) > 1:
        # Choose consistent ordering for determinism
        base_candidates.sort()
    return base_candidates[0]

def build_chain(joints: dict[str, dict], base_link: str, ee_link: str) -> list[dict]:
    child_to_joint = {j['child']: j for j in joints.values()}
    chain = []
    link = ee_link
    visited = set()
    while link != base_link:
        if link in visited:
            raise ValueError("Cycle detected in kinematic chain.")
        visited.add(link)
        if link not in child_to_joint:
            raise ValueError(f"No joint found from link '{link}' toward base '{base_link}'.")
        j = child_to_joint[link]
        chain.append(j)
        link = j['parent']
    chain.reverse()
    return chain

def poe_from_urdf(urdf_path: str, ee_link: str, base_link: str | None = None):
    """
    Returns:
      S: 6xn space screw axes (in base frame)
      M: 4x4 home configuration of EE in base frame
      joint_limits: 2xn array
      base_link, ee_link: names used
    """
    links, joints = load_urdf(urdf_path)
    if base_link is None:
        base_link = find_base_link(links, joints)

    chain = build_chain(joints, base_link, ee_link)

    T_base_to_here = np.eye(4, dtype=float)
    S_list: list[np.ndarray] = []
    joint_limits_list: list[list[float]] = []

    for j_data in chain:
        # Transform to the joint frame of this joint
        T_base_to_joint = T_base_to_here @ j_data['T_parent_joint']
        Rbj, pbj = T_base_to_joint[:3, :3], T_base_to_joint[:3, 3]
        axis_b = Rbj @ j_data['axis']

        if j_data['type'] in ('revolute', 'continuous'):
            w = axis_b
            v = -np.cross(w, pbj)
            S_list.append(np.hstack((w, v)))
            joint_limits_list.append([j_data['limits']['lower'], j_data['limits']['upper']])

        elif j_data['type'] == 'prismatic':
            w = np.zeros(3)
            v = axis_b
            S_list.append(np.hstack((w, v)))
            joint_limits_list.append([j_data['limits']['lower'], j_data['limits']['upper']])
        else:
            # fixed/floating joints do not add DoF; advance transform only
            pass

        # Advance to child link frame for next joint
        T_base_to_here = T_base_to_joint

    M = T_base_to_here
    if not S_list:
        raise ValueError("No active joints found in the chain.")

    S = np.array(S_list, dtype=float).T  # 6 x n
    joint_limits = np.array(joint_limits_list, dtype=float).T  # 2 x n
    return S, M, joint_limits, base_link, ee_link

import os
import json
import numpy as np

from deeps.robot_kinematics import RobotKinematics
from rb3_kinematics_gpt import DebugIKSolver


def test_fk_ik_roundtrip_sample():
    base = os.path.dirname(__file__) + '/..'
    base = os.path.abspath(base)
    json_path = os.path.join(base, 'Path_1751586417_20250704_084744.json')
    urdf_path = os.path.join(base, 'rb3_730es_u.urdf')

    with open(json_path, 'r') as f:
        data = json.load(f)

    robot = RobotKinematics(urdf_path)
    solver = DebugIKSolver(robot)

    waypoints = data.get('waypoints', [])[:3]
    assert len(waypoints) > 0

    for wp in waypoints:
        tcp = wp.get('tcp_position', [])
        target_pos = np.array(tcp[:3], dtype=float) / 1000.0
        q_rec_deg = np.array(wp.get('joint_angles', []), dtype=float)
        q_rec = np.radians(q_rec_deg)

        q_sol, err, success = solver.comprehensive_solve(target_pos, initial_guess=q_rec, verbose=False)
        assert success, f"IK failed for target {target_pos}"
        assert err < 0.002, f"Position error too large: {err} m"

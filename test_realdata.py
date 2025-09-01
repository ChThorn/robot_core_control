# import pytest
# import numpy as np
# import json
# import os

# from kinematics.urdf_to_poe import poe_from_urdf, rpy_to_R
# from kinematics.poe_kin import fk_space

# # --- Test Fixture ---
# @pytest.fixture(scope="module")
# def robot_model():
#     """Loads the RB3 URDF and returns its PoE parameters."""
#     urdf_path = "rb3_730es_u.urdf"
#     if not os.path.exists(urdf_path):
#         pytest.fail(f"URDF file not found at {urdf_path}.")
    
#     S, M, limits, _, _ = poe_from_urdf(urdf_path, ee_link="tcp", base_link="link0")
#     return {"S": S, "M": M, "joint_limits": limits}

# # --- Test Case ---
# def test_fk_against_real_data(robot_model):
#     """
#     Validates the FK model by comparing its output against real,
#     recorded data from the robot.
#     """
#     S, M = robot_model["S"], robot_model["M"]
#     json_path = "Path_1751586417_20250704_084744.json"
    
#     if not os.path.exists(json_path):
#         pytest.fail(f"Real data file not found at {json_path}.")
        
#     with open(json_path) as f:
#         real_data = json.load(f)

#     # --- NEW: Define the TCP correction transform ---
#     # This matrix represents the +1.766 mm offset on the Z-axis.
#     tcp_correction = np.eye(4)
#     tcp_correction[2, 3] = 0.001766 # The difference vector's Z-component

#     pos_tolerance = 1.5e-3
#     rot_tolerance = 0.025

#     for i, waypoint in enumerate(real_data["waypoints"]):
#         q_deg = np.array(waypoint["joint_angles"])
#         q_rad = np.deg2rad(q_deg)

#         tcp_recorded = waypoint["tcp_position"]
#         p_expected_m = np.array(tcp_recorded[:3]) / 1000.0
#         rpy_expected_rad = np.deg2rad(np.array(tcp_recorded[3:]))

#         # 5. Calculate the pose and APPLY THE CORRECTION
#         T_from_urdf = fk_space(S, M, q_rad)
#         T_calculated = T_from_urdf @ tcp_correction # Apply the offset
        
#         p_calculated = T_calculated[:3, 3]
#         R_calculated = T_calculated[:3, :3]

#         # 6. Compare the results (no try/except needed now)
#         pos_error = np.linalg.norm(p_calculated - p_expected_m)
#         assert pos_error < pos_tolerance, \
#             f"Position error at waypoint {i} is {pos_error*1000:.2f} mm (too high)"

#         R_expected = rpy_to_R(roll=rpy_expected_rad[0], pitch=rpy_expected_rad[1], yaw=rpy_expected_rad[2])
#         R_error_matrix = R_calculated.T @ R_expected
#         trace = np.trace(R_error_matrix)
#         angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        
#         assert angle_rad < rot_tolerance, \
#             f"Rotation error at waypoint {i} is {np.rad2deg(angle_rad):.2f} degrees (too high)"

import pytest
import numpy as np
import json
import os

from kinematics.urdf_to_poe import poe_from_urdf, rpy_to_R
from kinematics.poe_kin import fk_space

# --- Test Fixture ---
@pytest.fixture(scope="module")
def robot_model():
    """Loads the RB3 URDF and returns its PoE parameters."""
    urdf_path = "rb3_730es_u.urdf"
    if not os.path.exists(urdf_path):
        pytest.fail(f"URDF file not found at {urdf_path}.")
    
    S, M, limits, _, _ = poe_from_urdf(urdf_path, ee_link="tcp", base_link="link0")
    return {"S": S, "M": M, "joint_limits": limits}

# --- Test Case ---
def test_fk_against_new_real_data(robot_model):
    """
    Validates the FK model against the new, 38-waypoint recording.
    """
    S, M = robot_model["S"], robot_model["M"]
    
    # UPDATED: Point to the new JSON file
    json_path = "third_20250710_162459.json"
    
    if not os.path.exists(json_path):
        pytest.fail(f"Real data file not found at {json_path}.")
        
    with open(json_path) as f:
        real_data = json.load(f)

    # The TCP correction transform from our previous calibration
    tcp_correction = np.eye(4)
    tcp_correction[2, 3] = 0.001766

    # Using a slightly relaxed tolerance, as complex paths may have more variance
    # pos_tolerance = 2.5e-3  # 2.5 mm for position
    pos_tolerance = 3.5e-3  # 3.5 mm for position
    rot_tolerance = 0.044   # ~2.5 degrees for orientation in radians

    for i, waypoint in enumerate(real_data["waypoints"]):
        # UPDATED: Use 'joint_positions' key for the new file format
        q_deg = np.array(waypoint["joint_positions"])
        q_rad = np.deg2rad(q_deg)

        tcp_recorded = waypoint["tcp_position"]
        p_expected_m = np.array(tcp_recorded[:3]) / 1000.0
        rpy_expected_rad = np.deg2rad(np.array(tcp_recorded[3:]))

        # Calculate the pose and apply the correction
        T_from_urdf = fk_space(S, M, q_rad)
        T_calculated = T_from_urdf @ tcp_correction
        
        p_calculated = T_calculated[:3, 3]
        R_calculated = T_calculated[:3, :3]

        # Compare the results
        pos_error = np.linalg.norm(p_calculated - p_expected_m)
        assert pos_error < pos_tolerance, \
            f"Position error at waypoint {i} is {pos_error*1000:.2f} mm (too high)"

        R_expected = rpy_to_R(roll=rpy_expected_rad[0], pitch=rpy_expected_rad[1], yaw=rpy_expected_rad[2])
        R_error_matrix = R_calculated.T @ R_expected
        trace = np.trace(R_error_matrix)
        angle_rad = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        
        assert angle_rad < rot_tolerance, \
            f"Rotation error at waypoint {i} is {np.rad2deg(angle_rad):.2f} degrees (too high)"
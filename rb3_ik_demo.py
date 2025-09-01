# examples/rb3_ik_demo.py
import numpy as np
from numpy.linalg import norm
from kinematics.urdf_to_poe import poe_from_urdf
from kinematics.poe_kin import fk_space, ik_solve_robust

def main():
    # Use the RB3 URDF (generated from xacro in upstream repo)
    # Joint order and tcp link are consistent with the public URDF.
    URDF_PATH = "rb3_730es_u.urdf"
    EE_LINK = "tcp"
    BASE_LINK = "link0"

    S, M, joint_limits, base, ee = poe_from_urdf(URDF_PATH, ee_link=EE_LINK, base_link=BASE_LINK)

    # Known configuration test (adjust to your robot’s convention if needed)
    q_known = np.array([0.5, 0.1, 0.2, 0.8, -0.4, 0.6])
    T_fk = fk_space(S, M, q_known)
    print("Target FK Pose (T_des):\n", np.round(T_fk, 4))

    q_sol, converged = ik_solve_robust(
        S, M, T_fk, joint_limits,
        num_attempts=15,
        seed=42,
        pos_tol=1e-6,
        rot_tol=1e-6,
        damping=1e-2,
        step_scale=0.5,
        dq_max=0.2,
        limit_avoid_weight=0.0,
        verbose=True
    )

    print("\nRecovered q from IK:\n", np.round(q_sol, 6))
    print("Converged:", converged)

    if converged:
        T_check = fk_space(S, M, q_sol)
        pos_err = norm(T_check[:3, 3] - T_fk[:3, 3])
        R_err = np.linalg.inv(T_check[:3, :3]) @ T_fk[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
        print(f"\nPosition error: {pos_err:.6e} meters")
        print(f"Rotation error: {angle:.6e} radians")

if __name__ == "__main__":
    main()

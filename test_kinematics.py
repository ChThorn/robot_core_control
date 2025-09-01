import pytest
import numpy as np
import os
from kinematics.urdf_to_poe import poe_from_urdf
from kinematics.poe_kin import (
    skew,
    matrix_exp6,
    matrix_log6,
    fk_space,
    jacobian_body,
    ik_solve_robust
)

# Mark all tests in this file with a custom marker
pytestmark = pytest.mark.kinematics

# ----------------- Test Fixtures -----------------
# Fixtures provide a fixed baseline for tests. `scope="module"` means
# the fixture is set up only once for all tests in this file.

@pytest.fixture(scope="module")
def robot_model():
    """Loads the RB3 URDF and returns its PoE parameters."""
    urdf_path = "rb3_730es_u.urdf"
    if not os.path.exists(urdf_path):
        pytest.fail(f"URDF file not found at {urdf_path}. Make sure it's in the same directory.")
    
    S, M, limits, _, _ = poe_from_urdf(urdf_path, ee_link="tcp", base_link="link0")
    return {"S": S, "M": M, "joint_limits": limits}

# ----------------- Test Cases -----------------

class TestMathHelpers:
    """Tests for the low-level math utility functions."""

    def test_skew(self):
        """Verify the skew-symmetric matrix calculation."""
        v = np.array([1, 2, 3])
        v_hat = skew(v)
        expected = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        assert np.allclose(v_hat, expected)

    def test_exp_log_identity(self):
        """Test that matrix_log6 is the inverse of matrix_exp6."""
        # Test with a general twist (rotation and translation)
        xi_theta1 = np.array([0.2, -0.5, 1.0, 0.8, 0.3, -0.6])
        T1 = matrix_exp6(xi_theta1)
        xi_theta_recovered1 = matrix_log6(T1)
        assert np.allclose(xi_theta1, xi_theta_recovered1)

        # Test with a pure translation
        xi_theta2 = np.array([0.0, 0.0, 0.0, 0.1, 0.2, 0.3])
        T2 = matrix_exp6(xi_theta2)
        xi_theta_recovered2 = matrix_log6(T2)
        assert np.allclose(xi_theta2, xi_theta_recovered2, atol=1e-9)

class TestKinematics:
    """Tests for FK and Jacobian calculations."""

    def test_fk_space_known_config(self, robot_model):
        """Test FK against the known configuration from the demo script."""
        S, M = robot_model["S"], robot_model["M"]
        q_known = np.array([0.5, 0.1, 0.2, 0.8, -0.4, 0.6])
        
        T_fk = fk_space(S, M, q_known)
        
        # This is the expected result from your demo script's output
        T_expected = np.array([
            [-0.2622, -0.9540,  0.1453,  0.1319],
            [ 0.9264, -0.2911, -0.2389,  0.0329],
            [ 0.2702,  0.0720,  0.9601,  0.8545],
            [ 0.0,     0.0,     0.0,     1.0   ]
        ])
        
        assert np.allclose(T_fk, T_expected, atol=1e-4)

    def test_jacobian_body_numerical(self, robot_model):
        """
        Verify the body Jacobian by comparing it to a numerical approximation.
        The relationship Vb = Jb(q) * q_dot is used for verification.
        """
        S, M = robot_model["S"], robot_model["M"]
        q = np.random.uniform(-np.pi/4, np.pi/4, size=S.shape[1])
        q_dot = np.random.randn(S.shape[1])
        dt = 1e-7

        # 1. Calculate analytical Jacobian and resulting twist
        Jb = jacobian_body(S, M, q)
        Vb_analytical = Jb @ q_dot

        # 2. Calculate numerical twist by finite difference
        T_current = fk_space(S, M, q)
        T_next = fk_space(S, M, q + q_dot * dt)
        T_err = np.linalg.inv(T_current) @ T_next
        Vb_numerical = matrix_log6(T_err) / dt
        
        assert np.allclose(Vb_analytical, Vb_numerical, atol=1e-5)

class TestIKSolver:
    """Tests for the inverse kinematics solver."""

    def test_ik_convergence_to_known(self, robot_model):
        """Test if IK can recover a known set of joint angles."""
        S, M, limits = robot_model["S"], robot_model["M"], robot_model["joint_limits"]
        q_known = np.array([0.5, 0.1, 0.2, 0.8, -0.4, 0.6])
        T_des = fk_space(S, M, q_known)

        q_sol, converged = ik_solve_robust(S, M, T_des, limits, seed=42)

        assert converged
        # Check if the solution is close. Note: IK can find equivalent solutions
        # (e.g., q + 2*pi), so we check the resulting pose instead of just q.
        T_check = fk_space(S, M, q_sol)
        assert np.allclose(T_des, T_check, atol=1e-5)

    def test_ik_unreachable_target(self, robot_model):
        """Test that IK fails gracefully for a target outside the workspace."""
        S, M, limits = robot_model["S"], robot_model["M"], robot_model["joint_limits"]
        
        # Create a target pose that is physically impossible to reach
        T_unreachable = np.eye(4)
        T_unreachable[:3, 3] = np.array([10.0, 10.0, 10.0]) # 10 meters away

        # Use fewer attempts to speed up the test for a known failure
        q_sol, converged = ik_solve_robust(
            S, M, T_unreachable, limits, num_attempts=3, seed=1
        )
        
        assert not converged

    def test_ik_respects_joint_limits(self, robot_model):
        """Test that the IK solution adheres to specified joint limits."""
        S, M = robot_model["S"], robot_model["M"]
        # Make a copy of limits to modify them for this test
        limits = robot_model["joint_limits"].copy()

        # Define a very narrow limit for the elbow joint (joint 2)
        limits[:, 2] = [-0.1, 0.1] 
        
        # --- FIX IS HERE ---
        # Create a target pose using a q that is ALREADY VALID
        # under the new limits. Here, the elbow is at 0.05 rad.
        q_target = np.array([0.1, 0.2, 0.05, 0.1, 0.1, 0.1])
        T_des = fk_space(S, M, q_target)
        
        q_sol, converged = ik_solve_robust(S, M, T_des, limits, seed=2)

        # Now, the solver should be able to find a solution.
        assert converged
        
        # And the primary goal of the test can be verified: the solution
        # found must be within the specified narrow bounds.
        assert limits[0, 2] <= q_sol[2] <= limits[1, 2]

# monitoring.py
import logging
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class IKResult:
    q: np.ndarray
    converged: bool
    position_error: float
    rotation_error: float
    computation_time: float
    iterations: int

class KinematicsMonitor:
    def __init__(self):
        self.logger = logging.getLogger('robot_kinematics')
        self.logger.setLevel(logging.INFO)
        
        # Add handlers if not already added
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
    
    def log_ik_attempt(self, T_des: np.ndarray, q0: Optional[np.ndarray], 
                      result: IKResult, success: bool):
        if success:
            self.logger.info(f"IK solved successfully in {result.computation_time:.4f}s")
            self.logger.info(f"Position error: {result.position_error:.6e}, "
                           f"Rotation error: {result.rotation_error:.6e}")
        else:
            self.logger.warning(f"IK failed to converge after {result.iterations} iterations")
            self.logger.warning(f"Best position error: {result.position_error:.6e}, "
                              f"Rotation error: {result.rotation_error:.6e}")

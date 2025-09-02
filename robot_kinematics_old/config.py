# config.py
import yaml

class RobotConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.urdf_path = self.config.get('urdf_path', 'rb3_730es_u.urdf')
        self.ee_link = self.config.get('ee_link', 'tcp')
        self.base_link = self.config.get('base_link', 'link0')
        self.ik_params = self.config.get('ik_params', {})
        
    def get_ik_params(self):
        """Get IK parameters with defaults and ensure proper types."""
        params = {
            'pos_tol': float(self.ik_params.get('pos_tol', 1e-6)),
            'rot_tol': float(self.ik_params.get('rot_tol', 1e-6)),
            'max_iters': int(self.ik_params.get('max_iters', 300)),
            'damping': float(self.ik_params.get('damping', 1e-2)),
            'step_scale': float(self.ik_params.get('step_scale', 0.5)),
            'dq_max': float(self.ik_params.get('dq_max', 0.2)),
            'num_attempts': int(self.ik_params.get('num_attempts', 10))
        }
        return params
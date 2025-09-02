#!/usr/bin/env python3
"""
Production configuration management for robot kinematics system.
"""

import os
from typing import Dict, Any
import json

class KinematicsConfig:
    """Centralized configuration management for robot kinematics."""
    
    # Default production parameters
    DEFAULT_CONFIG = {
        "ik": {
            "position_tolerance": 2e-3,      # 2mm
            "rotation_tolerance": 5e-3,      # 0.3 degrees in radians
            "max_iterations": 300,
            "damping": 5e-4,
            "step_scale": 0.3,
            "max_step_size": 0.3,
            "max_attempts": 30,
            "combined_error_weight": 0.1,    # Weight for rotation in combined error
            "acceptance_threshold": 3e-3      # 3mm combined error threshold
        },
        "performance": {
            "enable_debug_logging": False,
            "random_seed": None,              # Set for reproducible behavior
            "max_fk_time_warning": 1e-3,     # 1ms warning threshold
            "max_ik_time_warning": 1.0       # 1s warning threshold  
        },
        "safety": {
            "joint_limit_margin": 0.05,      # 5% margin from joint limits
            "singularity_threshold": 1e-4,
            "max_condition_number": 1e6
        },
        "robot": {
            "urdf_path": "rb3_730es_u.urdf",
            "end_effector_link": "tcp",
            "base_link": "link0"
        }
    }
    
    def __init__(self, config_file: str = None):
        """Initialize configuration."""
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        
        # Check environment variables for overrides
        self._apply_env_overrides()
    
    def load_config(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            
            # Deep merge configuration
            self._deep_merge(self.config, user_config)
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load configuration from {config_file}: {e}")
    
    def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Recursively merge configuration dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Example: KINEMATICS_IK_MAX_ATTEMPTS=50
        env_prefix = "KINEMATICS_"
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(env_prefix):
                config_path = env_key[len(env_prefix):].lower().split('_')
                
                try:
                    # Convert string value to appropriate type
                    if env_value.lower() in ['true', 'false']:
                        value = env_value.lower() == 'true'
                    elif '.' in env_value:
                        value = float(env_value)
                    else:
                        value = int(env_value)
                    
                    # Apply to nested config
                    current = self.config
                    for key in config_path[:-1]:
                        current = current.setdefault(key, {})
                    current[config_path[-1]] = value
                    
                except (ValueError, KeyError):
                    # Ignore invalid environment variables
                    pass
    
    def get(self, section: str, key: str = None):
        """Get configuration value."""
        if key is None:
            return self.config.get(section, {})
        return self.config.get(section, {}).get(key)
    
    def save_config(self, config_file: str):
        """Save current configuration to file."""
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

# Global configuration instance
config = KinematicsConfig()

def get_config() -> KinematicsConfig:
    """Get global configuration instance."""
    return config

def set_config_file(config_file: str):
    """Set configuration file for global instance."""
    global config
    config = KinematicsConfig(config_file)

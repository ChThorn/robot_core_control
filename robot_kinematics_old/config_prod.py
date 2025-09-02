#!/usr/bin/env python3
"""
Enhanced production configuration system for robot kinematics.
Supports validation, environment-specific settings, and robust error handling.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class RobotConfig:
    """Enhanced robot configuration with validation and environment support."""
    
    def __init__(self, config_path: str, environment: str = 'production'):
        """
        Initialize robot configuration.
        
        Args:
            config_path: Path to YAML configuration file
            environment: Environment name (development, testing, production)
        """
        self.config_path = Path(config_path)
        self.environment = environment
        
        if not self.config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {e}")
        
        # Validate configuration structure
        self._validate_config()
        
        # Load environment-specific settings
        self._load_environment_config()
        
        logger.info(f"Configuration loaded from {config_path} for environment: {environment}")
    
    def _validate_config(self):
        """Validate configuration structure and required fields."""
        required_fields = ['robot', 'kinematics', 'logging']
        
        for field in required_fields:
            if field not in self.config:
                raise ConfigurationError(f"Missing required configuration section: {field}")
        
        # Validate robot section
        robot_config = self.config['robot']
        required_robot_fields = ['urdf_path', 'ee_link', 'base_link']
        
        for field in required_robot_fields:
            if field not in robot_config:
                raise ConfigurationError(f"Missing required robot configuration: {field}")
        
        # Validate URDF file exists
        urdf_path = Path(robot_config['urdf_path'])
        if not urdf_path.exists():
            raise ConfigurationError(f"URDF file not found: {urdf_path}")
        
        # Validate kinematics parameters
        ik_params = self.config['kinematics'].get('inverse_kinematics', {})
        self._validate_ik_params(ik_params)
    
    def _validate_ik_params(self, ik_params: Dict[str, Any]):
        """Validate inverse kinematics parameters."""
        param_ranges = {
            'pos_tol': (1e-8, 1e-3),
            'rot_tol': (1e-8, 1e-3),
            'max_iters': (10, 1000),
            'damping': (1e-6, 1e-1),
            'step_scale': (0.1, 1.0),
            'dq_max': (0.01, 1.0),
            'num_attempts': (1, 50)
        }
        
        for param, (min_val, max_val) in param_ranges.items():
            if param in ik_params:
                value = ik_params[param]
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    raise ConfigurationError(
                        f"Invalid {param}: {value}. Must be between {min_val} and {max_val}"
                    )
    
    def _load_environment_config(self):
        """Load environment-specific configuration overrides."""
        env_config = self.config.get('environments', {}).get(self.environment, {})
        
        if env_config:
            logger.info(f"Applying {self.environment} environment overrides")
            self._deep_update(self.config, env_config)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    @property
    def urdf_path(self) -> str:
        """Get URDF file path."""
        return self.config['robot']['urdf_path']
    
    @property
    def ee_link(self) -> str:
        """Get end effector link name."""
        return self.config['robot']['ee_link']
    
    @property
    def base_link(self) -> str:
        """Get base link name."""
        return self.config['robot']['base_link']
    
    def get_ik_params(self) -> Dict[str, Any]:
        """Get inverse kinematics parameters with proper type conversion."""
        ik_config = self.config['kinematics'].get('inverse_kinematics', {})
        
        # Default parameters
        defaults = {
            'pos_tol': 1e-6,
            'rot_tol': 1e-6,
            'max_iters': 300,
            'damping': 1e-2,
            'step_scale': 0.5,
            'dq_max': 0.2,
            'num_attempts': 10
        }
        
        # Merge with configuration
        params = {**defaults, **ik_config}
        
        # Ensure proper types
        type_conversions = {
            'pos_tol': float,
            'rot_tol': float,
            'max_iters': int,
            'damping': float,
            'step_scale': float,
            'dq_max': float,
            'num_attempts': int
        }
        
        for param, convert_func in type_conversions.items():
            if param in params:
                params[param] = convert_func(params[param])
        
        return params
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})
    
    def get_robot_controller_config(self) -> Dict[str, Any]:
        """Get robot controller configuration."""
        return self.config.get('robot_controller', {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self.config.get('validation', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance monitoring configuration."""
        return self.config.get('performance', {})
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to file."""
        if output_path is None:
            output_path = self.config_path
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        self._deep_update(self.config, updates)
        self._validate_config()
        logger.info("Configuration updated and validated")
    
    def get_joint_limits_deg(self) -> Optional[np.ndarray]:
        """Get joint limits in degrees if specified in config."""
        limits = self.config['robot'].get('joint_limits_deg')
        if limits:
            return np.array(limits)
        return None
    
    def get_workspace_limits(self) -> Optional[Dict[str, Any]]:
        """Get workspace limits if specified."""
        return self.config['robot'].get('workspace_limits')
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"RobotConfig(path={self.config_path}, env={self.environment})"
    
    def __repr__(self) -> str:
        return self.__str__()

def create_default_config(output_path: str = 'robot_config_prod.yaml'):
    """Create a default configuration file."""
    default_config = {
        'robot': {
            'urdf_path': 'rb3_730es_u.urdf',
            'ee_link': 'tcp',
            'base_link': 'link0',
            'joint_limits_deg': [
                [-180, 180],  # Base
                [-180, 180],  # Shoulder
                [-180, 180],  # Elbow
                [-180, 180],  # Wrist1
                [-180, 180],  # Wrist2
                [-180, 180]   # Wrist3
            ],
            'workspace_limits': {
                'x_range': [-1.0, 1.0],
                'y_range': [-1.0, 1.0],
                'z_range': [0.0, 1.5]
            }
        },
        'kinematics': {
            'inverse_kinematics': {
                'pos_tol': 1e-6,
                'rot_tol': 1e-6,
                'max_iters': 300,
                'damping': 1e-2,
                'step_scale': 0.5,
                'dq_max': 0.2,
                'num_attempts': 10
            }
        },
        'robot_controller': {
            'ip_address': '192.168.1.100',
            'port': 30003,
            'timeout': 5.0,
            'units': {
                'joint_angles': 'degrees',
                'positions': 'millimeters',
                'orientations': 'radians'
            }
        },
        'validation': {
            'position_tolerance': 0.005,  # 5mm
            'rotation_tolerance': 0.01,   # ~0.57 degrees
            'max_samples': 100
        },
        'performance': {
            'log_statistics': True,
            'max_ik_time': 1.0,  # seconds
            'warning_thresholds': {
                'position_error': 0.01,
                'rotation_error': 0.02,
                'ik_success_rate': 0.95
            }
        },
        'logging': {
            'level': 'INFO',
            'file': 'robot_kinematics.log',
            'max_size_mb': 10,
            'backup_count': 5
        },
        'environments': {
            'development': {
                'logging': {
                    'level': 'DEBUG'
                },
                'kinematics': {
                    'inverse_kinematics': {
                        'num_attempts': 5
                    }
                }
            },
            'testing': {
                'validation': {
                    'max_samples': 50
                }
            },
            'production': {
                'logging': {
                    'level': 'INFO'
                },
                'performance': {
                    'log_statistics': True
                }
            }
        }
    }
    
    try:
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        print(f"Default configuration created: {output_path}")
        return output_path
    except Exception as e:
        raise ConfigurationError(f"Failed to create default configuration: {e}")

if __name__ == "__main__":
    # Create default configuration if run directly
    create_default_config()


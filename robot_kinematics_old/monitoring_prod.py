#!/usr/bin/env python3
"""
Enhanced production monitoring and validation system for robot kinematics.
Provides real-time performance monitoring, error tracking, and validation tools.
"""

import numpy as np
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    timestamp: float
    operation_type: str  # 'FK', 'IK', 'validation'
    computation_time: float
    success: bool
    position_error: Optional[float] = None
    rotation_error: Optional[float] = None
    joint_angles: Optional[List[float]] = None
    
class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.session_start = time.time()
        self.alerts = []
        
        # Performance thresholds
        self.thresholds = {
            'max_ik_time': 1.0,
            'max_fk_time': 0.1,
            'min_ik_success_rate': 0.95,
            'max_position_error': 0.01,
            'max_rotation_error': 0.02
        }
    
    def record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric."""
        self.metrics_history.append(metric)
        self._check_thresholds(metric)
    
    def _check_thresholds(self, metric: PerformanceMetrics):
        """Check if metric exceeds thresholds and generate alerts."""
        alerts = []
        
        if metric.operation_type == 'IK' and metric.computation_time > self.thresholds['max_ik_time']:
            alerts.append(f"IK computation time exceeded threshold: {metric.computation_time:.3f}s")
        
        if metric.operation_type == 'FK' and metric.computation_time > self.thresholds['max_fk_time']:
            alerts.append(f"FK computation time exceeded threshold: {metric.computation_time:.3f}s")
        
        if metric.position_error and metric.position_error > self.thresholds['max_position_error']:
            alerts.append(f"Position error exceeded threshold: {metric.position_error:.6f}m")
        
        if metric.rotation_error and metric.rotation_error > self.thresholds['max_rotation_error']:
            alerts.append(f"Rotation error exceeded threshold: {metric.rotation_error:.6f}rad")
        
        for alert in alerts:
            logger.warning(alert)
            self.alerts.append({
                'timestamp': metric.timestamp,
                'message': alert,
                'severity': 'warning'
            })
    
    def get_statistics(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get performance statistics for a given time window."""
        if time_window:
            cutoff_time = time.time() - time_window
            metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        else:
            metrics = list(self.metrics_history)
        
        if not metrics:
            return {'error': 'No metrics available'}
        
        # Separate by operation type
        fk_metrics = [m for m in metrics if m.operation_type == 'FK']
        ik_metrics = [m for m in metrics if m.operation_type == 'IK']
        
        stats = {
            'session_duration': time.time() - self.session_start,
            'total_operations': len(metrics),
            'fk_operations': len(fk_metrics),
            'ik_operations': len(ik_metrics),
        }
        
        # FK statistics
        if fk_metrics:
            fk_times = [m.computation_time for m in fk_metrics]
            stats['fk_stats'] = {
                'count': len(fk_metrics),
                'avg_time': np.mean(fk_times),
                'max_time': np.max(fk_times),
                'min_time': np.min(fk_times)
            }
        
        # IK statistics
        if ik_metrics:
            ik_times = [m.computation_time for m in ik_metrics]
            ik_successes = [m.success for m in ik_metrics]
            
            stats['ik_stats'] = {
                'count': len(ik_metrics),
                'success_rate': np.mean(ik_successes),
                'avg_time': np.mean(ik_times),
                'max_time': np.max(ik_times),
                'min_time': np.min(ik_times)
            }
            
            # Error statistics for successful IK operations
            successful_ik = [m for m in ik_metrics if m.success and m.position_error is not None]
            if successful_ik:
                pos_errors = [m.position_error for m in successful_ik]
                rot_errors = [m.rotation_error for m in successful_ik if m.rotation_error is not None]
                
                stats['error_stats'] = {
                    'position_error_mean': np.mean(pos_errors),
                    'position_error_max': np.max(pos_errors),
                    'position_error_std': np.std(pos_errors)
                }
                
                if rot_errors:
                    stats['error_stats'].update({
                        'rotation_error_mean': np.mean(rot_errors),
                        'rotation_error_max': np.max(rot_errors),
                        'rotation_error_std': np.std(rot_errors)
                    })
        
        # Recent alerts
        recent_alerts = [a for a in self.alerts if a['timestamp'] >= (time.time() - 3600)]  # Last hour
        stats['recent_alerts'] = len(recent_alerts)
        
        return stats
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate a performance report."""
        stats = self.get_statistics()
        
        report_lines = [
            "=== Robot Kinematics Performance Report ===",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Session Duration: {stats['session_duration']:.1f} seconds",
            f"Total Operations: {stats['total_operations']}",
            ""
        ]
        
        if 'fk_stats' in stats:
            fk = stats['fk_stats']
            report_lines.extend([
                "Forward Kinematics:",
                f"  Operations: {fk['count']}",
                f"  Average Time: {fk['avg_time']:.6f}s",
                f"  Max Time: {fk['max_time']:.6f}s",
                ""
            ])
        
        if 'ik_stats' in stats:
            ik = stats['ik_stats']
            report_lines.extend([
                "Inverse Kinematics:",
                f"  Operations: {ik['count']}",
                f"  Success Rate: {ik['success_rate']:.3f}",
                f"  Average Time: {ik['avg_time']:.6f}s",
                f"  Max Time: {ik['max_time']:.6f}s",
                ""
            ])
        
        if 'error_stats' in stats:
            err = stats['error_stats']
            report_lines.extend([
                "Error Statistics:",
                f"  Position Error (mean): {err['position_error_mean']:.6f}m",
                f"  Position Error (max): {err['position_error_max']:.6f}m",
                f"  Position Error (std): {err['position_error_std']:.6f}m",
            ])
            
            if 'rotation_error_mean' in err:
                report_lines.extend([
                    f"  Rotation Error (mean): {err['rotation_error_mean']:.6f}rad",
                    f"  Rotation Error (max): {err['rotation_error_max']:.6f}rad",
                    f"  Rotation Error (std): {err['rotation_error_std']:.6f}rad",
                ])
            report_lines.append("")
        
        report_lines.extend([
            f"Recent Alerts (last hour): {stats['recent_alerts']}",
            ""
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Performance report saved to {output_path}")
        
        return report

class ValidationSuite:
    """Comprehensive validation suite for robot kinematics."""
    
    def __init__(self, robot_kinematics, config):
        self.robot = robot_kinematics
        self.config = config
        self.validation_results = []
    
    def validate_forward_kinematics(self, test_configurations: List[np.ndarray]) -> Dict[str, Any]:
        """Validate forward kinematics with known configurations."""
        logger.info("Running forward kinematics validation...")
        
        results = {
            'test_count': len(test_configurations),
            'success_count': 0,
            'failures': [],
            'computation_times': [],
            'poses': []
        }
        
        for i, q in enumerate(test_configurations):
            try:
                start_time = time.time()
                T = self.robot.forward_kinematics(q)
                computation_time = time.time() - start_time
                
                results['computation_times'].append(computation_time)
                results['poses'].append(T)
                results['success_count'] += 1
                
                # Check for reasonable pose values
                pos = T[:3, 3]
                if np.any(np.abs(pos) > 2.0):  # 2m workspace limit
                    results['failures'].append(f"Test {i}: Position out of reasonable range: {pos}")
                
            except Exception as e:
                results['failures'].append(f"Test {i}: {str(e)}")
        
        results['success_rate'] = results['success_count'] / results['test_count']
        results['avg_computation_time'] = np.mean(results['computation_times']) if results['computation_times'] else 0
        
        return results
    
    def validate_inverse_kinematics(self, target_poses: List[np.ndarray]) -> Dict[str, Any]:
        """Validate inverse kinematics with target poses."""
        logger.info("Running inverse kinematics validation...")
        
        results = {
            'test_count': len(target_poses),
            'success_count': 0,
            'convergence_failures': 0,
            'accuracy_failures': 0,
            'computation_times': [],
            'position_errors': [],
            'rotation_errors': [],
            'solutions': []
        }
        
        ik_params = self.config.get_ik_params()
        
        for i, T_target in enumerate(target_poses):
            try:
                start_time = time.time()
                q_sol, converged = self.robot.inverse_kinematics(T_target, **ik_params)
                computation_time = time.time() - start_time
                
                results['computation_times'].append(computation_time)
                
                if not converged:
                    results['convergence_failures'] += 1
                    continue
                
                # Check accuracy
                pos_err, rot_err = self.robot.check_pose_error(T_target, q_sol)
                results['position_errors'].append(pos_err)
                results['rotation_errors'].append(rot_err)
                results['solutions'].append(q_sol)
                
                # Check if within tolerance
                validation_config = self.config.get_validation_config()
                pos_tol = validation_config.get('position_tolerance', 0.005)
                rot_tol = validation_config.get('rotation_tolerance', 0.01)
                
                if pos_err <= pos_tol and rot_err <= rot_tol:
                    results['success_count'] += 1
                else:
                    results['accuracy_failures'] += 1
                
            except Exception as e:
                logger.error(f"IK validation test {i} failed: {e}")
        
        results['success_rate'] = results['success_count'] / results['test_count']
        results['convergence_rate'] = (results['test_count'] - results['convergence_failures']) / results['test_count']
        
        if results['position_errors']:
            results['position_error_stats'] = {
                'mean': np.mean(results['position_errors']),
                'max': np.max(results['position_errors']),
                'std': np.std(results['position_errors'])
            }
        
        if results['rotation_errors']:
            results['rotation_error_stats'] = {
                'mean': np.mean(results['rotation_errors']),
                'max': np.max(results['rotation_errors']),
                'std': np.std(results['rotation_errors'])
            }
        
        return results
    
    def validate_consistency(self, num_tests: int = 100) -> Dict[str, Any]:
        """Validate FK-IK consistency (round-trip test)."""
        logger.info(f"Running FK-IK consistency validation with {num_tests} tests...")
        
        results = {
            'test_count': num_tests,
            'success_count': 0,
            'joint_errors': [],
            'position_errors': [],
            'rotation_errors': []
        }
        
        # Generate random joint configurations within limits
        limits_lower, limits_upper = self.robot.joint_limits[0], self.robot.joint_limits[1]
        
        for i in range(num_tests):
            try:
                # Generate random joint configuration
                q_original = np.random.uniform(limits_lower, limits_upper)
                
                # Forward kinematics
                T_target = self.robot.forward_kinematics(q_original)
                
                # Inverse kinematics
                ik_params = self.config.get_ik_params()
                q_recovered, converged = self.robot.inverse_kinematics(T_target, **ik_params)
                
                if not converged:
                    continue
                
                # Check joint space error
                joint_error = np.linalg.norm(q_original - q_recovered)
                results['joint_errors'].append(joint_error)
                
                # Check Cartesian space error
                pos_err, rot_err = self.robot.check_pose_error(T_target, q_recovered)
                results['position_errors'].append(pos_err)
                results['rotation_errors'].append(rot_err)
                
                # Consider successful if within reasonable tolerances
                if joint_error < 0.1 and pos_err < 1e-3 and rot_err < 1e-3:
                    results['success_count'] += 1
                
            except Exception as e:
                logger.warning(f"Consistency test {i} failed: {e}")
        
        results['success_rate'] = results['success_count'] / results['test_count']
        
        if results['joint_errors']:
            results['joint_error_stats'] = {
                'mean': np.mean(results['joint_errors']),
                'max': np.max(results['joint_errors']),
                'std': np.std(results['joint_errors'])
            }
        
        return results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        logger.info("Starting full validation suite...")
        
        # Generate test configurations
        limits_lower, limits_upper = self.robot.joint_limits[0], self.robot.joint_limits[1]
        test_configs = [np.random.uniform(limits_lower, limits_upper) for _ in range(20)]
        
        # Generate target poses from test configurations
        target_poses = [self.robot.forward_kinematics(q) for q in test_configs[:10]]
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'forward_kinematics': self.validate_forward_kinematics(test_configs),
            'inverse_kinematics': self.validate_inverse_kinematics(target_poses),
            'consistency': self.validate_consistency(50)
        }
        
        # Overall assessment
        fk_success = validation_results['forward_kinematics']['success_rate']
        ik_success = validation_results['inverse_kinematics']['success_rate']
        consistency_success = validation_results['consistency']['success_rate']
        
        validation_results['overall_assessment'] = {
            'fk_pass': fk_success > 0.95,
            'ik_pass': ik_success > 0.90,
            'consistency_pass': consistency_success > 0.90,
            'overall_pass': all([fk_success > 0.95, ik_success > 0.90, consistency_success > 0.90])
        }
        
        self.validation_results.append(validation_results)
        return validation_results
    
    def save_validation_report(self, results: Dict[str, Any], output_path: str):
        """Save validation results to file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Validation report saved to {output_path}")

def create_performance_dashboard(monitor: PerformanceMonitor, output_path: str = "performance_dashboard.png"):
    """Create a visual performance dashboard."""
    if not monitor.metrics_history:
        logger.warning("No metrics available for dashboard")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Robot Kinematics Performance Dashboard')
    
    # Extract data
    timestamps = [m.timestamp for m in monitor.metrics_history]
    fk_times = [m.computation_time for m in monitor.metrics_history if m.operation_type == 'FK']
    ik_times = [m.computation_time for m in monitor.metrics_history if m.operation_type == 'IK']
    ik_successes = [m.success for m in monitor.metrics_history if m.operation_type == 'IK']
    
    # Computation times
    if fk_times:
        axes[0, 0].hist(fk_times, bins=20, alpha=0.7, label='FK')
    if ik_times:
        axes[0, 0].hist(ik_times, bins=20, alpha=0.7, label='IK')
    axes[0, 0].set_xlabel('Computation Time (s)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Computation Time Distribution')
    axes[0, 0].legend()
    
    # IK success rate over time
    if ik_successes:
        success_rate = np.cumsum(ik_successes) / np.arange(1, len(ik_successes) + 1)
        axes[0, 1].plot(success_rate)
        axes[0, 1].set_xlabel('IK Operation Number')
        axes[0, 1].set_ylabel('Cumulative Success Rate')
        axes[0, 1].set_title('IK Success Rate Over Time')
        axes[0, 1].set_ylim([0, 1])
    
    # Position errors
    pos_errors = [m.position_error for m in monitor.metrics_history 
                  if m.position_error is not None and m.success]
    if pos_errors:
        axes[1, 0].hist(pos_errors, bins=20)
        axes[1, 0].set_xlabel('Position Error (m)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Position Error Distribution')
    
    # Rotation errors
    rot_errors = [m.rotation_error for m in monitor.metrics_history 
                  if m.rotation_error is not None and m.success]
    if rot_errors:
        axes[1, 1].hist(rot_errors, bins=20)
        axes[1, 1].set_xlabel('Rotation Error (rad)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Rotation Error Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Performance dashboard saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    monitor = PerformanceMonitor()
    
    # Simulate some metrics
    for i in range(100):
        metric = PerformanceMetrics(
            timestamp=time.time(),
            operation_type='IK' if i % 2 == 0 else 'FK',
            computation_time=np.random.exponential(0.1),
            success=np.random.random() > 0.05,
            position_error=np.random.exponential(1e-4) if np.random.random() > 0.1 else None,
            rotation_error=np.random.exponential(1e-4) if np.random.random() > 0.1 else None
        )
        monitor.record_metric(metric)
    
    # Generate report
    report = monitor.generate_report()
    print(report)


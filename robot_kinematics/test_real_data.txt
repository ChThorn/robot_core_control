#!/usr/bin/env python3
"""
Comprehensive test script for validating robot kinematics against real JSON data.
Tests forward kinematics, inverse kinematics, and round-trip consistency.
"""

import numpy as np
import json
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from robot_controller import RobotController

class RealDataValidator:
    """Comprehensive validator for robot kinematics using real data."""
    
    def __init__(self, urdf_path: str, json_path: str):
        """Initialize validator with robot model and real data."""
        self.controller = RobotController(urdf_path)
        self.json_path = json_path
        self.results = {}
        
        # Load real data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data['waypoints'])} waypoints from {json_path}")
    
    def validate_forward_kinematics(self, num_samples: int = None) -> Dict[str, Any]:
        """Validate forward kinematics against real robot data."""
        print("\n" + "="*60)
        print("FORWARD KINEMATICS VALIDATION")
        print("="*60)
        
        waypoints = self.data['waypoints']
        if num_samples:
            indices = np.linspace(0, len(waypoints)-1, num_samples, dtype=int)
            waypoints = [waypoints[i] for i in indices]
        
        position_errors = []
        rotation_errors = []
        computation_times = []
        
        print(f"Testing {len(waypoints)} waypoints...")
        
        for i, wp in enumerate(waypoints):
            try:
                # Convert robot data to standard units
                q_deg = np.array(wp['joint_positions'])
                tcp_recorded = np.array(wp['tcp_position'])
                
                q_rad, T_recorded = self.controller.convert_from_robot_units(q_deg, tcp_recorded)
                
                # Compute forward kinematics
                start_time = time.time()
                T_fk = self.controller.forward_kinematics(q_rad)
                computation_time = time.time() - start_time
                computation_times.append(computation_time)
                
                # Calculate errors
                pos_err = np.linalg.norm(T_fk[:3, 3] - T_recorded[:3, 3])
                
                # Rotation error using rotation matrix comparison
                R_fk = T_fk[:3, :3]
                R_recorded = T_recorded[:3, :3]
                R_err = R_fk.T @ R_recorded
                cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
                rot_err = np.arccos(cos_angle)
                
                position_errors.append(pos_err)
                rotation_errors.append(rot_err)
                
                if i < 5:  # Show first 5 in detail
                    print(f"Waypoint {i+1}:")
                    print(f"  Joint angles (deg): {np.round(q_deg, 2)}")
                    print(f"  Position error: {pos_err*1000:.3f} mm")
                    print(f"  Rotation error: {np.rad2deg(rot_err):.3f}°")
                    print(f"  Computation time: {computation_time*1000:.2f} ms")
                
            except Exception as e:
                print(f"Error processing waypoint {i}: {e}")
        
        # Calculate statistics
        results = {
            'num_waypoints': len(waypoints),
            'position_errors_m': position_errors,
            'rotation_errors_rad': rotation_errors,
            'computation_times_s': computation_times,
            'mean_position_error_m': np.mean(position_errors),
            'max_position_error_m': np.max(position_errors),
            'std_position_error_m': np.std(position_errors),
            'mean_rotation_error_rad': np.mean(rotation_errors),
            'max_rotation_error_rad': np.max(rotation_errors),
            'std_rotation_error_rad': np.std(rotation_errors),
            'mean_computation_time_s': np.mean(computation_times),
            'max_computation_time_s': np.max(computation_times)
        }
        
        # Print summary
        print(f"\nFORWARD KINEMATICS RESULTS:")
        print(f"Position Accuracy:")
        print(f"  Mean error: {results['mean_position_error_m']*1000:.3f} mm")
        print(f"  Max error:  {results['max_position_error_m']*1000:.3f} mm")
        print(f"  Std dev:    {results['std_position_error_m']*1000:.3f} mm")
        print(f"Rotation Accuracy:")
        print(f"  Mean error: {np.rad2deg(results['mean_rotation_error_rad']):.3f}°")
        print(f"  Max error:  {np.rad2deg(results['max_rotation_error_rad']):.3f}°")
        print(f"  Std dev:    {np.rad2deg(results['std_rotation_error_rad']):.3f}°")
        print(f"Performance:")
        print(f"  Mean time:  {results['mean_computation_time_s']*1000:.2f} ms")
        print(f"  Max time:   {results['max_computation_time_s']*1000:.2f} ms")
        
        self.results['forward_kinematics'] = results
        return results
    
    def validate_inverse_kinematics(self, num_samples: int = 10) -> Dict[str, Any]:
        """Validate inverse kinematics using forward kinematics poses."""
        print("\n" + "="*60)
        print("INVERSE KINEMATICS VALIDATION")
        print("="*60)
        
        waypoints = self.data['waypoints']
        indices = np.linspace(0, len(waypoints)-1, num_samples, dtype=int)
        
        ik_successes = 0
        position_errors = []
        rotation_errors = []
        joint_errors = []
        computation_times = []
        
        print(f"Testing {num_samples} waypoints for IK validation...")
        
        for i, idx in enumerate(indices):
            wp = waypoints[idx]
            
            try:
                # Get original joint configuration
                q_deg_original = np.array(wp['joint_positions'])
                q_rad_original = np.deg2rad(q_deg_original)
                
                # Compute target pose using FK
                T_target = self.controller.forward_kinematics(q_rad_original)
                
                # Solve IK
                start_time = time.time()
                q_ik_solution, converged = self.controller.inverse_kinematics(T_target)
                computation_time = time.time() - start_time
                computation_times.append(computation_time)
                
                if converged:
                    ik_successes += 1
                    
                    # Check pose accuracy
                    T_check = self.controller.forward_kinematics(q_ik_solution)
                    pos_err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
                    
                    R_check = T_check[:3, :3]
                    R_target = T_target[:3, :3]
                    R_err = R_check.T @ R_target
                    cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
                    rot_err = np.arccos(cos_angle)
                    
                    # Check joint difference (accounting for multiple solutions)
                    joint_err = np.min([
                        np.linalg.norm(q_ik_solution - q_rad_original),
                        np.linalg.norm(q_ik_solution - q_rad_original + 2*np.pi),
                        np.linalg.norm(q_ik_solution - q_rad_original - 2*np.pi)
                    ])
                    
                    position_errors.append(pos_err)
                    rotation_errors.append(rot_err)
                    joint_errors.append(joint_err)
                    
                    if i < 5:  # Show first 5 in detail
                        print(f"Waypoint {idx} (Test {i+1}):")
                        print(f"  Original joints (deg): {np.round(q_deg_original, 2)}")
                        print(f"  IK solution (deg):     {np.round(np.rad2deg(q_ik_solution), 2)}")
                        print(f"  Joint difference:      {np.round(np.rad2deg(joint_err), 3)}°")
                        print(f"  Position error:        {pos_err*1e6:.1f} μm")
                        print(f"  Rotation error:        {np.rad2deg(rot_err)*3600:.1f} arcsec")
                        print(f"  Computation time:      {computation_time*1000:.1f} ms")
                else:
                    print(f"Waypoint {idx}: IK failed to converge")
                    
            except Exception as e:
                print(f"Error processing waypoint {idx}: {e}")
        
        # Calculate statistics
        success_rate = ik_successes / num_samples
        
        results = {
            'num_tests': num_samples,
            'ik_successes': ik_successes,
            'success_rate': success_rate,
            'position_errors_m': position_errors,
            'rotation_errors_rad': rotation_errors,
            'joint_errors_rad': joint_errors,
            'computation_times_s': computation_times
        }
        
        if position_errors:
            results.update({
                'mean_position_error_m': np.mean(position_errors),
                'max_position_error_m': np.max(position_errors),
                'mean_rotation_error_rad': np.mean(rotation_errors),
                'max_rotation_error_rad': np.max(rotation_errors),
                'mean_joint_error_rad': np.mean(joint_errors),
                'max_joint_error_rad': np.max(joint_errors),
                'mean_computation_time_s': np.mean(computation_times),
                'max_computation_time_s': np.max(computation_times)
            })
        
        # Print summary
        print(f"\nINVERSE KINEMATICS RESULTS:")
        print(f"Success Rate: {success_rate:.1%} ({ik_successes}/{num_samples})")
        
        if position_errors:
            print(f"Pose Accuracy (FK-IK round-trip):")
            print(f"  Mean position error: {results['mean_position_error_m']*1e6:.1f} μm")
            print(f"  Max position error:  {results['max_position_error_m']*1e6:.1f} μm")
            print(f"  Mean rotation error: {np.rad2deg(results['mean_rotation_error_rad'])*3600:.1f} arcsec")
            print(f"  Max rotation error:  {np.rad2deg(results['max_rotation_error_rad'])*3600:.1f} arcsec")
            print(f"Joint Accuracy:")
            print(f"  Mean joint error: {np.rad2deg(results['mean_joint_error_rad']):.3f}°")
            print(f"  Max joint error:  {np.rad2deg(results['max_joint_error_rad']):.3f}°")
            print(f"Performance:")
            print(f"  Mean time: {results['mean_computation_time_s']*1000:.1f} ms")
            print(f"  Max time:  {results['max_computation_time_s']*1000:.1f} ms")
        
        self.results['inverse_kinematics'] = results
        return results
    
    def validate_consistency(self, num_samples: int = 20) -> Dict[str, Any]:
        """Test FK-IK consistency across the workspace."""
        print("\n" + "="*60)
        print("FK-IK CONSISTENCY VALIDATION")
        print("="*60)
        
        # Generate test configurations across joint space
        joint_limits = self.controller.robot.joint_limits
        
        consistency_errors = []
        test_configs = []
        
        print(f"Testing {num_samples} random configurations...")
        
        for i in range(num_samples):
            # Generate random joint configuration
            q_test = np.random.uniform(joint_limits[0], joint_limits[1])
            test_configs.append(q_test)
            
            try:
                # FK: q -> T
                T_fk = self.controller.forward_kinematics(q_test)
                
                # IK: T -> q'
                q_ik, converged = self.controller.inverse_kinematics(T_fk, q_init=q_test)
                
                if converged:
                    # FK again: q' -> T'
                    T_check = self.controller.forward_kinematics(q_ik)
                    
                    # Calculate consistency error
                    pos_err = np.linalg.norm(T_check[:3, 3] - T_fk[:3, 3])
                    R_err = T_check[:3, :3].T @ T_fk[:3, :3]
                    cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
                    rot_err = np.arccos(cos_angle)
                    
                    total_err = pos_err + rot_err
                    consistency_errors.append(total_err)
                    
                    if i < 3:  # Show first 3 in detail
                        print(f"Test {i+1}:")
                        print(f"  Original q (deg): {np.round(np.rad2deg(q_test), 2)}")
                        print(f"  IK solution (deg): {np.round(np.rad2deg(q_ik), 2)}")
                        print(f"  Position consistency: {pos_err*1e6:.1f} μm")
                        print(f"  Rotation consistency: {np.rad2deg(rot_err)*3600:.1f} arcsec")
                else:
                    print(f"Test {i+1}: IK failed to converge")
                    
            except Exception as e:
                print(f"Error in test {i+1}: {e}")
        
        # Calculate statistics
        results = {
            'num_tests': num_samples,
            'num_successful': len(consistency_errors),
            'consistency_errors': consistency_errors,
            'success_rate': len(consistency_errors) / num_samples
        }
        
        if consistency_errors:
            results.update({
                'mean_consistency_error': np.mean(consistency_errors),
                'max_consistency_error': np.max(consistency_errors),
                'std_consistency_error': np.std(consistency_errors)
            })
        
        print(f"\nCONSISTENCY RESULTS:")
        print(f"Success Rate: {results['success_rate']:.1%}")
        if consistency_errors:
            print(f"Mean consistency error: {results['mean_consistency_error']:.2e}")
            print(f"Max consistency error:  {results['max_consistency_error']:.2e}")
        
        self.results['consistency'] = results
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE VALIDATION REPORT")
        print("="*60)
        
        report = []
        report.append("ROBOT KINEMATICS VALIDATION REPORT")
        report.append("=" * 50)
        report.append(f"Data source: {self.json_path}")
        report.append(f"Robot model: {self.controller.robot.urdf_path}")
        report.append(f"Test date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Forward Kinematics Summary
        if 'forward_kinematics' in self.results:
            fk = self.results['forward_kinematics']
            report.append("FORWARD KINEMATICS VALIDATION:")
            report.append(f"  Waypoints tested: {fk['num_waypoints']}")
            report.append(f"  Mean position error: {fk['mean_position_error_m']*1000:.3f} mm")
            report.append(f"  Mean rotation error: {np.rad2deg(fk['mean_rotation_error_rad']):.3f}°")
            report.append(f"  Mean computation time: {fk['mean_computation_time_s']*1000:.2f} ms")
            
            # Assessment
            if fk['mean_position_error_m'] < 0.005:  # 5mm
                report.append("  ✅ Position accuracy: EXCELLENT")
            elif fk['mean_position_error_m'] < 0.010:  # 10mm
                report.append("  ✅ Position accuracy: GOOD")
            else:
                report.append("  ⚠️  Position accuracy: NEEDS IMPROVEMENT")
                
            if fk['mean_rotation_error_rad'] < 0.1:  # ~5.7°
                report.append("  ✅ Rotation accuracy: EXCELLENT")
            elif fk['mean_rotation_error_rad'] < 0.2:  # ~11.4°
                report.append("  ✅ Rotation accuracy: GOOD")
            else:
                report.append("  ⚠️  Rotation accuracy: NEEDS IMPROVEMENT")
            report.append("")
        
        # Inverse Kinematics Summary
        if 'inverse_kinematics' in self.results:
            ik = self.results['inverse_kinematics']
            report.append("INVERSE KINEMATICS VALIDATION:")
            report.append(f"  Tests performed: {ik['num_tests']}")
            report.append(f"  Success rate: {ik['success_rate']:.1%}")
            
            if 'mean_position_error_m' in ik:
                report.append(f"  Mean position error: {ik['mean_position_error_m']*1e6:.1f} μm")
                report.append(f"  Mean rotation error: {np.rad2deg(ik['mean_rotation_error_rad'])*3600:.1f} arcsec")
                report.append(f"  Mean computation time: {ik['mean_computation_time_s']*1000:.1f} ms")
            
            # Assessment
            if ik['success_rate'] >= 0.95:
                report.append("  ✅ Reliability: EXCELLENT")
            elif ik['success_rate'] >= 0.90:
                report.append("  ✅ Reliability: GOOD")
            else:
                report.append("  ⚠️  Reliability: NEEDS IMPROVEMENT")
            report.append("")
        
        # Overall Assessment
        report.append("OVERALL ASSESSMENT:")
        
        # Check if system is production ready
        production_ready = True
        issues = []
        
        if 'forward_kinematics' in self.results:
            fk = self.results['forward_kinematics']
            if fk['mean_position_error_m'] > 0.010:  # 10mm
                production_ready = False
                issues.append("Position accuracy too low")
            if fk['mean_rotation_error_rad'] > 0.2:  # ~11.4°
                production_ready = False
                issues.append("Rotation accuracy too low")
        
        if 'inverse_kinematics' in self.results:
            ik = self.results['inverse_kinematics']
            if ik['success_rate'] < 0.90:
                production_ready = False
                issues.append("IK success rate too low")
        
        if production_ready:
            report.append("  ✅ SYSTEM IS PRODUCTION READY")
            report.append("  All validation tests passed successfully.")
        else:
            report.append("  ⚠️  SYSTEM NEEDS IMPROVEMENT")
            for issue in issues:
                report.append(f"  - {issue}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        return report_text
    
    def save_report(self, filename: str = "validation_report.txt"):
        """Save validation report to file."""
        report = self.generate_report()
        with open(filename, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {filename}")

def main():
    """Main test function."""
    print("ROBOT KINEMATICS REAL DATA VALIDATION")
    print("=" * 60)
    
    # Initialize validator
    validator = RealDataValidator("rb3_730es_u.urdf", "third_20250710_162459.json")
    
    # Run validation tests
    print("Starting comprehensive validation...")
    
    # Test 1: Forward Kinematics
    validator.validate_forward_kinematics(num_samples=20)
    
    # Test 2: Inverse Kinematics
    validator.validate_inverse_kinematics(num_samples=10)
    
    # Test 3: Consistency
    validator.validate_consistency(num_samples=15)
    
    # Generate and save report
    validator.save_report("kinematics_validation_report.txt")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print("Check 'kinematics_validation_report.txt' for detailed results.")

if __name__ == "__main__":
    main()


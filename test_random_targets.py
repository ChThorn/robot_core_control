#!/usr/bin/env python3
"""
Test Numerical IK Performance with Random TCP Targets
Comprehensive evaluation of consistency and robustness
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from main_kinematics import RB3Kinematics
import time

class RandomTargetTester:
    """Test numerical IK with random targets across the workspace"""
    
    def __init__(self):
        self.robot = RB3Kinematics()
        
    def generate_random_reachable_targets(self, num_targets: int = 100) -> List[np.ndarray]:
        """
        Generate random reachable TCP targets by sampling joint space
        
        Args:
            num_targets: Number of random targets to generate
            
        Returns:
            List of reachable TCP positions
        """
        targets = []
        
        # Extended joint limits based on recorded data
        joint_limits = [(-10.0, 10.0)] * 6
        
        np.random.seed(42)  # For reproducibility
        
        print(f"Generating {num_targets} random reachable targets...")
        
        for i in range(num_targets):
            # Generate random joint configuration
            random_joints = [np.random.uniform(low, high) for low, high in joint_limits]
            
            try:
                # Get TCP position for this joint configuration
                tcp_position = self.robot.get_tcp_position(random_joints)
                
                # Check if position is reasonable (within expected workspace)
                if (abs(tcp_position[0]) < 1.5 and 
                    abs(tcp_position[1]) < 1.5 and 
                    0.1 < tcp_position[2] < 1.5):
                    targets.append(tcp_position)
                    
                    if len(targets) % 20 == 0:
                        print(f"  Generated {len(targets)} targets...")
                        
            except:
                continue
                
            if len(targets) >= num_targets:
                break
        
        print(f"Successfully generated {len(targets)} reachable targets")
        return targets
    
    def test_ik_performance_on_random_targets(self, targets: List[np.ndarray]) -> Dict:
        """
        Test IK performance on random targets
        
        Args:
            targets: List of target TCP positions
            
        Returns:
            Dictionary with performance results
        """
        print(f"\nTesting numerical IK on {len(targets)} random targets...")
        print("=" * 60)
        
        results = {
            'total_targets': len(targets),
            'successful_solutions': 0,
            'failed_solutions': 0,
            'position_errors': [],
            'computation_times': [],
            'num_solutions_found': [],
            'target_positions': targets
        }
        
        print("Target | Position (m)              | Solutions | Time (ms) | Error (m)  | Status")
        print("-" * 75)
        
        for i, target_pos in enumerate(targets):
            try:
                # Measure computation time
                start_time = time.time()
                
                # Solve inverse kinematics (position only for better success rate)
                solutions = self.robot.inverse_kinematics_position_only(target_pos)
                
                end_time = time.time()
                computation_time = (end_time - start_time) * 1000  # Convert to ms
                
                if solutions:
                    # Use best solution
                    best_solution = solutions[0]
                    
                    # Verify solution
                    calculated_pos = self.robot.get_tcp_position(best_solution)
                    position_error = np.linalg.norm(calculated_pos - target_pos)
                    
                    # Record results
                    results['successful_solutions'] += 1
                    results['position_errors'].append(position_error)
                    results['computation_times'].append(computation_time)
                    results['num_solutions_found'].append(len(solutions))
                    
                    status = "✓ SUCCESS"
                    if position_error > 0.01:  # > 1cm
                        status = "⚠ HIGH_ERROR"
                    
                    print(f"{i:6d} | [{target_pos[0]:5.2f}, {target_pos[1]:5.2f}, {target_pos[2]:5.2f}] | "
                          f"{len(solutions):8d} | {computation_time:7.1f} | {position_error:8.5f} | {status}")
                    
                else:
                    results['failed_solutions'] += 1
                    results['computation_times'].append(computation_time)
                    
                    print(f"{i:6d} | [{target_pos[0]:5.2f}, {target_pos[1]:5.2f}, {target_pos[2]:5.2f}] | "
                          f"       0 | {computation_time:7.1f} |      N/A | ✗ FAILED")
                
                # Show progress for long tests
                if (i + 1) % 25 == 0:
                    success_rate = results['successful_solutions'] / (i + 1)
                    print(f"  Progress: {i+1}/{len(targets)} ({success_rate:.1%} success rate so far)")
                    
            except Exception as e:
                results['failed_solutions'] += 1
                print(f"{i:6d} | [{target_pos[0]:5.2f}, {target_pos[1]:5.2f}, {target_pos[2]:5.2f}] | "
                      f"       0 |     N/A |      N/A | ✗ ERROR: {str(e)[:20]}")
        
        # Calculate statistics
        if results['position_errors']:
            results['mean_error'] = np.mean(results['position_errors'])
            results['max_error'] = np.max(results['position_errors'])
            results['min_error'] = np.min(results['position_errors'])
            results['std_error'] = np.std(results['position_errors'])
            results['median_error'] = np.median(results['position_errors'])
            
            # Error distribution
            errors = np.array(results['position_errors'])
            results['sub_mm_count'] = np.sum(errors < 0.001)  # < 1mm
            results['sub_cm_count'] = np.sum(errors < 0.01)   # < 1cm
            results['high_error_count'] = np.sum(errors > 0.05)  # > 5cm
        
        if results['computation_times']:
            results['mean_time'] = np.mean(results['computation_times'])
            results['max_time'] = np.max(results['computation_times'])
            results['min_time'] = np.min(results['computation_times'])
        
        results['success_rate'] = results['successful_solutions'] / results['total_targets']
        
        return results
    
    def analyze_workspace_coverage(self, targets: List[np.ndarray], results: Dict):
        """
        Analyze how performance varies across the workspace
        
        Args:
            targets: List of target positions
            results: Performance results
        """
        print(f"\n" + "="*60)
        print("WORKSPACE COVERAGE ANALYSIS")
        print("="*60)
        
        if not results['position_errors']:
            print("No successful solutions to analyze")
            return
        
        targets = np.array(targets[:len(results['position_errors'])])
        errors = np.array(results['position_errors'])
        
        # Analyze by workspace regions
        print("\nPerformance by workspace region:")
        
        # X-axis regions
        x_coords = targets[:, 0]
        for region, (x_min, x_max) in [("Left", (-1.0, -0.1)), ("Center", (-0.1, 0.1)), ("Right", (0.1, 1.0))]:
            mask = (x_coords >= x_min) & (x_coords <= x_max)
            if np.any(mask):
                region_errors = errors[mask]
                print(f"  {region:6s} X: {np.mean(region_errors):.4f}m avg, {len(region_errors):2d} targets")
        
        # Z-axis regions  
        z_coords = targets[:, 2]
        for region, (z_min, z_max) in [("Low", (0.1, 0.6)), ("Mid", (0.6, 0.9)), ("High", (0.9, 1.5))]:
            mask = (z_coords >= z_min) & (z_coords <= z_max)
            if np.any(mask):
                region_errors = errors[mask]
                print(f"  {region:6s} Z: {np.mean(region_errors):.4f}m avg, {len(region_errors):2d} targets")
        
        # Distance from origin
        distances = np.linalg.norm(targets, axis=1)
        for region, (d_min, d_max) in [("Near", (0.0, 0.5)), ("Mid", (0.5, 1.0)), ("Far", (1.0, 2.0))]:
            mask = (distances >= d_min) & (distances <= d_max)
            if np.any(mask):
                region_errors = errors[mask]
                print(f"  {region:6s} Dist: {np.mean(region_errors):.4f}m avg, {len(region_errors):2d} targets")
    
    def print_performance_summary(self, results: Dict):
        """Print comprehensive performance summary"""
        
        print(f"\n" + "="*60)
        print("NUMERICAL IK PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"\n📊 Overall Results:")
        print(f"  Total targets tested: {results['total_targets']}")
        print(f"  Successful solutions: {results['successful_solutions']}")
        print(f"  Failed solutions: {results['failed_solutions']}")
        print(f"  Success rate: {results['success_rate']:.1%}")
        
        if results['position_errors']:
            print(f"\n🎯 Position Accuracy:")
            print(f"  Mean error: {results['mean_error']:.5f} m ({results['mean_error']*1000:.2f} mm)")
            print(f"  Median error: {results['median_error']:.5f} m ({results['median_error']*1000:.2f} mm)")
            print(f"  Min error: {results['min_error']:.5f} m ({results['min_error']*1000:.2f} mm)")
            print(f"  Max error: {results['max_error']:.5f} m ({results['max_error']*1000:.2f} mm)")
            print(f"  Std deviation: {results['std_error']:.5f} m")
            
            print(f"\n📈 Error Distribution:")
            print(f"  Sub-millimeter (< 1mm): {results['sub_mm_count']}/{results['successful_solutions']} ({results['sub_mm_count']/results['successful_solutions']:.1%})")
            print(f"  Sub-centimeter (< 1cm): {results['sub_cm_count']}/{results['successful_solutions']} ({results['sub_cm_count']/results['successful_solutions']:.1%})")
            print(f"  High error (> 5cm): {results['high_error_count']}/{results['successful_solutions']} ({results['high_error_count']/results['successful_solutions']:.1%})")
        
        if results['computation_times']:
            print(f"\n⚡ Computation Performance:")
            print(f"  Mean time: {results['mean_time']:.1f} ms")
            print(f"  Min time: {results['min_time']:.1f} ms")
            print(f"  Max time: {results['max_time']:.1f} ms")
        
        if results['num_solutions_found']:
            avg_solutions = np.mean(results['num_solutions_found'])
            print(f"  Average solutions found: {avg_solutions:.1f}")
        
        # Performance assessment
        print(f"\n🏆 Performance Assessment:")
        if results['success_rate'] > 0.9:
            print("  ✅ EXCELLENT success rate (>90%)")
        elif results['success_rate'] > 0.7:
            print("  ✅ GOOD success rate (>70%)")
        else:
            print("  ⚠️  LOW success rate (<70%)")
        
        if results['position_errors'] and results['mean_error'] < 0.001:
            print("  ✅ EXCELLENT accuracy (sub-millimeter)")
        elif results['position_errors'] and results['mean_error'] < 0.01:
            print("  ✅ GOOD accuracy (sub-centimeter)")
        else:
            print("  ⚠️  MODERATE accuracy (centimeter-level)")

def main():
    """Main testing function"""
    
    print("Random TCP Target Testing for Numerical IK")
    print("=" * 50)
    
    tester = RandomTargetTester()
    
    # Generate random reachable targets
    targets = tester.generate_random_reachable_targets(num_targets=50)  # Start with 50 for reasonable test time
    
    if not targets:
        print("Failed to generate reachable targets")
        return
    
    # Test IK performance
    results = tester.test_ik_performance_on_random_targets(targets)
    
    # Analyze workspace coverage
    tester.analyze_workspace_coverage(targets, results)
    
    # Print summary
    tester.print_performance_summary(results)
    
    # Comparison with recorded data performance
    print(f"\n" + "="*60)
    print("COMPARISON WITH RECORDED DATA")
    print("="*60)
    print("Recorded data performance: 0.0007-0.001m (sub-millimeter)")
    if results['position_errors']:
        print(f"Random targets performance: {results['mean_error']:.4f}m average")
        
        if results['mean_error'] < 0.002:
            print("✅ CONSISTENT: Random targets achieve similar accuracy!")
        elif results['mean_error'] < 0.01:
            print("✅ GOOD: Random targets achieve reasonable accuracy")
        else:
            print("⚠️  VARIABLE: Performance varies significantly from recorded data")

if __name__ == "__main__":
    main()


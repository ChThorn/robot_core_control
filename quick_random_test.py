#!/usr/bin/env python3
"""
Quick test of numerical IK with random targets
"""

import numpy as np
from main_kinematics import RB3Kinematics
import time

def quick_random_test():
    """Quick test with 10 random targets"""
    
    print("Quick Random TCP Target Test")
    print("=" * 40)
    
    robot = RB3Kinematics()
    
    # Generate 10 random reachable targets
    targets = []
    joint_limits = [(-5.0, 5.0)] * 6
    
    np.random.seed(42)
    print("Generating random targets...")
    
    for i in range(100):  # Try up to 100 to get 10 good ones
        random_joints = [np.random.uniform(low, high) for low, high in joint_limits]
        try:
            tcp_position = robot.get_tcp_position(random_joints)
            if (abs(tcp_position[0]) < 1.0 and 
                abs(tcp_position[1]) < 1.0 and 
                0.2 < tcp_position[2] < 1.2):
                targets.append(tcp_position)
                if len(targets) >= 10:
                    break
        except:
            continue
    
    print(f"Generated {len(targets)} targets")
    
    # Test each target
    results = []
    print("\nTesting IK performance:")
    print("Target | Position (m)              | Solutions | Time (ms) | Error (m)  | Status")
    print("-" * 75)
    
    for i, target_pos in enumerate(targets):
        try:
            start_time = time.time()
            solutions = robot.inverse_kinematics_position_only(target_pos)
            end_time = time.time()
            
            computation_time = (end_time - start_time) * 1000
            
            if solutions:
                best_solution = solutions[0]
                calculated_pos = robot.get_tcp_position(best_solution)
                position_error = np.linalg.norm(calculated_pos - target_pos)
                
                results.append({
                    'success': True,
                    'error': position_error,
                    'time': computation_time,
                    'solutions': len(solutions)
                })
                
                status = "✓ SUCCESS"
                if position_error > 0.01:
                    status = "⚠ HIGH_ERROR"
                
                print(f"{i:6d} | [{target_pos[0]:5.2f}, {target_pos[1]:5.2f}, {target_pos[2]:5.2f}] | "
                      f"{len(solutions):8d} | {computation_time:7.1f} | {position_error:8.5f} | {status}")
            else:
                results.append({
                    'success': False,
                    'error': float('inf'),
                    'time': computation_time,
                    'solutions': 0
                })
                print(f"{i:6d} | [{target_pos[0]:5.2f}, {target_pos[1]:5.2f}, {target_pos[2]:5.2f}] | "
                      f"       0 | {computation_time:7.1f} |      N/A | ✗ FAILED")
        except Exception as e:
            print(f"{i:6d} | Error: {e}")
    
    # Calculate statistics
    successful = [r for r in results if r['success']]
    
    print(f"\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    if successful:
        errors = [r['error'] for r in successful]
        times = [r['time'] for r in successful]
        solutions_counts = [r['solutions'] for r in successful]
        
        print(f"Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results):.1%})")
        print(f"Position accuracy:")
        print(f"  Mean error: {np.mean(errors):.5f} m ({np.mean(errors)*1000:.2f} mm)")
        print(f"  Min error:  {np.min(errors):.5f} m ({np.min(errors)*1000:.2f} mm)")
        print(f"  Max error:  {np.max(errors):.5f} m ({np.max(errors)*1000:.2f} mm)")
        print(f"  Std error:  {np.std(errors):.5f} m")
        
        print(f"Computation time:")
        print(f"  Mean time: {np.mean(times):.1f} ms")
        print(f"  Min time:  {np.min(times):.1f} ms")
        print(f"  Max time:  {np.max(times):.1f} ms")
        
        print(f"Solutions found: {np.mean(solutions_counts):.1f} average")
        
        # Error distribution
        sub_mm = sum(1 for e in errors if e < 0.001)
        sub_cm = sum(1 for e in errors if e < 0.01)
        
        print(f"Error distribution:")
        print(f"  Sub-millimeter (< 1mm): {sub_mm}/{len(successful)} ({sub_mm/len(successful):.1%})")
        print(f"  Sub-centimeter (< 1cm): {sub_cm}/{len(successful)} ({sub_cm/len(successful):.1%})")
        
        # Assessment
        print(f"\nPerformance Assessment:")
        if np.mean(errors) < 0.001:
            print("✅ EXCELLENT: Sub-millimeter accuracy maintained!")
        elif np.mean(errors) < 0.01:
            print("✅ GOOD: Sub-centimeter accuracy achieved")
        else:
            print("⚠️  MODERATE: Centimeter-level accuracy")
            
        if len(successful)/len(results) > 0.9:
            print("✅ EXCELLENT: High success rate (>90%)")
        elif len(successful)/len(results) > 0.7:
            print("✅ GOOD: Reasonable success rate (>70%)")
        else:
            print("⚠️  LOW: Success rate needs improvement")
    else:
        print("❌ No successful solutions found")
    
    print(f"\nComparison with recorded data:")
    print("Recorded data: 0.0007-0.001m (sub-millimeter)")
    if successful:
        print(f"Random targets: {np.mean(errors):.4f}m average")
        if np.mean(errors) < 0.002:
            print("✅ CONSISTENT performance across workspace!")
        else:
            print("⚠️  Performance varies from recorded data")

if __name__ == "__main__":
    quick_random_test()


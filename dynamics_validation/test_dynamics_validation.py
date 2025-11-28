"""
Dynamics Validation 통합 테스트

전체 파이프라인을 테스트합니다:
TDL v1 → Dynamics Validation → TDL v2
"""

import sys
import numpy as np
from pathlib import Path

# 경로 설정
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

from robot_dynamics_db import load_robot
from rnea_calculator import RNEACalculator
from feasibility_checker import FeasibilityChecker
from parameter_scaler import ParameterScaler


def test_single_tdl_validation():
    """단일 TDL 검증 테스트"""
    print("=" * 80)
    print(" Test 1: Single TDL Validation")
    print("=" * 80)

    # Robot_B 로드
    robot_b = load_robot(robot_name='Robot_B')

    # Scaler 생성
    scaler = ParameterScaler(robot_b, safety_margin=0.9)

    # TDL v1 예시 (적당한 가속도)
    tdl_v1 = {
        'task': 'move_linear',
        'object': 'part1',
        'robot': 'Robot_B',
        'accel_percent': 40,
        'speed_percent': 50
    }

    print(f"\n[Input TDL v1]")
    print(f"  Task: {tdl_v1['task']}")
    print(f"  Robot: {tdl_v1['robot']}")
    print(f"  Accel: {tdl_v1['accel_percent']}%")

    # 검증 및 스케일링
    result = scaler.scale_tdl_parameters(tdl_v1)

    # 결과 출력
    scaler.print_scaling_report(result)

    return result


def test_infeasible_tdl():
    """Infeasible TDL 테스트 (스케일링 필요)"""
    print("\n\n" + "=" * 80)
    print(" Test 2: Infeasible TDL (Requires Scaling)")
    print("=" * 80)

    # Robot_A 로드 (UR5e)
    robot_a = load_robot(robot_name='Robot_A')

    # Scaler 생성
    scaler = ParameterScaler(robot_a, safety_margin=0.9)

    # TDL v1 예시 (매우 공격적인 가속도)
    tdl_v1 = {
        'task': 'pick',
        'object': 'apple',
        'robot': 'Robot_A',
        'accel_percent': 95,  # 너무 높음!
        'speed_percent': 90
    }

    print(f"\n[Input TDL v1]")
    print(f"  Task: {tdl_v1['task']}")
    print(f"  Robot: {tdl_v1['robot']}")
    print(f"  Accel: {tdl_v1['accel_percent']}% (Very aggressive!)")

    # 검증 및 스케일링
    result = scaler.scale_tdl_parameters(tdl_v1)

    # 결과 출력
    scaler.print_scaling_report(result)

    if result['scaled']:
        print(f"\n✓ TDL was automatically scaled to ensure safety!")
        print(f"  Original Accel: {tdl_v1['accel_percent']}%")
        print(f"  Scale Factor: {result['scale_factor']:.3f}")
        print(f"  Effective Accel: {tdl_v1['accel_percent'] * result['scale_factor']:.1f}%")

    return result


def test_robot_comparison():
    """로봇별 비교 테스트"""
    print("\n\n" + "=" * 80)
    print(" Test 3: Robot Comparison (Same TDL, Different Robots)")
    print("=" * 80)

    # 동일한 TDL v1
    tdl_v1 = {
        'task': 'move',
        'accel_percent': 80,
    }

    print(f"\n[TDL v1 (Same for all robots)]")
    print(f"  Task: {tdl_v1['task']}")
    print(f"  Accel: {tdl_v1['accel_percent']}%")

    robots = ['Robot_A', 'Robot_B', 'ABB_IRB140']
    results = {}

    for robot_name in robots:
        print(f"\n{'─' * 80}")
        print(f" Testing with {robot_name}")
        print('─' * 80)

        # 로봇 로드
        robot_db = load_robot(robot_name=robot_name)

        # Scaler 생성
        scaler = ParameterScaler(robot_db, safety_margin=0.9)

        # 검증
        result = scaler.scale_tdl_parameters(tdl_v1)
        results[robot_name] = result

        print(f"\n  Feasible: {result['feasible']}")
        print(f"  Scale Factor: {result['scale_factor']:.3f}")

        if not result['feasible']:
            print(f"  ⚠ This robot cannot execute Accel=80% safely!")

    # 요약
    print("\n" + "=" * 80)
    print(" Comparison Summary")
    print("=" * 80)

    print(f"\n{'Robot':<15} {'Feasible':<12} {'Scale Factor':<15} {'Effective Accel'}")
    print("─" * 80)

    for robot_name, result in results.items():
        feasible_str = "✓ Yes" if result['feasible'] else "✗ No"
        scale = result['scale_factor']
        effective_accel = tdl_v1['accel_percent'] * scale

        print(f"{robot_name:<15} {feasible_str:<12} {scale:<15.3f} {effective_accel:.1f}%")

    print("=" * 80)


def test_trajectory_validation():
    """궤적 검증 테스트"""
    print("\n\n" + "=" * 80)
    print(" Test 4: Trajectory Validation")
    print("=" * 80)

    # Robot_B 로드
    robot_b = load_robot(robot_name='Robot_B')

    # Scaler 생성
    scaler = ParameterScaler(robot_b)

    # 테스트 궤적 생성 (간단한 사인파)
    timesteps = 100
    dof = 7
    t = np.linspace(0, 2*np.pi, timesteps)

    q_traj = np.tile(np.sin(t)[:, None], (1, dof)) * 0.5
    qd_traj = np.tile(np.cos(t)[:, None], (1, dof)) * 0.5
    qdd_traj = np.tile(-np.sin(t)[:, None], (1, dof)) * 2.0  # 높은 가속도

    print(f"\n[Trajectory Info]")
    print(f"  Timesteps: {timesteps}")
    print(f"  DOF: {dof}")
    print(f"  Max |q̈|: {np.max(np.abs(qdd_traj)):.2f} rad/s²")

    # 궤적 검증
    result = scaler.validate_and_scale_trajectory(q_traj, qd_traj, qdd_traj)

    print(f"\n[Validation Result]")
    print(f"  Feasible: {result['feasible']}")
    print(f"  Scale Factor: {result['scale_factor']:.3f}")

    if not result['feasible']:
        print(f"\n  Original Max |q̈|: {np.max(np.abs(result['qdd_original'])):.2f}")
        print(f"  Scaled Max |q̈|: {np.max(np.abs(result['qdd_scaled'])):.2f}")

    return result


if __name__ == '__main__':
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Dynamics Validation - Full Test Suite" + " " * 20 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        # Test 1: 정상 TDL
        test_single_tdl_validation()

        # Test 2: Infeasible TDL
        test_infeasible_tdl()

        # Test 3: 로봇별 비교
        test_robot_comparison()

        # Test 4: 궤적 검증
        test_trajectory_validation()

        print("\n\n" + "=" * 80)
        print(" ✓ All Tests Completed Successfully!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

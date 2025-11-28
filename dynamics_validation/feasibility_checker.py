"""
Feasibility Checker - 동역학적 실현가능성 검증

계산된 토크가 로봇의 물리적 한계를 초과하는지 검증합니다.

검증 항목:
1. 토크 한계 (τ_max)
2. 속도 한계 (q̇_max) - 선택적
3. 가속도 한계 (q̈_max) - 선택적
"""

import numpy as np
from enum import Enum


class FeasibilityStatus(Enum):
    """실현가능성 상태"""
    FEASIBLE = "feasible"  # 모든 한계 내
    INFEASIBLE_TORQUE = "infeasible_torque"  # 토크 한계 초과
    INFEASIBLE_VELOCITY = "infeasible_velocity"  # 속도 한계 초과
    INFEASIBLE_ACCELERATION = "infeasible_acceleration"  # 가속도 한계 초과


class FeasibilityChecker:
    """
    동역학적 실현가능성 검증기

    로봇의 물리적 한계와 계산된 요구값을 비교하여 실행 가능성을 판단합니다.
    """

    def __init__(self, robot_db, safety_margin=0.9):
        """
        Args:
            robot_db: RobotDynamicsDB 인스턴스
            safety_margin: 안전 마진 (0.9 = 최대 토크의 90%까지만 허용)
        """
        self.robot_db = robot_db
        self.safety_margin = safety_margin

        # 한계값 로드
        self.tau_max = robot_db.get_torque_limits() * safety_margin
        self.vel_max = robot_db.get_velocity_limits() * safety_margin if hasattr(robot_db, 'vel_max') else None
        self.acc_max = robot_db.get_acceleration_limits() * safety_margin if hasattr(robot_db, 'acc_max') else None

    def check_torque_feasibility(self, tau_required):
        """
        토크 실현가능성 검증

        Args:
            tau_required: 필요 토크 (shape: [dof] or [timesteps, dof])

        Returns:
            dict: {
                'feasible': bool,
                'status': FeasibilityStatus,
                'exceeded_joints': list,  # 초과한 관절 인덱스
                'max_ratio': float,  # 최대 사용률 (>1.0 이면 초과)
                'ratios': ndarray  # 각 관절의 사용률
            }
        """
        tau_required = np.asarray(tau_required)

        # 단일 상태인 경우
        if tau_required.ndim == 1:
            tau_abs = np.abs(tau_required)
        # 궤적인 경우 (timesteps, dof)
        else:
            tau_abs = np.max(np.abs(tau_required), axis=0)

        # 토크 사용률 계산
        ratios = tau_abs / self.tau_max

        # 초과 여부
        exceeded = ratios > 1.0
        exceeded_joints = np.where(exceeded)[0].tolist()

        feasible = len(exceeded_joints) == 0
        status = FeasibilityStatus.FEASIBLE if feasible else FeasibilityStatus.INFEASIBLE_TORQUE

        return {
            'feasible': feasible,
            'status': status,
            'exceeded_joints': exceeded_joints,
            'max_ratio': float(np.max(ratios)),
            'ratios': ratios,
            'tau_required': tau_abs,
            'tau_max': self.tau_max
        }

    def check_velocity_feasibility(self, qd):
        """
        속도 실현가능성 검증

        Args:
            qd: 관절 속도 (shape: [dof] or [timesteps, dof])

        Returns:
            dict: 검증 결과
        """
        if self.vel_max is None:
            return {
                'feasible': True,
                'status': FeasibilityStatus.FEASIBLE,
                'message': 'Velocity limits not available'
            }

        qd = np.asarray(qd)

        if qd.ndim == 1:
            qd_abs = np.abs(qd)
        else:
            qd_abs = np.max(np.abs(qd), axis=0)

        ratios = qd_abs / self.vel_max
        exceeded = ratios > 1.0
        exceeded_joints = np.where(exceeded)[0].tolist()

        feasible = len(exceeded_joints) == 0
        status = FeasibilityStatus.FEASIBLE if feasible else FeasibilityStatus.INFEASIBLE_VELOCITY

        return {
            'feasible': feasible,
            'status': status,
            'exceeded_joints': exceeded_joints,
            'max_ratio': float(np.max(ratios)),
            'ratios': ratios
        }

    def check_acceleration_feasibility(self, qdd):
        """
        가속도 실현가능성 검증

        Args:
            qdd: 관절 가속도 (shape: [dof] or [timesteps, dof])

        Returns:
            dict: 검증 결과
        """
        if self.acc_max is None:
            return {
                'feasible': True,
                'status': FeasibilityStatus.FEASIBLE,
                'message': 'Acceleration limits not available'
            }

        qdd = np.asarray(qdd)

        if qdd.ndim == 1:
            qdd_abs = np.abs(qdd)
        else:
            qdd_abs = np.max(np.abs(qdd), axis=0)

        ratios = qdd_abs / self.acc_max
        exceeded = ratios > 1.0
        exceeded_joints = np.where(exceeded)[0].tolist()

        feasible = len(exceeded_joints) == 0
        status = FeasibilityStatus.FEASIBLE if feasible else FeasibilityStatus.INFEASIBLE_ACCELERATION

        return {
            'feasible': feasible,
            'status': status,
            'exceeded_joints': exceeded_joints,
            'max_ratio': float(np.max(ratios)),
            'ratios': ratios
        }

    def check_full_feasibility(self, tau_required, qd=None, qdd=None):
        """
        전체 실현가능성 검증 (토크 + 속도 + 가속도)

        Args:
            tau_required: 필요 토크
            qd: 관절 속도 (선택)
            qdd: 관절 가속도 (선택)

        Returns:
            dict: {
                'feasible': bool,  # 전체 실현가능 여부
                'torque': dict,    # 토크 검증 결과
                'velocity': dict,  # 속도 검증 결과 (선택)
                'acceleration': dict  # 가속도 검증 결과 (선택)
            }
        """
        # 토크 검증
        torque_result = self.check_torque_feasibility(tau_required)

        # 속도 검증
        if qd is not None:
            velocity_result = self.check_velocity_feasibility(qd)
        else:
            velocity_result = {'feasible': True, 'status': FeasibilityStatus.FEASIBLE}

        # 가속도 검증
        if qdd is not None:
            acceleration_result = self.check_acceleration_feasibility(qdd)
        else:
            acceleration_result = {'feasible': True, 'status': FeasibilityStatus.FEASIBLE}

        # 전체 feasibility
        overall_feasible = (
            torque_result['feasible'] and
            velocity_result['feasible'] and
            acceleration_result['feasible']
        )

        return {
            'feasible': overall_feasible,
            'torque': torque_result,
            'velocity': velocity_result,
            'acceleration': acceleration_result
        }

    def get_required_scale_factor(self, tau_required):
        """
        Infeasible한 경우 필요한 스케일 팩터 계산

        Args:
            tau_required: 필요 토크

        Returns:
            float: 스케일 팩터 (0 < scale <= 1.0)
                   1.0 = feasible (스케일 불필요)
                   0.9 = 10% 감소 필요
        """
        result = self.check_torque_feasibility(tau_required)

        if result['feasible']:
            return 1.0

        # 최대 사용률의 역수
        scale_factor = 1.0 / result['max_ratio']

        return scale_factor

    def print_feasibility_report(self, result):
        """실현가능성 검증 결과 출력"""
        print("=" * 80)
        print(" Feasibility Check Report")
        print("=" * 80)

        # 토크 검증
        torque = result['torque']
        print(f"\n[Torque]")
        print(f"  Status: {torque['status'].value}")
        print(f"  Feasible: {torque['feasible']}")
        print(f"  Max Ratio: {torque['max_ratio']:.2f} ({'EXCEEDED' if torque['max_ratio'] > 1.0 else 'OK'})")

        if not torque['feasible']:
            print(f"  Exceeded Joints: {torque['exceeded_joints']}")
            for joint_idx in torque['exceeded_joints']:
                ratio = torque['ratios'][joint_idx]
                tau_req = torque['tau_required'][joint_idx]
                tau_max = torque['tau_max'][joint_idx]
                print(f"    Joint {joint_idx}: {tau_req:.2f} N·m / {tau_max:.2f} N·m = {ratio:.2f}x")

        # 속도 검증
        if 'velocity' in result and 'message' not in result['velocity']:
            vel = result['velocity']
            print(f"\n[Velocity]")
            print(f"  Status: {vel['status'].value}")
            print(f"  Feasible: {vel['feasible']}")

        # 가속도 검증
        if 'acceleration' in result and 'message' not in result['acceleration']:
            acc = result['acceleration']
            print(f"\n[Acceleration]")
            print(f"  Status: {acc['status'].value}")
            print(f"  Feasible: {acc['feasible']}")

        # 전체 결과
        print(f"\n{'=' * 80}")
        print(f" Overall Feasibility: {result['feasible']}")
        print(f"{'=' * 80}\n")


if __name__ == '__main__':
    # 테스트
    print("=== Testing Feasibility Checker ===\n")

    from robot_dynamics_db import load_robot
    from rnea_calculator import RNEACalculator

    # Robot_B (Panda) 로드
    robot_b = load_robot(robot_name='Robot_B')

    # 검증기 생성
    checker = FeasibilityChecker(robot_b, safety_margin=0.9)

    # RNEA 계산
    calculator = RNEACalculator(robot_b)

    # 테스트 케이스 1: Feasible
    print("[Test 1: Feasible case]")
    q1 = np.zeros(7)
    qd1 = np.zeros(7)
    qdd1 = np.ones(7) * 0.5  # 낮은 가속도

    tau1 = calculator.calculate_required_torque(q1, qd1, qdd1)
    result1 = checker.check_full_feasibility(tau1, qd1, qdd1)
    checker.print_feasibility_report(result1)

    # 테스트 케이스 2: Infeasible
    print("\n[Test 2: Infeasible case]")
    qdd2 = np.ones(7) * 10.0  # 매우 높은 가속도

    tau2 = calculator.calculate_required_torque(q1, qd1, qdd2)
    result2 = checker.check_full_feasibility(tau2, qd1, qdd2)
    checker.print_feasibility_report(result2)

    # 스케일 팩터 계산
    scale = checker.get_required_scale_factor(tau2)
    print(f"Required Scale Factor: {scale:.3f}")
    print(f"Scaled Acceleration: {qdd2 * scale}")

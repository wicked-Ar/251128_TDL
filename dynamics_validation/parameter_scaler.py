"""
Parameter Scaler - 동역학 기반 파라미터 스케일링

TDL의 추상적 파라미터(Accel_Percent=80)를 로봇의 동역학적 한계 내에서
실현 가능한 실제 파라미터로 변환합니다.

알고리즘:
1. TDL 의도 → 목표 가속도 프로파일 (q̈_desired)
2. RNEA → 필요 토크 계산 (τ_calc)
3. Feasibility Check → τ_calc vs τ_max
4. If Infeasible → Scale Factor 계산
5. q̈_actual = q̈_desired × ScaleFactor
"""

import numpy as np
from .robot_dynamics_db import RobotDynamicsDB
from .rnea_calculator import RNEACalculator
from .feasibility_checker import FeasibilityChecker


class ParameterScaler:
    """
    TDL 파라미터 스케일링

    로봇의 동역학적 한계를 고려하여 TDL 파라미터를 안전한 값으로 변환합니다.
    """

    def __init__(self, robot_db, safety_margin=0.9):
        """
        Args:
            robot_db: RobotDynamicsDB 인스턴스
            safety_margin: 안전 마진 (기본 0.9 = 90%)
        """
        self.robot_db = robot_db
        self.calculator = RNEACalculator(robot_db)
        self.checker = FeasibilityChecker(robot_db, safety_margin=safety_margin)

    def scale_tdl_parameters(self, tdl_v1, robot_state=None):
        """
        TDL v1 파라미터를 동역학 기반으로 스케일링

        Args:
            tdl_v1: TDL v1 딕셔너리 (예: {'task': 'move', 'accel_percent': 80, ...})
            robot_state: 현재 로봇 상태 (q, qd) - 선택적

        Returns:
            dict: {
                'tdl_v2': dict,  # 스케일링된 TDL v2
                'feasible': bool,  # 원본이 실행가능했는지
                'scale_factor': float,  # 적용된 스케일 팩터
                'validation_report': dict  # 상세 검증 결과
            }
        """
        # 1. TDL 의도 해석 → 목표 가속도
        qdd_desired = self._interpret_tdl_intent(tdl_v1)

        # 2. 현재 로봇 상태 (없으면 기본값 사용)
        if robot_state is None:
            q = np.zeros(self.robot_db.dof)
            qd = np.zeros(self.robot_db.dof)
        else:
            q = robot_state.get('q', np.zeros(self.robot_db.dof))
            qd = robot_state.get('qd', np.zeros(self.robot_db.dof))

        # 3. RNEA로 필요 토크 계산
        tau_required = self.calculator.calculate_required_torque(q, qd, qdd_desired)

        # 4. Feasibility Check
        validation_result = self.checker.check_full_feasibility(
            tau_required, qd, qdd_desired
        )

        # 5. 스케일 팩터 계산
        if validation_result['feasible']:
            scale_factor = 1.0
            qdd_actual = qdd_desired
        else:
            scale_factor = self.checker.get_required_scale_factor(tau_required)
            qdd_actual = qdd_desired * scale_factor

        # 6. TDL v2 생성 (실제 파라미터)
        tdl_v2 = self._create_tdl_v2(tdl_v1, qdd_actual, scale_factor)

        return {
            'tdl_v2': tdl_v2,
            'feasible': validation_result['feasible'],
            'scale_factor': scale_factor,
            'qdd_desired': qdd_desired,
            'qdd_actual': qdd_actual,
            'tau_required': tau_required,
            'validation_report': validation_result
        }

    def _interpret_tdl_intent(self, tdl_v1):
        """
        TDL v1의 추상적 의도를 목표 가속도로 변환

        Args:
            tdl_v1: TDL v1 딕셔너리

        Returns:
            qdd_desired: 목표 가속도 [rad/s²]
        """
        # TDL에서 가속도 퍼센트 추출 (기본값 50%)
        accel_percent = tdl_v1.get('accel_percent', 50)

        # 퍼센트를 실제 가속도로 변환
        # 예: 100% = 로봇의 최대 가속도 한계
        acc_max = self.robot_db.get_acceleration_limits()

        # 안전하게 최대 가속도의 일정 비율만 사용
        qdd_desired = (accel_percent / 100.0) * acc_max

        return qdd_desired

    def _create_tdl_v2(self, tdl_v1, qdd_actual, scale_factor):
        """
        TDL v2 생성 (로봇 특화 파라미터)

        Args:
            tdl_v1: 원본 TDL v1
            qdd_actual: 실제 가속도
            scale_factor: 적용된 스케일 팩터

        Returns:
            tdl_v2: 로봇별 실제 파라미터가 포함된 TDL
        """
        tdl_v2 = tdl_v1.copy()

        # 실제 가속도 추가
        tdl_v2['acceleration'] = qdd_actual.tolist()

        # 스케일 팩터 기록
        tdl_v2['scale_factor'] = scale_factor

        # 원본 의도 보존
        tdl_v2['original_accel_percent'] = tdl_v1.get('accel_percent', 50)

        # 스케일링 여부 표시
        tdl_v2['scaled'] = (scale_factor < 1.0)

        # 버전 표시
        tdl_v2['tdl_version'] = 'v2'

        return tdl_v2

    def validate_and_scale_trajectory(self, q_traj, qd_traj, qdd_traj):
        """
        궤적 전체에 대한 검증 및 스케일링

        Args:
            q_traj: 관절 위치 궤적 (shape: [timesteps, dof])
            qd_traj: 관절 속도 궤적
            qdd_traj: 관절 가속도 궤적

        Returns:
            dict: 스케일링 결과
        """
        # 궤적에 대한 토크 계산
        tau_traj = self.calculator.calculate_torque_trajectory(
            q_traj, qd_traj, qdd_traj
        )

        # Feasibility Check
        validation_result = self.checker.check_full_feasibility(
            tau_traj, qd_traj, qdd_traj
        )

        # 스케일 팩터 계산
        if validation_result['feasible']:
            scale_factor = 1.0
            qdd_traj_scaled = qdd_traj
        else:
            scale_factor = self.checker.get_required_scale_factor(tau_traj)
            qdd_traj_scaled = qdd_traj * scale_factor

        return {
            'feasible': validation_result['feasible'],
            'scale_factor': scale_factor,
            'qdd_original': qdd_traj,
            'qdd_scaled': qdd_traj_scaled,
            'validation_report': validation_result
        }

    def print_scaling_report(self, result):
        """스케일링 결과 출력"""
        print("=" * 80)
        print(" Parameter Scaling Report")
        print("=" * 80)

        print(f"\n[Original TDL v1]")
        print(f"  Accel Percent: {result['tdl_v2'].get('original_accel_percent', 'N/A')}%")

        print(f"\n[Dynamics Validation]")
        print(f"  Feasible: {result['feasible']}")

        if not result['feasible']:
            print(f"  ⚠ Original parameters exceed robot limits!")
            print(f"  Scale Factor Applied: {result['scale_factor']:.3f}")

        print(f"\n[Scaled TDL v2]")
        print(f"  Acceleration (q̈): {np.array(result['tdl_v2']['acceleration'])}")
        print(f"  Scaled: {result['tdl_v2']['scaled']}")

        print(f"\n[Torque Requirements]")
        tau_req = result['tau_required']
        tau_max = self.robot_db.get_torque_limits()
        for i, (tau, tau_lim) in enumerate(zip(tau_req, tau_max)):
            ratio = abs(tau) / tau_lim
            status = "✓" if ratio <= 1.0 else "✗"
            print(f"  Joint {i}: {abs(tau):6.2f} / {tau_lim:6.2f} N·m = {ratio:.2f}x {status}")

        print(f"\n{'=' * 80}\n")


# 편의 함수
def scale_tdl(robot_name, tdl_v1, robot_state=None, safety_margin=0.9):
    """
    TDL v1을 로봇별 TDL v2로 스케일링

    Args:
        robot_name: 'Robot_A', 'Robot_B' 등
        tdl_v1: TDL v1 딕셔너리
        robot_state: 현재 로봇 상태 (선택)
        safety_margin: 안전 마진

    Returns:
        dict: 스케일링 결과
    """
    from .robot_dynamics_db import load_robot

    robot_db = load_robot(robot_name=robot_name)
    scaler = ParameterScaler(robot_db, safety_margin=safety_margin)

    return scaler.scale_tdl_parameters(tdl_v1, robot_state)


if __name__ == '__main__':
    # 테스트
    print("=== Testing Parameter Scaler ===\n")

    from robot_dynamics_db import load_robot

    # Robot_B (Panda) 로드
    robot_b = load_robot(robot_name='Robot_B')

    # Scaler 생성
    scaler = ParameterScaler(robot_b, safety_margin=0.9)

    # 테스트 케이스 1: Feasible TDL
    print("[Test 1: Feasible TDL (Accel=30%)]")
    tdl1 = {
        'task': 'move_linear',
        'accel_percent': 30,
        'robot': 'Robot_B'
    }

    result1 = scaler.scale_tdl_parameters(tdl1)
    scaler.print_scaling_report(result1)

    # 테스트 케이스 2: Infeasible TDL (너무 공격적)
    print("\n[Test 2: Infeasible TDL (Accel=95%)]")
    tdl2 = {
        'task': 'move_linear',
        'accel_percent': 95,
        'robot': 'Robot_B'
    }

    result2 = scaler.scale_tdl_parameters(tdl2)
    scaler.print_scaling_report(result2)

    print(f"\n✓ TDL v2 (Scaled): {result2['tdl_v2']}")

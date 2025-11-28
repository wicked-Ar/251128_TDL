"""
RNEA Calculator - Recursive Newton-Euler Algorithm 기반 토크 계산

주어진 관절 상태(q, q̇, q̈)에 대해 필요한 토크(τ)를 계산합니다.

τ = M(q)q̈ + C(q, q̇)q̇ + G(q)

여기서:
- M(q): 질량 행렬 (Mass/Inertia matrix)
- C(q, q̇): 코리올리/원심력 항
- G(q): 중력 항
"""

import numpy as np
try:
    import roboticstoolbox as rtb
    ROBOTICS_TOOLBOX_AVAILABLE = True
except ImportError:
    ROBOTICS_TOOLBOX_AVAILABLE = False


class RNEACalculator:
    """
    RNEA 기반 역동역학 계산기

    로봇의 동역학 모델을 사용하여 주어진 궤적에 필요한 토크를 계산합니다.
    """

    def __init__(self, robot_db):
        """
        Args:
            robot_db: RobotDynamicsDB 인스턴스
        """
        self.robot_db = robot_db
        self.robot_model = robot_db.get_robot_model()
        self.dof = robot_db.dof

    def calculate_required_torque(self, q, qd, qdd, use_gravity=True):
        """
        주어진 관절 상태에 대한 필요 토크 계산

        Args:
            q: 관절 위치 (shape: [dof])
            qd: 관절 속도 (shape: [dof])
            qdd: 관절 가속도 (shape: [dof])
            use_gravity: 중력 포함 여부

        Returns:
            tau: 필요 토크 (shape: [dof]) [N·m]
        """
        q = np.asarray(q)
        qd = np.asarray(qd)
        qdd = np.asarray(qdd)

        if q.shape[0] != self.dof:
            raise ValueError(f"q must have shape [{self.dof}], got {q.shape}")

        # roboticstoolbox 사용 가능한 경우
        if ROBOTICS_TOOLBOX_AVAILABLE and self.robot_model is not None:
            return self._calculate_with_rtb(q, qd, qdd, use_gravity)
        else:
            # Fallback: 간단한 추정 모델 사용
            return self._calculate_fallback(q, qd, qdd, use_gravity)

    def _calculate_with_rtb(self, q, qd, qdd, use_gravity):
        """roboticstoolbox를 사용한 정확한 RNEA 계산"""
        try:
            # RNEA: Recursive Newton-Euler Algorithm
            tau = self.robot_model.rne(q, qd, qdd, gravity=[0, 0, -9.81] if use_gravity else [0, 0, 0])
            return np.array(tau)

        except Exception as e:
            print(f"[WARNING] RNEA calculation failed: {e}")
            return self._calculate_fallback(q, qd, qdd, use_gravity)

    def _calculate_fallback(self, q, qd, qdd, use_gravity):
        """
        Fallback: 간단한 동역학 모델

        실제 URDF가 없거나 RTB가 없을 때 사용하는 근사 모델입니다.
        실제 프로덕션에서는 반드시 정확한 URDF 모델을 사용해야 합니다.
        """
        # 간단한 추정: τ ≈ I·q̈ + damping·q̇ + gravity_comp
        # 여기서는 로봇의 전체 질량을 기반으로 추정

        # 관성 추정 (매우 단순화된 모델)
        mass_total = getattr(self.robot_db, 'mass_total', 20.0)
        link_mass = mass_total / self.dof

        # 각 관절의 등가 관성 추정 (link_mass × typical_link_length²)
        typical_link_length = 0.4  # m (예시)
        inertia_estimate = link_mass * (typical_link_length ** 2)

        # 토크 계산 (매우 간단한 모델)
        tau_inertia = inertia_estimate * qdd  # 관성 토크
        tau_damping = 0.5 * qd  # 댐핑 (점성 마찰)

        # 중력 보상 (간단한 추정)
        if use_gravity:
            # 중력은 관절 위치에 따라 변함 (간단하게는 sin(q))
            gravity_torque = link_mass * 9.81 * typical_link_length * np.sin(q)
        else:
            gravity_torque = np.zeros(self.dof)

        tau = tau_inertia + tau_damping + gravity_torque

        return tau

    def calculate_torque_trajectory(self, q_traj, qd_traj, qdd_traj, use_gravity=True):
        """
        궤적 전체에 대한 토크 계산

        Args:
            q_traj: 관절 위치 궤적 (shape: [timesteps, dof])
            qd_traj: 관절 속도 궤적 (shape: [timesteps, dof])
            qdd_traj: 관절 가속도 궤적 (shape: [timesteps, dof])
            use_gravity: 중력 포함 여부

        Returns:
            tau_traj: 토크 궤적 (shape: [timesteps, dof])
        """
        timesteps = q_traj.shape[0]
        tau_traj = np.zeros_like(q_traj)

        for t in range(timesteps):
            tau_traj[t] = self.calculate_required_torque(
                q_traj[t], qd_traj[t], qdd_traj[t], use_gravity
            )

        return tau_traj

    def get_max_torque_in_trajectory(self, q_traj, qd_traj, qdd_traj):
        """
        궤적에서 각 관절의 최대 토크 계산

        Args:
            q_traj, qd_traj, qdd_traj: 궤적 데이터

        Returns:
            tau_max_per_joint: 각 관절의 최대 토크 절댓값 (shape: [dof])
        """
        tau_traj = self.calculate_torque_trajectory(q_traj, qd_traj, qdd_traj)

        # 각 관절별 최대 절댓값
        tau_max_per_joint = np.max(np.abs(tau_traj), axis=0)

        return tau_max_per_joint


# 편의 함수
def calculate_torque(robot_db, q, qd, qdd, use_gravity=True):
    """
    단일 상태에 대한 토크 계산

    Args:
        robot_db: RobotDynamicsDB 인스턴스
        q, qd, qdd: 관절 위치/속도/가속도
        use_gravity: 중력 포함 여부

    Returns:
        tau: 필요 토크 [N·m]
    """
    calculator = RNEACalculator(robot_db)
    return calculator.calculate_required_torque(q, qd, qdd, use_gravity)


if __name__ == '__main__':
    # 테스트
    print("=== Testing RNEA Calculator (Panda / Robot_B) ===\n")

    # robot_dynamics_db.py에서 load_robot 함수를 가져옵니다.
    from robot_dynamics_db import load_robot

    # 'Robot_A' 대신 'Robot_B' (Panda) 로드
    try:
        # robot_dynamics_db.py에 'Robot_B'로 정의되어 있습니다.
        robot_panda = load_robot(robot_name='Robot_B') 
    except Exception as e:
        print(f"Failed to load 'Robot_B'. Error: {e}")
        # robot_dynamics_db.py 파일이 같은 폴더에 있는지 확인하세요.
        exit()

    # RNEA Calculator 생성
    calculator = RNEACalculator(robot_panda)

    # === Panda는 7-DoF 이므로 배열 크기를 7로 변경 ===
    # (이 궤적은 예시일 뿐이므로 자유롭게 변경하세요)
    q = np.array([0.0, -np.pi/4, 0.0, -np.pi/2, 0.0, np.pi/3, 0.0])  # 관절 위치 (7-DoF)
    qd = np.zeros(7)  # 정지 상태 (7-DoF)
    qdd = np.ones(7) * 1.0  # 1 rad/s² 가속 (7-DoF)

    print(f"Robot: {robot_panda.name} (Loaded as 'Robot_B')")
    print(f"DOF: {robot_panda.dof}")
    print(f"Joint positions (q): {q}")
    print(f"Joint velocities (qd): {qd}")
    print(f"Joint accelerations (qdd): {qdd}\n")

    # 토크 계산
    tau = calculator.calculate_required_torque(q, qd, qdd, use_gravity=True)

    # 보기 편하게 소수점 3자리까지만 출력
    print(f"Required torque (τ): {np.round(tau, 3)}")
    print(f"Torque limits (τ_max): {robot_panda.get_torque_limits()}")

    # 한계 초과 여부
    exceeded = np.abs(tau) > robot_panda.get_torque_limits()
    print(f"\nExceeded limits: {exceeded}")

    if np.any(exceeded):
        print(f"  Joints exceeding limits: {np.where(exceeded)[0]}")
    else:
        print("  All joints within limits ✓")

"""
Robot Dynamics DB - 로봇별 동역학 파라미터 데이터베이스

URDF 파일에서 로봇의 동역학 파라미터를 추출하고 관리합니다.
- DH 파라미터
- 질량, 관성 텐서
- 토크 한계 (τ_max)
- 속도/가속도 한계
"""

import numpy as np
from pathlib import Path
try:
    import roboticstoolbox as rtb
    from spatialmath import SE3
    ROBOTICS_TOOLBOX_AVAILABLE = True
except ImportError:
    ROBOTICS_TOOLBOX_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Using fallback dynamics parameters (roboticstoolbox-python not installed)")


class RobotDynamicsDB:
    """
    로봇 동역학 파라미터 데이터베이스

    URDF 파일에서 파라미터를 로드하거나, 사전 정의된 값을 사용합니다.
    """

    # 사전 정의된 로봇 토크 한계 (N·m)
    PREDEFINED_TORQUE_LIMITS = {
        'Robot_A': {  # UR5e (Universal Robots)
            'name': 'UR5e',
            'manufacturer': 'Universal Robots',
            'dof': 6,
            'tau_max': np.array([150, 150, 150, 28, 28, 28]),  # N·m
            'vel_max': np.array([180, 180, 180, 360, 360, 360]) * np.pi / 180,  # rad/s
            'acc_max': np.array([300, 300, 300, 600, 600, 600]) * np.pi / 180,  # rad/s²
            'mass_total': 20.6,  # kg
        },
        'Robot_B': {  # Panda (Franka Emika)
            'name': 'Panda',
            'manufacturer': 'Franka Emika',
            'dof': 7,
            'tau_max': np.array([87, 87, 87, 87, 12, 12, 12]),  # N·m
            'vel_max': np.array([150, 150, 150, 150, 180, 180, 180]) * np.pi / 180,  # rad/s
            'acc_max': np.array([500, 500, 500, 500, 600, 600, 600]) * np.pi / 180,  # rad/s²
            'mass_total': 18.0,  # kg (with gripper)
        },
        'ABB_IRB140': {  # ABB IRB 140
            'name': 'ABB IRB 140',
            'manufacturer': 'ABB',
            'dof': 6,
            'tau_max': np.array([200, 200, 100, 50, 50, 30]),  # N·m (estimated)
            'vel_max': np.array([180, 180, 260, 320, 320, 420]) * np.pi / 180,  # rad/s
            'acc_max': np.array([400, 400, 500, 700, 700, 900]) * np.pi / 180,  # rad/s²
            'mass_total': 98.0,  # kg
        }
    }

    def __init__(self, urdf_path=None, robot_name=None):
        """
        Args:
            urdf_path: URDF 파일 경로 (선택)
            robot_name: 사전 정의된 로봇 이름 (Robot_A, Robot_B 등)
        """
        self.urdf_path = urdf_path
        self.robot_name = robot_name
        self.robot_model = None

        if urdf_path and ROBOTICS_TOOLBOX_AVAILABLE:
            self._load_from_urdf(urdf_path)
        elif robot_name:
            self._load_predefined(robot_name)
        else:
            raise ValueError("Either urdf_path or robot_name must be provided")

    def _load_from_urdf(self, urdf_path):
        """URDF 파일에서 로봇 모델 로드"""
        try:
            # roboticstoolbox로 URDF 로드
            self.robot_model = rtb.Robot.URDF(urdf_path)

            # 토크 한계 추출 (URDF에 정의되어 있는 경우)
            self.tau_max = np.array([
                joint.effort_limit if hasattr(joint, 'effort_limit') else 100.0
                for joint in self.robot_model.joints
            ])

            self.dof = self.robot_model.n

            print(f"[RobotDynamicsDB] Loaded robot from URDF: {urdf_path}")
            print(f"  DOF: {self.dof}")
            print(f"  Torque limits: {self.tau_max}")

        except Exception as e:
            print(f"[ERROR] Failed to load URDF: {e}")
            print("  Falling back to predefined parameters")
            if self.robot_name:
                self._load_predefined(self.robot_name)

    def _load_predefined(self, robot_name):
        """사전 정의된 파라미터 로드"""
        if robot_name not in self.PREDEFINED_TORQUE_LIMITS:
            raise ValueError(f"Unknown robot: {robot_name}. Available: {list(self.PREDEFINED_TORQUE_LIMITS.keys())}")

        params = self.PREDEFINED_TORQUE_LIMITS[robot_name]

        self.name = params['name']
        self.manufacturer = params['manufacturer']
        self.dof = params['dof']
        self.tau_max = params['tau_max']
        self.vel_max = params['vel_max']
        self.acc_max = params['acc_max']
        self.mass_total = params['mass_total']

        print(f"[RobotDynamicsDB] Loaded predefined robot: {robot_name} ({self.name})")
        print(f"  DOF: {self.dof}")
        print(f"  Torque limits: {self.tau_max}")
        print(f"  Velocity limits: {self.vel_max}")
        print(f"  Acceleration limits: {self.acc_max}")

    def get_torque_limits(self):
        """토크 한계 반환"""
        return self.tau_max.copy()

    def get_velocity_limits(self):
        """속도 한계 반환"""
        return self.vel_max.copy()

    def get_acceleration_limits(self):
        """가속도 한계 반환"""
        return self.acc_max.copy()

    def get_robot_model(self):
        """roboticstoolbox Robot 모델 반환 (URDF 로드된 경우)"""
        return self.robot_model

    def get_info(self):
        """로봇 정보 딕셔너리 반환"""
        return {
            'name': getattr(self, 'name', self.robot_name),
            'manufacturer': getattr(self, 'manufacturer', 'Unknown'),
            'dof': self.dof,
            'tau_max': self.tau_max,
            'vel_max': getattr(self, 'vel_max', None),
            'acc_max': getattr(self, 'acc_max', None),
            'mass_total': getattr(self, 'mass_total', None),
        }


# 편의 함수
def load_robot(robot_name=None, urdf_path=None):
    """
    로봇 동역학 DB 로드

    Args:
        robot_name: 'Robot_A', 'Robot_B', 'ABB_IRB140' 등
        urdf_path: URDF 파일 경로 (선택)

    Returns:
        RobotDynamicsDB 인스턴스
    """
    return RobotDynamicsDB(urdf_path=urdf_path, robot_name=robot_name)


if __name__ == '__main__':
    # 테스트
    print("=== Testing Robot Dynamics DB ===\n")

    # Robot_A (UR5e) 로드
    robot_a = load_robot(robot_name='Robot_A')
    print(f"\n{robot_a.get_info()}\n")

    # Robot_B (Panda) 로드
    robot_b = load_robot(robot_name='Robot_B')
    print(f"\n{robot_b.get_info()}\n")

    # ABB IRB 140 로드
    abb = load_robot(robot_name='ABB_IRB140')
    print(f"\n{abb.get_info()}\n")

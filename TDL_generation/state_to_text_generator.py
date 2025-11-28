"""
State to Text Generator - MuJoCo Ground Truth TSD 생성

MuJoCo Observation을 파싱하여 Ground Truth 기반의
Textual State Description (TSD)를 생성합니다.
"""
import numpy as np
from typing import Any, Dict, List


class StateToTextGenerator:
    """
    MuJoCo Observation → TSD 변환기

    기능:
    1. 로봇 상태 추출 (End-effector 위치/자세)
    2. 물체 상태 추출 (이름, 위치)
    3. LLM 프롬프트용 마크다운 TSD 생성
    """

    def __init__(self):
        """초기화"""
        # 로봇 매핑 (obs 필드명 → 표시명)
        self.robot_mapping = {
            'ur5e_robotiq': 'Robot_A (UR5e)',
            'panda': 'Robot_B (Panda)'
        }

        # 무시할 물체 (환경 구성 요소)
        self.ignore_objects = {
            'table_top',
            'bin',
            'slot_1',
            'slot_2',
            'slot_3',
            'slot_4'
        }

    def generate_tsd_context(self, obs: Any) -> str:
        """
        MuJoCo Observation에서 TSD 생성

        Args:
            obs: MuJoCo EnvState 객체
                - obs.ur5e_robotiq: Robot_A state
                - obs.panda: Robot_B state
                - obs.objects: 물체 상태 딕셔너리

        Returns:
            str: 마크다운 형식의 TSD 컨텍스트
        """
        # 1. 로봇 상태 추출
        robot_status = self._extract_robot_states(obs)

        # 2. 물체 상태 추출
        object_status = self._extract_object_states(obs)

        # 3. TSD 마크다운 생성
        tsd_markdown = self._format_tsd_markdown(robot_status, object_status)

        return tsd_markdown

    def _extract_robot_states(self, obs: Any) -> List[Dict]:
        """
        로봇 상태 추출

        Returns:
            List[Dict]: [
                {
                    'name': 'Robot_A (UR5e)',
                    'ee_pos': [x, y, z],
                    'ee_quat': [w, x, y, z]
                },
                ...
            ]
        """
        robot_states = []

        for field_name, display_name in self.robot_mapping.items():
            robot_state = getattr(obs, field_name, None)

            if robot_state is None:
                continue

            robot_states.append({
                'name': display_name,
                'ee_pos': np.array(robot_state.ee_xpos),
                'ee_quat': np.array(robot_state.ee_xquat)
            })

        return robot_states

    def _extract_object_states(self, obs: Any) -> List[Dict]:
        """
        물체 상태 추출

        Returns:
            List[Dict]: [
                {
                    'name': 'apple',
                    'type': 'fruit',
                    'pos': [x, y, z]
                },
                ...
            ]
        """
        object_states = []

        # obs.objects는 딕셔너리: {name: ObjectState}
        for obj_name, obj_state in obs.objects.items():
            # 환경 구성 요소는 제외
            if obj_name in self.ignore_objects:
                continue

            # 물체 타입 추론
            obj_type = self._infer_object_type(obj_name)

            object_states.append({
                'name': obj_name,
                'type': obj_type,
                'pos': np.array(obj_state.xpos)
            })

        return object_states

    def _infer_object_type(self, obj_name: str) -> str:
        """
        물체 이름에서 타입 추론

        Args:
            obj_name: 물체 이름 (예: 'apple', 'milk')

        Returns:
            str: 물체 타입 (예: 'fruit', 'container')
        """
        # 간단한 휴리스틱 분류
        fruit_names = {'apple', 'banana', 'orange', 'grape'}
        container_names = {'milk', 'bottle', 'cup', 'box'}
        food_names = {'bread', 'sandwich', 'cookie'}

        if obj_name in fruit_names:
            return 'fruit'
        elif obj_name in container_names:
            return 'container'
        elif obj_name in food_names:
            return 'food'
        else:
            return 'object'

    def _format_tsd_markdown(
        self,
        robot_status: List[Dict],
        object_status: List[Dict]
    ) -> str:
        """
        TSD 마크다운 형식 생성

        Args:
            robot_status: 로봇 상태 리스트
            object_status: 물체 상태 리스트

        Returns:
            str: 마크다운 형식 TSD
        """
        lines = []

        # 헤더
        lines.append("# Current Scene Information (GROUND TRUTH TSD)")
        lines.append("")
        lines.append("The following state is parsed directly from MuJoCo Observation data:")
        lines.append("")

        # 로봇 상태
        lines.append("## Robot Status")
        for robot in robot_status:
            pos = robot['ee_pos']
            lines.append(
                f"- {robot['name']}: "
                f"EE Position (X, Y, Z): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m"
            )
        lines.append("")

        # 물체 상태
        lines.append("## Detected Objects (Name, Type, Position)")
        for obj in object_status:
            pos = obj['pos']
            lines.append(
                f"- {obj['name']} ({obj['type']}): "
                f"Position [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m, "
                f"State: On Table"
            )

        return "\n".join(lines)


def test_tsd_generator():
    """
    TSD Generator 테스트

    간단한 Mock EnvState를 생성하여 TSD 생성 테스트
    """
    from dataclasses import dataclass
    from typing import Dict

    @dataclass
    class MockRobotState:
        ee_xpos: np.ndarray
        ee_xquat: np.ndarray

    @dataclass
    class MockObjectState:
        xpos: np.ndarray
        xquat: np.ndarray

    @dataclass
    class MockEnvState:
        ur5e_robotiq: MockRobotState
        panda: MockRobotState
        objects: Dict[str, MockObjectState]

    # Mock 데이터 생성
    mock_obs = MockEnvState(
        ur5e_robotiq=MockRobotState(
            ee_xpos=np.array([0.520, 0.110, 0.450]),
            ee_xquat=np.array([0.99, 0.0, 0.0, 0.0])
        ),
        panda=MockRobotState(
            ee_xpos=np.array([0.300, -0.200, 0.500]),
            ee_xquat=np.array([1.0, 0.0, 0.0, 0.0])
        ),
        objects={
            'apple': MockObjectState(
                xpos=np.array([0.250, 0.150, 0.050]),
                xquat=np.array([1.0, 0.0, 0.0, 0.0])
            ),
            'banana': MockObjectState(
                xpos=np.array([0.500, -0.100, 0.050]),
                xquat=np.array([1.0, 0.0, 0.0, 0.0])
            ),
            'milk': MockObjectState(
                xpos=np.array([-0.300, 0.400, 0.100]),
                xquat=np.array([1.0, 0.0, 0.0, 0.0])
            ),
            'table_top': MockObjectState(
                xpos=np.array([0.0, 0.0, 0.0]),
                xquat=np.array([1.0, 0.0, 0.0, 0.0])
            )
        }
    )

    # TSD 생성
    generator = StateToTextGenerator()
    tsd = generator.generate_tsd_context(mock_obs)

    print("=" * 80)
    print(" TSD Generator Test")
    print("=" * 80)
    print()
    print(tsd)
    print()
    print("=" * 80)


if __name__ == "__main__":
    test_tsd_generator()

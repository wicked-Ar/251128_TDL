"""
Dynamics Validation Module - 동역학 기반 실현가능성 검증

TDL 파라미터가 선택된 로봇의 물리적 한계(토크, 속도 등) 내에서
실행 가능한지 검증하고, 필요시 스케일링하여 안전한 파라미터로 변환합니다.

Architecture:
    [TDL v1] → [Robot Selection] → [Dynamics Validation] → [TDL v2]
                                         ↓
                                 Feasibility Check
                                 Parameter Scaling
"""

from .robot_dynamics_db import RobotDynamicsDB
from .rnea_calculator import RNEACalculator
from .feasibility_checker import FeasibilityChecker
from .parameter_scaler import ParameterScaler

__all__ = [
    'RobotDynamicsDB',
    'RNEACalculator',
    'FeasibilityChecker',
    'ParameterScaler'
]

__version__ = '1.0.0'

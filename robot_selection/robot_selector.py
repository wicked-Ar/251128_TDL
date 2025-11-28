# robot_selector.py
"""
Robot Selector Module (Contribution 2)
Selects the optimal robot based on TDL requirements to prevent over-specification
"""

import json
import math
import os
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


# --- 1. 점수 계산 함수 (수식) ---

def calculate_payload_score(robot_payload: float, required_payload: float, return_details: bool = False):
    """
    Payload 점수(Sp)를 계산합니다. (개선된 상대 비율 방식)
    '과소비' 방지를 위해 최적 비율(안전 마진)에서 가우시안 함수를 적용합니다.

    Args:
        robot_payload: 로봇의 최대 페이로드 용량 (kg)
        required_payload: 작업에 필요한 페이로드 (kg)
        return_details: True면 상세 정보도 반환

    Returns:
        Payload 점수 (0.0 ~ 1.0) 또는 (점수, 상세정보dict)

    Formula:
        S_p = 0                                                      if p_c < p_r
              max(exp(-((p_c/p_r - α)^2) / (2 * σ_rel^2)), MIN)    otherwise

        where:
        - p_c: robot_payload (로봇 용량)
        - p_r: required_payload (요구 페이로드)
        - α = 1.2 (안전 마진, 최적 비율)
        - σ_rel = 0.20 (상대적 허용 편차, 이전 0.15에서 완화)
        - MIN = 0.1 (최소 보장 점수)

    특징:
        - 상대 비율 방식으로 모든 페이로드 범위에서 일관된 엄격도 유지
        - 최적(1.2배)에서 ±20% 벗어나면 약 0.61점
        - 최적에서 ±40% 벗어나면 약 0.14점
        - 과잉 스펙이라도 최소 0.1점 보장 (Payload가 Reach에 완전히 묻히지 않음)
    """
    # 파라미터
    ALPHA_P = 1.2      # Safety Margin (1.2배가 최적)
    SIGMA_REL = 0.20   # 상대적 허용 편차 (0.20 = 좀 더 관대하게, 이전 0.15)
    MIN_SCORE = 0.1    # 최소 보장 점수 (능력은 있지만 과잉인 경우)

    if robot_payload < required_payload:
        if return_details:
            return 0.0, {'ratio': 0, 'gaussian': 0, 'min_applied': False, 'reason': 'insufficient'}
        return 0.0

    # 상대 비율 계산
    ratio = robot_payload / required_payload

    # 최적 비율(α)로부터의 편차
    deviation = ratio - ALPHA_P

    # 가우시안 점수: exp(-deviation^2 / (2 * σ_rel^2))
    gaussian_score = math.exp(-(deviation ** 2) / (2 * SIGMA_REL ** 2))

    # 최소 점수 보장 (능력은 있으므로 완전히 0이 되지 않도록)
    score = max(gaussian_score, MIN_SCORE)

    if return_details:
        details = {
            'ratio': ratio,
            'gaussian': gaussian_score,
            'min_applied': gaussian_score < MIN_SCORE,
            'reason': 'ok'
        }
        return score, details
    return score


def calculate_reach_score(robot_reach: float, required_reach: float) -> float:
    """
    Reach 점수(Sr)를 계산합니다. (PDF 수식 3페이지)
    요구사항보다 클수록 좋지만, 점차 1에 수렴합니다.

    Args:
        robot_reach: 로봇의 작업 반경 (m)
        required_reach: 작업에 필요한 반경 (m)

    Returns:
        Reach 점수 (0.0 ~ 1.0)

    Formula:
        S_r = 1 - exp(-ALPHA_R * (robot_reach - required_reach))
        if robot_reach >= required_reach, else 0
    """
    # 이 값은 태스크의 특성에 따라 조절 가능
    ALPHA_R = 1.5  # Growth Rate (값이 클수록 점수 상승이 빠름)

    if robot_reach < required_reach:
        return 0.0
    else:
        # 요구사항 대비 초과된 Reach
        diff = robot_reach - required_reach

        # 1 - exp(-alpha * diff)
        score = 1.0 - math.exp(-ALPHA_R * diff)
        return score


def calculate_dof_score(robot_dof: int, required_dof: int) -> float:
    """
    DoF 점수(Sd)를 계산합니다. (PDF 수식 4페이지)
    같으면 1점, 더 많으면 과잉 스펙으로 감점(0.8점)

    Args:
        robot_dof: 로봇의 자유도 수
        required_dof: 작업에 필요한 자유도 수

    Returns:
        DoF 점수 (0.0, 0.8, 또는 1.0)

    Formula:
        S_d = 1.0 if robot_dof == required_dof
              0.8 if robot_dof > required_dof (over-specification penalty)
              0.0 if robot_dof < required_dof (cannot perform task)
    """
    if robot_dof < required_dof:
        return 0.0
    elif robot_dof == required_dof:
        return 1.0
    else:  # robot_dof > required_dof
        return 0.8


# --- 2. TDL 파서 ---

def parse_requirements_from_tdl(tdl_content: str) -> Dict:
    """
    TDL 파일에서 요구사항을 추출합니다.
    실제 TDL 구조에서 payload, reach, dof 요구사항을 파싱합니다.

    Args:
        tdl_content: TDL 파일 내용 (문자열)

    Returns:
        요구사항 딕셔너리 {'payload': float, 'reach': float, 'dof': int}
    """
    import re

    requirements = {
        'payload': 1.0,  # 기본값 (kg) - 작은 물체를 위한 현실적인 기본값
        'reach': 0.8,    # 기본값 (m)
        'dof': 6         # 기본값
    }

    # 파싱 플래그 (우선순위 관리)
    payload_found = False

    # TDL 라인을 읽으며 요구사항 파싱
    for line in tdl_content.split('\n'):
        line = line.strip()

        # 1. 명시적 요구사항 키워드 찾기 (최고 우선순위)
        if "PAYLOAD_KG" in line:
            try:
                # 예: "PAYLOAD_KG: 15.0" or "// PAYLOAD_KG: 15.0"
                parts = line.split(':')
                if len(parts) >= 2:
                    requirements['payload'] = float(parts[1].strip())
                    payload_found = True
                    logger.info(f"[TDL Parser] Found explicit PAYLOAD_KG: {requirements['payload']} kg")
            except (ValueError, IndexError):
                logger.warning(f"Failed to parse PAYLOAD_KG from line: {line}")

        # 2. SetWorkpieceWeight 파싱 (우선순위: 명시적 PAYLOAD_KG가 없을 때만)
        if "SetWorkpieceWeight" in line and not payload_found:
            try:
                # 예: "SetWorkpieceWeight(15.0, Trans(...))" or "SetWorkpieceWeight(15, ...)"
                match = re.search(r'SetWorkpieceWeight\s*\(\s*([\d.]+)', line)
                if match:
                    weight = float(match.group(1))
                    requirements['payload'] = weight
                    payload_found = True
                    logger.info(f"[TDL Parser] Found SetWorkpieceWeight: {weight} kg")
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to parse SetWorkpieceWeight from line: {line}, error: {e}")

        # REQUIRED_REACH_M 키워드 찾기
        if "REQUIRED_REACH_M" in line or "REACH_M" in line:
            try:
                # 예: "REQUIRED_REACH_M: 1.1"
                parts = line.split(':')
                if len(parts) >= 2:
                    requirements['reach'] = float(parts[1].strip())
                    logger.info(f"[TDL Parser] Found explicit REQUIRED_REACH_M: {requirements['reach']} m")
            except (ValueError, IndexError):
                logger.warning(f"Failed to parse REQUIRED_REACH_M from line: {line}")

        # 3. PosX에서 reach 추정 (대략적)
        if "PosX" in line and requirements['reach'] == 0.8:  # 기본값일 때만
            try:
                # 예: "PosX(1200, 300, 500, ...)" → reach ≈ sqrt(x^2 + y^2) / 1000
                match = re.search(r'PosX\s*\(\s*([\d.]+)\s*,\s*([\d.]+)', line)
                if match:
                    x = float(match.group(1))
                    y = float(match.group(2))
                    estimated_reach = ((x**2 + y**2) ** 0.5) / 1000.0  # mm → m
                    # 안전마진 추가
                    requirements['reach'] = max(requirements['reach'], estimated_reach * 1.1)
            except (ValueError, AttributeError):
                pass

        # 3. REQUIRED_DOF 키워드 찾기
        if "REQUIRED_DOF" in line or "DOF" in line:
            try:
                # 예: "REQUIRED_DOF: 6"
                parts = line.split(':')
                if len(parts) >= 2:
                    requirements['dof'] = int(float(parts[1].strip()))
            except (ValueError, IndexError):
                logger.warning(f"Failed to parse REQUIRED_DOF from line: {line}")

    # 최종 로깅 (소스 표시)
    payload_source = "SetWorkpieceWeight" if payload_found else "default(1.0kg)"
    logger.info(f"[RobotSelector] TDL 요구사항 파싱 완료: {requirements}")
    logger.info(f"  Payload source: {payload_source}")

    return requirements


# --- 3. 메인 선택 함수 ---

def select_best_robot(
    tdl_v1_content: str,
    robot_db_path: str = None,
    weights: Dict[str, float] = None
) -> Tuple[str, float, Dict]:
    """
    모듈 2의 메인 함수입니다.
    TDL(v1)과 로봇 DB를 받아, 최고 점수의 로봇 ID를 반환합니다.

    Args:
        tdl_v1_content: TDL v1 코드 (문자열)
        robot_db_path: robot_db.json 파일 경로 (None이면 기본 경로 사용)
        weights: 점수 가중치 딕셔너리 {'payload': float, 'reach': float, 'dof': float}

    Returns:
        Tuple of (best_robot_id, best_score, all_scores_dict)

    Raises:
        FileNotFoundError: robot_db.json을 찾을 수 없을 때
        ValueError: 적합한 로봇이 없을 때
    """

    # 1. 로봇 DB 로드
    if robot_db_path is None:
        # 기본 경로: 현재 파일의 data/ 폴더
        current_dir = os.path.dirname(os.path.abspath(__file__))
        robot_db_path = os.path.join(current_dir, 'data', 'robot_db.json')

    try:
        with open(robot_db_path, 'r', encoding='utf-8') as f:
            robot_db = json.load(f)
        logger.info(f"[RobotSelector] Loaded {len(robot_db)} robots from: {robot_db_path}")
    except FileNotFoundError:
        error_msg = f"Error: 로봇 DB 로드 실패 - {robot_db_path} 파일을 찾을 수 없습니다."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Error: 로봇 DB JSON 파싱 실패 - {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 2. TDL 요구사항 파싱
    reqs = parse_requirements_from_tdl(tdl_v1_content)
    req_payload = reqs.get('payload', 0.0)
    req_reach = reqs.get('reach', 0.0)
    req_dof = reqs.get('dof', 6)  # 기본 6축

    # 3. 가중치 설정 (태스크에 따라 변경 가능)
    if weights is None:
        # 기본 가중치: 페이로드가 가장 중요
        weights = {
            'payload': 0.6,
            'reach': 0.2,
            'dof': 0.2
        }

    # 가중치 합계가 1.0인지 확인
    total_weight = sum(weights.values())
    if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
        logger.warning(f"[RobotSelector] 가중치 합계가 1.0이 아닙니다: {total_weight}. 정규화합니다.")
        weights = {k: v / total_weight for k, v in weights.items()}

    # 4. 모든 로봇에 대해 점수 계산
    all_scores = {}
    logger.info("="*80)
    logger.info(f"[RobotSelector] 로봇 선정 시작")
    logger.info(f"요구사항: Payload={req_payload}kg, Reach={req_reach}m, DoF={req_dof}")
    logger.info(f"가중치: Payload={weights['payload']}, Reach={weights['reach']}, DoF={weights['dof']}")
    logger.info("="*80)

    for robot_id, specs in robot_db.items():
        # 4-1. 개별 점수 계산
        s_p, p_details = calculate_payload_score(specs['payload'], req_payload, return_details=True)
        s_r = calculate_reach_score(specs['reach'], req_reach)
        s_d = calculate_dof_score(specs['dof'], req_dof)

        # 4-2. 최종 가중합 계산
        s_total = (
            weights['payload'] * s_p +
            weights['reach'] * s_r +
            weights['dof'] * s_d
        )

        all_scores[robot_id] = {
            'total': s_total,
            'payload_score': s_p,
            'reach_score': s_r,
            'dof_score': s_d,
            'specs': specs,
            'payload_details': p_details
        }

        # 상세 로깅 (가우시안 점수와 MIN_SCORE 적용 여부 표시)
        p_info = f"P: {s_p:.3f}"
        if p_details.get('min_applied'):
            p_info += f" (gaussian={p_details['gaussian']:.4f}, MIN applied)"
        else:
            p_info += f" (ratio={p_details['ratio']:.1f}x)"

        logger.info(
            f"  로봇: {robot_id:<15} | "
            f"총점: {s_total:.4f} | "
            f"{p_info} | R: {s_r:.3f} | D: {s_d:.1f} | "
            f"(Payload: {specs['payload']}kg, Reach: {specs['reach']}m, DoF: {specs['dof']})"
        )

    # 5. 최고 점수 로봇 반환
    if not all_scores:
        error_msg = "Error: 점수를 계산할 로봇이 없습니다."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 점수가 0인 로봇은 제외
    valid_robots = {k: v for k, v in all_scores.items() if v['total'] > 0}

    if not valid_robots:
        error_msg = "Error: 요구사항을 만족하는 로봇이 없습니다. 모든 로봇의 점수가 0입니다."
        logger.error(error_msg)
        raise ValueError(error_msg)

    best_robot_id = max(valid_robots, key=lambda k: valid_robots[k]['total'])
    best_score = valid_robots[best_robot_id]['total']

    logger.info("="*80)
    logger.info(f"[RobotSelector] ✓ 선택 완료")
    logger.info(f"최적 로봇: {best_robot_id} (점수: {best_score:.4f})")
    logger.info("="*80)

    return best_robot_id, best_score, all_scores


# --- 4. 유틸리티 함수 ---

def print_selection_report(best_robot_id: str, all_scores: Dict):
    """
    로봇 선택 결과를 보기 좋게 출력합니다.

    Args:
        best_robot_id: 선택된 로봇 ID
        all_scores: 모든 로봇의 점수 딕셔너리
    """
    print("\n" + "="*80)
    print("ROBOT SELECTION REPORT")
    print("="*80)

    # 점수 순으로 정렬
    sorted_robots = sorted(
        all_scores.items(),
        key=lambda x: x[1]['total'],
        reverse=True
    )

    print(f"\n{'Rank':<6} {'Robot ID':<20} {'Total Score':<15} {'Payload':<12} {'Reach':<12} {'DoF':<12}")
    print("-"*80)

    for rank, (robot_id, scores) in enumerate(sorted_robots, 1):
        marker = " [SELECTED]" if robot_id == best_robot_id else ""
        print(
            f"{rank:<6} "
            f"{robot_id:<20} "
            f"{scores['total']:.4f}{marker:<15} "
            f"{scores['payload_score']:.3f}        "
            f"{scores['reach_score']:.3f}        "
            f"{scores['dof_score']:.1f}"
        )

    print("="*80)

    # 선택된 로봇 상세 정보
    best_specs = all_scores[best_robot_id]['specs']
    print(f"\nSELECTED ROBOT DETAILS:")
    print(f"  Robot ID: {best_robot_id}")
    print(f"  Manufacturer: {best_specs.get('manufacturer', 'N/A')}")
    print(f"  Payload: {best_specs['payload']} kg")
    print(f"  Reach: {best_specs['reach']} m")
    print(f"  DoF: {best_specs['dof']}")
    print(f"  Max Velocity: {best_specs.get('max_velocity', 'N/A')} rad/s")
    print(f"  Max Acceleration: {best_specs.get('max_acceleration', 'N/A')} rad/s^2")
    print(f"  Description: {best_specs.get('description', 'N/A')}")
    print("="*80 + "\n")


# --- 5. 실행 예제 ---

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*80)
    print("ROBOT SELECTOR TEST")
    print("="*80 + "\n")

    # 가짜 TDL (v1) 내용 (테스트용)
    # PAYLOAD_KG: 15.0
    # REQUIRED_REACH_M: 1.1
    mock_tdl_content = """
TASK_NAME: WELDING_TASK_001
DESCRIPTION: 15kg 용접기를 A에서 B로 이동

// 작업 요구사항
PAYLOAD_KG: 15.0
REQUIRED_REACH_M: 1.1
REQUIRED_DOF: 6

GOAL Initialize_Process()
{
    SPAWN SetTool(0) WITH WAIT;
    SPAWN SetJointVelocity(50) WITH WAIT;
    SPAWN SetJointAcceleration(50) WITH WAIT;
}

GOAL Execute_Process()
{
    SPAWN MoveJoint(PosJ(0,0,90,0,90,0), 50, 50, 0, 0.0, None) WITH WAIT;
    SPAWN MoveLinear(PosX(300,0,200,0,180,0), 50, 50, 0, 0.0, None) WITH WAIT;
}

GOAL Finalize_Process()
{
    SPAWN MoveJoint(PosJ(0,0,0,0,0,0), 50, 50, 0, 0.0, None) WITH WAIT;
}
"""

    try:
        # 로봇 선택 실행
        best_robot_id, best_score, all_scores = select_best_robot(
            tdl_v1_content=mock_tdl_content
        )

        # 결과 출력
        print_selection_report(best_robot_id, all_scores)

        print(f"\n[OK] Test completed successfully!")
        print(f"Selected robot: {best_robot_id} (score: {best_score:.4f})\n")

    except Exception as e:
        print(f"\n[ERROR] Test failed with error:")
        print(f"  {e}\n")
        import traceback
        traceback.print_exc()

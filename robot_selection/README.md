# Robot Selection Module

**NL2TDL Framework - Contribution 2: Optimal Robot Selection**

이 모듈은 TDL(v1)에서 주어진 작업 요구사항을 바탕으로 '과소비(Over-specification)' 문제를 방지하는 최적의 로봇을 선택합니다.

## 📋 목차

- [개요](#개요)
- [핵심 알고리즘](#핵심-알고리즘)
- [설치 및 실행](#설치-및-실행)
- [사용 방법](#사용-방법)
- [파일 구조](#파일-구조)
- [테스트](#테스트)

## 개요

### 문제점
LLM 기반 로봇 제어 시스템에서 **"10kg을 들어야 하는데 30kg 로봇을 쓰는 것은 과소비 아닌가?"** 라는 피드백에 대응하기 위해 개발되었습니다.

### 해결 방법
단순히 "로봇 페이로드 > 요구 페이로드" 조건만 확인하는 것이 아니라:
- **가우시안 함수**를 사용하여 최적 페이로드(optimal_payload = 1.2 × required_payload)에 가까운 로봇에 높은 점수 부여
- Reach, DoF 등 다중 기준을 종합적으로 고려
- 과잉 스펙에 대한 페널티 적용 (예: DoF가 필요 이상으로 많으면 0.8점)

## 핵심 알고리즘

### 1. Payload 점수 (S_p)

**가우시안 함수**를 사용하여 최적 지점과의 거리를 평가:

```
S_p = exp(-(robot_payload - optimal_payload)² / (2σ²))

where:
  optimal_payload = α_p × required_payload
  α_p = 1.2 (safety margin)
  σ² = 0.1 (matching strictness)
```

**특징:**
- `robot_payload < required_payload`: **0점** (불가능)
- `robot_payload = optimal_payload`: **1점** (최적)
- `robot_payload >> optimal_payload`: **점수 감소** (과소비)

### 2. Reach 점수 (S_r)

요구사항보다 작업 반경이 클수록 좋지만, 지수적으로 1에 수렴:

```
S_r = 1 - exp(-α_r × (robot_reach - required_reach))

where:
  α_r = 1.5 (growth rate)
```

**특징:**
- `robot_reach < required_reach`: **0점** (불가능)
- `robot_reach = required_reach`: **~0.77점**
- `robot_reach >> required_reach`: **~1점** (수렴)

### 3. DoF 점수 (S_d)

자유도 일치 여부를 평가:

```
S_d = 1.0  if robot_dof == required_dof
      0.8  if robot_dof > required_dof (over-specification)
      0.0  if robot_dof < required_dof (insufficient)
```

### 4. 최종 점수 (S_total)

가중합으로 종합 점수 산출:

```
S_total = w_p × S_p + w_r × S_r + w_d × S_d

Default weights:
  w_p = 0.6  (payload가 가장 중요)
  w_r = 0.2
  w_d = 0.2
```

## 설치 및 실행

### 1. 필요 조건

- Python 3.8 이상
- 필수 패키지 없음 (표준 라이브러리만 사용)

### 2. 폴더 구조

```
NL2TDL/robot_selection/
├── __init__.py              # 모듈 초기화
├── robot_selector.py        # 메인 로직
├── test_selector.py         # 테스트 스크립트
├── README.md                # 이 문서
└── data/
    └── robot_db.json        # 로봇 데이터베이스
```

### 3. 빠른 시작

```bash
# 현재 디렉토리로 이동
cd NL2TDL/robot_selection

# 기본 테스트 실행
python robot_selector.py

# 종합 테스트 실행
python test_selector.py
```

## 사용 방법

### 1. Python 스크립트에서 사용

```python
from NL2TDL.robot_selection import select_best_robot

# TDL v1 코드 준비
tdl_content = """
// 작업 요구사항
PAYLOAD_KG: 15.0
REQUIRED_REACH_M: 1.2
REQUIRED_DOF: 6

GOAL Execute_Process()
{
    SPAWN MoveLinear(PosX(1000,0,500,0,180,0), 50, 50, 0, 0.0, None) WITH WAIT;
}
"""

# 로봇 선택 실행
best_robot_id, best_score, all_scores = select_best_robot(tdl_content)

print(f"선택된 로봇: {best_robot_id}")
print(f"점수: {best_score:.4f}")
```

### 2. 사용자 정의 가중치

```python
# 가중치 커스터마이징
custom_weights = {
    'payload': 0.5,  # Payload 중요도 낮춤
    'reach': 0.3,    # Reach 중요도 높임
    'dof': 0.2
}

best_robot_id, best_score, all_scores = select_best_robot(
    tdl_content,
    weights=custom_weights
)
```

### 3. TDL 요구사항 형식

TDL 파일에 다음 키워드를 포함하여 요구사항을 명시:

```tdl
// 작업 요구사항 (주석 또는 별도 라인으로 명시)
PAYLOAD_KG: 10.0           // 필요한 페이로드 (kg)
REQUIRED_REACH_M: 1.0      // 필요한 작업 반경 (m)
REQUIRED_DOF: 6            // 필요한 자유도 수
```

**기본값:**
- `PAYLOAD_KG`: 5.0 kg
- `REQUIRED_REACH_M`: 0.8 m
- `REQUIRED_DOF`: 6

## 파일 구조

### robot_selector.py

메인 로직을 포함하는 핵심 파일:

**주요 함수:**

| 함수 | 설명 |
|------|------|
| `calculate_payload_score()` | Payload 점수 계산 (가우시안 함수) |
| `calculate_reach_score()` | Reach 점수 계산 (지수 함수) |
| `calculate_dof_score()` | DoF 점수 계산 (조건 분기) |
| `parse_requirements_from_tdl()` | TDL에서 요구사항 추출 |
| `select_best_robot()` | 메인 로봇 선택 함수 |
| `print_selection_report()` | 결과 출력 (시각화) |

### data/robot_db.json

로봇 데이터베이스 (JSON 형식):

```json
{
  "panda": {
    "robot_id": "panda",
    "payload": 3.0,      // kg
    "reach": 0.855,      // m
    "dof": 7,
    "max_velocity": 2.175,      // rad/s
    "max_acceleration": 15.0,   // rad/s²
    "manufacturer": "Franka Emika",
    "description": "High-precision collaborative robot"
  }
}
```

**포함된 로봇:**
1. `panda` - Franka Emika (3kg, 7-DoF)
2. `ur5e` - Universal Robots (5kg, 6-DoF)
3. `doosan_h2515` - Doosan Robotics (25kg, 6-DoF)
4. `doosan_m1013` - Doosan Robotics (10kg, 6-DoF)
5. `doosan_m0609` - Doosan Robotics (6kg, 6-DoF)
6. `kuka_iiwa14` - KUKA (14kg, 7-DoF)

## 테스트

### 1. 기본 테스트

```bash
python robot_selector.py
```

**테스트 케이스:** 15kg 용접 작업
- 예상 결과: `doosan_m1013` 또는 `kuka_iiwa14`

### 2. 종합 테스트

```bash
python test_selector.py
```

**테스트 시나리오:**
1. Heavy Payload (25kg) → `doosan_h2515`
2. Medium Payload (10kg) → `doosan_m1013`
3. Light Payload (3kg) → `panda` 또는 `ur5e`
4. High DoF (7-axis) → `panda` 또는 `kuka_iiwa14`
5. Long Reach (1.5m) → `doosan_h2515`
6. Default Values → 기본값으로 선택

### 3. 예상 출력

```
================================================================================
ROBOT SELECTION REPORT
================================================================================

Rank   Robot ID             Total Score     Payload      Reach        DoF
--------------------------------------------------------------------------------
1      doosan_m1013         0.8543 ✓ SELECTED 0.923        0.884        1.0
2      kuka_iiwa14          0.7891          0.856        0.765        0.8
3      doosan_h2515         0.6234          0.512        0.932        1.0
4      ur5e                 0.5123          0.434        0.721        1.0
5      panda                0.4567          0.389        0.689        0.8
6      doosan_m0609         0.3421          0.267        0.598        1.0
================================================================================
```

## 기여 (Contribution)

이 모듈은 NL2TDL 프레임워크의 **Contribution 2**에 해당합니다:

> "로봇 선택 시 단순 비교가 아닌 **가우시안 함수 기반 최적화**를 통해 '과소비' 문제를 수학적으로 해결하고, 이를 통해 이종 로봇 환경에서 비용 효율적인 로봇 선택이 가능함을 증명"

### 교수님 피드백 대응

**참석자 3, 4의 피드백:**
> "알고리즘이 하나도 없다", "10kg 드는데 30kg 로봇 쓰는 건 과소비 아닌가?"

**해결 방법:**
✅ 가우시안 함수, 지수 함수 등 수학적 알고리즘 적용
✅ 최적 지점(optimal_payload = 1.2 × required) 기반 점수 계산
✅ 과잉 스펙에 대한 명시적 페널티 (DoF 점수 0.8)

## 라이선스

이 프로젝트는 NL2TDL 프레임워크의 일부입니다.

## 문의

- 프로젝트 리포지토리: wicked-ar/20251112_nl2tdl
- 이슈 및 버그 리포트: GitHub Issues

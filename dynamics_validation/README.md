# Dynamics Validation - 동역학 기반 실현가능성 검증

## 개요

TDL(Task Description Language)의 추상적 파라미터를 **로봇의 물리적 한계(토크, 속도, 가속도)** 내에서 실행 가능한 실제 파라미터로 변환하는 모듈입니다.

### 핵심 문제 해결

**문제**: "서로 다른 로봇의 다이나믹스, 구조 역학이 다르다. 그걸 동일한 파라미터로 제어할 수 있냐?" (피드백 #3)

**해결**: RNEA(Recursive Newton-Euler Algorithm) 기반 토크 계산과 동역학 모델을 활용하여, TDL의 추상적 의도를 각 로봇에 맞게 스케일링합니다.

---

## 아키텍처

```
[NL] → [TDL v1] (추상적 파라미터: Accel=80%)
         ↓
    [Robot Selection]
         ↓
    [Dynamics Validation] ⭐ 이 모듈!
         ├─ Robot DB 로드 (DH, 질량, 관성, τ_max)
         ├─ RNEA 계산 (τ_calc = M(q)q̈ + C(q,q̇)q̇ + G(q))
         ├─ Feasibility Check (τ_calc vs τ_max)
         └─ Parameter Scaling (ScaleFactor 계산)
         ↓
    [TDL v2] (실제 파라미터: q̈=[1.2, 0.9, ...] rad/s²)
         ↓
    [Job Code Generation / Simulation]
```

---

## 핵심 알고리즘

### 동역학 기반 파라미터 스케일링

**Input**: TDL v1의 추상적 의도 (예: `Accel_Percent=80`)

**Process**:

1. **의도 해석**: 80% → 목표 가속도 프로파일 (q̈_desired)
2. **모델 로드**: 선택된 로봇(예: Robot_B)의 동역학 파라미터 로드
   - DH 파라미터, 질량, 관성, 토크 한계 (τ_max)
3. **토크 계산** (RNEA):
   ```
   τ_calc = M(q)q̈ + C(q,q̇)q̇ + G(q)
   ```
4. **검증** (Feasibility Check):
   ```
   if any(|τ_calc[i]| > τ_max[i]):
       Infeasible!
   ```
5. **스케일링** (Parameter Scaling):
   - **Case 1 (Feasible)**: 그대로 사용
     ```
     q̈_actual = q̈_desired
     ScaleFactor = 1.0
     ```
   - **Case 2 (Infeasible)**: 스케일 팩터 계산
     ```
     ScaleFactor = min(τ_max[i] / |τ_calc[i]|)  # 0 < S < 1
     q̈_actual = q̈_desired × ScaleFactor
     ```

**Output**: TDL v2 (로봇별 실제 파라미터)

---

## 파일 구조

```
dynamics_validation/
├── __init__.py                    # 모듈 초기화
├── robot_dynamics_db.py           # 로봇별 동역학 파라미터 DB
├── rnea_calculator.py             # RNEA 토크 계산
├── feasibility_checker.py         # 실현가능성 검증
├── parameter_scaler.py            # 파라미터 스케일링 ⭐
├── test_dynamics_validation.py    # 테스트 스크립트
├── quick_example.py               # 간단한 사용 예제
└── README.md                      # 이 파일
```

---

## 🚀 사용 방법

### 1. 기본 사용 (Quick Example)

```python
from dynamics_validation import scale_tdl

# TDL v1 정의
tdl_v1 = {
    'task': 'pick',
    'object': 'apple',
    'robot': 'Robot_B',
    'accel_percent': 85  # 85% 가속도
}

# 동역학 검증 및 스케일링
result = scale_tdl(
    robot_name='Robot_B',
    tdl_v1=tdl_v1,
    safety_margin=0.9
)

# 결과 확인
print(f"Feasible: {result['feasible']}")
print(f"Scale Factor: {result['scale_factor']}")

# TDL v2 사용
tdl_v2 = result['tdl_v2']
```

**출력 예시**:
```
Feasible: False
Scale Factor: 0.887

⚠ TDL parameters were too aggressive for Robot_B!
✓ Automatically scaled to safe values
```

---

### 2. 상세 사용 (전체 파이프라인)

```python
from dynamics_validation import (
    RobotDynamicsDB,
    RNEACalculator,
    FeasibilityChecker,
    ParameterScaler
)

# 1. 로봇 DB 로드
robot_db = RobotDynamicsDB(robot_name='Robot_A')

# 2. RNEA Calculator 생성
calculator = RNEACalculator(robot_db)

# 3. 토크 계산
import numpy as np
q = np.zeros(6)  # 현재 관절 위치
qd = np.zeros(6)  # 관절 속도
qdd = np.ones(6) * 2.0  # 관절 가속도 (목표)

tau = calculator.calculate_required_torque(q, qd, qdd)

# 4. Feasibility Check
checker = FeasibilityChecker(robot_db, safety_margin=0.9)
result = checker.check_torque_feasibility(tau)

if not result['feasible']:
    print(f"Exceeded joints: {result['exceeded_joints']}")

    # 5. 스케일 팩터 계산
    scale_factor = checker.get_required_scale_factor(tau)
    qdd_scaled = qdd * scale_factor

    print(f"Scale Factor: {scale_factor:.3f}")
    print(f"Scaled Acceleration: {qdd_scaled}")
```

---

### 3. 로봇별 비교

```python
# 동일한 TDL v1을 여러 로봇에 적용
tdl_v1 = {'task': 'move', 'accel_percent': 80}

for robot_name in ['Robot_A', 'Robot_B', 'ABB_IRB140']:
    result = scale_tdl(robot_name=robot_name, tdl_v1=tdl_v1)

    print(f"{robot_name}: Scale={result['scale_factor']:.3f}")
```

**출력 예시**:
```
Robot_A: Scale=0.921
Robot_B: Scale=0.887
ABB_IRB140: Scale=0.956
```

→ 같은 TDL이라도 로봇마다 다른 스케일 팩터 적용!

---

## 📋 로봇 데이터베이스

### 지원하는 로봇

| 로봇 이름 | 실제 모델 | DOF | 최대 토크 (N·m) | 비고 |
|-----------|-----------|-----|----------------|------|
| `Robot_A` | UR5e (Universal Robots) | 6 | [150, 150, 150, 28, 28, 28] | 협동 로봇 |
| `Robot_B` | Panda (Franka Emika) | 7 | [87, 87, 87, 87, 12, 12, 12] | 7축 협동 로봇 |
| `ABB_IRB140` | ABB IRB 140 | 6 | [200, 200, 100, 50, 50, 30] | 산업용 로봇 |

### URDF에서 로드 (선택적)

```python
# URDF 파일이 있는 경우
robot_db = RobotDynamicsDB(urdf_path="/path/to/robot.urdf")

# 토크 한계, 관성 등이 URDF에서 자동 추출됨
```

**참고**: 현재는 사전 정의된 파라미터를 우선 사용합니다. URDF 로드는 `roboticstoolbox-python` 설치 필요.

---

## 🔧 의존성

### 필수
- `numpy` - 수치 계산

### 선택적 (정확한 RNEA 계산)
- `roboticstoolbox-python` - URDF 로드 및 정확한 동역학 계산
- `spatialmath-python` - 공간 변환

```bash
pip install roboticstoolbox-python spatialmath-python
```

**참고**: 라이브러리가 없어도 fallback 모드로 동작합니다 (간단한 추정 모델 사용).

---

## 📊 검증 결과 예시

### Case 1: Feasible TDL

```
TDL v1: {'task': 'move', 'accel_percent': 40}
Robot: Robot_B

[Torque Requirements]
  Joint 0:  12.45 /  78.30 N·m = 0.16x ✓
  Joint 1:  13.21 /  78.30 N·m = 0.17x ✓
  Joint 2:  11.87 /  78.30 N·m = 0.15x ✓
  Joint 3:  12.03 /  78.30 N·m = 0.15x ✓
  Joint 4:   3.45 /  10.80 N·m = 0.32x ✓
  Joint 5:   3.21 /  10.80 N·m = 0.30x ✓
  Joint 6:   3.09 /  10.80 N·m = 0.29x ✓

✓ All joints within limits
Scale Factor: 1.000 (No scaling needed)
```

### Case 2: Infeasible TDL (스케일링 적용)

```
TDL v1: {'task': 'pick', 'accel_percent': 95}
Robot: Robot_A

[Torque Requirements]
  Joint 0:  85.04 /  135.00 N·m = 0.63x ✓
  Joint 1:  87.69 /  135.00 N·m = 0.65x ✓
  Joint 2:  82.31 /  135.00 N·m = 0.61x ✓
  Joint 3:  31.45 /   25.20 N·m = 1.25x ✗  ← 초과!
  Joint 4:  29.87 /   25.20 N·m = 1.19x ✗  ← 초과!
  Joint 5:  28.12 /   25.20 N·m = 1.12x ✗  ← 초과!

✗ Joints [3, 4, 5] exceed limits
Scale Factor: 0.800 (20% reduction)

→ TDL v2: accel_percent=76% (scaled from 95%)
```

---

## 🎯 통합 예제: 전체 파이프라인

```python
from dynamics_validation import scale_tdl

# Stage 1: NL → TDL v1 (기존 모듈)
tdl_v1 = {
    'task': 'pick',
    'object': 'apple',
    'robot': 'Robot_A',
    'accel_percent': 80
}

# Stage 2: Dynamics Validation (이 모듈)
result = scale_tdl(robot_name='Robot_A', tdl_v1=tdl_v1)

tdl_v2 = result['tdl_v2']
# → {'task': 'pick', 'acceleration': [1.2, 0.9, ...], 'scale_factor': 0.92, ...}

# Stage 3: Job Code Generation
# tdl_v2['acceleration']를 사용하여 실제 Job Code 생성

# Stage 4: Simulation Validation (validation_integration/)
# 실제 동작 확인
```

---

## 🧪 테스트 실행

```bash
cd dynamics_validation

# 1. Quick Example
python quick_example.py

# 2. Full Test Suite
python test_dynamics_validation.py

# 3. 개별 모듈 테스트
python robot_dynamics_db.py
python rnea_calculator.py
python feasibility_checker.py
python parameter_scaler.py
```

---

## 📖 vs. validation_integration (시뮬레이션 검증)

| 항목 | dynamics_validation | validation_integration |
|------|---------------------|------------------------|
| **목적** | 물리적 실현가능성 검증 | 동작 정확성 검증 |
| **입력** | TDL v1 (추상 파라미터) | TDL v2 (실제 파라미터) |
| **출력** | TDL v2 + Feasibility Report | validation.mp4 |
| **방법** | RNEA + 토크/속도/가속도 한계 비교 | MuJoCo 시뮬레이션 + RRT |
| **검증 대상** | 로봇의 물리적 한계 | 경로 계획, 충돌 회피 |
| **실행 시점** | TDL v1 → v2 변환 시 | TDL v2 → 실행 전 |

**통합 워크플로우**:
```
TDL v1 → dynamics_validation → TDL v2 → validation_integration → Video
         (물리적 가능성)                  (동작 정확성)
```

---

## 🔬 수학적 배경

### RNEA (Recursive Newton-Euler Algorithm)

역동역학 방정식:
```
τ = M(q)q̈ + C(q, q̇)q̇ + G(q)
```

여기서:
- `M(q)`: 질량/관성 행렬
- `C(q, q̇)`: 코리올리/원심력 항
- `G(q)`: 중력 항
- `q, q̇, q̈`: 관절 위치, 속도, 가속도
- `τ`: 관절 토크

RNEA는 이 방정식을 **O(n)** 복잡도로 효율적으로 계산합니다.

### 스케일링 팩터 계산

```python
# 각 관절별 토크 사용률
ratio[i] = |τ_calc[i]| / τ_max[i]

# 초과한 관절 중 최대 사용률의 역수
ScaleFactor = 1.0 / max(ratio[i])

# 예: ratio = [0.5, 0.8, 1.2] → ScaleFactor = 1.0 / 1.2 = 0.833
```

---

## 🐛 문제 해결

### ImportError: No module named 'roboticstoolbox'

**해결**: 라이브러리 설치 (선택적)
```bash
pip install roboticstoolbox-python spatialmath-python
```

또는 사전 정의된 파라미터 사용 (fallback 모드).

### ValueError: Unknown robot

**해결**: `robot_dynamics_db.py`의 `PREDEFINED_TORQUE_LIMITS`에 로봇 추가:
```python
PREDEFINED_TORQUE_LIMITS = {
    'My_Robot': {
        'dof': 6,
        'tau_max': np.array([...]),
        ...
    }
}
```

---

## 📚 참고 문서

- `../validation_integration/README.md` - 시뮬레이션 검증
- `../../Basic_Info/Project_Briefing.md` - 프로젝트 전체 구조
- [Robotics Toolbox Documentation](https://petercorke.github.io/robotics-toolbox-python/)
- [RNEA Algorithm](https://en.wikipedia.org/wiki/Newton%E2%80%93Euler_equations)

---

## 🎓 확장 가이드

### 새 로봇 추가

```python
# robot_dynamics_db.py에 추가
PREDEFINED_TORQUE_LIMITS = {
    'My_New_Robot': {
        'name': 'Custom Robot',
        'dof': 6,
        'tau_max': np.array([100, 100, 50, 30, 30, 20]),
        'vel_max': np.array([...]) * np.pi / 180,
        'acc_max': np.array([...]) * np.pi / 180,
    }
}
```

### 속도/가속도 검증 추가

```python
# 현재는 토크 검증이 주요 기능
# 속도/가속도 검증도 지원되며, 필요시 활성화:

result = checker.check_full_feasibility(
    tau_required=tau,
    qd=joint_velocities,    # 속도 검증 활성화
    qdd=joint_accelerations # 가속도 검증 활성화
)
```

---

**작성일**: 2025-11-18
**버전**: 1.0.0
**상태**: ✅ 구현 완료, 테스트 대기

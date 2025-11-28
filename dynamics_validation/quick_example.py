"""
Quick Example - Dynamics Validation 빠른 사용 예제

가장 간단한 사용법을 보여줍니다.
"""

import sys
from pathlib import Path

# 경로 설정
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

from parameter_scaler import scale_tdl

print("=" * 80)
print(" Dynamics Validation - Quick Example")
print("=" * 80)

# 1. TDL v1 정의 (추상적 파라미터)
print("\n[Step 1] Define TDL v1 (Abstract Parameters)")
tdl_v1 = {
    'task': 'pick',
    'object': 'apple',
    'robot': 'Robot_B',
    'accel_percent': 85,  # 85% 가속도 (공격적!)
    'speed_percent': 70
}
print(f"  TDL v1: {tdl_v1}")

# 2. 동역학 검증 및 스케일링
print("\n[Step 2] Validate & Scale with Robot Dynamics")
result = scale_tdl(
    robot_name='Robot_B',
    tdl_v1=tdl_v1,
    safety_margin=0.9
)

# 3. 결과 확인
print("\n[Step 3] Check Result")
print(f"  Original Feasible: {result['feasible']}")
print(f"  Scale Factor: {result['scale_factor']:.3f}")

if result['scaled']:
    print(f"\n  ⚠ TDL parameters were too aggressive for Robot_B!")
    print(f"  ✓ Automatically scaled to safe values")
else:
    print(f"\n  ✓ TDL parameters are safe for Robot_B")

# 4. TDL v2 출력 (실제 파라미터)
print("\n[Step 4] TDL v2 (Robot-Specific Parameters)")
tdl_v2 = result['tdl_v2']
print(f"  Task: {tdl_v2['task']}")
print(f"  Robot: {tdl_v2['robot']}")
print(f"  Acceleration (q̈): {tdl_v2['acceleration'][:3]}... (7 joints)")
print(f"  Scaled: {tdl_v2['scaled']}")
print(f"  Scale Factor: {tdl_v2['scale_factor']:.3f}")

print("\n" + "=" * 80)
print(" Example Complete!")
print("=" * 80)
print("\n📝 TDL v2 can now be used for:")
print("   - Parameter conversion to Job Code")
print("   - Simulation validation")
print("   - Actual robot execution")

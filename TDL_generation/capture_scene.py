"""
MuJoCo 장면 캡처 스크립트

validation_integration 환경을 로드하고 현재 장면을 이미지로 저장합니다.
"""

import sys
from pathlib import Path

# 경로 설정
CURRENT_DIR = Path(__file__).parent
NL2TDL_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(NL2TDL_DIR))

print("=" * 80)
print(" MuJoCo Scene Capture")
print("=" * 80)

# validation_integration에서 환경 로드
print("\n[1/3] Loading MuJoCo environment...")

validation_path = NL2TDL_DIR / "validation_integration"
sys.path.insert(0, str(validation_path))

try:
    from validation_executor import ValidationExecutor

    # 환경 초기화
    executor = ValidationExecutor()
    env = executor.env

    print(f"✓ MuJoCo environment loaded")
    print(f"  Robots: {list(env.get_sim_robots().keys())}")
    print(f"  Parts: {env.part_names}")

except Exception as e:
    print(f"✗ Failed to load environment: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 장면 캡처
print("\n[2/3] Capturing scene...")

try:
    from vision_scene_analyzer import capture_mujoco_scene

    output_path = "scene_capture.png"
    image_path = capture_mujoco_scene(env, output_path=output_path)

    print(f"✓ Scene captured: {image_path}")

except Exception as e:
    print(f"✗ Capture failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 비전 분석 테스트
print("\n[3/3] Testing vision analysis...")

try:
    from vision_scene_analyzer import VisionSceneAnalyzer

    analyzer = VisionSceneAnalyzer()
    result = analyzer.analyze_scene_from_image(image_path)

    if result['success']:
        print(f"✓ Vision analysis successful")
        print(f"\nScene Description:")
        print(f"  {result['description'][:200]}...")

        print(f"\nDetected Objects ({len(result['objects'])}):")
        for obj in result['objects'][:5]:  # 처음 5개만 출력
            print(f"  • {obj.get('name', 'unknown')} ({obj.get('type', 'unknown')})")
            print(f"    Position: {obj.get('position', 'unknown')}")

        if len(result['objects']) > 5:
            print(f"  ... and {len(result['objects']) - 5} more objects")

    else:
        print(f"✗ Vision analysis failed: {result.get('error', 'Unknown error')}")

except Exception as e:
    print(f"✗ Vision analysis error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print(" Capture Complete!")
print("=" * 80)
print(f"\nYou can now run:")
print(f"  python interactive_pipeline_with_vision.py --mode standalone")
print("=" * 80)

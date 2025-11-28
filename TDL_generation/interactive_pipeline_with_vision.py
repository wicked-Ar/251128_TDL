"""
Interactive NL2TDL Pipeline with Vision

MuJoCo 시뮬레이션 장면을 비전으로 인식하고,
인식된 정보를 활용하여 TDL을 생성하는 통합 파이프라인
"""

import sys
import os
import logging
from pathlib import Path

# 경로 설정
CURRENT_DIR = Path(__file__).parent
NL2TDL_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(NL2TDL_DIR))

from nl2tdl_converter import NL2TDLConverter
from vision_scene_analyzer import VisionSceneAnalyzer, capture_mujoco_scene

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisionEnhancedNL2TDL:
    """
    비전 인식 기능이 통합된 NL2TDL 파이프라인

    MuJoCo 환경을 캡처하고 비전으로 분석한 후,
    인식된 물체 정보를 활용하여 더 정확한 TDL을 생성합니다.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize Vision-Enhanced Pipeline

        Args:
            api_key: Gemini API key (선택)
        """
        print("\n" + "=" * 80)
        print(" Vision-Enhanced NL2TDL Pipeline")
        print("=" * 80)

        # NL2TDL Converter 초기화
        print("\n[1/2] Initializing NL2TDL Converter...")
        self.nl2tdl = NL2TDLConverter(api_key=api_key)

        # Vision Analyzer 초기화
        print("[2/2] Initializing Vision Scene Analyzer...")
        self.vision = VisionSceneAnalyzer(api_key=api_key)

        print("\n✓ Pipeline Ready!")

    def analyze_scene(self, env=None, image_path: str = None, image_array=None):
        """
        장면 분석

        Args:
            env: MuJoCo 환경 (선택)
            image_path: 이미지 파일 경로 (선택)
            image_array: 이미지 배열 (선택)

        Returns:
            dict: 비전 분석 결과
        """
        print("\n" + "─" * 80)
        print(" STEP 1: Scene Analysis (Vision)")
        print("─" * 80)

        # MuJoCo 환경에서 캡처
        if env is not None:
            print("\n[Vision] Capturing scene from MuJoCo environment...")
            image_path = capture_mujoco_scene(env, output_path="scene_capture.png")

        # 이미지 분석
        if image_path is not None:
            print(f"[Vision] Analyzing image: {image_path}")
            result = self.vision.analyze_scene_from_image(image_path)
        elif image_array is not None:
            print(f"[Vision] Analyzing image array (shape: {image_array.shape})")
            result = self.vision.analyze_scene_from_array(image_array)
        else:
            raise ValueError("Either env, image_path, or image_array must be provided")

        # 결과 출력
        if result['success']:
            print(f"\n✓ Vision Analysis Complete")
            print(f"\nScene Description:")
            print(f"  {result['description']}")

            print(f"\nDetected Objects ({len(result['objects'])}):")
            for obj in result['objects']:
                name = obj.get('name', 'unknown')
                obj_type = obj.get('type', 'unknown')
                position = obj.get('position', 'unknown')
                confidence = obj.get('confidence', 'medium')

                print(f"  • {name} ({obj_type})")
                print(f"    Position: {position}")
                print(f"    Confidence: {confidence}")

            return result
        else:
            print(f"\n✗ Vision Analysis Failed: {result.get('error', 'Unknown error')}")
            return None

    def generate_tdl_with_vision(self, user_nl: str, scene_analysis: dict = None):
        """
        비전 정보를 활용한 TDL 생성

        Args:
            user_nl: 사용자 자연어 명령
            scene_analysis: 비전 분석 결과 (선택)

        Returns:
            dict: TDL 생성 결과
        """
        print("\n" + "─" * 80)
        print(" STEP 2: TDL Generation (with Vision Context)")
        print("─" * 80)

        print(f"\nUser Input: \"{user_nl}\"")

        # 비전 컨텍스트 추가
        if scene_analysis and scene_analysis.get('success'):
            print("\n[TDL] Adding vision context to prompt...")
            vision_context = self.vision.generate_scene_context_prompt(scene_analysis)

            # NL에 컨텍스트 추가
            enhanced_nl = f"{user_nl}\n\n{vision_context}"
        else:
            print("\n[TDL] No vision context available (proceeding without it)")
            enhanced_nl = user_nl

        # TDL 생성
        print("\n[TDL] Generating TDL code...")
        tdl_result = self.nl2tdl.convert(enhanced_nl)

        if tdl_result['success']:
            print(f"\n✓ TDL Generation Complete")
            print(f"\n{tdl_result['tdl_code']}")

            return tdl_result
        else:
            print(f"\n✗ TDL Generation Failed: {tdl_result.get('error', 'Unknown error')}")
            return None

    def run_interactive_pipeline(self, env=None):
        """
        대화형 파이프라인 실행

        Args:
            env: MuJoCo 환경 (선택)
        """
        print("\n" + "=" * 80)
        print(" Interactive Mode")
        print("=" * 80)
        print("\nCommands:")
        print("  'analyze' - Analyze current scene")
        print("  'generate <command>' - Generate TDL from natural language")
        print("  'full <command>' - Analyze scene + Generate TDL")
        print("  'quit' - Exit")
        print("=" * 80)

        scene_analysis = None

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nExiting pipeline. Goodbye!")
                    break

                elif user_input.lower() == 'analyze':
                    # 장면 분석만 수행
                    if env is None:
                        print("Error: No MuJoCo environment provided. Please provide image_path instead.")
                        continue

                    scene_analysis = self.analyze_scene(env=env)

                elif user_input.lower().startswith('generate '):
                    # TDL 생성만 수행 (기존 비전 분석 활용)
                    command = user_input[9:].strip()
                    if not command:
                        print("Error: Please provide a command after 'generate'")
                        continue

                    self.generate_tdl_with_vision(command, scene_analysis)

                elif user_input.lower().startswith('full '):
                    # 비전 분석 + TDL 생성
                    command = user_input[5:].strip()
                    if not command:
                        print("Error: Please provide a command after 'full'")
                        continue

                    if env is None:
                        print("Error: No MuJoCo environment provided")
                        continue

                    # 장면 분석
                    scene_analysis = self.analyze_scene(env=env)

                    if scene_analysis and scene_analysis.get('success'):
                        # TDL 생성
                        self.generate_tdl_with_vision(command, scene_analysis)

                else:
                    print(f"Unknown command: {user_input}")
                    print("Try 'analyze', 'generate <cmd>', 'full <cmd>', or 'quit'")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                logger.exception("Pipeline error")


def run_standalone_example():
    """
    MuJoCo 환경 없이 테스트 (이미지 파일 사용)
    """
    print("\n" + "=" * 80)
    print(" Vision-Enhanced NL2TDL - Standalone Example")
    print("=" * 80)

    # Pipeline 초기화
    pipeline = VisionEnhancedNL2TDL()

    # 테스트 이미지 경로
    test_image = "scene_capture.png"

    if not os.path.exists(test_image):
        print(f"\nError: Test image not found: {test_image}")
        print("Please capture a MuJoCo scene first or provide a test image.")
        return

    # 1. 장면 분석
    scene_analysis = pipeline.analyze_scene(image_path=test_image)

    # 2. TDL 생성
    if scene_analysis and scene_analysis.get('success'):
        # 예제 명령
        user_command = "Pick the apple and place it in the bin"

        tdl_result = pipeline.generate_tdl_with_vision(user_command, scene_analysis)

        if tdl_result and tdl_result.get('success'):
            print("\n" + "=" * 80)
            print(" Pipeline Complete!")
            print("=" * 80)


def run_with_mujoco():
    """
    MuJoCo 환경과 통합하여 실행
    """
    print("\n" + "=" * 80)
    print(" Vision-Enhanced NL2TDL - MuJoCo Integration")
    print("=" * 80)

    # MuJoCo 환경 로드
    print("\n[Setup] Loading MuJoCo environment...")

    # validation_integration에서 환경 가져오기
    validation_path = Path(__file__).parent.parent / "validation_integration"
    sys.path.insert(0, str(validation_path))

    try:
        from validation_executor import ValidationExecutor

        # 환경 초기화
        executor = ValidationExecutor()
        env = executor.env

        print(f"✓ MuJoCo environment loaded")
        print(f"  Robots: {list(env.get_sim_robots().keys())}")
        print(f"  Parts: {env.part_names}")

        # Pipeline 초기화
        pipeline = VisionEnhancedNL2TDL()

        # 대화형 모드 실행
        pipeline.run_interactive_pipeline(env=env)

    except Exception as e:
        print(f"\n✗ Failed to load MuJoCo environment: {e}")
        print("Falling back to standalone mode...")
        run_standalone_example()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Vision-Enhanced NL2TDL Pipeline')
    parser.add_argument('--mode', choices=['standalone', 'mujoco', 'interactive'],
                        default='standalone',
                        help='Execution mode')
    parser.add_argument('--image', type=str, default='scene_capture.png',
                        help='Test image path (for standalone mode)')

    args = parser.parse_args()

    try:
        if args.mode == 'mujoco':
            run_with_mujoco()
        elif args.mode == 'interactive':
            run_with_mujoco()
        else:
            run_standalone_example()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()

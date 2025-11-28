"""
Master Pipeline - Complete NL2TDL2Validation Pipeline

자연어 → TDL v1 → 로봇 선택 → 동역학 검증 → TDL v2 → 시뮬레이션 검증

전체 파이프라인:
1. Ground Truth TSD (선택) - MuJoCo Observation → Textual State Description
2. NL → TDL v1 (Gemini LLM + TSD context)
3. Robot Selection - 최적 로봇 선택
4. Dynamics Validation - 물리적 실현가능성 검증
5. TDL v2 생성 - 로봇별 실제 파라미터
6. Simulation Validation (validation_integration) - Roco 시뮬레이션 + 비디오
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# 경로 설정
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MasterPipeline:
    """
    NL2TDL 완전 통합 파이프라인

    자연어 명령을 받아 TDL 생성, 검증, 시뮬레이션 실행까지 전체 과정 수행
    """

    def __init__(self, use_tsd: bool = True, api_key: str = None):
        """
        Args:
            use_tsd: Ground Truth TSD 생성 사용 여부 (기본: True)
            api_key: Gemini API 키 (선택)
        """
        print("=" * 80)
        print(" NL2TDL Master Pipeline - Initializing")
        print("=" * 80)

        self.use_tsd = use_tsd
        self.history = []

        # 1. NL2TDL Converter
        print("\n[1/5] Loading NL2TDL Converter...")
        from TDL_generation.nl2tdl_converter import NL2TDLConverter
        self.nl2tdl = NL2TDLConverter(api_key=api_key)

        # 2. TSD Generator (Ground Truth State Description)
        if use_tsd:
            print("[2/5] Loading Ground Truth TSD Generator...")
            from TDL_generation.state_to_text_generator import StateToTextGenerator
            self.tsd_generator = StateToTextGenerator()
        else:
            print("[2/5] TSD disabled (using NL only)")
            self.tsd_generator = None

        # 3. Robot Selector
        print("[3/5] Loading Robot Selector...")
        from robot_selection.robot_selector import select_best_robot
        self.select_robot_func = select_best_robot

        # 4. Dynamics Validator
        print("[4/5] Loading Dynamics Validator...")
        from dynamics_validation.parameter_scaler import ParameterScaler
        from dynamics_validation.robot_dynamics_db import load_robot

        # 로봇 DB는 나중에 로봇 선택 후 로드
        self.dynamics_validator = None

        # 5. Simulation Validator (PyBullet)
        print("[5/5] Loading Simulation Validator (PyBullet)...")
        from pybullet_adapter import PyBulletExecutor
        self.sim_validator = PyBulletExecutor(render=True)  # GUI mode for video recording

        print("\n[OK] Master Pipeline Ready!")
        print("=" * 80)

    def execute_full_pipeline(self,
                            user_nl: str,
                            robot_requirements: Dict = None,
                            output_video: str = None,
                            enable_dynamics: bool = True) -> Dict:
        """
        전체 파이프라인 실행

        Args:
            user_nl: 사용자 자연어 명령
            robot_requirements: 로봇 요구사항 (선택)
            output_video: 출력 비디오 경로 (기본값: auto)
            enable_dynamics: 동역학 검증 활성화 여부

        Returns:
            dict: 전체 실행 결과
        """
        print("\n" + "=" * 80)
        print(f" Executing Pipeline: \"{user_nl}\"")
        print("=" * 80)

        result = {
            'user_nl': user_nl,
            'timestamp': datetime.now().isoformat(),
            'success': False
        }

        # Step 1: Ground Truth TSD Generation (선택적)
        scene_context = None
        if self.use_tsd and self.tsd_generator:
            print("\n" + "-" * 80)
            print(" STEP 1: Ground Truth TSD Generation")
            print("-" * 80)

            try:
                # PyBullet에서 현재 씬 상태 추출
                scene_context = self.sim_validator.get_scene_description()

                # 결과 저장
                result['tsd_analysis'] = {'context': scene_context}
                print("[OK] TSD context generated from PyBullet scene")
                print(f"\n{scene_context}\n")

            except Exception as e:
                print(f"[X] TSD generation error: {e}")
                logger.exception("TSD generation failed")
                scene_context = None

        # Step 2: NL → TDL v1
        print("\n" + "-" * 80)
        print(" STEP 2: NL → TDL v1 Generation")
        print("-" * 80)

        try:
            # 비전 컨텍스트 추가
            enhanced_nl = user_nl
            if scene_context:
                enhanced_nl = f"{user_nl}\n\n{scene_context}"

            tdl_result = self.nl2tdl.convert_with_metadata(enhanced_nl)

            # convert_with_metadata returns {'tdl_code': ..., 'metadata': ...}
            if 'tdl_code' not in tdl_result:
                result['error'] = "TDL generation failed: No TDL code returned"
                return result

            tdl_v1 = tdl_result['tdl_code']
            print(f"[OK] TDL v1 generated ({len(tdl_v1)} chars)")
            result['tdl_v1'] = tdl_v1

            # TDL 파일 저장
            try:
                output_dir = Path(CURRENT_DIR) / "TDL_generation" / "output"
                output_dir.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                tdl_filename = f"tdl_v1_{timestamp}.txt"
                tdl_filepath = output_dir / tdl_filename

                with open(tdl_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# Generated TDL v1\n")
                    f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"# User Command: {user_nl}\n")
                    f.write(f"# {'='*60}\n\n")
                    f.write(tdl_v1)

                print(f"  TDL saved to: {tdl_filepath}")
                result['tdl_filepath'] = str(tdl_filepath)
            except Exception as e:
                logger.warning(f"Failed to save TDL file: {e}")

        except Exception as e:
            result['error'] = f"TDL generation error: {e}"
            logger.exception("TDL generation failed")
            return result

        # Step 3: Robot Selection
        print("\n" + "-" * 80)
        print(" STEP 3: Robot Selection")
        print("-" * 80)

        try:
            # select_best_robot returns Tuple[str, float, Dict]
            robot_id, confidence, all_scores = self.select_robot_func(
                tdl_v1_content=tdl_v1,
                robot_db_path=None,
                weights=robot_requirements if robot_requirements else None
            )

            # Build robot_result structure
            robot_result = {
                'selected_robot': {
                    'id': robot_id,
                    'name': robot_id,
                    'confidence': confidence
                },
                'reason': f"Selected with confidence {confidence:.2f}",
                'all_scores': all_scores
            }

            selected_robot = robot_result['selected_robot']
            print(f"[OK] Selected Robot: {selected_robot['name']}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  Reason: {robot_result['reason']}")
            result['robot_selection'] = robot_result

        except Exception as e:
            result['error'] = f"Robot selection error: {e}"
            logger.exception("Robot selection failed")
            return result

        # Step 4: Dynamics Validation (선택적)
        tdl_v2 = None
        if enable_dynamics:
            print("\n" + "-" * 80)
            print(" STEP 4: Dynamics Validation")
            print("-" * 80)

            try:
                from dynamics_validation.parameter_scaler import ParameterScaler
                from dynamics_validation.robot_dynamics_db import load_robot

                # 선택된 로봇의 동역학 DB 로드
                robot_name_mapping = {
                    'Robot_A': 'Robot_A',  # UR5e
                    'Robot_B': 'Robot_B',  # Panda
                }

                robot_db_name = robot_name_mapping.get(selected_robot['name'], 'Robot_A')
                robot_db = load_robot(robot_name=robot_db_name)

                # 동역학 검증
                scaler = ParameterScaler(robot_db, safety_margin=0.9)

                # TDL v1에서 파라미터 추출 (간단한 예시)
                tdl_params = self._extract_tdl_parameters(tdl_v1)

                scaling_result = scaler.scale_tdl_parameters(tdl_params)

                print(f"[OK] Dynamics validation complete")
                print(f"  Feasible: {scaling_result['feasible']}")
                print(f"  Scale Factor: {scaling_result['scale_factor']:.3f}")

                result['dynamics_validation'] = scaling_result
                tdl_v2 = scaling_result['tdl_v2']

            except Exception as e:
                print(f"[!] Dynamics validation skipped: {e}")
                logger.warning("Dynamics validation failed, continuing without it")
                tdl_v2 = None

        # Step 5: Simulation Validation (PyBullet)
        print("\n" + "-" * 80)
        print(" STEP 5: Simulation Validation (PyBullet)")
        print("-" * 80)

        try:
            # Map robot selector ID to PyBullet robot name
            robot_to_pybullet_mapping = {
                'panda': 'Robot_B',
                'ur5e': 'Robot_A',
                'Robot_A': 'Robot_A',
                'Robot_B': 'Robot_B',
                'doosan_m0609': 'Robot_A',
                'doosan_m1013': 'Robot_A',
                'doosan_h2515': 'Robot_A',
                'kuka_iiwa14': 'Robot_A',
            }
            pybullet_robot = robot_to_pybullet_mapping.get(selected_robot['name'], 'Robot_A')
            print(f"  Mapping {selected_robot['name']} -> {pybullet_robot}")

            # TDL v1을 간단한 형식으로 변환
            tdl_dict = self._tdl_to_dict(tdl_v1, pybullet_robot, user_nl)

            # PyBullet 실행 계획 생성
            plan_text = f"{pybullet_robot} {tdl_dict['task']} {tdl_dict['object']}"
            print(f"  Generated plan: {plan_text}")

            # 비디오 경로 생성
            if output_video is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_video = f"simulation_{timestamp}.mp4"

            # PyBullet 실행 (비디오 녹화 포함)
            success, message = self.sim_validator.execute_plan(
                plan_text,
                record_video=True,
                video_path=output_video
            )

            if success:
                print(f"\n[OK] Simulation validation SUCCESS!")
                print(f"  Message: {message}")
                print(f"  Video saved: {output_video}")

                result['success'] = True
                result['simulation'] = {
                    'success': True,
                    'message': message,
                    'plan': plan_text,
                    'video_path': output_video
                }
            else:
                print(f"\n[X] Simulation validation FAILED")
                print(f"  Error: {message}")

                result['error'] = f"Simulation failed: {message}"
                result['simulation'] = {
                    'success': False,
                    'error': message,
                    'plan': plan_text
                }

        except Exception as e:
            result['error'] = f"Simulation error: {e}"
            logger.exception("Simulation validation failed")
            return result

        # 히스토리 저장
        self.history.append(result)

        return result

    def _extract_tdl_parameters(self, tdl_code: str) -> Dict:
        """
        TDL 코드에서 파라미터 추출 (간단한 파서)

        실제로는 더 정교한 파싱이 필요하지만, 여기서는 간단히 구현
        """
        # 기본값
        params = {
            'task': 'pick',
            'accel_percent': 50,
            'speed_percent': 50
        }

        # TDL에서 숫자 추출 시도
        import re

        # 속도/가속도 파라미터 찾기
        velocity_match = re.search(r'velocity[:\s]+(\d+)', tdl_code, re.IGNORECASE)
        if velocity_match:
            params['speed_percent'] = int(velocity_match.group(1))

        accel_match = re.search(r'accel[eration]*[:\s]+(\d+)', tdl_code, re.IGNORECASE)
        if accel_match:
            params['accel_percent'] = int(accel_match.group(1))

        return params

    def _tdl_to_dict(self, tdl_code: str, robot_name: str, user_nl: str = "") -> Dict:
        """
        TDL 코드를 validation_integration이 기대하는 딕셔너리 형식으로 변환

        Args:
            tdl_code: TDL 코드 (LLM 생성 또는 fallback)
            robot_name: 로봇 이름
            user_nl: 원본 자연어 명령 (fallback용)

        Returns:
            TDL 딕셔너리
        """
        import re

        # 기본 TDL 딕셔너리
        tdl_dict = {
            'task': 'pick',
            'object': 'apple',
            'robot': robot_name,
            'speed': 50
        }

        # TDL 코드에서 정보 추출
        # Pick, Place, Move 등 찾기
        search_text = tdl_code.lower()
        if 'pick' in search_text:
            tdl_dict['task'] = 'pick'
        elif 'place' in search_text:
            tdl_dict['task'] = 'place'
            tdl_dict['location'] = 'bin'  # 기본값
        elif 'move' in search_text:
            tdl_dict['task'] = 'move'
            tdl_dict['location'] = 'home'
        elif 'inspect' in search_text:
            tdl_dict['task'] = 'inspect'

        # 물체 이름 추출 - 우선순위:
        # 1. TDL 코드에서 찾기
        # 2. 원본 NL에서 찾기 (TDL 생성 실패 시)
        # 3. TSD context에서 찾기 (환경에 있는 물체)
        objects = ['apple', 'banana', 'milk', 'bread', 'soda', 'part']

        # 1. TDL 코드에서 찾기
        found = False
        for obj in objects:
            if obj in search_text:
                tdl_dict['object'] = obj
                found = True
                break

        # 2. TDL에서 못 찾았으면 원본 NL에서 찾기
        if not found and user_nl:
            nl_lower = user_nl.lower()
            for obj in objects:
                if obj in nl_lower:
                    tdl_dict['object'] = obj
                    print(f"  [Fallback] Object '{obj}' extracted from NL command")
                    found = True
                    break

        # 3. 여전히 못 찾았으면 경고 출력
        if not found:
            print(f"  [WARNING] No object found in TDL or NL. Using default: {tdl_dict['object']}")

        return tdl_dict

    def run_interactive(self):
        """대화형 모드 실행"""
        print("\n" + "=" * 80)
        print(" Interactive Mode - Master Pipeline")
        print("=" * 80)
        print("\nCommands:")
        print("  /help     - Show help")
        print("  /tsd      - Toggle TSD (Ground Truth State) on/off")
        print("  /dynamics - Toggle dynamics validation on/off")
        print("  /history  - Show execution history")
        print("  /quit     - Exit")
        print("\nOr enter natural language command:")
        print("=" * 80)

        enable_dynamics = True

        while True:
            try:
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['/quit', '/exit', '/q']:
                    print("\nExiting. Goodbye!")
                    break

                elif user_input.lower() == '/help':
                    self._print_help()

                elif user_input.lower() == '/tsd':
                    self.use_tsd = not self.use_tsd
                    print(f"TSD (Ground Truth State): {'ON' if self.use_tsd else 'OFF'}")

                elif user_input.lower() == '/dynamics':
                    enable_dynamics = not enable_dynamics
                    print(f"Dynamics Validation: {'ON' if enable_dynamics else 'OFF'}")

                elif user_input.lower() == '/history':
                    self._print_history()

                else:
                    # 자연어 명령 실행
                    result = self.execute_full_pipeline(
                        user_nl=user_input,
                        enable_dynamics=enable_dynamics
                    )

                    print("\n" + "=" * 80)
                    if result['success']:
                        print(" [OK] PIPELINE SUCCESS!")
                        print("=" * 80)
                        print(f"\n  Message: {result['simulation'].get('message', 'Success')}")
                        print(f"  Plan: {result['simulation'].get('plan', 'N/A')}")
                        if 'video_path' in result['simulation']:
                            print(f"  Video: {result['simulation']['video_path']}")
                    else:
                        print(" [X] PIPELINE FAILED")
                        print("=" * 80)
                        print(f"\n  Error: {result.get('error', 'Unknown error')}")
                    print("=" * 80)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                logger.exception("Interactive mode error")

    def _print_help(self):
        """도움말 출력"""
        print("\n" + "-" * 80)
        print(" Help - Master Pipeline Commands")
        print("-" * 80)
        print("\nPipeline Steps:")
        print("  1. Ground Truth TSD (optional) - MuJoCo Observation → Text State")
        print("  2. NL → TDL v1 - Generate task description (with TSD context)")
        print("  3. Robot Selection - Choose best robot")
        print("  4. Dynamics Validation (optional) - Check physics feasibility")
        print("  5. Simulation - Execute in Roco and save video")
        print("\nExample Commands:")
        print('  "Pick the apple and place it in the bin"')
        print('  "Move the banana to the left side"')
        print('  "Inspect the milk container"')
        print("-" * 80)

    def _print_history(self):
        """실행 히스토리 출력"""
        if not self.history:
            print("\nNo execution history yet.")
            return

        print("\n" + "-" * 80)
        print(f" Execution History ({len(self.history)} items)")
        print("-" * 80)

        for i, item in enumerate(self.history, 1):
            status = "[OK]" if item['success'] else "[X]"
            print(f"\n{i}. {status} {item['user_nl']}")
            print(f"   Time: {item['timestamp']}")
            if item['success']:
                print(f"   Plan: {item['simulation'].get('plan', 'N/A')}")
                print(f"   Message: {item['simulation'].get('message', 'Success')}")
            else:
                print(f"   Error: {item.get('error', 'Unknown')}")

        print("-" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='NL2TDL Master Pipeline')
    parser.add_argument('--tsd', action='store_true', default=True, help='Enable Ground Truth TSD generation (default: True)')
    parser.add_argument('--no-tsd', action='store_true', help='Disable Ground Truth TSD generation')
    parser.add_argument('--command', type=str, help='Execute single command')
    parser.add_argument('--no-dynamics', action='store_true', help='Disable dynamics validation')

    args = parser.parse_args()

    try:
        # TSD 플래그 결정 (--no-tsd가 있으면 False, 없으면 True)
        use_tsd = not args.no_tsd

        # 파이프라인 초기화
        pipeline = MasterPipeline(use_tsd=use_tsd)

        if args.command:
            # 단일 명령 실행
            result = pipeline.execute_full_pipeline(
                user_nl=args.command,
                enable_dynamics=not args.no_dynamics
            )

            if result['success']:
                print(f"\n[OK] SUCCESS")
                print(f"Plan: {result['simulation'].get('plan', 'N/A')}")
                print(f"Message: {result['simulation'].get('message', 'Success')}")
                if 'video_path' in result['simulation']:
                    print(f"Video: {result['simulation']['video_path']}")
                sys.exit(0)
            else:
                print(f"\n[X] FAILED: {result.get('error')}")
                sys.exit(1)
        else:
            # 대화형 모드
            pipeline.run_interactive()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

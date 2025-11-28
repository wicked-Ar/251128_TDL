# interactive_full_pipeline.py
"""
Interactive Full Pipeline: NL → TDL v1 → Robot Selection
사용자가 자연어 명령을 입력하면 TDL 코드를 생성하고 최적의 로봇을 자동으로 선택
"""

import os
import sys
import logging
from datetime import datetime

# 상위 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TDL_generation.nl2tdl_converter import NL2TDLConverter
from robot_selection.robot_selector import select_best_robot, print_selection_report

# Import path planner
try:
    from path_planning.path_planner import PathPlanner
    PATH_PLANNER_AVAILABLE = True
except ImportError:
    PATH_PLANNER_AVAILABLE = False
    print("[WARNING] Path planner not available. Install required dependencies.")

# 로깅 설정
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


def print_banner():
    """시작 배너 출력"""
    print("\n" + "="*80)
    print("  NL2TDL FULL PIPELINE - INTERACTIVE MODE")
    print("  Natural Language → TDL v1 → Robot Selection")
    print("="*80 + "\n")


def print_help():
    """도움말 출력"""
    print("\n" + "-"*80)
    print("AVAILABLE COMMANDS:")
    print("  /help     - Show this help message")
    print("  /examples - Show example instructions with requirements")
    print("  /robots   - Show available robots")
    print("  /history  - Show conversion history")
    print("  /last     - Show last generated TDL and selected robot")
    print("  /save     - Save last generated TDL to file")
    print("  /path     - Generate path planning for last TDL")
    print("  /clear    - Clear screen")
    print("  /quit     - Exit the program")
    print("\nOtherwise, type your natural language instruction and press Enter.")
    print("You can include requirements like: PAYLOAD_KG: 15.0, REQUIRED_REACH_M: 1.2")
    print("-"*80 + "\n")


def print_examples():
    """예제 명령어 출력"""
    print("\n" + "-"*80)
    print("EXAMPLE INSTRUCTIONS (with implicit requirements):")
    print("-"*80)

    examples = [
        {
            "title": "1. Light Assembly (3kg)",
            "instruction": "Pick up a 3kg component at (300, 0, 100) and place it at (300, 200, 100).",
            "implicit_req": "PAYLOAD_KG: 3.0, REACH: ~0.3m"
        },
        {
            "title": "2. Medium Welding (15kg)",
            "instruction": "Weld a 15kg steel plate at position (1200, 300, 500) then return home.",
            "implicit_req": "PAYLOAD_KG: 15.0, REACH: ~1.2m"
        },
        {
            "title": "3. Heavy Material Handling (25kg)",
            "instruction": "Pick up a 25kg box from (1400, 0, 100) and place on pallet at (1400, 600, 100).",
            "implicit_req": "PAYLOAD_KG: 25.0, REACH: ~1.4m"
        },
        {
            "title": "4. Precision Task (5kg, Long Reach)",
            "instruction": "Move a 5kg tool to inspection position at (1500, 0, 300) and take measurement.",
            "implicit_req": "PAYLOAD_KG: 5.0, REACH: 1.5m"
        },
    ]

    for example in examples:
        print(f"\n{example['title']}:")
        print(f"  Instruction: {example['instruction']}")
        print(f"  Implicit Requirements: {example['implicit_req']}")

    print("\n" + "-"*80)
    print("TIP: You can also explicitly add requirements in your instruction:")
    print("     'Pick up 10kg object. PAYLOAD_KG: 10.0, REQUIRED_REACH_M: 1.0'")
    print("-"*80 + "\n")


def show_available_robots():
    """사용 가능한 로봇 목록 출력"""
    import json

    robot_db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'robot_selection', 'data', 'robot_db.json'
    )

    try:
        with open(robot_db_path, 'r', encoding='utf-8') as f:
            robot_db = json.load(f)

        print("\n" + "-"*80)
        print("AVAILABLE ROBOTS:")
        print("-"*80)
        print(f"\n{'Robot ID':<20} {'Payload':<12} {'Reach':<12} {'DoF':<8} {'Manufacturer':<20}")
        print("-"*80)

        for robot_id, specs in robot_db.items():
            print(
                f"{robot_id:<20} "
                f"{specs['payload']:.1f} kg      "
                f"{specs['reach']:.3f} m     "
                f"{specs['dof']:<8} "
                f"{specs.get('manufacturer', 'N/A'):<20}"
            )

        print("-"*80 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Failed to load robot database: {e}\n")


def run_path_planning(tdl_code, robot_id):
    """경로 계획 실행"""
    if not PATH_PLANNER_AVAILABLE:
        print("\n[ERROR] Path planner not available. Please install required dependencies.")
        return None

    print("\n" + "="*80)
    print("STEP 3: Generating Path Planning...")
    print("="*80)

    try:
        # Initialize path planner with enhanced planning
        planner = PathPlanner(
            output_dir=os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'path_planning', 'output'
            ),
            use_advanced_planning=True,
            enable_collision_checking=False
        )

        # Generate trajectory
        trajectory_path = planner.plan_trajectory(tdl_code, robot_id)

        print("\n" + "-"*80)
        print(f"[OK] Trajectory generated successfully!")
        print(f"     Saved to: {trajectory_path}")
        print("-"*80)

        return trajectory_path

    except Exception as e:
        print(f"\n[ERROR] Path planning failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_results(nl_instruction, tdl_code, robot_id, score, trajectory_path=None):
    """결과를 파일로 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"pipeline_output_{timestamp}"

    print(f"\nEnter filename prefix (default: {default_filename}): ", end='')
    filename_prefix = input().strip()

    if not filename_prefix:
        filename_prefix = default_filename

    # 출력 디렉토리 생성
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'TDL_generation', 'output'
    )
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 1. TDL 파일 저장
        tdl_path = os.path.join(output_dir, f"{filename_prefix}.tdl")
        with open(tdl_path, 'w', encoding='utf-8') as f:
            f.write(tdl_code)

        # 2. 메타데이터 저장
        meta_path = os.path.join(output_dir, f"{filename_prefix}_metadata.txt")
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("NL2TDL FULL PIPELINE - OUTPUT METADATA\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"Natural Language Input:\n")
            f.write("-"*80 + "\n")
            f.write(f"{nl_instruction}\n")
            f.write("-"*80 + "\n\n")
            f.write(f"Selected Robot: {robot_id}\n")
            f.write(f"Selection Score: {score:.4f}\n\n")
            f.write(f"TDL Code saved to: {tdl_path}\n")
            if trajectory_path:
                f.write(f"Trajectory saved to: {trajectory_path}\n")
            f.write("="*80 + "\n")

        print(f"\n[OK] Results saved:")
        print(f"     TDL Code: {tdl_path}")
        print(f"     Metadata: {meta_path}")
        if trajectory_path:
            print(f"     Trajectory: {trajectory_path}")

    except Exception as e:
        print(f"\n[ERROR] Failed to save: {e}")


def clear_screen():
    """화면 클리어"""
    os.system('cls' if os.name == 'nt' else 'clear')


def interactive_mode():
    """인터랙티브 모드 메인 루프"""
    print_banner()
    print("Initializing NL2TDL Converter and Robot Selector...")

    try:
        converter = NL2TDLConverter()
        print("[OK] Systems initialized successfully!\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return

    print_help()

    # 세션 데이터
    history = []
    last_result = None

    while True:
        try:
            # 사용자 입력 받기
            print("\n" + "="*80)
            print("Enter your instruction (or /help for commands):")
            print("-"*80)
            user_input = input("> ").strip()

            # 빈 입력 처리
            if not user_input:
                print("\n[WARNING] Empty input. Type /help for available commands.")
                continue

            # 명령어 처리
            if user_input.startswith('/'):
                command = user_input.lower()

                if command in ['/quit', '/exit', '/q']:
                    print("\n" + "="*80)
                    print("Thank you for using NL2TDL Full Pipeline!")
                    print("="*80 + "\n")
                    break

                elif command in ['/help', '/h']:
                    print_help()

                elif command in ['/examples', '/ex']:
                    print_examples()

                elif command in ['/robots', '/r']:
                    show_available_robots()

                elif command == '/history':
                    if not history:
                        print("\n[INFO] No conversion history yet.")
                    else:
                        print("\n" + "-"*80)
                        print("CONVERSION HISTORY:")
                        print("-"*80)
                        for i, (nl, robot, score) in enumerate(history, 1):
                            print(f"\n{i}. Input: {nl[:50]}...")
                            print(f"   Robot: {robot} (score: {score:.4f})")
                        print("-"*80)

                elif command in ['/last', '/l']:
                    if last_result:
                        nl, tdl, robot, score = last_result
                        print("\n" + "-"*80)
                        print("LAST RESULT:")
                        print("-"*80)
                        print(f"Input: {nl}")
                        print(f"Robot: {robot} (score: {score:.4f})")
                        print("\nTDL Code:")
                        print(tdl)
                        print("-"*80)
                    else:
                        print("\n[INFO] No results yet. Generate TDL first.")

                elif command in ['/save', '/s']:
                    if last_result:
                        nl, tdl, robot, score = last_result
                        # Check if trajectory exists in last_result
                        trajectory = last_result[4] if len(last_result) > 4 else None
                        save_results(nl, tdl, robot, score, trajectory)
                    else:
                        print("\n[ERROR] No results to save. Generate TDL first.")

                elif command in ['/path', '/p']:
                    if last_result:
                        nl, tdl, robot, score = last_result[:4]
                        trajectory_path = run_path_planning(tdl, robot)
                        if trajectory_path:
                            # Update last_result with trajectory
                            last_result = (nl, tdl, robot, score, trajectory_path)
                    else:
                        print("\n[ERROR] No results yet. Generate TDL first.")

                elif command in ['/clear', '/cls']:
                    clear_screen()
                    print_banner()

                else:
                    print(f"\n[ERROR] Unknown command: {user_input}")
                    print("Type /help to see available commands.")

                continue

            # 자연어 명령어 처리
            nl_instruction = user_input

            print("\n" + "="*80)
            print("STEP 1: Converting Natural Language to TDL v1...")
            print("="*80)

            # TDL 생성
            tdl_code = converter.convert(nl_instruction, temperature=0.1)

            if tdl_code.startswith("// ERROR"):
                print("\n[ERROR] TDL generation failed:")
                print(tdl_code)
                continue

            # TDL 출력
            print("\n" + "-"*80)
            print("GENERATED TDL (v1):")
            print("-"*80)
            print(tdl_code)
            print("-"*80)

            print("\n" + "="*80)
            print("STEP 2: Selecting Optimal Robot...")
            print("="*80)

            # 로봇 선택
            try:
                best_robot_id, best_score, all_scores = select_best_robot(tdl_code)

                # 결과 출력
                print_selection_report(best_robot_id, all_scores)

                # 히스토리 저장
                history.append((nl_instruction[:50], best_robot_id, best_score))
                last_result = (nl_instruction, tdl_code, best_robot_id, best_score)

                # Ask if user wants to run path planning
                print("\n" + "="*80)
                print("PIPELINE STEP 1-2 COMPLETED!")
                print("="*80)
                print(f"\n1. Natural Language → TDL v1: [OK]")
                print(f"2. Robot Selection: [OK] → {best_robot_id} (score: {best_score:.4f})")

                if PATH_PLANNER_AVAILABLE:
                    print("\n" + "-"*80)
                    print("Would you like to generate path planning now? (y/n): ", end='')
                    run_path = input().strip().lower()

                    trajectory_path = None
                    if run_path in ['y', 'yes']:
                        trajectory_path = run_path_planning(tdl_code, best_robot_id)
                        if trajectory_path:
                            last_result = (nl_instruction, tdl_code, best_robot_id, best_score, trajectory_path)
                    else:
                        print("\nYou can run path planning later with /path command.")
                else:
                    trajectory_path = None
                    print("\n[INFO] Path planner not available. Skipping path planning step.")

                print("\n" + "="*80)
                print("AVAILABLE NEXT STEPS:")
                print("="*80)
                print("  /path  - Generate path planning (if not done)")
                print("  /save  - Save all results to files")
                print("  /last  - Review generated TDL and robot selection")
                print("\nFuture steps:")
                print("  3. Parameter Conversion (TDL v1 → TDL v2)")
                print("  4. Job Code Generation (TDL v2 → DRL)")
                print("="*80)

            except Exception as e:
                print(f"\n[ERROR] Robot selection failed: {e}")
                import traceback
                traceback.print_exc()

        except KeyboardInterrupt:
            print("\n\n[INFO] Interrupted by user. Type /quit to exit.")
            continue

        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")
            import traceback
            traceback.print_exc()
            continue


def batch_mode(nl_instruction):
    """배치 모드 (단일 명령어 실행)"""
    print_banner()
    print(f"Input: {nl_instruction}\n")

    try:
        # TDL 생성
        converter = NL2TDLConverter()
        tdl_code = converter.convert(nl_instruction)

        print("="*80)
        print("GENERATED TDL CODE:")
        print("="*80)
        print(tdl_code)
        print("="*80)

        # 로봇 선택
        if not tdl_code.startswith("// ERROR"):
            print("\n" + "="*80)
            print("SELECTING OPTIMAL ROBOT...")
            print("="*80)

            best_robot_id, best_score, all_scores = select_best_robot(tdl_code)
            print_selection_report(best_robot_id, all_scores)

            print(f"\n[OK] Pipeline completed: {best_robot_id} selected (score: {best_score:.4f})")
            return tdl_code, best_robot_id, best_score

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def main():
    """메인 함수"""
    # 명령줄 인자 처리
    if len(sys.argv) > 1:
        # 배치 모드: 명령줄 인자를 자연어 명령어로 사용
        nl_instruction = ' '.join(sys.argv[1:])
        batch_mode(nl_instruction)
    else:
        # 인터랙티브 모드
        interactive_mode()


if __name__ == "__main__":
    main()

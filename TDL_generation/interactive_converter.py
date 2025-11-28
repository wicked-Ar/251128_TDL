# interactive_converter.py
"""
Interactive NL2TDL Converter
사용자가 실시간으로 자연어 명령을 입력하고 TDL 코드를 생성할 수 있는 인터랙티브 인터페이스
"""

import os
import sys
import logging
from datetime import datetime
from nl2tdl_converter import NL2TDLConverter

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def print_banner():
    """시작 배너 출력"""
    print("\n" + "="*80)
    print("  NL2TDL INTERACTIVE CONVERTER")
    print("  Natural Language → Task Description Language (TDL)")
    print("="*80 + "\n")


def print_help():
    """도움말 출력"""
    print("\n" + "-"*80)
    print("AVAILABLE COMMANDS:")
    print("  /help     - Show this help message")
    print("  /examples - Show example natural language instructions")
    print("  /history  - Show conversion history")
    print("  /clear    - Clear screen")
    print("  /save     - Save last generated TDL to file")
    print("  /quit     - Exit the program")
    print("\nOtherwise, type your natural language instruction and press Enter.")
    print("-"*80 + "\n")


def print_examples():
    """예제 명령어 출력"""
    print("\n" + "-"*80)
    print("EXAMPLE NATURAL LANGUAGE INSTRUCTIONS:")
    print("-"*80)

    examples = [
        {
            "title": "1. Simple Pick and Place",
            "instruction": "Pick up an object at position (300, 0, 100) and place it at (300, 200, 100)."
        },
        {
            "title": "2. Welding Task",
            "instruction": "Move to welding position (1000, 500, 300), perform welding for 5 seconds, then return home."
        },
        {
            "title": "3. Assembly Task",
            "instruction": "Pick up the component at (200, 100, 50), move it to assembly position (400, 100, 50), and tighten the screw."
        },
        {
            "title": "4. Inspection Task",
            "instruction": "Move to inspection position (500, 0, 200), take a photo, then move to next position (600, 0, 200)."
        },
        {
            "title": "5. Material Handling",
            "instruction": "Pick up a 15kg object from conveyor at (800, 200, 100) and place it on pallet at (1200, 500, 50)."
        }
    ]

    for example in examples:
        print(f"\n{example['title']}:")
        print(f"  {example['instruction']}")

    print("\n" + "-"*80 + "\n")


def save_tdl_interactive(converter, tdl_code):
    """TDL 코드를 파일로 저장"""
    if not tdl_code or tdl_code.startswith("// ERROR"):
        print("\n[ERROR] No valid TDL code to save.")
        return

    # 기본 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_filename = f"tdl_output_{timestamp}.tdl"

    print(f"\nEnter filename (default: {default_filename}): ", end='')
    filename = input().strip()

    if not filename:
        filename = default_filename

    if not filename.endswith('.tdl'):
        filename += '.tdl'

    # 출력 디렉토리 생성
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)

    try:
        converter.save_tdl(tdl_code, output_path)
        print(f"\n[OK] TDL code saved to: {output_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to save: {e}")


def clear_screen():
    """화면 클리어"""
    os.system('cls' if os.name == 'nt' else 'clear')


def interactive_mode():
    """인터랙티브 모드 메인 루프"""
    print_banner()
    print("Initializing NL2TDL Converter...")

    try:
        converter = NL2TDLConverter()
        print("[OK] Converter initialized successfully!\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize converter: {e}")
        return

    print_help()

    # 세션 데이터
    history = []
    last_tdl = None
    last_nl = None

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

                if command == '/quit' or command == '/exit' or command == '/q':
                    print("\n" + "="*80)
                    print("Thank you for using NL2TDL Converter!")
                    print("="*80 + "\n")
                    break

                elif command == '/help' or command == '/h':
                    print_help()

                elif command == '/examples' or command == '/ex':
                    print_examples()

                elif command == '/history':
                    if not history:
                        print("\n[INFO] No conversion history yet.")
                    else:
                        print("\n" + "-"*80)
                        print("CONVERSION HISTORY:")
                        print("-"*80)
                        for i, (nl, tdl_preview) in enumerate(history, 1):
                            print(f"\n{i}. Input: {nl[:60]}...")
                            print(f"   Output: {tdl_preview[:60]}...")
                        print("-"*80)

                elif command == '/clear' or command == '/cls':
                    clear_screen()
                    print_banner()

                elif command == '/save' or command == '/s':
                    if last_tdl:
                        save_tdl_interactive(converter, last_tdl)
                    else:
                        print("\n[ERROR] No TDL code to save. Generate TDL first.")

                else:
                    print(f"\n[ERROR] Unknown command: {user_input}")
                    print("Type /help to see available commands.")

                continue

            # 자연어 명령어 처리
            print("\n" + "="*80)
            print("CONVERTING...")
            print("="*80)

            nl_instruction = user_input

            # TDL 생성
            tdl_code = converter.convert(nl_instruction, temperature=0.1)

            # 결과 출력
            print("\n" + "="*80)
            print("GENERATED TDL CODE:")
            print("="*80)
            print(tdl_code)
            print("="*80)

            # 히스토리 저장
            if not tdl_code.startswith("// ERROR"):
                tdl_preview = tdl_code.split('\n')[0] if tdl_code else ""
                history.append((nl_instruction, tdl_preview))
                last_tdl = tdl_code
                last_nl = nl_instruction

                # 자동 저장 여부 확인
                print("\n[INFO] TDL code generated successfully!")
                print("       Type /save to save this TDL to a file.")
            else:
                print("\n[ERROR] TDL generation failed. Please check your input.")

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
        converter = NL2TDLConverter()
        tdl_code = converter.convert(nl_instruction)

        print("="*80)
        print("GENERATED TDL CODE:")
        print("="*80)
        print(tdl_code)
        print("="*80)

        return tdl_code

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None


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

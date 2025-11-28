# demo.py
"""
Robot Selector Demo
Demonstrates integration with TDL_generation module
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_selection import select_best_robot, print_selection_report


def demo_with_real_tdl():
    """
    Demo: Use robot selector with a realistic TDL scenario
    """
    print("\n" + "="*80)
    print("ROBOT SELECTOR DEMO - Integration with TDL Generation")
    print("="*80 + "\n")

    # Simulated TDL output from TDL_generation module
    tdl_v1_content = """
// Auto-generated TDL (v1) - Robot-agnostic format
// Generated from NL: "Weld a 20kg steel plate at position (1200, 300, 500)"

// Task Requirements
PAYLOAD_KG: 20.0
REQUIRED_REACH_M: 1.3
REQUIRED_DOF: 6

GOAL Initialize_Process()
{
    SPAWN SetTool(0) WITH WAIT;
    SPAWN SetJointVelocity(50) WITH WAIT;
    SPAWN SetJointAcceleration(50) WITH WAIT;
}

GOAL Execute_Process()
{
    // Move to home position
    SPAWN MoveJoint(PosJ(0,0,0,0,90,0), 50, 50, 0, 0.0, None) WITH WAIT;

    // Approach welding position
    SPAWN MoveLinear(PosX(1200,300,600,0,180,0), 40, 40, 0, 5.0, None) WITH WAIT;

    // Move to welding position
    SPAWN MoveLinear(PosX(1200,300,500,0,180,0), 30, 30, 0, 0.0, None) WITH WAIT;

    // Perform welding (simulate with wait)
    SPAWN Wait(3.0) WITH WAIT;

    // Retract
    SPAWN MoveLinear(PosX(1200,300,600,0,180,0), 40, 40, 0, 5.0, None) WITH WAIT;
}

GOAL Finalize_Process()
{
    // Return to home
    SPAWN MoveJoint(PosJ(0,0,0,0,0,0), 50, 50, 0, 0.0, None) WITH WAIT;
}
"""

    print("Scenario: Welding a 20kg steel plate at position (1200, 300, 500)")
    print("="*80)
    print("\nGenerated TDL (v1):")
    print("-"*80)
    print(tdl_v1_content)
    print("-"*80)

    # Select optimal robot
    print("\n\nSELECTING OPTIMAL ROBOT...")
    print("="*80)

    try:
        best_robot_id, best_score, all_scores = select_best_robot(tdl_v1_content)

        # Display results
        print_selection_report(best_robot_id, all_scores)

        # Next steps
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print(f"1. Selected Robot: {best_robot_id}")
        print(f"2. TDL v1 → TDL v2 conversion (Parameter Conversion Module)")
        print(f"   - Convert velocity: 50% → {all_scores[best_robot_id]['specs'].get('max_velocity', 'N/A')} * 0.5 rad/s")
        print(f"   - Convert acceleration: 50% → {all_scores[best_robot_id]['specs'].get('max_acceleration', 'N/A')} * 0.5 rad/s^2")
        print(f"3. TDL v2 → Job Code (DRL) generation")
        print(f"4. Execute on {best_robot_id}")
        print("="*80 + "\n")

        return best_robot_id, tdl_v1_content

    except Exception as e:
        print(f"\n[ERROR] Robot selection failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def demo_comparison_scenarios():
    """
    Demo: Compare robot selection for different task requirements
    """
    print("\n" + "="*80)
    print("ROBOT SELECTOR DEMO - Task Comparison")
    print("="*80 + "\n")

    scenarios = [
        ("Light Assembly (3kg)", "PAYLOAD_KG: 3.0\nREQUIRED_REACH_M: 0.8\nREQUIRED_DOF: 6"),
        ("Medium Welding (15kg)", "PAYLOAD_KG: 15.0\nREQUIRED_REACH_M: 1.1\nREQUIRED_DOF: 6"),
        ("Heavy Material Handling (25kg)", "PAYLOAD_KG: 25.0\nREQUIRED_REACH_M: 1.4\nREQUIRED_DOF: 6"),
        ("Complex Manipulation (7-DoF, 10kg)", "PAYLOAD_KG: 10.0\nREQUIRED_REACH_M: 0.8\nREQUIRED_DOF: 7"),
    ]

    results = []

    for scenario_name, requirements in scenarios:
        print(f"\n{'='*80}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*80}\n")

        tdl_content = f"""
TASK: {scenario_name}

{requirements}

GOAL Execute_Process()
{{
    SPAWN MoveLinear(PosX(1000,0,500,0,180,0), 50, 50, 0, 0.0, None) WITH WAIT;
}}
"""

        try:
            best_robot_id, best_score, all_scores = select_best_robot(tdl_content)
            results.append((scenario_name, best_robot_id, best_score))

            print(f"[SELECTED] {best_robot_id} (score: {best_score:.4f})")
            print(f"  Payload: {all_scores[best_robot_id]['specs']['payload']} kg")
            print(f"  Reach: {all_scores[best_robot_id]['specs']['reach']} m")
            print(f"  DoF: {all_scores[best_robot_id]['specs']['dof']}")

        except Exception as e:
            print(f"[ERROR] {e}")
            results.append((scenario_name, "ERROR", 0.0))

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Robot Selection Results")
    print("="*80)
    print(f"\n{'Task':<35} {'Selected Robot':<20} {'Score':<10}")
    print("-"*80)
    for scenario_name, robot_id, score in results:
        print(f"{scenario_name:<35} {robot_id:<20} {score:<10.4f}")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ROBOT SELECTOR MODULE - INTERACTIVE DEMO")
    print("="*80)

    # Demo 1: Real TDL integration
    print("\n[DEMO 1] Integration with TDL Generation")
    best_robot, tdl_content = demo_with_real_tdl()

    input("\n\nPress Enter to continue to Demo 2...")

    # Demo 2: Compare scenarios
    print("\n[DEMO 2] Comparison Across Different Tasks")
    demo_comparison_scenarios()

    print("\n" + "="*80)
    print("DEMO COMPLETED")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Robot selection prevents over-specification ('과소비' 방지)")
    print("2. Gaussian scoring ensures optimal payload matching")
    print("3. Multi-criteria optimization (Payload, Reach, DoF)")
    print("4. Ready for integration with Parameter Conversion module")
    print("="*80 + "\n")

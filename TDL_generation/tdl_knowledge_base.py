# tdl_knowledge_base.py
"""
TDL Knowledge Base
Loads and manages TDL grammar rules from TDLset.md
"""

import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class TDLKnowledgeBase:
    """
    Manages TDL grammar rules and command definitions.
    Loads the TDL syntax from TDLset.md and provides structured access.
    """

    def __init__(self, tdl_grammar_path: str = None):
        """
        Initialize TDL Knowledge Base

        Args:
            tdl_grammar_path: Path to TDLset.md file. If None, searches in default locations.
        """
        self.tdl_grammar_path = tdl_grammar_path
        self.grammar_content = ""
        self.goal_templates = []
        self.command_definitions = {}

        # Load TDL grammar
        self._load_grammar()
        self._parse_grammar()

    def _find_grammar_file(self) -> str:
        """
        Find TDLset.md file in default locations

        Returns:
            Path to TDLset.md file
        """
        # Search in rag_documents folder
        # __file__ is in TDL_generation/, so go up 1 level to AYN/
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        possible_paths = [
            os.path.join(base_path, "rag_documents", "TDLset.md"),
            os.path.join(base_path, "rag_documents", "TDLset2.md"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found TDL grammar file: {path}")
                return path

        raise FileNotFoundError("TDLset.md not found in rag_documents folder")

    def _load_grammar(self):
        """Load TDL grammar from file"""
        try:
            if self.tdl_grammar_path is None:
                self.tdl_grammar_path = self._find_grammar_file()

            with open(self.tdl_grammar_path, 'r', encoding='utf-8') as f:
                self.grammar_content = f.read()

            logger.info(f"Successfully loaded TDL grammar from: {self.tdl_grammar_path}")
        except Exception as e:
            logger.error(f"Failed to load TDL grammar: {e}")
            raise

    def _parse_grammar(self):
        """Parse TDL grammar to extract GOAL templates and COMMAND definitions"""
        lines = self.grammar_content.split('\n')

        for line in lines:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Extract GOAL templates
            if line.startswith('GOAL'):
                self.goal_templates.append(line)

            # Extract COMMAND definitions
            elif line.startswith('COMMAND'):
                # Parse command name and definition
                # Format: COMMAND CommandName(params) { implementation; }
                parts = line.split('(', 1)
                if len(parts) >= 2:
                    command_name = parts[0].replace('COMMAND', '').strip()
                    self.command_definitions[command_name] = line

        logger.info(f"Parsed {len(self.goal_templates)} GOAL templates and {len(self.command_definitions)} COMMAND definitions")

    def get_full_grammar(self) -> str:
        """
        Get full TDL grammar content

        Returns:
            Complete TDL grammar as string
        """
        return self.grammar_content

    def get_goal_templates(self) -> List[str]:
        """
        Get all GOAL templates

        Returns:
            List of GOAL template strings
        """
        return self.goal_templates

    def get_command_definitions(self) -> Dict[str, str]:
        """
        Get all COMMAND definitions

        Returns:
            Dictionary mapping command names to their definitions
        """
        return self.command_definitions

    def get_command_definition(self, command_name: str) -> str:
        """
        Get specific COMMAND definition

        Args:
            command_name: Name of the command

        Returns:
            Command definition string, or None if not found
        """
        return self.command_definitions.get(command_name)

    def get_system_prompt_context(self) -> str:
        """
        Generate system prompt context for LLM
        This provides the TDL grammar rules to the LLM for NL→TDL conversion

        Returns:
            Formatted TDL grammar context for system prompt
        """
        prompt_context = f"""
# TDL (Task Description Language) Grammar Reference

You are an expert TDL code generator. Your task is to convert natural language instructions into valid TDL code.

## TDL Structure

TDL programs consist of three main GOAL blocks:
1. **Initialize_Process()**: Setup and initialization commands
2. **Execute_Process()**: Main task execution commands
3. **Finalize_Process()**: Cleanup and finalization commands

## Available Commands

Below are all available TDL commands with their syntax:

{self.grammar_content}

## TDL Generation Rules

1. **Command Format**: Each command must be wrapped in a SPAWN statement:
   ```
   SPAWN CommandName(parameters) WITH WAIT;
   ```

2. **Parameter Values**: Use normalized 0-100% scale for robot-agnostic parameters:
   - velocity: 0-100 (percentage of max speed)
   - acceleration: 0-100 (percentage of max acceleration)
   - blending_radius: in mm (absolute value)

3. **Pose Representation**:
   - Joint poses: Use PosJ(j1, j2, j3, j4, j5, j6) with angles in degrees
   - Cartesian poses: Use PosX(x, y, z, rx, ry, rz) with positions in mm and rotations in degrees

4. **Code Structure**:
   - Always include all three GOAL blocks (Initialize, Execute, Finalize)
   - Place initialization commands (SetTool, SetJointVelocity, etc.) in Initialize_Process
   - Place main motion commands in Execute_Process
   - Place cleanup commands in Finalize_Process

5. **Best Practices**:
   - Use descriptive variable names if needed
   - Add comments to explain complex logic
   - Keep commands simple and atomic
   - Use proper indentation for readability

## Output Format

Your output must be valid TDL code following this structure:

```
GOAL Initialize_Process()
{{
    SPAWN CommandName(params) WITH WAIT;
    // ... more initialization commands
}}

GOAL Execute_Process()
{{
    SPAWN CommandName(params) WITH WAIT;
    // ... more execution commands
}}

GOAL Finalize_Process()
{{
    SPAWN CommandName(params) WITH WAIT;
    // ... more finalization commands
}}
```
"""
        return prompt_context.strip()


if __name__ == "__main__":
    # Test the TDLKnowledgeBase
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        kb = TDLKnowledgeBase()
        print("\n=== TDL Knowledge Base Test ===\n")
        print(f"Loaded {len(kb.get_goal_templates())} GOAL templates")
        print(f"Loaded {len(kb.get_command_definitions())} COMMAND definitions")
        print("\nSample GOAL templates:")
        for template in kb.get_goal_templates()[:3]:
            print(f"  - {template}")
        print("\nSample COMMAND definitions:")
        for i, (name, definition) in enumerate(list(kb.get_command_definitions().items())[:3]):
            print(f"  - {name}: {definition[:80]}...")
    except Exception as e:
        print(f"Error: {e}")

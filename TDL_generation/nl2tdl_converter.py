# nl2tdl_converter.py
"""
NL to TDL Converter (v1)
Converts natural language instructions to TDL code using LLM
"""

import os
import json
import logging
import re
from typing import Dict, Optional
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .tdl_knowledge_base import TDLKnowledgeBase

logger = logging.getLogger(__name__)


class NL2TDLConverter:
    """
    Converts Natural Language to TDL (v1) format

    This class uses an LLM with TDL grammar knowledge to convert
    user's natural language instructions into standardized TDL code.

    TDL v1 characteristics:
    - Robot-agnostic (0-100% scale for speed/acceleration)
    - General pose values (not robot-specific)
    - Standardized command format
    """

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-pro"):
        """
        Initialize NL2TDL Converter

        Args:
            api_key: Google Gemini API key. If None, tries to load from config.json
            model_name: Gemini model to use
        """
        self.model_name = model_name
        self.api_key = api_key or self._load_api_key()

        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

        # Load TDL knowledge base
        self.knowledge_base = TDLKnowledgeBase()

        logger.info(f"NL2TDLConverter initialized with model: {model_name}")

    def _load_api_key(self) -> str:
        """
        Load API key from multiple sources (priority order):
        1. Environment variable (GEMINI_API_KEY)
        2. .env file (project root)
        3. api_key.txt file (project root)
        4. config.json file (project root)

        Returns:
            API key string

        Raises:
            ValueError: If API key is not found in any source
        """
        import os
        import json

        # 프로젝트 루트 경로
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # 1. 환경 변수에서 로드 시도 (최우선)
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            logger.info("API key loaded from GEMINI_API_KEY environment variable")
            return api_key

        # 2. .env 파일에서 로드 시도
        env_file = os.path.join(project_root, '.env')
        if os.path.exists(env_file):
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('GEMINI_API_KEY='):
                            api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                            if api_key:
                                logger.info("API key loaded from .env file")
                                return api_key
            except Exception as e:
                logger.warning(f"Failed to read .env file: {e}")

        # 3. api_key.txt 파일에서 로드 시도
        key_file = os.path.join(project_root, 'api_key.txt')
        if os.path.exists(key_file):
            try:
                with open(key_file, 'r', encoding='utf-8') as f:
                    api_key = f.read().strip()
                    if api_key:
                        logger.info("API key loaded from api_key.txt")
                        return api_key
            except Exception as e:
                logger.warning(f"Failed to read api_key.txt: {e}")

        # 4. config.json 파일에서 로드 시도
        config_file = os.path.join(project_root, 'config.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    api_key = config.get('GEMINI_API_KEY') or config.get('api_key')
                    if api_key:
                        logger.info("API key loaded from config.json")
                        return api_key
            except Exception as e:
                logger.warning(f"Failed to read config.json: {e}")

        # API 키를 찾지 못한 경우
        raise ValueError(
            "API key not found. Please provide a valid Gemini API key using one of these methods:\n"
            "1. Set environment variable: GEMINI_API_KEY='your-api-key'\n"
            "2. Create .env file with: GEMINI_API_KEY=your-api-key\n"
            "3. Create api_key.txt file with your API key\n"
            "4. Create config.json with: {\"GEMINI_API_KEY\": \"your-api-key\"}"
        )

    def _build_system_prompt(self) -> str:
        """
        Build system prompt with TDL grammar knowledge

        Returns:
            Complete system prompt for LLM
        """
        grammar_context = self.knowledge_base.get_system_prompt_context()

        system_prompt = f"""
{grammar_context}

## Common Object Weights (for reference)
- Apple: ~0.2 kg
- Banana: ~0.12 kg (with peel)
- Orange: ~0.15 kg
- Milk carton (1L): ~1.0 kg
- Soda can: ~0.4 kg
- Tuna can: ~0.2 kg
- Small parts: ~0.05-0.1 kg

**IMPORTANT:** Always include SetWorkpieceWeight() in Initialize_Process based on the object being manipulated.

## Your Task

You are an expert TDL code generator. Convert the user's natural language instruction into valid TDL code.

**CRITICAL OUTPUT RULES:**

1. **Output ONLY valid TDL code** - no explanations, no markdown, no extra text
2. **Follow the exact format** shown in the grammar reference
3. **Include all three GOAL blocks** even if some are empty
4. **Use SPAWN ... WITH WAIT;** for each command
5. **Use robot-agnostic values**:
   - Velocities: 0-100 (percentage)
   - Accelerations: 0-100 (percentage)
   - Positions: reasonable mm values
   - Angles: reasonable degree values

6. **Infer reasonable defaults** when specific values are not provided:
   - Default velocity: 50 (moderate speed)
   - Default acceleration: 50 (moderate acceleration)
   - Default tool: 0 (standard tool)
   - Default blending_radius: 0.0 (precise positioning)

7. **Handle common tasks intelligently**:
   - "pick and place" → Move to pickup, close gripper, move to place, open gripper
   - "move to position" → Use MoveLinear or MoveJoint depending on context
   - "go home" → Move to home position PosJ(0,0,0,0,0,0)

8. **CRITICAL: Add task requirements at the top as comments:**
   - Extract payload (kg) from the instruction
   - Estimate reach (m) from target positions
   - Specify DoF if mentioned (default: 6)

   **Required header format:**
   ```
   // ========================================
   // TASK REQUIREMENTS (for Robot Selection)
   // ========================================
   // PAYLOAD_KG: <weight in kg>
   // REQUIRED_REACH_M: <reach in meters>
   // REQUIRED_DOF: <degrees of freedom>
   // ========================================
   ```

**Example Output Format:**

```
// ========================================
// TASK REQUIREMENTS (for Robot Selection)
// ========================================
// PAYLOAD_KG: 15.0
// REQUIRED_REACH_M: 1.2
// REQUIRED_DOF: 6
// ========================================

GOAL Initialize_Process()
{{
    SPAWN SetTool(0) WITH WAIT;
    SPAWN SetWorkpieceWeight(15.0, Trans(0, 0, 100, 0, 0, 0)) WITH WAIT;
    SPAWN SetJointVelocity(50) WITH WAIT;
    SPAWN SetJointAcceleration(50) WITH WAIT;
}}

GOAL Execute_Process()
{{
    SPAWN MoveJoint(PosJ(0,0,90,0,90,0), 50, 50, 0, 0.0, None) WITH WAIT;
    SPAWN MoveLinear(PosX(300,0,200,0,180,0), 50, 50, 0, 0.0, None) WITH WAIT;
}}

GOAL Finalize_Process()
{{
    SPAWN MoveJoint(PosJ(0,0,0,0,0,0), 50, 50, 0, 0.0, None) WITH WAIT;
}}
```

Now convert the following natural language instruction to TDL code:
"""
        return system_prompt.strip()

    def convert(self, nl_input: str, temperature: float = 0.0) -> str:
        """
        Convert natural language to TDL code

        Args:
            nl_input: Natural language instruction from user
            temperature: LLM temperature (0.0 = deterministic, 1.0 = creative)

        Returns:
            Generated TDL code as string
        """
        logger.info("Converting NL to TDL...")
        logger.info(f"Input: {nl_input[:100]}...")

        # Build complete prompt
        system_prompt = self._build_system_prompt()
        full_prompt = f"{system_prompt}\n\n**User Instruction:**\n{nl_input}"

        try:
            # Configure safety settings
            safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }

            # Call Gemini API
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8192,
                    temperature=temperature
                ),
                safety_settings=safety_settings
            )

            if not response.parts:
                finish_reason = response.candidates[0].finish_reason.name if response.candidates else "UNKNOWN"
                error_msg = f"API returned no content. Finish reason: {finish_reason}"
                logger.error(error_msg)
                return f"// ERROR: {error_msg}"

            tdl_code = response.text.strip()

            # Clean up the response (remove markdown code blocks if present)
            tdl_code = self._clean_output(tdl_code)

            logger.info("TDL conversion successful")
            logger.info(f"Generated {len(tdl_code.split(chr(10)))} lines of TDL code")

            return tdl_code

        except Exception as e:
            error_msg = f"TDL conversion failed: {e}"
            logger.error(error_msg)
            return f"// ERROR: {error_msg}"

    def _clean_output(self, output: str) -> str:
        """
        Clean LLM output to extract pure TDL code

        Args:
            output: Raw LLM output

        Returns:
            Cleaned TDL code
        """
        # Remove markdown code blocks
        output = re.sub(r'```tdl\s*', '', output)
        output = re.sub(r'```\s*', '', output)

        # Remove any leading/trailing explanatory text
        lines = output.split('\n')
        tdl_lines = []
        in_tdl = False

        for line in lines:
            # Start capturing when we see GOAL
            if 'GOAL' in line:
                in_tdl = True

            if in_tdl:
                tdl_lines.append(line)

        if tdl_lines:
            return '\n'.join(tdl_lines).strip()
        else:
            # If no GOAL blocks found, return original output
            return output.strip()

    def convert_with_metadata(self, nl_input: str, temperature: float = 0.0) -> Dict:
        """
        Convert natural language to TDL with metadata

        Args:
            nl_input: Natural language instruction from user
            temperature: LLM temperature

        Returns:
            Dictionary with TDL code and metadata
        """
        from datetime import datetime
        import pytz

        seoul_tz = pytz.timezone("Asia/Seoul")

        tdl_code = self.convert(nl_input, temperature)

        metadata = {
            "generated_at": datetime.now(seoul_tz).isoformat(),
            "nl_input": nl_input,
            "tdl_version": "v1",
            "model": self.model_name,
            "converter": "NL2TDLConverter",
        }

        return {
            "tdl_code": tdl_code,
            "metadata": metadata
        }

    def save_tdl(self, tdl_code: str, output_path: str):
        """
        Save TDL code to file

        Args:
            tdl_code: TDL code to save
            output_path: Output file path
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(tdl_code)
            logger.info(f"TDL code saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save TDL code: {e}")
            raise


if __name__ == "__main__":
    # Test the NL2TDLConverter
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        print("\n=== NL2TDL Converter Test ===\n")

        # Initialize converter
        converter = NL2TDLConverter()

        # Test case 1: Simple pick and place
        nl_instruction = """
        Pick up an object at position (300, 0, 100) and place it at position (300, 200, 100).
        Use a moderate speed of 60% for safety.
        """

        print(f"Input: {nl_instruction.strip()}\n")
        print("Converting...\n")

        tdl_code = converter.convert(nl_instruction)

        print("="*80)
        print("GENERATED TDL CODE:")
        print("="*80)
        print(tdl_code)
        print("="*80)

        # Save to file
        output_path = os.path.join(
            os.path.dirname(__file__),
            "output",
            "test_output.tdl"
        )
        converter.save_tdl(tdl_code, output_path)

        print(f"\nTDL code saved to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

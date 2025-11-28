# NL2TDL Converter (v1)

Natural Language to Task Description Language converter module.

## Overview

This module converts natural language instructions into standardized TDL (Task Description Language) code. This is **Contribution 1** of the project: defining and generating a robot-agnostic intermediate representation.

### Key Features

- **Robot-Agnostic TDL Generation**: Creates TDL v1 with normalized 0-100% scale values
- **LLM-Powered**: Uses Google Gemini for intelligent NL understanding
- **TDL Grammar Based**: Leverages complete TDL syntax from `TDLset.md`
- **No RAG Dependencies**: Basic v1 version works without scene context (RAG will be added in future versions)

## Architecture

### Components

1. **TDLKnowledgeBase** (`tdl_knowledge_base.py`)
   - Loads and parses TDL grammar from `TDLset.md`
   - Provides structured access to GOAL templates and COMMAND definitions
   - Generates system prompt context for LLM

2. **NL2TDLConverter** (`nl2tdl_converter.py`)
   - Main converter class
   - Converts natural language to TDL v1 code
   - Uses Gemini API with TDL grammar knowledge
   - Handles output cleaning and validation

3. **Test Suite** (`test_converter.py`)
   - Example test cases
   - Interactive testing mode
   - Automated test execution

## Installation

### Prerequisites

```bash
pip install google-generativeai pytz
```

### Directory Structure

```
xz/
├── NL2TDL/
│   └── TDL_generation/
│       ├── tdl_knowledge_base.py
│       ├── nl2tdl_converter.py
│       ├── test_converter.py
│       ├── README.md
│       └── output/
│           ├── tests/
│           └── interactive/
├── rag_documents/
│   └── TDLset.md
└── config.json
```

## Usage

### Basic Usage

```python
from nl2tdl_converter import NL2TDLConverter

# Initialize converter
converter = NL2TDLConverter()

# Convert natural language to TDL
nl_input = "Pick up an object at (300, 0, 100) and place it at (300, 200, 100)"
tdl_code = converter.convert(nl_input)

print(tdl_code)
```

### With Metadata

```python
# Get TDL with metadata
result = converter.convert_with_metadata(nl_input)

print(result['tdl_code'])
print(result['metadata'])
```

### Save to File

```python
# Save TDL to file
converter.save_tdl(tdl_code, "output/my_task.tdl")
```

### Running Tests

```bash
# Run test suite
cd NL2TDL/TDL_generation
python test_converter.py
```

The test suite offers two modes:
1. **Example Test Cases**: Runs predefined test scenarios
2. **Interactive Mode**: Enter custom natural language instructions

## TDL v1 Characteristics

### Robot-Agnostic Values

TDL v1 uses normalized, robot-independent values:

- **Velocity**: 0-100 (percentage of max speed)
- **Acceleration**: 0-100 (percentage of max acceleration)
- **Positions**: Millimeters (mm)
- **Angles**: Degrees (°)
- **Blending Radius**: Millimeters (mm)

### Structure

All TDL code follows this structure:

```tdl
GOAL Initialize_Process()
{
    // Initialization commands
    SPAWN SetTool(0) WITH WAIT;
    SPAWN SetJointVelocity(50) WITH WAIT;
}

GOAL Execute_Process()
{
    // Main task commands
    SPAWN MoveJoint(PosJ(0,0,90,0,90,0), 50, 50, 0, 0.0, None) WITH WAIT;
    SPAWN MoveLinear(PosX(300,0,200,0,180,0), 50, 50, 0, 0.0, None) WITH WAIT;
}

GOAL Finalize_Process()
{
    // Cleanup commands
    SPAWN MoveJoint(PosJ(0,0,0,0,0,0), 50, 50, 0, 0.0, None) WITH WAIT;
}
```

## Example Conversions

### Example 1: Pick and Place

**Input:**
```
Pick up an object at position (300, 0, 100) and place it at position (300, 200, 100).
Use a speed of 50% for safety.
```

**Output:**
```tdl
GOAL Initialize_Process()
{
    SPAWN SetTool(0) WITH WAIT;
    SPAWN SetJointVelocity(50) WITH WAIT;
    SPAWN SetJointAcceleration(50) WITH WAIT;
}

GOAL Execute_Process()
{
    SPAWN MoveLinear(PosX(300, 0, 100, 0, 180, 0), 50, 50, 0, 0.0, None) WITH WAIT;
    SPAWN SetDigitalOutput(1, 1) WITH WAIT;
    SPAWN Delay(0.5) WITH WAIT;
    SPAWN MoveLinear(PosX(300, 200, 100, 0, 180, 0), 50, 50, 0, 0.0, None) WITH WAIT;
    SPAWN SetDigitalOutput(1, 0) WITH WAIT;
}

GOAL Finalize_Process()
{
    SPAWN MoveJoint(PosJ(0, 0, 0, 0, 0, 0), 50, 50, 0, 0.0, None) WITH WAIT;
}
```

### Example 2: Go Home

**Input:**
```
Move the robot to home position safely.
```

**Output:**
```tdl
GOAL Initialize_Process()
{
    SPAWN SetJointVelocity(50) WITH WAIT;
    SPAWN SetJointAcceleration(50) WITH WAIT;
}

GOAL Execute_Process()
{
    SPAWN MoveJoint(PosJ(0, 0, 0, 0, 0, 0), 50, 50, 0, 0.0, None) WITH WAIT;
}

GOAL Finalize_Process()
{
}
```

## API Reference

### NL2TDLConverter

#### Constructor

```python
NL2TDLConverter(api_key: str = None, model_name: str = "gemini-2.5-pro")
```

**Parameters:**
- `api_key`: Google Gemini API key (optional, loads from config if not provided)
- `model_name`: Gemini model to use (default: "gemini-2.5-pro")

#### Methods

##### convert(nl_input: str, temperature: float = 0.0) -> str

Convert natural language to TDL code.

**Parameters:**
- `nl_input`: Natural language instruction
- `temperature`: LLM temperature (0.0 = deterministic, 1.0 = creative)

**Returns:** Generated TDL code as string

##### convert_with_metadata(nl_input: str, temperature: float = 0.0) -> Dict

Convert with metadata.

**Returns:** Dictionary with `tdl_code` and `metadata` keys

##### save_tdl(tdl_code: str, output_path: str)

Save TDL code to file.

### TDLKnowledgeBase

#### Constructor

```python
TDLKnowledgeBase(tdl_grammar_path: str = None)
```

**Parameters:**
- `tdl_grammar_path`: Path to TDLset.md (optional, auto-detected if not provided)

#### Methods

##### get_full_grammar() -> str

Get complete TDL grammar content.

##### get_goal_templates() -> List[str]

Get all GOAL templates.

##### get_command_definitions() -> Dict[str, str]

Get all COMMAND definitions.

##### get_system_prompt_context() -> str

Generate system prompt context for LLM.

## Configuration

### API Key

The module uses the Google Gemini API key. Configure it in one of these ways:

1. Pass directly to constructor:
```python
converter = NL2TDLConverter(api_key="your_api_key_here")
```

2. Use existing key from codebase (default behavior)

### Model Selection

Choose different Gemini models based on your needs:

```python
# Fast and economical
converter = NL2TDLConverter(model_name="gemini-1.5-flash")

# High quality (default)
converter = NL2TDLConverter(model_name="gemini-2.5-pro")

# Maximum capability
converter = NL2TDLConverter(model_name="gemini-2.5-pro")
```

## Limitations (v1)

This is the **basic v1 version** with the following limitations:

1. **No RAG Integration**: Does not use scene context from MuJoCo XML
2. **No Hallucination Prevention**: May generate references to non-existent objects
3. **Fixed Default Values**: Uses predefined defaults for unspecified parameters
4. **No Validation**: Does not validate against physical constraints

These will be addressed in future versions:
- **v2** (Planned): Add RAG integration with `mujoco_scene_parser.py`
- **v3** (Planned): Add validation and constraint checking

## Troubleshooting

### Common Issues

**Issue: "API key not found"**
- Solution: Ensure `config.json` exists or pass API key to constructor

**Issue: "TDLset.md not found"**
- Solution: Ensure `rag_documents/TDLset.md` exists in parent directory

**Issue: "Import error"**
- Solution: Install required packages: `pip install google-generativeai pytz`

**Issue: "LLM returns no content"**
- Solution: Check your API key and internet connection
- Try adjusting the temperature parameter

## Project Context

This module is part of the larger **LLM-based Heterogeneous Robot Control Framework**:

1. **Module 1 (Current)**: NL → TDL(v1) conversion ✓
2. **Module 2**: Robot selection based on TDL requirements
3. **Module 3**: Path planning and validation with MuJoCo
4. **Module 4**: Parameter conversion TDL(v1) → TDL(v2)
5. **Module 5**: TDL(v2) → Job Code generation

See `Basic_Info/Project_Briefing.md` for complete project overview.

## Contributing

When extending this module:

1. Maintain backward compatibility with v1 format
2. Add new features in separate functions/classes
3. Update tests when adding new functionality
4. Document all public APIs

## License

Part of the xz robotics project.

## Contact

For questions or issues, refer to the main project documentation.

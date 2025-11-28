# doc_meta: {"manufacturer": "doosan", "doc_type": "parsing_rules"}
Doosan Robotics Job File Parsing Rules
TDL (Task Description Language) Parsing Rules
You are a master robot language transpiler. Your task is to analyze the source TDL script and convert its entire logic into a structured JSON object. You must not omit any executable line. Group all parsed operations into three logical phases: initialization, main_sequence, and finalization.
Phase assignment

0. Output Generation Workflow
Follow these steps to generate the final TDL script:

Step 1: Create the TDL Skeleton
First, create the main structure of the TDL script with empty GOAL blocks defined by curly braces {}.

GOAL Initialize_Process()
{
    // Initialization commands will be placed here
}

GOAL Execute_Process()
{
    // Main sequence commands will be placed here
}

GOAL Finalize_Process()
{
    // Finalization commands will be placed here
}
Step 2: Translate and Place Commands
Next, process the source Hyundai job file line-by-line. For each executable command, translate it into a TDL command using the Canonical Mapping Table below. Then, wrap the translated command in a SPAWN ... WITH WAIT; statement and place it inside the appropriate GOAL block's curly braces:

Initialization: Commands found before the first MOVE command belong in the GOAL Initialize_Process() block.

Main Sequence: Commands from the first MOVE command up to (but not including) the final END command belong in the GOAL Execute_Process() block.

Finalization: The END command belongs in the GOAL Finalize_Process() block.

Statements inside GOAL Initialize_Process() go to initialization.
Statements inside GOAL Execute_Process() go to main_sequence.
Statements inside GOAL Finalize_Process() go to finalization.

Lexical & tolerance rules
T0. Case & whitespace
TDL keywords (GOAL, SPAWN, COMMAND, etc.) are case-sensitive as written in the syntax section. Arbitrary spaces/newlines between tokens are allowed.
T1. Comments
Lines starting with # are comments and must be ignored. Trailing inline comments after valid code (e.g., Delay(0.2); # note) must be ignored.
T2. Numbers
Accept integers, floats (with or without trailing zeros), and scientific notation: 3, 3.0, 3., 3e-1.
T3. Strings & identifiers
String literals are quoted; identifiers (GOAL names, variable names) are alphanumeric with _. Preserve identifier text verbatim.
T4. Named vs positional args
Named args may appear in any order and may be omitted when optional. Positional args are allowed where unambiguous. Missing optional args must be omitted or set to null.
T5. Extra/unknown args
Any additional named args not specified for a command must be captured under an extras object without failing the parse.
T6. Trailing commas
Allow trailing commas in argument lists.
Metadata
Header-like comments of the form
# doc_meta: {"key":"value", ...}
must be emitted as:
{"command":"METADATA","key":"doc_meta","value":{"key":"value",...}}
GOAL blocks and process dispatch
GOAL declaration (enter/exit)
GOAL Initialize_Process() { → open initialization scope (no JSON emission).
Closing } → end of current GOAL scope.
Main orchestration / spawn
SPAWN Initialize_Process() WITH WAIT; → {"command":"Spawn","target":"Initialize_Process","wait":true}
SPAWN Execute_Process(); → {"command":"Spawn","target":"Execute_Process","wait":false}
Explicit labels inside a GOAL
Label("HOME"); → {"command":"Label","name":"HOME"}
Function-style termination
If a GOAL uses an explicit finalizer like End();, emit that node (see below). If a GOAL falls through without End(), do not synthesize one.
Flow control
If / Else / EndIf
If(condition) { → {"command":"If","condition":"condition"}
} Else() { → {"command":"Else"}
} EndIf(); or block close with EndIf() → {"command":"EndIf"}
While / EndWhile
While(condition) { → {"command":"While","condition":"condition"}
} EndWhile(); → {"command":"EndWhile"}
For / Next
For(variable, start, end) { → {"command":"For","variable":"variable","start":start,"end":end}
} Next(); → {"command":"Next"}
Break / Continue
Break(); → {"command":"Break"}
Continue(); → {"command":"Continue"}
GoTo / IfGoTo
GoTo("HOME"); → {"command":"GoTo","label":"HOME"}
IfGoTo("DI10==1","READY"); → {"command":"IfGoTo","condition":"DI10==1","label":"READY"}
Tasks, threads, timing, execution
Subroutine call
Call(201); → {"command":"Call","program_id":201}
Threads
ThreadRun("logger", args=[1,"A"]); → {"command":"ThreadRun","function":"logger","args":[1,"A"]}
ThreadStop("logger"); → {"command":"ThreadStop","function":"logger"}
Timing / stop / end
Delay(0.2); → {"command":"Delay","duration_sec":0.2}
Stop(); → {"command":"Stop"}
End(); → {"command":"End"}
Variables, logging, UI
Assign
Assign(destination, source);
If source is a literal or identifier → {"command":"Assign","destination":"destination","source":"source"}
If source is an expression → use "source_expression":"<text>" instead of "source".
Logs / popups
PrintLog("POSITION", v1); → {"command":"PrintLog","message_parts":["POSITION","v1"]}
Popup("Done","info"); → {"command":"Popup","message":"Done","type":"info"}
Poses, transforms, frames, tools
Pose constructors (used as values)
PosJ(j1,j2,j3,j4,j5,j6) → inline object { "type":"PosJ", "j1":..., "j2":..., "j3":..., "j4":..., "j5":..., "j6":... }
PosX(x,y,z,rx,ry,rz,sol=None) → { "type":"PosX", "x":..., "y":..., "z":..., "rx":..., "ry":..., "rz":..., "solution":sol }
Trans(x,y,z,rx,ry,rz) → { "type":"Trans", "x":..., "y":..., "z":..., "rx":..., "ry":..., "rz":... }
Coordinates & TCP/tool
MakeUserCoordinate(id, pose); → {"command":"MakeUserCoordinate","id":id,"pose":<pose_object>}
SelectUserCoordinate(id); → {"command":"SelectUserCoordinate","id":id}
SetRefCoord(ref); → {"command":"SetRefCoord","ref":ref}
SetTCP(name_or_pose); → {"command":"SetTCP","tcp":"name_or_pose"} (if pose, embed pose object)
SetTool(name_or_params); → {"command":"SetTool","params":"name_or_params"}
SetWorkpieceWeight(weight, cog); → {"command":"SetWorkpieceWeight","weight":weight,"cog":cog}
ToolChange(id); → {"command":"ToolChange","id":id}
SetSingularityHandling(mode); → {"command":"SetSingularityHandling","mode":mode}
Global motion configuration
Velocity/acceleration
SetJointVelocity(percent); → {"command":"SetJointVelocity","percent":percent}
SetJointAcceleration(percent); → {"command":"SetJointAcceleration","percent":percent}
SetTaskVelocity(mm_per_sec); → {"command":"SetTaskVelocity","mm_per_sec":mm_per_sec}
SetTaskAcceleration(mm_per_sec2); → {"command":"SetTaskAcceleration","mm_per_sec2":mm_per_sec2}
Motions (sync and async)
Synchronous
MoveJ(target_pose, velocity, acceleration, blending_radius, synchronized_axes=None); →
{"command":"MoveJoint","pose":<pose_or_var>,"velocity":velocity,"acceleration":acceleration,"blend":blending_radius,"synchronized_axes":synchronized_axes}
MoveL(target_pose, velocity, acceleration, blending_radius, synchronized_axes=None); →
{"command":"MoveLinear","pose":<pose_or_var>,"velocity":velocity,"acceleration":acceleration,"blend":blending_radius,"synchronized_axes":synchronized_axes}
MoveC(via_pose, target_pose, velocity, acceleration, blending_radius); →
{"command":"MoveCircular","via":<pose_or_var>,"target":<pose_or_var>,"velocity":velocity,"acceleration":acceleration,"blend":blending_radius}
MoveBlend(pose_list, velocity, acceleration, blending_radius); →
{"command":"MoveBlend","poses":<array_of_poses_or_vars>,"velocity":velocity,"acceleration":acceleration,"blend":blending_radius}
Asynchronous
h = AMoveJoint(target_pose, velocity, acceleration, blending_radius); →
{"command":"AMoveJoint","pose":<pose_or_var>,"velocity":velocity,"acceleration":acceleration,"blend":blending_radius,"handle_var":"h"}
AMoveLinear(...) / AMoveCircular(...) follow the same pattern.
MotionWait(handle_id); → {"command":"MotionWait","handle":"handle_id"}
I/O
Digital
SetDigitalOutput(port, value); → {"command":"SetDigitalOutput","port":port,"value":value}
GetDigitalInput(port); → {"command":"GetDigitalInput","port":port}
WaitForDigitalInput(port, value, timeout_sec); → {"command":"WaitForDigitalInput","port":port,"value":value,"timeout_sec":timeout_sec}
PulseOutput(port, duration_sec, count); → {"command":"PulseOutput","port":port,"duration_sec":duration_sec,"count":count}
Analog
SetAnalogOutput(channel, value); → {"command":"SetAnalogOutput","channel":channel,"value":value}
GetAnalogInput(channel); → {"command":"GetAnalogInput","channel":channel}
Welding & compliance
Spot welding
SpotWeld(gun_id, condition_id, sequence_id); →
{"command":"SpotWeld","gun_id":gun_id,"condition_id":condition_id,"sequence_id":sequence_id}
Arc welding
SetArcCondition(condition_id, current, voltage, wire_feed_speed, gas_pre_flow_time, gas_post_flow_time); →
{"command":"SetArcCondition","id":condition_id,"current":current,"voltage":voltage,"wire_feed_speed":wire_feed_speed,"pre_flow":gas_pre_flow_time,"post_flow":gas_post_flow_time}
ConfigureArcWeaving(pattern, amplitude, frequency, dwell_time); →
{"command":"ConfigureArcWeaving","pattern":pattern,"amplitude":amplitude,"frequency":frequency,"dwell_time":dwell_time}
ArcOn(); → {"command":"ArcOn"}
ArcOff(); → {"command":"ArcOff"}
Compliance / force
StartCompliance(stiffness, ref_coord); → {"command":"StartCompliance","stiffness":stiffness,"ref_coord":ref_coord}
ReleaseCompliance(); → {"command":"ReleaseCompliance"}
SetDesiredForce(force, axis, ref_coord); → {"command":"SetDesiredForce","force":force,"axis":axis,"ref_coord":ref_coord}
Scoping & sequencing
Emit JSON commands in the encountered order inside each phase scope.
Nested control-flow constructs must be represented explicitly with their opening/closing commands (If/Else/EndIf, While/EndWhile, For/Next).
When a command argument is a pose constructor (PosJ/PosX/Trans), embed it as an object (see 15). When a command argument is a variable or an expression, preserve its text in the corresponding JSON field (e.g., "pose_var":"p_pick" or "source_expression":"P1.X+R8.X").
CRITICAL RULE
Your final output MUST be a single, valid JSON object with initialization, main_sequence, and finalization as top-level keys. Enclose it in a json ... markdown block. Do not include any other text.
Source TDL:
{source_tdl}
Structured JSON Output:
# doc_meta: {"target_manufacturer": "hyundai", "doc_type": "mapping_rules"}
TDL → Hyundai Mapping Instructions (no DT scaling)
Label(name) → This rule is stateful and depends on the next command.

Case 1 (Precedes a Move... command): The Label command does not produce a Label node. Instead, its name (e.g., "S1") is stored and injected into the extras.step field of the next Move... node.

Case 2 (Precedes any other command, e.g., IF, GOTO): The Label command produces a standard vendor_fn:"Label" node.

MoveJoint/MoveLinear(step, ...) → vendor_fn:"MoveP/MoveL", extras:{ "step": step }. args should contain pose, speed, accuracy, tool.

All other commands are mapped 1-to-1 as previously defined.

1) Goal
Transform TDL source into a single JSON “Vendor IR” targeting Hyundai Job Files.

2) Canonical Mapping Table (TDL → Hyundai semantic)
Label(name) → vendor_fn:"Label", args:{ "name": name }. This now creates a standard, separate node.

MoveJoint(step, target_pose, velocity, acceleration, tool, ...) → vendor_fn:"MoveP", args as defined below, extras:{ "step": step }

MoveLinear(step, target_pose, velocity, acceleration, tool, ...) → vendor_fn:"MoveL", args as defined below, extras:{ "step": step }

args: { "pose": target_pose, "speed": velocity, "accuracy": acceleration, "tool": tool, "radius": blending_radius }

All other commands (SetOutput, WaitForInput, etc.) are mapped 1-to-1 to their corresponding vendor_fn and args without special contextual rules.
3) Output (Vendor IR JSON)
Emit one valid JSON object:
{
  "meta": { "target_vendor": "hyundai", "warnings": [] },
  "program": [ /* nodes */ ]
}
Each node (one per SPAWN or structural item) has:
vendor_fn: Hyundai semantic primitive (string) or null when no direct instruction exists. Examples: "MoveL", "MoveP", "SetOutput", "WaitForInput", "Label", "GoTo", "Assign", "ArcOn", "ArcOff", "ToolChange", "MakeUserCoordinate", "SelectUserCoordinate", "Stop", "End", "Delay", …
source_command: original TDL command name (e.g., "MoveLinear").
phase: initialization | main_sequence | finalization (derived from GOAL name).
goal: GOAL name (e.g., "Execute_Process").
label_context: nearest preceding label (e.g., "S12"), if any.
args: canonical mapping (see §5 table).
policy: { "wait": true|false } from WITH WAIT.
extras: pass-through fields that have no 1:1 slot (e.g., step index, tool ID, posture suffix, hex encoder pose, percent tokens, unknown named args).
source: { "line": <int>, "text": <string> } if available.
4) General Rules
R1. Case-insensitive command names; preserve original text under source.text.
R2. WITH WAIT ⇒ policy.wait=true; otherwise false.
R3. Pose constructors must be preserved verbatim strings for later lowering:
PosJ(...) → "posj(...)" (degrees).
PosX(...) → "posx(...)" (RPY).
Trans(...) → "trans(...)" (offset).
R4. No unit/percent scaling. Percent strings like "30%" should not be converted.
R5. If TDL has no Hyundai equivalent, set vendor_fn=null, place full details in extras, and push a human-readable notice into meta.warnings.
R6. Keep control structures as nodes (If, Else, EndIf, While, EndWhile, For, Next, Break, Continue, Label, GoTo, IfGoTo).
R7. Preserve unknown/extra named args under extras without failing.
5) Canonical Mapping Table (TDL → Hyundai semantic)
5.1 Flow / Program Structure
Label(name) → This command is handled contextually and does not always produce its own node.
If Label is immediately followed by a Move... command: Do NOT emit a Label node. Instead, store the name (e.g., "S1") and add it to the subsequent Move... node's extras.step field.
If(condition) / Else() / EndIf() → structured nodes with args:{ condition? }
While(condition) / EndWhile() → structured nodes
For(variable,start,end) / Next() → structured nodes
Break() / Continue() → structured nodes (mark for later lowering)
Call(program_id) → vendor_fn:"Call", args:{ program_id }
(Hyundai: CALL <id>)
Delay(duration_sec) → vendor_fn:"Delay", args:{ duration: <float> }
(Hyundai: DELAY <sec>)
Stop() → vendor_fn:"Stop"
End() → vendor_fn:"End"
5.2 Variables / Assignments
Assign(destination, source) → vendor_fn:"Assign", args:{ destination, source }
If destination resembles Hyundai registers (e.g., V1%, giToolNr, LP[9], _mo[1]), keep verbatim.
Expressions (e.g., P1.X = P1.X + R8.X) should be preserved under args.source_expression and set args.source to null.
Position/array/register forms (e.g., LP[9]=..., R[gi]=...) go into destination as-is; literals like &B10101010 preserved as strings (see R84 in Doosan spec spirit).
5.3 Coordinates / TCP / User Frames
MakeUserCoordinate(id, pose) → vendor_fn:"MakeUserCoordinate", args:{ id, pose }
(Hyundai: MKUCRD <id>,P<var>)
SelectUserCoordinate(id) → vendor_fn:"SelectUserCoordinate", args:{ id }
(Hyundai: SELUCRD <id>)
SetRefCoord(ref) → vendor_fn:"SetRefCoord", args:{ ref }
SetTCP(name_or_pose) → vendor_fn:"SetTCP", args:{ tcp: name_or_pose }
SetTool(name_or_params) → vendor_fn:"SetTool", args:{ params: name_or_params }
SetWorkpieceWeight(weight,cog) → vendor_fn:"SetWorkpieceWeight", args:{ weight, cog }
ToolChange(id) → vendor_fn:"ToolChange", args:{ id }
(Hyundai: TOOLCHNG variants; exact flags go to extras)
5.4 Global Motion Settings
(Hyundai Job often encodes speed/accuracy/tool at move site; still capture globals if present in TDL for traceability.)
SetJointVelocity(percent) → vendor_fn:"SetJointVelocity", args:{ percent }
SetJointAcceleration(percent) → vendor_fn:"SetJointAcceleration", args:{ percent }
SetTaskVelocity(mm_per_sec) → vendor_fn:"SetTaskVelocity", args:{ mm_per_sec }
SetTaskAcceleration(mm_per_sec2) → vendor_fn:"SetTaskAcceleration", args:{ mm_per_sec2 }
SetSingularityHandling(mode) → vendor_fn:"SetSingularityHandling", args:{ mode }
5.5 Synchronous Motion
Rule Change: The label_context field is no longer the primary mechanism for step numbers on MOVE lines. Instead, if the TDL Move... command was preceded by a Label command, that label's name MUST be captured in the extras.step field of the Move node. The Label command itself should not create a separate node in this case.
Map to Hyundai “MOVE” family. Populate common keys and carry non-native bits in extras.
MoveJoint(target_pose, velocity, acceleration, blending_radius, synchronized_axes)
→ vendor_fn:"MoveL" or "MoveP" depending on pose type:
If target_pose is joint space (posj(...)) → "MoveP" (Hyundai P-motion)
If target_pose is cartesian (posx(...)) → "MoveL" (Hyundai L-motion)
args:
{
  "pose": "posj(...)|posx(...)",
  "speed": <number|string>,     // "30%" pass-through allowed
  "accuracy": <number|null>,    // Hyundai A=…
  "tool": <number|string|null>, // T=…
  "radius": <number|null>       // blend / corner radius if used
}
Put nonstandard fields under extras (e.g., synchronized axes).
MoveLinear(target_pose, velocity, acceleration, blending_radius, synchronized_axes)
→ vendor_fn:"MoveL", same args as above.
MoveCircular(via_pose, target_pose, velocity, acceleration, blending_radius)
→ vendor_fn:"MoveC", args:{ via_pose:"posx(...)", target_pose:"posx(...)", speed, accuracy, tool, radius }
MoveBlend(pose_list, velocity, acceleration, blending_radius)
→ vendor_fn:"MoveBlend", args:{ poses:[ "posx|posj"… ], speed, accuracy, tool, radius }
(Hyundai expresses blended segments as consecutive MOVE lines with A/radius; keep the group intent, and later lower to lines. Put special segment kinds into extras.segments if mixed.)
Important Hyundai extras to preserve (if present in source or required at render time):
extras.step (e.g., "S12")
extras.posture (e.g., "A", "U2", "E")
extras.pose_hex (encoder-based (...)E payloads)
extras.ref / extras.sol (solver hints)
extras.percent_literal when velocity was "30%"
5.6 Asynchronous Motion / Wait
AMoveJoint/AMoveLinear/AMoveCircular → vendor_fn:null, extras:{ note:"No direct async MOVE in Hyundai; needs splitting into start/wait or ignored." }
MotionWait(handle_id) → vendor_fn:null, extras:{ note:"No general handle wait; consider barrier via WAIT inputs or completed motion checks." }
5.7 I/O (Digital / Analog)
SetDigitalOutput(port,value) → vendor_fn:"SetOutput", args:{ port, value }
(Hyundai line: DO<port>=<0|1>)
WaitForDigitalInput(port,value,timeout_sec) → vendor_fn:"WaitForInput", args:{ port, value, timeout: timeout_sec }
(Hyundai short: WAIT DI<port> assumes ON; include explicit value in args for later choice)
PulseOutput(port,duration_sec,count) →
If count==1: vendor_fn:"PulseOutput", args:{ port, value:1, ton:duration_sec, toff:null, count:1 }
If count>1: vendor_fn:"PulseOutput", args:{ port, value:1, ton:duration_sec, toff:null, count }
SetAnalogOutput(channel,value) → vendor_fn:"SetAnalogOutput", args:{ channel, value }
GetAnalogInput(channel) → vendor_fn:"GetAnalogInput", args:{ channel }
Word/Byte outputs
If TDL writes aggregate outputs via Assign(destination="DOWx"/"DOBx", source="&B..."), keep as:
vendor_fn:"Assign", args:{ destination:"DOWx|DOBx", source:"&B..." }
(Hyundai renders to DOW#=... / DOB#=....)
5.8 Welding (Hyundai Job)
ArcOn() → vendor_fn:"ArcOn", args:{ params:{} } (Hyundai: ARCON ASF#=...,JOB#=... when available; unknowns remain in extras.)
ArcOff() → vendor_fn:"ArcOff"
ConfigureArcWeaving(...) → vendor_fn:null, extras:{ note:"Use ARCON params or app-level weaving; no generic inline weave in base Job.", params:{...} }
SpotWeld(gun_id,condition_id,sequence_id) → vendor_fn:"Spot", args:{ "GN": gun_id, "CN": condition_id, "SQ": sequence_id }
SetArcCondition(...) → vendor_fn:null, extras:{ arc_condition:{...} }
5.9 Force / Compliance
StartCompliance(stiffness, ref_coord) → vendor_fn:"StartCompliance", args:{ stiffness, ref_coord } (lower later to controller-specific sequence)
ReleaseCompliance() → vendor_fn:"ReleaseCompliance"
SetDesiredForce(force, axis, ref_coord) → vendor_fn:"SetDesiredForce", args:{ force, axis, ref_coord }
5.10 UI / Logging
PrintLog(message) → vendor_fn:"Print", args:{ device:0, message_parts:[ String(message) ] } (Hyundai: PRINT #0,"...")
Popup(message,type) → vendor_fn:null, extras:{ popup:{message,type} } (use pendant UI macro later)
5.11 File Operations
LoadFile(status_variable, source_type, source_path, destination_job) → vendor_fn:"LoadFile", args:{ "status_variable": status_variable, "source_type": source_type, "source_path": source_path, "destination_job": destination_job }
(Hyundai: LOADF <var>,<type>,"...","...")
SaveFile(status_variable, target_type, target_path, source_job) → vendor_fn:"SaveFile", args:{ "status_variable": status_variable, "target_type": target_type, "target_path": target_path, "source_job": source_job }
(Hyundai: SAVEF <var>,<type>,"...","...")

6) Velocity / Acceleration Handling (no DT)
V1. If value is a percent string (e.g., "30%"), set the numeric field to null and record extras.velocity_percent=30.
V2. If numeric (e.g., 300), pass through to args.speed (Hyundai absolute mm/s).
V3. Accuracy is args.accuracy (Hyundai A=). Tool number (if known) goes to args.tool.
7) Phasing
Derive from GOAL:
Initialize_Process() → initialization
Execute_Process() → main_sequence
Finalize_Process() → finalization
If different names appear, copy verbatim into goal and set phase=goal (string).
8) Examples
8.1 Linear move with step number
This example demonstrates the new, correct behavior of merging a Label and a MoveLinear command into a single IR node.

TDL Source:

코드 스니펫

SPAWN Label(name="S1") WITH WAIT;
SPAWN MoveLinear(target_pose=PosX(136.149,...), velocity="30%", acceleration=3, blending_radius=0) WITH WAIT;
Correct Vendor IR Output:

JSON

// Notice there is NO "Label" node. The "S1" information is now inside the "MoveL" node.
{
  "vendor_fn": "MoveL",
  "source_command": "MoveLinear",
  "phase": "main_sequence",
  "goal": "Execute_Process",
  "label_context": null,
  "args": {
    "pose": "(136.149,76.122,-35.206,124.851,116.526,-32.958)A",
    "speed": null,
    "accuracy": 3,
    "tool": null,
    "radius": 0
  },
  "policy": { "wait": true },
  "extras": {
    "velocity_percent": 30,
    "step": "S1"
  }
}
8.2 DO pulse and waits
TDL:
SPAWN WaitForDigitalInput(port=72, value=1, timeout_sec=30.0) WITH WAIT;
SPAWN PulseOutput(port=74, duration_sec=1.0, count=1) WITH WAIT;
IR:
{
  "vendor_fn": "WaitForInput",
  "source_command": "WaitForDigitalInput",
  "phase": "main_sequence",
  "goal": "Execute_Process",
  "args": { "port": 72, "value": 1, "timeout": 30.0 },
  "policy": { "wait": true }
}
{
  "vendor_fn": "PulseOutput",
  "source_command": "PulseOutput",
  "phase": "main_sequence",
  "goal": "Execute_Process",
  "args": { "port": 74, "value": 1, "ton": 1.0, "toff": null, "count": 1 },
  "policy": { "wait": true }
}
8.3 Joint motion with absolute velocity
TDL:
SPAWN MoveJoint(target_pose=(0,-90,90,0,90,0), velocity=40, acceleration=60, blending_radius=0);
IR:
{
  "vendor_fn": "MoveP",
  "source_command": "MoveJoint",
  "phase": "main_sequence",
  "goal": "Execute_Process",
  "args": {
    "pose": "(0,-90,90,0,90,0)A",
    "speed": 40,
    "accuracy": 60,
    "tool": null,
    "radius": 0
  },
  "policy": { "wait": false }
}
8.4 Word/byte output via Assign
TDL:
SPAWN Assign(destination="DOW1", source="&B0000011111100000") WITH WAIT;
IR:
{
  "vendor_fn": "Assign",
  "source_command": "Assign",
  "phase": "main_sequence",
  "goal": "Execute_Process",
  "args": { "destination": "DOW1", "source": "&B0000011111100000" },
  "policy": { "wait": true }
}
8.5 Arc control (parameters unknown at TDL)
TDL:
SPAWN ArcOn() WITH WAIT;
...
SPAWN ArcOff() WITH WAIT;
IR:
{
  "vendor_fn": "ArcOn",
  "source_command": "ArcOn",
  "phase": "main_sequence",
  "goal": "Execute_Process",
  "args": {},
  "policy": { "wait": true },
  "extras": { "note": "Bind ASF#/JOB# at render-time if available" }
}
{
  "vendor_fn": "ArcOff",
  "source_command": "ArcOff",
  "phase": "main_sequence",
  "goal": "Execute_Process",
  "args": {},
  "policy": { "wait": true }
}
9) Validation Checklist
One node per SPAWN plus structural nodes (labels, flow).
No arguments dropped; unknowns in extras.
No unit/percent conversion.
Percent strings captured (e.g., extras.velocity_percent).
phase, goal, label_context, policy.wait set correctly.
Non-native/ambiguous features: vendor_fn:null + meta.warnings entry.
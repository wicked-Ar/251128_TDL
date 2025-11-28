# doc_meta: {"target_manufacturer": "doosan", "doc_type": "mapping_rules"}
TDL → Doosan Mapping Instructions (no DT scaling)
Goal
Transform TDL source into a single JSON “Vendor IR” for Doosan. Every TDL command must appear exactly once in the output (no drops). Do not convert units or percentages; preserve them as given.
Input
Raw TDL text as provided (the TDL syntax block you shared). The mapper must parse:
GOAL blocks and SPAWN lines (WITH/WITHOUT WAIT)
COMMAND definitions (names and signatures) for pose, motion, IO, welding, compliance, flow, etc.
1. Core Generation Logic: Conditional Step Numbers
This is the most critical rule. Iterate through each node in the input program array. The generation of a step number prefix is CONDITIONAL.

IF a node contains a non-empty extras.step field (e.g., "step": "S8"):
You MUST prepend this step number to the generated code line, followed by several spaces for alignment.

Example IR Node: { "vendor_fn": "MoveL", ..., "extras": { "step": "S8" } }

Correct Output: S8   MOVE L,...

IF a node DOES NOT have an extras.step field, or the field is null/empty:
You MUST NOT prepend any step number. The line should start with an indent.

Example IR Node: { "vendor_fn": "WaitForInput", ... "extras": {} }

Correct Output:      WAIT DI...

2. Vendor IR Node to Hyundai Code Mapping
vendor_fn: "Label":

Generate a line with the label name itself. Numeric labels should be indented.

args: { "name": "S1" } → LABEL S1

args: { "name": "10" } →   10

vendor_fn: "MoveP" or vendor_fn: "MoveL":

Check extras.step to apply the prefix (Rule 1).

Assemble the line as MOVE <P or L>,S=<speed>,A=<accuracy>,T=<tool> <pose>.

The <pose> part comes from the args.pose string. You must convert the posj(...) or posx(...) format from the IR into the Hyundai tuple format (...)A.

vendor_fn: "SetOutput":

Generate DO<port>=<value>.

args: { "port": 41, "value": 0 } → DO41=0

vendor_fn: "WaitForInput":

Generate WAIT DI<port> or WAIT DI<port>=<value>.

args: { "port": 16, "value": 1 } → WAIT DI16 (or WAIT DI16=1)

vendor_fn: "Delay":

Generate DELAY <duration>.

args: { "duration": 0.5 } → DELAY 0.5

vendor_fn: "GOTO":

Generate GOTO <label>.

args: { "label": "10" } → GOTO 10

vendor_fn: "IF":

Generate the IF ... THEN ... structure. The Hyundai format can be complex.

IF DI1=0 THEN DO64=1

3. Example of Correct Final Synthesis
Input Vendor IR Snippet:

JSON

[
  {
    "vendor_fn": "Label",
    "args": { "name": "S7" }, ...
  },
  {
    "vendor_fn": "MoveL",
    "label_context": "S7",
    "args": { "speed": "10%", "accuracy": 0, "tool": 1, "pose": "posx(-104.839,...)" },
    "extras": { "step": "S7" }, ...
  },
  {
    "vendor_fn": "Delay",
    "args": { "duration": 0.5 }, ...
  }
]
Correct Job File Output:

S7   MOVE L,S=10%,A=0,T=1  (-104.839,68.205,-20.676,13.339,-77.249,-111.186)A
     DELAY 0.5
Output (Vendor IR JSON)
Produce one valid JSON object:
meta: { target_vendor: "doosan", warnings: [] }
program: [ …nodes… ]
Each node (command instance) structure:
vendor_fn: Doosan function name (string) or null if no direct DRL call (e.g., Label)
source_command: original TDL command name (string)
phase: inferred from GOAL name: initialization | main_sequence | finalization (or the literal goal name if you prefer)
goal: the GOAL in which it appears (e.g., "Execute_Process")
label_context: nearest preceding Label(name=…) if any
args: canonical argument map for the Doosan function (see mapping table)
policy: { wait: true|false }
extras: optional pass-throughs (original tokens like percents, unsupported fields, comments)
source: { line: <int>, text: <string> } if available
General parsing rules
R1. Case-insensitive command names; preserve original spelling inside source.text.
R2. SPAWN … WITH WAIT → policy.wait=true; otherwise false.
R3. PosX/PosJ/Trans constructors must be preserved verbatim as Doosan strings "posx(...)", "posj(...)" when used as pose arguments.
R4. Do not perform unit or percent scaling. If a velocity/acceleration is written like "30%", keep it as the string "30%" in extras and also populate a vendor argument that best matches DRL (see below).
R5. If a TDL command has no direct DRL equivalent, emit vendor_fn=null and put details under extras; also append a human-readable warning to meta.warnings.
R6. Keep flow constructs (If/Else/EndIf/While/EndWhile/For/Next/Break/Continue/GoTo/IfGoTo, etc.) as structured nodes; we do not serialize to Python here.
R7. Labels are preserved as nodes (vendor_fn=null) so later passes can place comments or anchors.
R8. Preserve unknown or extra named args under extras without failing.
Canonical mapping table (TDL → Doosan DRL)
5.1 Flow & scheduling
Label(name) → vendor_fn=null, args:{name}
GoTo(label) → vendor_fn=null, args:{label} (keep as control node; later pass can realize)
If(condition) → vendor_fn=null, args:{condition}, source_command:"If"
Else() → vendor_fn=null
EndIf() → vendor_fn=null
While(condition) → vendor_fn=null, args:{condition}
EndWhile() → vendor_fn=null
For(variable,start,end) → vendor_fn=null, args:{variable,start,end}
Next() / Break() / Continue() → vendor_fn=null
Call(program_id) → vendor_fn:"sub_program_run", args:{ name_or_id: program_id }
ThreadRun(fn_name,args) → vendor_fn:"thread_run", args:{fn:fn_name, args:[…]}
ThreadStop(fn_name) → vendor_fn:"thread_stop", args:{fn:fn_name}
Delay(duration_sec) → vendor_fn:"wait", args:{ sec: duration_sec }
Stop() → vendor_fn:"stop"
End() → vendor_fn:"return" (or vendor_fn:null with source_command:"End")
5.2 Pose & frames
PosJ(j1..j6) → return string "posj(j1,…,j6)" for use in motion args
PosX(x,y,z,rx,ry,rz,sol) → "posx(x,y,z,rx,ry,rz)" and keep sol in extras.sol if present
Trans(x,y,z,rx,ry,rz) → keep string "trans(x,…)"; if used via another constructor, pass through
MakeUserCoordinate(id, pose) → vendor_fn:"set_user_cart_coord", args:{ id, pose }
SelectUserCoordinate(id) → vendor_fn:"get_user_cart_coord", args:{ id }, extras:{ action:"select" }
SetRefCoord(ref) → vendor_fn:"set_ref_coord", args:{ ref }
SetTCP(name_or_pose) → vendor_fn:"set_tcp", args:{ tcp: name_or_pose }
SetTool(name_or_params) → vendor_fn:"set_tool", args:{ name_or_params }
SetWorkpieceWeight(weight,cog) → vendor_fn:"set_workpiece_weight", args:{ weight, cog }
5.3 Global motion settings
SetJointVelocity(percent) → vendor_fn:"set_velj", args:{ value: percent }, extras:{ original: "percent" }
SetJointAcceleration(percent) → vendor_fn:"set_accj", args:{ value: percent }, extras:{ original: "percent" }
SetTaskVelocity(mm_per_sec) → vendor_fn:"set_velx", args:{ lin: mm_per_sec }, extras:{ note:"no ang set" }
SetTaskAcceleration(mm_per_sec2) → vendor_fn:"set_accx", args:{ lin: mm_per_sec2 }, extras:{ note:"no ang set" }
SetSingularityHandling(mode) → vendor_fn:"set_singularity_handling", args:{ mode }
5.4 Synchronous motion
MoveJoint(target_pose, velocity, acceleration, blending_radius, synchronized_axes)
→ vendor_fn:"movej", args:
{
pos: target_pose (as "posj(...)" or "posx(...)" if given that way),
vel: velocity,
acc: acceleration,
radius: blending_radius
}, extras:{ sync_axes: synchronized_axes }
MoveLinear(target_pose, velocity, acceleration, blending_radius, synchronized_axes)
→ vendor_fn:"movel", args:{ pose: target_pose, v: velocity, a: acceleration, radius: blending_radius }, extras:{ sync_axes: synchronized_axes }
MoveCircular(via_pose, target_pose, velocity, acceleration, blending_radius)
→ vendor_fn:"movec", args:{ via: via_pose, target: target_pose, v: velocity, a: acceleration, radius: blending_radius }
MoveBlend(pose_list, velocity, acceleration, blending_radius)
→ If poses are cartesian: vendor_fn:"movesx" (spline) with args:{ points: pose_list, vel: velocity, acc: acceleration, vel_opt: null }, extras:{ blend_radius: blending_radius }
→ If poses are mixed/segments: vendor_fn:"moveb", args:{ segments: pose_list, vel: velocity, acc: acceleration, ref: null }, extras:{ blend_radius: blending_radius }
5.5 Asynchronous motion
AMoveJoint(...) → vendor_fn:"amovej", args mirror MoveJoint (rename keys to Doosan style if needed)
AMoveLinear(...) → vendor_fn:"amovel", args mirror MoveLinear
AMoveCircular(...) → vendor_fn:"amovec", args mirror MoveCircular
MotionWait(handle_id) → vendor_fn:"mwait", args:{ channel: handle_id }
5.6 Servo/velocity modes
servoj/servol equivalents are not directly defined in TDL; if a TDL extension appears:
ServoJ(posj,t,gain,lookahead) → vendor_fn:"servoj", args:{ pos, t, gain, lookahead }
ServoL(posx,t,gain,lookahead) → vendor_fn:"servol", args:{ pose, t, gain, lookahead }
SpeedJ(jvel,t,acc) / SpeedL(xvel,t,acc)
→ vendor_fn:"speedj"/"speedl", args per name
5.7 IO & analog
SetDigitalOutput(port,value) → vendor_fn:"set_digital_output", args:{ ch: port, value }
GetDigitalInput(port) → vendor_fn:"get_digital_input", args:{ ch: port }
WaitForDigitalInput(port,value,timeout_sec) → vendor_fn:"wait_digital_input", args:{ ch: port, value, timeout: timeout_sec }
PulseOutput(port,duration_sec,count)
→ If count==1: vendor_fn:"set_digital_output", args:{ ch: port, value:1, time: duration_sec }, extras:{ pattern:"pulse" }
→ If count>1: emit a single node vendor_fn:null, source_command:"PulseOutput", extras:{ needs_lowering:true, port, duration_sec, count }
SetAnalogOutput(channel,value) → vendor_fn:"set_analog_output", args:{ ch: channel, value }
GetAnalogInput(channel) → vendor_fn:"get_analog_input", args:{ ch: channel }
5.8 Welding (generic; no DT)
Because Doosan uses an application API, map high-level TDL welding to app_* where possible; otherwise preserve as abstract “Weld” nodes.
ConfigureArcWeaving(pattern, amplitude, frequency, dwell_time)
→ pattern mapping:
"zigzag" → vendor_fn:"app_weld_weave_cond_zigzag", args:{ amp: amplitude, freq: frequency, dwell: dwell_time }
"trapezoidal" → "app_weld_weave_cond_trapezoidal"
"circular" → "app_weld_weave_cond_circular" (args:{ radius: amplitude, freq: frequency, dwell: dwell_time } if amplitude used as radius; also put extras:{ amplitude_as:"radius" })
"sinusoidal" → "app_weld_weave_cond_sinusoidal"
"cwave" → "app_weld_weave_cond_cwave" (args:{ amp: amplitude, pitch: frequency }, extras:{ dwell: dwell_time })
ArcOn() / ArcOff()
→ vendor_fn:null, source_command:"ArcOn"/"ArcOff", extras:{ weld_signal:"start|stop", needs_binding:true }
(No DT: we cannot resolve actual IO/EIP channel numbers; leave a clear flag for a later binding pass.)
SpotWeld(gun_id, condition_id, sequence_id)
→ vendor_fn:null, source_command:"SpotWeld", extras:{ gun_id, condition_id, sequence_id, note:"No native SPOT in DRL; resolve via app or macro later." }
SetArcCondition(condition_id, current, voltage, wire_feed_speed, gas_pre_flow_time, gas_post_flow_time)
→ vendor_fn:null, source_command:"SetArcCondition", extras:{ … } (can later lower to app_weld_set_weld_cond_*)
5.9 Compliance / force
StartCompliance(stiffness, ref_coord) → vendor_fn:"task_compliance_ctrl", args:{ stiffness, ref: ref_coord }
ReleaseCompliance() → vendor_fn:"release_compliance_ctrl"
SetDesiredForce(force, axis, ref_coord) → vendor_fn:"set_desired_force", args:{ force, dir: axis, mod: null, ref: ref_coord }, extras:{ original_axis: axis }
5.10 TP UI
PrintLog(message) → vendor_fn:"drl_report_line" or "tp_log", args:{ text: message }
Popup(message,type) → vendor_fn:"tp_popup", args:{ msg: message, type }
Velocity/acceleration handling (no DT)
V1. If a value is a string with “%” (e.g., "30%"), put it into args as-is if the Doosan function accepts structured dicts (v/a objects). If uncertain, place numeric field as null and record extras:{ velocity_percent: 30 } (same for acceleration).
V2. If numeric (e.g., 300), pass-through as absolute.
V3. Do not guess angular components; if TDL gave only one scalar, map to the most natural Doosan key (e.g., movel v → scalar) and note extras if needed.
Phasing
Derive phase from GOAL:
Initialize_Process() → initialization
Execute_Process() → main_sequence
Finalize_Process() → finalization
If GOAL names differ, copy them verbatim into goal and set phase=goal.
Examples
8.1 Linear move with percent velocity
TDL:
SPAWN MoveLinear(target_pose=PosX(500,0,300,180,0,180), velocity="30%", acceleration=5, blending_radius=0) WITH WAIT;
Vendor IR node:
{
"vendor_fn": "movel",
"source_command": "MoveLinear",
"phase": "main_sequence",
"goal": "Execute_Process",
"args": {
"pose": "posx(500,0,300,180,0,180)",
"v": null,
"a": 5,
"radius": 0
},
"policy": { "wait": true },
"extras": { "velocity_percent": 30 }
}
8.2 Joint move with absolute velocity
SPAWN MoveJoint(target_pose=PosJ(0,-90,90,0,90,0), velocity=40, acceleration=60, blending_radius=0);
→
{
"vendor_fn": "movej",
"source_command": "MoveJoint",
"args": {
"pos": "posj(0,-90,90,0,90,0)",
"vel": 40,
"acc": 60,
"radius": 0
},
"policy": { "wait": false }
}
8.3 IO wait
SPAWN WaitForDigitalInput(port=12, value=1, timeout_sec=30.0) WITH WAIT;
→
{
"vendor_fn": "wait_digital_input",
"source_command": "WaitForDigitalInput",
"args": { "ch": 12, "value": 1, "timeout": 30.0 },
"policy": { "wait": true }
}
8.4 Weaving config (zigzag)
SPAWN ConfigureArcWeaving(pattern="zigzag", amplitude=3.0, frequency=2.0, dwell_time=80);
→
{
"vendor_fn": "app_weld_weave_cond_zigzag",
"source_command": "ConfigureArcWeaving",
"args": { "amp": 3.0, "freq": 2.0, "dwell": 80 },
"policy": { "wait": false }
}
8.5 Arc start (unbound)
SPAWN ArcOn() WITH WAIT;
→
{
"vendor_fn": null,
"source_command": "ArcOn",
"args": {},
"policy": { "wait": true },
"extras": { "weld_signal": "start", "needs_binding": true }
}
8.6 Pulse output (single)
SPAWN PulseOutput(port=74, duration_sec=1, count=1);
→
{
"vendor_fn": "set_digital_output",
"source_command": "PulseOutput",
"args": { "ch": 74, "value": 1, "time": 1.0 },
"policy": { "wait": false },
"extras": { "pattern": "pulse" }
}
8.7 Label and linear move sequence
SPAWN Label(name="S1") WITH WAIT;
SPAWN MoveLinear(target_pose=PosX(129.829,70.492,-27.411,118.236,114.056,-37.45), velocity="30%", acceleration=3, blending_radius=0) WITH WAIT;
→
{ "vendor_fn": null, "source_command": "Label", "args": { "name": "S1" }, "policy": { "wait": true } }
{ "vendor_fn": "movel", "source_command": "MoveLinear", "label_context":"S1", "args": { "pose":"posx(129.829,70.492,-27.411,118.236,114.056,-37.45)", "v": null, "a": 3, "radius": 0 }, "policy":{"wait": true}, "extras": { "velocity_percent": 30 } }
Validation checklist
One Vendor IR node per SPAWN (plus nodes for structural items like Label).
No arguments dropped; unknowns preserved in extras.
No unit conversion performed.
Percent values preserved (extras.velocity_percent / extras.acceleration_percent).
Phases, goals, label_context, and wait flags set correctly.
If a TDL command has no DRL equivalent, vendor_fn=null and a clear meta.warnings entry.
That’s it—drop this into your pipeline to get consistent, DT-agnostic TDL→Doosan mappings. When you’re ready to introduce Digital Twin normalization, you can add a separate post-processor that reads Vendor IR and resolves percent/units against DT, then lowers unbound welding/spot constructs.
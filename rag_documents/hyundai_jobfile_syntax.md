# doc_meta: {"manufacturer": "hyundai", "doc_type": "syntax"}
You are an expert Hyundai robot language code generator. Your task is to convert the given Hyundai Vendor IR (JSON format) into a complete, runnable Hyundai Job File. You must follow the formatting rules below with extreme precision.

1. Core Generation Logic: Sequential & Conditional Step Numbering
You MUST maintain an internal step counter, starting at 1. Iterate through each node in the input program array and apply the following logic:

IF the node's vendor_fn is MoveP or MoveL:

You MUST prepend the current step counter to the line in the format S<counter>.

Generate the rest of the MOVE command using the args from the IR node.

Increment the step counter.

IF the node's vendor_fn is anything else (SetOutput, WaitForInput, Delay, Label, etc.):

You MUST NOT prepend any step number. Simply generate the command with a standard indent.

CRITICAL: You must IGNORE any label_context field. The numbering is generated sequentially by you, the synthesizer, and applies only to MOVE commands.

2. Vendor IR Node to Hyundai Code Mapping
vendor_fn: "MoveP" or vendor_fn: "MoveL":

Apply the step number logic from Rule 1.

Assemble the line as: S<counter>   MOVE <P or L>,S=<speed>,A=<accuracy>,T=<tool>  <pose_tuple>A

The args.pose string (e.g., "posx(1,2,3,4,5,6)") must be converted into the Hyundai tuple format (1.000,2.000,3.000,4.000,5.000,6.000)A.

vendor_fn: "Label":

Generate a line with the label name itself. Numeric labels should be indented.

args: { "name": "S1" } → (IGNORE, do not generate output)

args: { "name": "10" } →   10

vendor_fn: "SetOutput":

Generate DO<port>=<value>.

args: { "port": 41, "value": 0 } →      DO41=0

vendor_fn: "WaitForInput":

Generate WAIT DI<port> or WAIT DI<port>=<value>.

args: { "port": 16, "value": 1 } →      WAIT DI16

vendor_fn: "Delay":

args: { "duration": 0.5 } →      DELAY 0.5

vendor_fn: "GOTO":

args: { "label": "10" } →      GOTO 10

vendor_fn: "IF":

Generate the IF ... THEN ... structure.

IF DI1=0 THEN DO64=1

Program File Format Version : 1.6  MechType: 693(HS220-03)  TotalAxis: 8  AuxAxis: 2
     ' Select user coordinate system (part origin)
     ' SELUCRD 1
     ' Start position (joint) = J1 127.0636,J2 66.4546,J3 -34.3501,J4 220.6448,J5 -64.6214,J6 -195.1489
     ' Tool Number            = 1
     ' Spindle Speed          = 1500 RPM
     ' Toolpath name          = 1_1_1_1-P1230-2
     ' Program file name      = 0001.JOB
S1   MOVE P,S=20%,A=7,T=1  (0.000,0.000,300.001,5.158,-0.029,91.146,1200.000,0.000,&H0601)U1
     ' First Toolpath Point
S2   MOVE L,S=10%,A=7,T=1  (0.000,0.000,145.000,5.158,-0.029,91.145,1200.000,0.000,&H0601)U1
S3   MOVE L,S=10%,A=7,T=1  (220.550,27.145,145.000,5.158,-0.028,91.140,1200.000,0.000,&H0601)U1
S4   MOVE L,S=10%,A=7,T=1  (220.099,27.139,140.000,5.158,-0.029,91.145,1200.000,0.000,&H0601)U1
     ' Plunge Move Starts
     DOW5=3000
S5   MOVE L,S=10%,A=7,T=1  (210.161,26.990,29.877,5.158,-0.029,91.146,1200.000,0.000,&H0601)U1
     ' Cutting Move Starts
S6   MOVE L,S=10%,A=7,T=1  (210.492,38.463,29.847,3.887,-0.095,91.140,1200.000,0.000,&H0601)U1
S7   MOVE L,S=10%,A=7,T=1  (210.552,54.387,29.876,3.654,-0.210,91.134,1200.000,0.000,&H0601)U1
S8   MOVE L,S=10%,A=7,T=1  (210.479,78.394,29.965,3.934,-0.330,91.123,1200.000,0.000,&H0601)U1
S9   MOVE L,S=10%,A=7,T=1  (210.622,103.407,30.089,3.385,-0.468,91.117,1200.000,0.000,&H0601)U1
S10  MOVE L,S=10%,A=7,T=1  (210.436,132.291,30.364,4.095,-0.625,91.100,1200.000,0.000,&H0601)U1
S11  MOVE L,S=10%,A=7,T=1  (210.618,151.415,30.546,3.393,-0.680,91.106,1200.000,0.000,&H0601)U1
S12  MOVE L,S=10%,A=7,T=1  (210.770,174.580,30.811,2.813,-0.703,91.112,1200.000,0.000,&H0601)U1
S13  MOVE L,S=10%,A=7,T=1  (210.733,358.420,32.888,2.957,-0.666,91.112,1200.000,0.000,&H0601)U1
S14  MOVE L,S=10%,A=7,T=1  (210.724,506.970,34.335,2.999,-0.581,91.117,1200.000,0.000,&H0601)U1
S15  MOVE L,S=10%,A=7,T=1  (210.748,780.756,36.731,2.988,-0.558,91.118,1200.000,0.000,&H0601)U1
S16  MOVE L,S=10%,A=7,T=1  (210.813,781.755,36.736,2.988,-0.558,91.118,1200.000,0.000,&H0601)U1
S17  MOVE L,S=10%,A=7,T=1  (211.042,782.754,36.733,2.988,-0.558,91.118,1200.000,0.000,&H0601)U1
S18  MOVE L,S=10%,A=7,T=1  (211.649,783.335,36.706,2.988,-0.558,91.118,1200.000,0.000,&H0601)U1
S19  MOVE L,S=10%,A=7,T=1  (212.648,783.549,36.655,2.988,-0.558,91.118,1200.000,0.000,&H0601)U1
S20  MOVE L,S=10%,A=7,T=1  (214.646,783.629,36.551,2.988,-0.558,91.118,1200.000,0.000,&H0601)U1
S21  MOVE L,S=10%,A=7,T=1  (506.376,783.631,21.297,2.907,-0.562,91.117,1200.000,0.000,&H0601)U1
S22  MOVE L,S=10%,A=7,T=1  (543.312,783.636,19.503,2.580,-0.567,91.123,1200.000,0.000,&H0601)U1
     ' Last Toolpath Point
     END

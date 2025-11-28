# doc_meta: {"manufacturer": "doosan", "doc_type": "syntax"}
Title : h2017_tire_doosan2
Time : 2025-07-29 21:12:03
set_singular_handling(DR_AVOID)
set_velj(60.0)
set_accj(60.0)
set_velx(250.0, 80.625, DR_OFF)
set_accx(1000.0, 322.5)
gLoop211203046 = 0
while gLoop211203046 < 1:
    # CommentNode
    tp_log("move command")
    # MoveJNode
    movej(posj(0.00, 0.01, 132.64, 0.00, 15.00, -0.01), radius=0.00, ra=DR_MV_RA_DUPLICATE)
    # CommentNode
    tp_log("wait command")
    # WaitNode
    wait(3.00)
    # CommentNode
    tp_log("DIO command")
    # WaitNode
    wait_digital_input(11,ON)
    # IfBlockNode
    if True:
        set_digital_output(1,ON)
    else:
        exit()
        sub_program_run("subend")
    # WaitNode
    wait_digital_input(1,ON)
    # SetNode
    set_digital_output(1,ON,1.0,ON)
    # CommentNode
    tp_log("Call command")
    gLoop211203046 = gLoop211203046 + 1

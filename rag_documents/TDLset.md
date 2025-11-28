# doc_meta: {"doc_type": "tdl_syntax"}
# TDL (Task Description Language) Syntax and Command Set

GOAL Initialize_Process()
{
}
GOAL Execute_Process()
{
}
GOAL Finalize_Process()
{
}
COMMAND If(condition) { system.flow.if(condition=condition); }
COMMAND Else() { system.flow.else(); }
COMMAND EndIf() { system.flow.endif(); }
COMMAND While(condition) { system.flow.while(condition=condition); }
COMMAND EndWhile() { system.flow.endwhile(); }
COMMAND For(variable, start, end) { system.flow.for(variable=variable, start=start, end=end); }
COMMAND Next() { system.flow.next(); }
COMMAND Break() { system.flow.break(); }
COMMAND Continue() { system.flow.continue(); }
COMMAND Call(program_id) { system.task.call(program_id=program_id); }
COMMAND ThreadRun(fn_name, args=[]) { system.task.thread_run(function=fn_name, args=args); }
COMMAND ThreadStop(fn_name) { system.task.thread_stop(function=fn_name); }
COMMAND Delay(duration_sec) { system.time.delay(duration_sec=duration_sec); }
COMMAND Stop() { system.execution.stop(); }
COMMAND End() { system.execution.end(); }
COMMAND Label(name) { system.flow.define_label(name=name); }
COMMAND GoTo(label) { system.flow.goto(label=label); }
COMMAND IfGoTo(condition, label) { system.flow.if_goto(condition=condition, label=label); }
COMMAND Assign(destination, source) { system.variable.set(destination=destination, source=source); }
COMMAND PrintLog(message) { system.io.print_log(message=message); }
COMMAND Popup(message, type) { system.io.popup(message=message, type=type); }
COMMAND PosJ(j1, j2, j3, j4, j5, j6) { return types.pose.joint(j1=j1, j2=j2, j3=j3, j4=j4, j5=j5, j6=j6); }
COMMAND PosX(x, y, z, rx, ry, rz, sol=None) { return types.pose.cartesian(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz, solution=sol); }
COMMAND Trans(x, y, z, rx, ry, rz) { return types.pose.offset(x=x, y=y, z=z, rx=rx, ry=ry, rz=rz); }
COMMAND MakeUserCoordinate(id, pose) { robot.config.set_user_coordinate(id=id, pose=pose); }
COMMAND SelectUserCoordinate(id) { robot.config.select_user_coordinate(id=id); }
COMMAND SetRefCoord(ref) { robot.config.set_reference_coordinate(ref=ref); }
COMMAND SetTool(name_or_params) { robot.config.set_tool_parameters(params=name_or_params); }
COMMAND SetWorkpieceWeight(weight, cog) { robot.config.set_workpiece(weight=weight, center_of_gravity=cog); }
COMMAND ToolChange(id) { robot.tool_changer.execute(id=id); }
COMMAND SetJointVelocity(percent) { robot.config.set_joint_velocity(percent=percent); }
COMMAND SetJointAcceleration(percent) { robot.config.set_joint_acceleration(percent=percent); }
COMMAND SetTaskVelocity(mm_per_sec) { robot.config.set_task_velocity(velocity=mm_per_sec); }
COMMAND SetTaskAcceleration(mm_per_sec2) { robot.config.set_task_acceleration(acceleration=mm_per_sec2); }
COMMAND SetSingularityHandling(mode) { robot.config.set_singularity_mode(mode=mode); }
COMMAND MoveJoint(target_pose, velocity, acceleration, tool, blending_radius, synchronized_axes=None) { motion.execute(type="Joint", pose=target_pose, vel=velocity, acc=acceleration, tool=tool, blend=blending_radius, sync=synchronized_axes); }
COMMAND MoveLinear(target_pose, velocity, acceleration, tool, blending_radius, synchronized_axes=None) { motion.execute(type="Linear", pose=target_pose, vel=velocity, acc=acceleration, tool=tool, blend=blending_radius, sync=synchronized_axes); }
COMMAND MoveCircular(via_pose, target_pose, velocity, acceleration, tool, blending_radius) { motion.execute(type="Circular", via=via_pose, target=target_pose, vel=velocity, acc=acceleration, tool=tool, blend=blending_radius); }
COMMAND MoveBlend(pose_list, velocity, acceleration, blending_radius) { motion.execute(type="Blend", poses=pose_list, vel=velocity, acc=acceleration, tool=tool, blend=blending_radius); }
COMMAND AMoveJoint(target_pose, velocity, acceleration, tool, blending_radius) { return motion.execute_async(type="Joint", pose=target_pose, vel=velocity, acc=acceleration, tool=tool, blend=blending_radius); }
COMMAND AMoveLinear(target_pose, velocity, acceleration, tool, blending_radius) { return motion.execute_async(type="Linear", pose=target_pose, vel=velocity, acc=acceleration, tool=tool, blend=blending_radius); }
COMMAND AMoveCircular(via_pose, target_pose, velocity, acceleration, tool, blending_radius) { return motion.execute_async(type="Circular", via=via_pose, target=target_pose, vel=velocity, tool=tool, acc=acceleration, blend=blending_radius); }
COMMAND MotionWait(handle_id) { motion.wait(handle=handle_id); }
COMMAND SetDigitalOutput(port, value) { io.digital.set(port=port, value=value); }
COMMAND GetDigitalInput(port) { return io.digital.get(port=port); }
COMMAND WaitForDigitalInput(port, value, timeout_sec) { return io.digital.wait(port=port, value=value, timeout=timeout_sec); }
COMMAND PulseOutput(port, duration_sec, count) { io.digital.pulse(port=port, duration=duration_sec, count=count); }
COMMAND SetAnalogOutput(channel, value) { io.analog.set(channel=channel, value=value); }
COMMAND GetAnalogInput(channel) { return io.analog.get(channel=channel); }
COMMAND SpotWeld(gun_id, condition_id, sequence_id)    { application.spot_weld.run(gun=gun_id, condition=condition_id, sequence=sequence_id); }
COMMAND SetArcCondition(condition_id, current, voltage, wire_feed_speed, gas_pre_flow_time, gas_post_flow_time) { application.arc_weld.set_condition(id=condition_id, current=current, voltage=voltage, wire_speed=wire_feed_speed, pre_flow=gas_pre_flow_time, post_flow=gas_post_flow_time); }
COMMAND ConfigureArcWeaving(pattern, amplitude, frequency, dwell_time) { application.arc_weld.configure_weaving(pattern=pattern, amplitude=amplitude, frequency=frequency, dwell=dwell_time); }
COMMAND ArcOn() { application.arc_weld.start(); }
COMMAND ArcOff() { application.arc_weld.stop(); }
COMMAND StartCompliance(stiffness, ref_coord) { robot.force_control.start_compliance(stiffness=stiffness, ref=ref_coord); }
COMMAND ReleaseCompliance() { robot.force_control.release_compliance(); }
COMMAND SetDesiredForce(force, axis, ref_coord) { robot.force_control.set_force(force=force, axis=axis, ref=ref_coord); }
# doc_meta: {"doc_type": "digital_twin"}
{
  "meta": {
    "schema_version": "1.0.0",
    "source": "RAG/DT/manual",
    "last_updated": "YYYY-MM-DD",
    "notes": ""
  },
  "identity": {
    "brand": "hyundai | doosan | ...",
    "model": "e.g., HR-12",
    "controller": "e.g., Hi5 | DRCF",
    "firmware": "e.g., v3.4.1",
    "serial": "optional"
  },
  "units": {
    "angle": "deg | rad",
    "linear": "mm",
    "time": "s"
  },
  "kinematics": {
    "dof": 6,
    "joint_order": ["J1","J2","J3","J4","J5","J6"],
    "joint_limits": {
      "J1": { "min": -180, "max": 180, "unit": "deg" },
      "J2": { "min": -120, "max": 120, "unit": "deg" },
      "J3": { "min": -140, "max": 140, "unit": "deg" },
      "J4": { "min": -360, "max": 360, "unit": "deg" },
      "J5": { "min": -135, "max": 135, "unit": "deg" },
      "J6": { "min": -360, "max": 360, "unit": "deg" }
    },
    "reach_mm": 1300,
    "payload_kg": 12,
    "mount": "floor | wall | ceiling"
  },
  "tcp": {
    "active": "default",
    "list": {
      "default": { "pose": "posx(0,0,0,0,0,0)", "weight_kg": 0.5, "cog_mm": [0,0,50] },
      "welding_gun_tcp": { "pose": "posx(0,0,220,0,0,0)", "weight_kg": 3.2, "cog_mm": [10,0,120] }
    }
  },
  "frames": {
    "ref_default": "BASE",
    "user_frames": {
      "USER_BASE_1": { "id": 1, "pose": "posx(1000,0,0,0,0,0)" },
      "USER_BASE_2": { "id": 2, "pose": "posx(0,500,0,0,0,90)" }
    }
  },
  "io": {
    "digital_out_count": 128,
    "digital_in_count": 128,
    "tool_dout_count": 8,
    "tool_din_count": 8,
    "latency_ms_nominal": 5,
    "groups": {
      "word": { "DOW1": { "ports": [1,16] }, "DOW2": { "ports": [17,32] } },
      "byte": { "DOB1": { "ports": [1,8] }, "DOB2": { "ports": [9,16] } }
    },
    "semantics": {
      "Welder_Ready": { "type": "DO", "port": 10 },
      "Welder_Ack": { "type": "DI", "port": 11 },
      "Weld_Start": { "type": "DO", "port": 12 },
      "Weld_In_Progress": { "type": "DI", "port": 13 }
    }
  },
  "limits": {
    "joint": {
      "max_vel_deg_s": { "J1": 180, "J2": 180, "J3": 180, "J4": 360, "J5": 360, "J6": 360 },
      "max_acc_deg_s2": { "J1": 300, "J2": 300, "J3": 300, "J4": 600, "J5": 600, "J6": 600 }
    },
    "task": {
      "max_lin_vel_mm_s": 1500,
      "max_lin_acc_mm_s2": 6000,
      "max_ang_vel_deg_s": 360,
      "max_ang_acc_deg_s2": 1500
    },
    "blending": {
      "radius_mm": { "min": 0, "max": 50 }
    }
  },
  "percent_maps": {
    "joint_velocity": {
      "scale": "piecewise_linear",
      "points": [[0,0],[10,30],[50,120],[100,180]],
      "unit": "deg/s",
      "per_joint_multiplier": { "J4": 2.0, "J5": 2.0, "J6": 2.0 }
    },
    "joint_acceleration": {
      "scale": "piecewise_linear",
      "points": [[0,0],[50,150],[100,300]],
      "unit": "deg/s^2"
    },
    "task_linear_velocity": {
      "scale": "piecewise_linear",
      "points": [[0,0],[20,150],[50,400],[80,900],[100,1200]],
      "unit": "mm/s"
    },
    "task_linear_acceleration": {
      "scale": "piecewise_linear",
      "points": [[0,0],[50,3000],[100,6000]],
      "unit": "mm/s^2"
    },
    "task_angular_velocity": {
      "scale": "piecewise_linear",
      "points": [[0,0],[50,180],[100,360]],
      "unit": "deg/s"
    },
    "task_angular_acceleration": {
      "scale": "piecewise_linear",
      "points": [[0,0],[50,750],[100,1500]],
      "unit": "deg/s^2"
    }
  },
  "vendor_equivalents": {
    "hyundai": {
      "move_linear": { "name": "MoveL", "args": ["pose","speed","accuracy","tool","radius"] },
      "move_joint_cartesian": { "name": "MoveL" },
      "move_joint_jointspace": { "name": "MoveP" },
      "accuracy_to_A": { "scale": "identity", "unit": "A_index" }
    },
    "doosan": {
      "move_linear": { "name": "movel", "v_field": "linear_speed", "a_field": "linear_acc" },
      "move_joint_jointspace": { "name": "movej", "v_field": "vel", "a_field": "acc" },
      "radius_alias": ["radius","r"]
    }
  },
  "render_policies": {
    "percent_velocity_interpretation": "task_linear_velocity", 
    "percent_joint_velocity_interpretation": "joint_velocity",
    "default_accuracy_A": 1,
    "tool_default": null,
    "allow_percent_literals_on_vendor": ["hyundai"], 
    "clip_to_limits": true,
    "interpolation": "linear"
  },
  "welding": {
    "supported": true,
    "arc": {
      "asf_default": 1,
      "job_default": 1,
      "start_delay_ms": 100,
      "stop_postflow_ms": 300
    },
    "weaving": {
      "patterns": ["trapezoidal","zigzag","circular","sinusoidal","cwave"],
      "amplitude_mm_max": 10,
      "frequency_hz_max": 10
    }
  },
  "force_control": {
    "supported": true,
    "stiffness_range": { "xyz": [100, 5000], "rxyz": [10, 500] },
    "ref_frames": ["BASE","TASK","TCP"]
  }
}

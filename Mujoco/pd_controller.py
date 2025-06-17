import mujoco
import mujoco.viewer
import numpy as np

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path("biped.xml")
data = mujoco.MjData(model)

# Viewer for real-time visualization (optional, needs a display)
viewer = mujoco.viewer.launch_passive(model, data)

# Define PD control parameters
Kp = 50
Kd = 1

# Target joint angles (standing pose)
target_angles = {
    "left_hip": 0.0,
    "left_knee": -0.5,
    "left_ankle": 0.5,
    "right_hip": 0.0,
    "right_knee": -0.5,
    "right_ankle": 0.5,
}

# Simulation loop
while viewer.is_running():
    mujoco.mj_step(model, data)

    for name, target in target_angles.items():
        j_id = model.joint(name).id
        qpos = data.qpos[j_id]
        qvel = data.qvel[j_id]
        torque = Kp * (target - qpos) - Kd * qvel
        data.ctrl[j_id] = torque

    viewer.sync()

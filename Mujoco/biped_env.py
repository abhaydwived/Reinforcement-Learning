import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco as m
import mujoco.viewer
import os

class BipedMujocoEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()

        # Load MuJoCo model
        urdf_path = os.path.join(os.path.dirname(__file__), "biped.xml")
        self.model = m.MjModel.from_xml_path(urdf_path)
        self.data = m.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        # Parameters
        self.sim_steps_per_action = 5
        self.timestep = self.model.opt.timestep
        self.max_steps = 1000
        self.step_counter = 0

        # Identify joints and DOFs
        self.actuated_joint_names = [name for name in self.model.joint_names if self.model.jnt_type[self.model.joint(name).id] == m.mjtJoint.mjJNT_HINGE]
        self.joint_ids = [self.model.joint(name).qposadr for name in self.actuated_joint_names]
        self.n_actuated_joints = len(self.joint_ids)

        # Action and Observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_actuated_joints,), dtype=np.float32)
        obs_dim = self.n_actuated_joints * 2 + 6 + 6 + 2  # joint pos + vel + base orientation + lin/ang vel + foot contact
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[2] = 0.7  # initial z height
        self.step_counter = 0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)

        for i, joint_idx in enumerate(self.joint_ids):
            self.data.ctrl[i] = action[i]  # assumes actuator is mapped 1:1 with joints

        for _ in range(self.sim_steps_per_action):
            m.mj_step(self.model, self.data)

        self.step_counter += 1
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = self._check_termination(obs)
        truncated = self.step_counter >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        joint_angles = self.data.qpos[3:3 + self.n_actuated_joints].copy()
        joint_vels = self.data.qvel[3:3 + self.n_actuated_joints].copy()

        base_linvel = self.data.qvel[:3]
        base_angvel = self.data.qvel[3:6]

        base_quat = self.data.qpos[3:7]  # Orientation quaternion
        base_euler = self._quat_to_euler(base_quat)

        left_contact = 1.0 if self.data.cfrc_ext[2][2] > 10 else 0.0  # crude contact
        right_contact = 1.0 if self.data.cfrc_ext[3][2] > 10 else 0.0

        return np.concatenate([joint_angles, joint_vels, base_euler, base_linvel, base_angvel, [left_contact, right_contact]]).astype(np.float32)

    def _compute_reward(self, obs):
        pitch, roll = obs[self.n_actuated_joints * 2 + 0], obs[self.n_actuated_joints * 2 + 1]
        forward_vel = obs[self.n_actuated_joints * 2 + 4]
        upright_bonus = 1 - (abs(pitch) + abs(roll))

        joint_effort = np.sum(np.square(self.data.ctrl))
        height = self.data.qpos[2]

        reward = (
            1.5 * forward_vel +
            0.5 * upright_bonus -
            0.01 * joint_effort +
            1.0 * (height > 0.6)
        )
        return reward

    def _check_termination(self, obs):
        pitch, roll = obs[self.n_actuated_joints * 2 + 0], obs[self.n_actuated_joints * 2 + 1]
        height = self.data.qpos[2]
        return height < 0.5 or abs(pitch) > 1.2 or abs(roll) > 1.2

    def _quat_to_euler(self, quat):
        w, x, y, z = quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return np.array([roll_x, pitch_y, yaw_z])

    def render(self):
        if self.viewer:
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()

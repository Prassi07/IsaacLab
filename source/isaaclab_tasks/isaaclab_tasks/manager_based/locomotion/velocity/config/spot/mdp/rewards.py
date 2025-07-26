# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the reward functions that can be used for Spot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from isaaclab.utils.math import quat_rotate_inverse, euler_xyz_from_quat
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


##
# Task Rewards
##


def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


def base_angular_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / std)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, ramp_at_vel: float = 1.0, ramp_rate: float = 0.5
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    # fixed 1.0 multiple for tracking below the ramp_at_vel value, then scale by the rate above
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple

def standing_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, velocity_threshold: float = 0.25
) -> torch.Tensor:
    
    """Reward tracking of for 0 velocity using abs exponential kernel."""
     
    # Extract the asset (make sure type hints match your actual classes)
    asset: RigidObject = env.scene[asset_cfg.name] # Replace PlaceholderRigidObject with RigidObject

    # Get commanded linear velocity (x, y)
    target_vel_xy = env.command_manager.get_command("base_velocity")[:, :2]
    
    # Get current linear velocity (x, y) in base frame
    current_vel_xy = asset.data.root_lin_vel_b[:, :2]

    # Calculate the magnitude (speed) of commanded and current velocities
    target_speed = torch.linalg.norm(target_vel_xy, dim=1)
    current_speed = torch.linalg.norm(current_vel_xy, dim=1)

    # --- Core Logic ---
    # 1. Check if the command is to stand still (magnitude below threshold)
    #    Result is a boolean tensor (True where command is near zero)
    is_command_zero = (target_speed < velocity_threshold)

    # 2. Calculate the reward potential based on how close the *current* speed is to zero
    #    using an exponential kernel. Reward is high (near 1.0) when current_speed is near zero.
    standing_potential = torch.exp(-current_speed / std)

    # 3. Grant the reward *only* when the command was effectively zero.
    #    Multiply the potential reward by the boolean mask (converted to float: True->1.0, False->0.0)
    reward = standing_potential * is_command_zero.float()

    return reward

def standing_reward_xy(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, velocity_threshold: float = 0.25
) -> torch.Tensor:
    """Penalize for not standing still when commanded to do so.

    This reward is 1.0 if the commanded XY-velocity is below the `velocity_threshold`
    but the actual XY-velocity is not, and 0.0 otherwise.
    """
    # Extract the asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Get commanded linear velocity (x, y)
    target_vel_xy = env.command_manager.get_command("base_velocity")[:, :2]
    # Get current linear velocity (x, y) in base frame
    current_vel_xy = asset.data.root_lin_vel_b[:, :2]

    # Calculate the magnitude (speed) of commanded and current velocities
    target_speed = torch.linalg.norm(target_vel_xy, dim=1)
    current_speed = torch.linalg.norm(current_vel_xy, dim=1)

    # Check if the command is to stand still
    is_command_zero = (target_speed < velocity_threshold)
    # Check if the robot is actually standing still
    is_actually_standing = (current_speed < velocity_threshold)

    # Penalize/ if commanded to stand but not actually standing
    condition = is_command_zero & is_actually_standing
    reward = condition.float()

    return reward

class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


##
# Regularization Penalties
##


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


# ! look into simplifying the kernel here; it's a little oddly complex
def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)


def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.1, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)


def body_termination_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    
    # if(torch.sum(is_contact, dim=1) > 0):
    #     return torch.ones(is_contact.size(dim=0))
    # else:
    #     return torch.zeros(is_contact.size(dim=0))
    
    violation_present = torch.sum(is_contact, dim=1) > 0
    return violation_present.float()


def base_linear_z_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

def base_angular_xy_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.norm(asset.data.root_ang_vel_b[:, :2], dim=1)

def joint_acceleration_square_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    acc_norm = torch.linalg.norm((asset.data.joint_acc), dim=1)
    return torch.square(acc_norm)


def joint_torques_square_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    torq_norm = torch.linalg.norm((asset.data.applied_torque), dim=1)
    return torch.square(torq_norm)

def joint_velocity_square_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    vel_norm = torch.linalg.norm((asset.data.joint_vel), dim=1)
    return torch.square(vel_norm)

def action_smoothness_square_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    act_norm =torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)
    return torch.square(act_norm)

# Pedipulation Rewards - Task based
def pedipulation_goal_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    leg_asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward Foot Reaching the Goal. (left leg for now)"""
    # extract the used quantities (to enable type-hinting)
    
    asset: Articulation = env.scene[asset_cfg.name]
    leg: RigidObject = env.scene[leg_asset_cfg.name]
    
    target_b = env.command_manager.get_command("foot_position")
    
    curr_pos_w = leg.data.body_pos_w[:, leg_asset_cfg.body_ids, :].squeeze()
    # print("LEG: ", curr_pos_w.size())
    
    robot_pos_w = asset.data.root_pos_w
    robot_quat_w = asset.data.root_quat_w
    # print("BODy: ", robot_pos_w.size(), robot_quat_w.size())
    
    curr_pos_b = quat_rotate_inverse(robot_quat_w, curr_pos_w - robot_pos_w)
    
    pos_error = torch.linalg.norm((target_b - curr_pos_b), dim = 1)

    return torch.exp(-pos_error / std)


# def multileg_pedipulation_reward(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg,
#     left_leg_asset_cfg: SceneEntityCfg,
#     right_leg_asset_cfg: SceneEntityCfg,
#     std: float,
# ) -> torch.Tensor:
#     """Reward the commanded leg for reaching its 3D position goal in the base frame.

#     The command includes which leg to target (0 for left, 1 for right).
#     The reward is calculated based on the distance error of the commanded leg to its target.
#     """
    
#     # Extract assets
#     robot_asset: Articulation = env.scene[asset_cfg.name]
#     # Note: left_leg_asset_cfg and right_leg_asset_cfg are used to get body_ids for the respective feet.
#     # The actual position data comes from robot_asset.data.body_pos_w using these IDs.

#     # Get the 4D command: (local_x, local_y, local_z, leg_switch)
#     # leg_switch is 0 for left, 1 for right.
#     full_command = env.command_manager.get_command("foot_position")
#     target_pos_b = full_command[:, :3]  # Target position in base frame
    
#     # leg_command = torch.zeros_like(full_command[:, 2])
#     # is_right_leg_commanded_mask = (leg_command == 1.0) 
    
#     # leg_switch_command will be (num_envs,). True if right leg is commanded.
#     is_right_leg_commanded_mask = (full_command[:, 3] == 1.0)

#     # Get current world positions of the feet
#     left_foot_pos_w = robot_asset.data.body_pos_w[:, left_leg_asset_cfg.body_ids, :].squeeze()
#     right_foot_pos_w = robot_asset.data.body_pos_w[:, right_leg_asset_cfg.body_ids, :].squeeze()

#     # Get robot's root pose for transformation
#     robot_pos_w = robot_asset.data.root_pos_w
#     robot_quat_w = robot_asset.data.root_quat_w

#     # Transform foot positions from world to robot's base frame
#     left_foot_pos_b = quat_rotate_inverse(robot_quat_w, left_foot_pos_w - robot_pos_w)
#     right_foot_pos_b = quat_rotate_inverse(robot_quat_w, right_foot_pos_w - robot_pos_w)

#     # Calculate position error for both legs
#     error_left_leg = torch.linalg.norm((target_pos_b - left_foot_pos_b), dim=1)
#     error_right_leg = torch.linalg.norm((target_pos_b - right_foot_pos_b), dim=1)

#     # Select the error corresponding to the commanded leg
#     commanded_leg_error = torch.where(is_right_leg_commanded_mask, error_right_leg, error_left_leg)

#     return torch.exp(-commanded_leg_error / std)

def multileg_pedipulation_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    left_leg_asset_cfg: SceneEntityCfg,
    right_leg_asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Reward the commanded leg for reaching its 3D position goal in the base frame.

    The command is 5D: (x, y, z, left_leg_cmd, right_leg_cmd).
    - [1, 0] for left leg command.
    - [0, 1] for right leg command.
    - [0, 0] for standing command.

    For leg swing commands, the reward is based on the distance error of the commanded leg.
    For standing commands, the reward is zero.
    """
    
    # Extract assets
    robot_asset: Articulation = env.scene[robot_cfg.name]

    # Get the 5D command: (local_x, local_y, local_z, left_cmd, right_cmd)
    full_command = env.command_manager.get_command("foot_position")
    target_pos_b = full_command[:, :3]  # Target position in base frame
    leg_commands = full_command[:, 3:5] # Leg command bits

    # Create masks based on the leg commands
    is_left_leg_commanded_mask = (leg_commands[:, 0] == 1.0)
    is_right_leg_commanded_mask = (leg_commands[:, 1] == 1.0)
    # Standing command is when both leg commands are 0
    is_standing_command_mask = ~is_left_leg_commanded_mask & ~is_right_leg_commanded_mask

    # Get current world positions of the feet
    left_foot_pos_w = robot_asset.data.body_pos_w[:, left_leg_asset_cfg.body_ids, :].squeeze()
    right_foot_pos_w = robot_asset.data.body_pos_w[:, right_leg_asset_cfg.body_ids, :].squeeze()

    # Get robot's root pose for transformation
    robot_pos_w = robot_asset.data.root_pos_w
    robot_quat_w = robot_asset.data.root_quat_w

    # Transform foot positions from world to robot's base frame
    left_foot_pos_b = quat_rotate_inverse(robot_quat_w, left_foot_pos_w - robot_pos_w)
    right_foot_pos_b = quat_rotate_inverse(robot_quat_w, right_foot_pos_w - robot_pos_w)

    # Calculate position error for both legs
    error_left_leg = torch.linalg.norm((target_pos_b - left_foot_pos_b), dim=1)
    error_right_leg = torch.linalg.norm((target_pos_b - right_foot_pos_b), dim=1)

    # Select the error corresponding to the commanded leg.
    # Initialize error to zero. It will remain zero for standing commands.
    commanded_leg_error = torch.zeros_like(error_left_leg)
    commanded_leg_error = torch.where(is_left_leg_commanded_mask, error_left_leg, commanded_leg_error)
    commanded_leg_error = torch.where(is_right_leg_commanded_mask, error_right_leg, commanded_leg_error)

    # Calculate reward - it will be 1.0 for standing commands since error is 0.
    reward = torch.exp(-commanded_leg_error / std)

    # Zero out the reward for standing commands as requested.
    reward = torch.where(is_standing_command_mask, 0.0, reward)

    return reward

def zero_velocity_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    velocity_std: float,
) -> torch.Tensor:
    """
    Rewards the robot for standing stably on four feet when a standing command is issued.

    This reward is only active when the command is [0, 0] (standing). It checks for:
    1. Contact on all four feet, verified by checking contact forces.
    2. Minimal linear and angular velocity of the base to ensure stillness.
    3. An upright body orientation (minimal roll and pitch).
    """
    # --- 1. Extract assets and check for standing command ---

    # Extract robot and contact sensor from the environment scene
    robot: Articulation = env.scene[robot_cfg.name]
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]

    # Get the 5D command from the command manager
    # Command format: (x, y, z, left_cmd, right_cmd)
    full_command = env.command_manager.get_command("foot_position")
    # A standing command is active when both leg command bits are 0
    is_standing_command_mask = (full_command[:, 3] == 0.0) & (full_command[:, 4] == 0.0)

    # --- 2. Check for four-leg contact ---

    # Get contact forces on all bodies tracked by the sensor
    net_contact_forces = contact_sensor.data.net_forces_w
    # Get the vertical component (Z-axis) of the forces on the four feet
    feet_forces_z = net_contact_forces[:, contact_sensor_cfg.body_ids, 2]
    
    # Check if each foot has a contact force greater than the threshold
    # The result is a boolean tensor of shape (num_envs, 4)
    feet_in_contact = feet_forces_z > force_threshold
    # Check if all four feet are in contact for each environment
    # The result is a boolean tensor of shape (num_envs,)
    all_four_feet_in_contact = torch.all(feet_in_contact, dim=1)

    # --- 3. Reward for stability (low velocity) ---
    base_lin_vel = robot.data.root_lin_vel_b
    base_ang_vel = robot.data.root_ang_vel_b
    velocity_error = torch.norm(base_lin_vel, dim=1) + 0.5 * torch.norm(base_ang_vel, dim=1)
    velocity_reward = torch.exp(-velocity_error / velocity_std)

    # The reward is only applied if the command is "stand" AND all four feet are on the ground, Otherwise, the reward is zero.
    final_reward = torch.where(is_standing_command_mask & all_four_feet_in_contact, velocity_reward, 0.0)

    return final_reward

def body_terrain_alignment_reward(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    leg_asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    """
    Computes a reward for aligning the robot's torso parallel to the terrain.

    This reward encourages the robot's 'up' vector to align with the terrain normal.
    The terrain normal is robustly estimated using a fully vectorized SVD on the
    positions of the feet currently in contact with the ground.

    Args:
        root_quat (torch.Tensor): Quaternion of the robot's base link (root)
            in shape (num_envs, 4).
        end_effector_pos (torch.Tensor): Positions of the end-effectors (feet)
            in the world frame, in shape (num_envs, num_feet, 3).
        end_effector_contact (torch.Tensor): Boolean tensor indicating if an
            end-effector is in contact, in shape (num_envs, num_feet).
        weight (float, optional): The scaling factor for the reward. Defaults to 1.0.

    Returns:
        torch.Tensor: The calculated alignment reward tensor of shape (num_envs,).
    """
    device = env.device
    robot: Articulation = env.scene[robot_cfg.name]
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name] 

    # Get the 5D command from the command manager
    # Command format: (x, y, z, left_cmd, right_cmd)
    full_command = env.command_manager.get_command("foot_position")
    # A standing command is active when both leg command bits are 0
    is_standing_command_mask = (full_command[:, 3] == 0.0) & (full_command[:, 4] == 0.0)

    # Get contact forces on all bodies tracked by the sensor
    net_contact_forces = contact_sensor.data.net_forces_w
    # Get the vertical component (Z-axis) of the forces on the four feet
    feet_forces_z = net_contact_forces[:, contact_sensor_cfg.body_ids, 2]
    
    # Check if each foot has a contact force greater than the threshold
    # The result is a boolean tensor of shape (num_envs, 4)
    feet_in_contact = feet_forces_z > force_threshold
    
    # Check if all feet are in contact for each environment. The result is a boolean tensor of shape (num_envs,)
    # valid_envs_mask = torch.all(feet_in_contact, dim=1)
    num_contacting_feet = torch.sum(feet_in_contact, dim=1)
    valid_envs_mask = num_contacting_feet >= 3  
    
    robot_quat_w = robot.data.root_quat_w
    feet_pos_w = robot.data.body_pos_w[:, leg_asset_cfg.body_ids, :]
    
    # Get the torso's 'up' vector from its orientation quaternion
    x, y, z, w = robot_quat_w[:, 0], robot_quat_w[:, 1], robot_quat_w[:, 2], robot_quat_w[:, 3]
    torso_up_vector = torch.stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y)
    ], dim=1)

    alignment_reward = torch.zeros(env.num_envs, device=device)
    
    if torch.any(valid_envs_mask):
        
        # Perform calculations only on the environments with enough contacts
        valid_feet_pos = feet_pos_w[valid_envs_mask]
        
        # Calculate the centroid of the feet for each valid environment
        centroid = torch.mean(valid_feet_pos, dim=1, keepdim=True)
        
        # Center the points by subtracting the centroid
        centered_points = valid_feet_pos - centroid
        
        # Perform SVD on the batch of centered points
        U, S, Vh = torch.linalg.svd(centered_points)
        
        # The normal to the plane is the last row of Vh
        plane_normal = Vh[..., -1, :]

        # Ensure the normal points upwards relative to the world's Z-axis
        up_dot = torch.sum(plane_normal * torch.tensor([0.0, 0.0, 1.0], device=device), dim=1)
        plane_normal[up_dot < 0] *= -1.0
        
        # Calculate alignment for the valid environments
        valid_torso_up = torso_up_vector[valid_envs_mask]
        dot_product = torch.sum(valid_torso_up * plane_normal, dim=1)
        
        # Update the alignment reward tensor only for the valid environments
        alignment_reward[valid_envs_mask] = dot_product ** 2

    final_reward = torch.where(is_standing_command_mask & valid_envs_mask, alignment_reward, 0.0)
    
    return final_reward

def penalize_foot_contact_w_obstacle( 
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    left_leg_contact_cfg: SceneEntityCfg,
    right_leg_contact_cfg: SceneEntityCfg,
    contact_force_threshold: float,
) -> torch.Tensor:
    
    # Get the 6D command: (local_x, local_y, local_z, left_cmd, right_cmd, avoid_cmd)
    full_command = env.command_manager.get_command("foot_position")

    # Extract the command bits for each leg and the avoid obstacle flag
    leg_commands = full_command[:, 3:5]
    # Convert the avoid command flag to a boolean tensor for logical operations
    is_avoid_obstacle_command = full_command[:, 5] == 1.0

    # Create boolean masks based on which leg is commanded to move
    is_left_leg_commanded = leg_commands[:, 0] == 1.0
    is_right_leg_commanded = leg_commands[:, 1] == 1.0

    # --- 2. Check for Foot Contact ---

    # Access the contact sensors for each foot
    left_foot_sensor: ContactSensor = env.scene.sensors[left_leg_contact_cfg.name]
    right_foot_sensor: ContactSensor = env.scene.sensors[right_leg_contact_cfg.name]

    # Get the latest net contact force from the sensor's history buffer
    # The shape is (num_envs, 3)
    left_foot_forces_full = left_foot_sensor.data.net_forces_w
    left_foot_forces = left_foot_forces_full[:, left_leg_contact_cfg.body_ids, :]
    
    right_foot_forces_full = right_foot_sensor.data.net_forces_w
    right_foot_forces = right_foot_forces_full[:, right_leg_contact_cfg.body_ids, :]

    # Check if the magnitude (norm) of the contact force exceeds the threshold
    is_left_foot_in_contact = (torch.norm(left_foot_forces, dim=-1) > contact_force_threshold).squeeze(-1)
    is_right_foot_in_contact = (torch.norm(left_foot_forces, dim=-1) > contact_force_threshold).squeeze(-1)
    
    # --- 3. Apply Penalty Logic ---

    # Penalize the left foot only if:
    # (it's an avoid command) AND (the left leg is commanded) AND (the left foot makes contact)
    left_penalty = (
        is_avoid_obstacle_command & is_left_leg_commanded & is_left_foot_in_contact
    ).float()

    # Penalize the right foot only if:
    # (it's an avoid command) AND (the right leg is commanded) AND (the right foot makes contact)
    right_penalty = (
        is_avoid_obstacle_command & is_right_leg_commanded & is_right_foot_in_contact
    ).float()
    
    
    return left_penalty + right_penalty



def default_joint_pos_reward(
    env: ManagerBasedRLEnv, 
    robot_cfg: SceneEntityCfg,
    joint_pos_std: float = 0.5
) -> torch.Tensor:
    """
    Computes a reward for keeping the robot's joints close to their default positions.

    This reward uses a Gaussian (RBF kernel) function, which is maximized when the
    current joint positions match the default positions and falls off smoothly
    as they deviate.

    Returns:
        torch.Tensor: The calculated pose reward tensor of shape (num_envs,).
    """
    asset: Articulation = env.scene[robot_cfg.name]
    
    full_command = env.command_manager.get_command("foot_position")
    is_standing_command_mask = (full_command[:, 3] == 0.0) & (full_command[:, 4] == 0.0)
    
    # Calculate the squared error between current and default joint positions
    joint_error = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    
    # Use a Gaussian (RBF kernel) to create the reward
    pose_reward = torch.exp(-joint_error / (2 * joint_pos_std ** 2))
    
    final_reward = torch.where(is_standing_command_mask, pose_reward, 0.0)
    
    return final_reward


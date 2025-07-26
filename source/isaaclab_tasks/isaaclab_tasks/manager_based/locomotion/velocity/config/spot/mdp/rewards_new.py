# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the reward functions that can be used for Spot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from isaaclab.utils.math import quat_rotate_inverse
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def base_motion_penalty_paper(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    vz_2 = 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2])
    
    vw_2 = 0.2 * torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    
    return torch.exp(-(vz_2 + vw_2)/std)


def body_terrain_alignment_reward_walking(
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
    full_command = env.command_manager.get_command("base_velocity")
    # A standing command is active when both leg command bits are 0
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    is_standing_command_mask = cmd < 0.1

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

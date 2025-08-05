# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the reward functions that can be used for Spot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import GridPatternCfg
from isaaclab.sensors.ray_caster.ray_caster import RayCaster
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

def roll_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Penalize base roll velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    v_roll_2 = torch.square(asset.data.root_ang_vel_b[:, 0])
    
    return 1.0 - torch.exp(-v_roll_2/std)

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

def zero_velocity_reward_walking(
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
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    is_standing_command_mask = cmd < 0.1

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

def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel_xy = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    body_vel_z = torch.abs(asset.data.root_ang_vel_b[:, 2])
    body_vel = body_vel_xy + body_vel_z
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_and(cmd > 0.1, body_vel > velocity_threshold), reward, stand_still_scale * reward)

def get_terrain_height_from_scan(
    robot_positions_w: torch.Tensor, robot_orienations_w: torch.Tensor, foot_pos_w: torch.Tensor, height_scan_data: torch.Tensor, height_scan_config: GridPatternCfg, num_envs, device
) -> torch.Tensor:
    """
    Gets the terrain height under the feet by finding the NEAREST point in a height scan grid.

    Args:
        env: The simulation environment.
        foot_pos_w: The world coordinates of the feet. Shape: (num_envs, num_feet, 3).
        height_scan_data: The height scan grid data. Shape: (num_envs, grid_rows, grid_cols).

    Returns:
        A tensor of shape (num_envs, num_feet) with the nearest terrain z-coordinate.
    """
    # -- 1. Get necessary data from environment --
    # You must find these values in your environment's configuration.
    # The number of columns is no longer in the shape, so it must be from the config.
    resolution = height_scan_config.resolution
    grid_cols = 1 + round(height_scan_config.size[0] / resolution)
    grid_rows = 1 + round(height_scan_config.size[1] / resolution)
    
    # -- 2. Transform foot positions from world frame to robot's local frame --
    foot_pos_relative = foot_pos_w - robot_positions_w.unsqueeze(1)
    foot_pos_local = quat_rotate_inverse(robot_orienations_w.unsqueeze(1), foot_pos_relative)

    # -- 3. Convert local coordinates to integer 2D grid indices --
    center_col_idx = (grid_cols - 1) / 2.0
    center_row_idx = (grid_rows - 1) / 2.0
    col_idx_float = foot_pos_local[..., 0] / resolution + center_col_idx
    row_idx_float = foot_pos_local[..., 1] / resolution + center_row_idx

    col_indices = torch.round(col_idx_float).long()
    row_indices = torch.round(row_idx_float).long()

    # Clamp 2D indices before flattening
    col_indices.clamp_(0, grid_cols - 1)
    row_indices.clamp_(0, grid_rows - 1)

    # -- 4. NEW: Calculate the flattened 1D index --
    # The data is ordered by row, then column (row-major).
    flat_indices = row_indices * grid_cols + col_indices

    # -- 5. NEW: Direct 1D Lookup --
    num_feet = foot_pos_w.shape[1]
    env_indices = torch.arange(num_envs, device=device).unsqueeze(1).expand(-1, num_feet)

    # Use the calculated 1D indices to retrieve height values from the flattened vector.
    return height_scan_data[env_indices, flat_indices]

def foot_clearance_reward_hs(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    feet_cfg: SceneEntityCfg,
    height_scan_cfg: SceneEntityCfg,
    target_clearance: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground terrain."""
    
    robot: Articulation = env.scene[robot_cfg.name]
    
    feet_asset: RigidObject = env.scene[feet_cfg.name]
    foot_pos_w = feet_asset.data.body_pos_w[:, feet_cfg.body_ids]
    
    # --- The ONLY change is calling the new helper function ---

    sensor: RayCaster = env.scene.sensors[height_scan_cfg.name]
    
    # height scan: height = sensor_height - hit_point_z 
    # height_scan_data = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] 
    height_scan_data = sensor.data.ray_hits_w[..., 2]
    terrain_z_under_foot = get_terrain_height_from_scan(robot.data.root_pos_w, robot.data.root_quat_w, foot_pos_w, height_scan_data, sensor.cfg.pattern_cfg, env.num_envs, env.device)

    # The rest of the function remains identical
    dynamic_target_height = terrain_z_under_foot + target_clearance
    foot_z_target_error = torch.square(foot_pos_w[..., 2] - dynamic_target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.norm(feet_asset.data.body_lin_vel_w[:, feet_cfg.body_ids, :2], dim=2)
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)
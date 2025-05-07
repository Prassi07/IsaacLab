# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from isaaclab.utils.math import quat_rotate
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())

def pedipulation_levels_size(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
     Curriculum for Pedipulation, as more success, range of sampled commands goes up 
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    left_leg_idx = asset.find_bodies(["fl_foot"])[0]
    left_foot_pos_w = asset.data.body_pos_w[:, left_leg_idx, :].squeeze()
        
    command_b = env.command_manager.get_command("foot_position")
    command_w = quat_rotate(asset.data.root_quat_w[:, :4], command_b[:, :3]) + asset.data.root_pos_w[:, :3]
    
    all_errors = torch.norm(left_foot_pos_w - command_w, dim=1)
    mean_error = torch.mean(all_errors)
    
    if(mean_error < 0.06):
        command_term = env.command_manager.get_term("foot_position")
        command_term.update_curriculums(curriculum_factor = 0.2)
    
    return mean_error
    

def pedipulation_multileg_levels_size(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
     Curriculum for Pedipulation, as more success, range of sampled commands goes up 
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    full_command_b = env.command_manager.get_command("foot_position")
    target_pos_b = full_command_b[:, :3] # Target position in base frame
    
    is_right_leg_commanded_mask = (full_command_b[:, 3] == 1.0) 
    
    # leg_command = torch.zeros_like(full_command_b[:, 2])
    # is_right_leg_commanded_mask = (leg_command == 1.0) 
    
    target_pos_w = quat_rotate(asset.data.root_quat_w[:, :4], target_pos_b) + asset.data.root_pos_w[:, :3]

    left_leg_idx = asset.find_bodies(["fl_foot"])[0]
    left_foot_pos_w = asset.data.body_pos_w[:, left_leg_idx, :].squeeze()
    
    right_leg_idx = asset.find_bodies(["fr_foot"])[0]
    right_foot_pos_w = asset.data.body_pos_w[:, right_leg_idx, :].squeeze()

    error_left_leg_w = torch.linalg.norm(left_foot_pos_w - target_pos_w, dim=1)
    error_right_leg_w = torch.linalg.norm(right_foot_pos_w - target_pos_w, dim=1)
    
    all_errors = torch.where(is_right_leg_commanded_mask, error_right_leg_w, error_left_leg_w)
    
    mean_error = torch.mean(all_errors)
    
    if(mean_error < 0.06):
        command_term = env.command_manager.get_term("foot_position")
        command_term.update_curriculums(curriculum_factor = 0.2)
    
    return mean_error
    
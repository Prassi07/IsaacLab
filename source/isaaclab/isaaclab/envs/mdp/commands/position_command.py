# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformPosition3dCommandCfg


class UniformPosition3dCommand(CommandTerm):
    """Command generator that generates postion commands containing a 3-D position.

    The command generator samples uniform 3D positions around the environment origin.
    """

    cfg: UniformPosition3dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPosition3dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.left_leg_name = cfg.left_leg_name
        self.right_leg_name = cfg.right_leg_name
        
        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        
        self.zero_orientations = torch.zeros(self.num_envs, 1)
        # -- metrics
        self.metrics["error_pos_3d_left"] = torch.ones(self.num_envs, device=self.device)
        self.metrics["error_pos_3d_right"] = torch.ones(self.num_envs, device=self.device)
    
        
    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 3)."""
        return self.pos_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        left_leg_idx = self.robot.find_bodies([self.left_leg_name])[0]
        left_foot_pos_w = self.robot.data.body_pos_w[:, left_leg_idx, :].squeeze()
        self.metrics["error_pos_3d_left"] = torch.norm(self.pos_command_w - left_foot_pos_w, dim=-1)
        
        right_leg_idx = self.robot.find_bodies([self.right_leg_name])[0]
        right_foot_pos_w = self.robot.data.body_pos_w[:, right_leg_idx, :].squeeze()
        self.metrics["error_pos_3d_right"] = torch.norm(self.pos_command_w - right_foot_pos_w, dim=-1)
        
        # print("---------------------------------------")
        # print("Ranges: ", self.cfg.ranges.pos_x, self.cfg.ranges.pos_y, self.cfg.ranges.pos_z)
        # print("---------------------------------------")
        

    def _resample_command(self, env_ids: Sequence[int]):
        # obtain env origins for the environments
        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        # offset the position command by the current root position
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        self.pos_command_w[env_ids, 2] += r.uniform_(*self.cfg.ranges.pos_z)
        


    def update_curriculums(self, curriculum_factor: float = 0.2):
        """Updates the command ranges, potentially for curriculum learning."""
        # Calculate the new lower bound for pos_x, ensuring it stays within max_ranges
        
        def _update_single_range(current_range, max_range, factor, device):
            new_min = torch.clip(
                torch.tensor(current_range[0] - factor, device=device),
                min=max_range[0],
                max=max_range[1] # Clip against upper bound too
            ).item()
            new_max = torch.clip(
                torch.tensor(current_range[1] + factor, device=device),
                min=max_range[0], # Clip against lower bound too
                max=max_range[1]
            ).item()
            # Ensure min is still less than or equal to max after clipping
            return (min(new_min, new_max), max(new_min, new_max))

        # Update pos_x range
        self.cfg.ranges.pos_x = _update_single_range(self.cfg.ranges.pos_x, self.cfg.max_ranges.pos_x, curriculum_factor, self.device)
        # Update pos_y range
        self.cfg.ranges.pos_y = _update_single_range(self.cfg.ranges.pos_y, self.cfg.max_ranges.pos_y, curriculum_factor, self.device)
        # Update pos_z range
        self.cfg.ranges.pos_z = _update_single_range(self.cfg.ranges.pos_z, self.cfg.max_ranges.pos_z, curriculum_factor, self.device)
        
        
    def _update_command(self):
        """Re-target the position command to the current root state."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_rotate_inverse(self.robot.data.root_quat_w, target_vec)
        

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_position_visualizer"):
                self.goal_position_visualizer = VisualizationMarkers(self.cfg.goal_position_visualizer_cfg)
            # set their visibility to true
            self.goal_position_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_position_visualizer"):
                self.goal_position_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.goal_position_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(self.zero_orientations, self.zero_orientations, self.zero_orientations).squeeze()
        )

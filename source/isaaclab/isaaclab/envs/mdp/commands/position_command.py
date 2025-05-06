"""Sub-module containing command generator pedipulation task."""

from __future__ import annotations

from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate, quat_rotate_inverse, wrap_to_pi, yaw_quat

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
        
        self.pos_leg_command_b = torch.zeros(self.num_envs, 4, device=self.device)
        
        self.leg_switch_command = torch.zeros(self.num_envs, 1, device=self.device)
        
        self.zero_orientations = torch.zeros(self.num_envs, 1)
        # -- metrics
        self.metrics["error_pos_3d_left"] = torch.ones(self.num_envs, device=self.device)
        self.metrics["error_pos_3d_right"] = torch.ones(self.num_envs, device=self.device)
        self.metrics["error_pos_3d_commanded_leg"] = torch.ones(self.num_envs, device=self.device)
    
        
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
        """The desired 3D position (x,y,z) and leg choice (0 for left, 1 for right) in base frame. Shape is (num_envs, 4)."""
        return self.pos_leg_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        
        # World position of the command target
        target_pos_w = self.pos_command_w

        # Calculate error for the left leg
        # self.robot.find_bodies([self.left_leg_name])[0] returns a list like [[index_for_left_foot]]
        # So, self.robot.find_bodies([self.left_leg_name])[0][0] gives [index_for_left_foot]
        
        left_leg_idx = self.robot.find_bodies([self.left_leg_name])[0]
        left_foot_pos_w = self.robot.data.body_pos_w[:, left_leg_idx, :].squeeze()
        error_left = torch.norm(target_pos_w - left_foot_pos_w, dim=-1)
        self.metrics["error_pos_3d_left"] = error_left
        
        # Calculate error for the right leg
        right_leg_idx = self.robot.find_bodies([self.right_leg_name])[0]
        right_foot_pos_w = self.robot.data.body_pos_w[:, right_leg_idx, :].squeeze()
        error_right = torch.norm(target_pos_w - right_foot_pos_w, dim=-1)
        self.metrics["error_pos_3d_right"] = error_right
        
        # Calculate error for the commanded leg
        # self.leg_switch_command is (num_envs, 1): 0 for left, 1 for right.
        # Squeeze to (num_envs,) for torch.where condition.
        is_left_command_mask = (self.leg_switch_command.squeeze(dim=-1) == 0)
        self.metrics["error_pos_3d_commanded_leg"] = torch.where(is_left_command_mask, error_left, error_right)
        
        

    def _resample_command(self, env_ids: Sequence[int]):
        
        # tensor `r_for_sampling` for random sampling, shape: (len(env_ids),)
        r_for_sampling = torch.empty(len(env_ids), device=self.device)
        
        # Determine leg choice
        is_left_leg_mask = r_for_sampling.uniform_(0, 1) < 0.5  # LEFT LEG will be True
        self.leg_switch_command[env_ids, 0] = (~is_left_leg_mask).float()   # LEFT LEG will be 0, Right will be 1
        
        self.pos_command_b[env_ids, 0] = r_for_sampling.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_b[env_ids, 2] = r_for_sampling.uniform_(*self.cfg.ranges.pos_z)

        # Sample y-offsets
        y_offsets_candidate = r_for_sampling.uniform_(*self.cfg.ranges.pos_y)
        
        # Apply negation for the right leg if it's chosen
        final_y_offsets = torch.where(is_left_leg_mask, y_offsets_candidate, -y_offsets_candidate) # Negative for red, as left leg is true
        
        self.pos_command_b[env_ids, 1] = final_y_offsets
        
        # Update the world-frame command for ALL environments to be consistent with the current base-frame command
        self.pos_command_w[:] = self.robot.data.root_pos_w + quat_rotate(self.robot.data.root_quat_w, self.pos_command_b)

        # # obtain env origins for the environments
        # self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        # # offset the position command by the current root position
        # r = torch.empty(len(env_ids), device=self.device)
        
        # self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        # self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        # self.pos_command_w[env_ids, 2] += r.uniform_(*self.cfg.ranges.pos_z)
        

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
        """
        Assembles the final agent command `self.pos_leg_command_b`.
        
        The 3D position part is derived from `self.pos_command_w` (world-frame target)
        by transforming it back to the robot's base frame. The leg switch command is appended.
        """
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_leg_command_b[:, :3] = quat_rotate_inverse(self.robot.data.root_quat_w, target_vec)
        self.pos_leg_command_b[:, 3] = self.leg_switch_command.squeeze()
        

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "left_goal_position_visualizer"):
                self.left_goal_position_visualizer = VisualizationMarkers(self.cfg.left_goal_position_visualizer_cfg)
                
            # set their visibility to true
            self.left_goal_position_visualizer.set_visibility(True)
            
            if not hasattr(self, "right_goal_position_visualizer"):
                self.right_goal_position_visualizer = VisualizationMarkers(self.cfg.right_goal_position_visualizer_cfg)
                
            # set their visibility to true
            self.right_goal_position_visualizer.set_visibility(True)
        else:
            if hasattr(self, "right_goal_position_visualizer"):
                self.right_goal_position_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        
        # Default orientation for the markers (e.g., identity quaternion)
        # self.zero_orientations is (num_envs, 1)
        # quat_from_euler_xyz output is (num_envs, 1, 4), squeeze to (num_envs, 4)
        default_orientations_quat = quat_from_euler_xyz(
            self.zero_orientations, self.zero_orientations, self.zero_orientations
        ).squeeze(dim=1) # Squeeze only dim 1 to be safe if num_envs is 1

        # Create a tensor for far-away positions to "hide" inactive markers
        far_away_translations = torch.full_like(self.pos_command_w, 0.0)
        far_away_translations[:, 2] = -1000.0 

        # Determine which leg is commanded for each environment
        # self.leg_switch_command is (num_envs, 1): 0 for left, 1 for right
        is_left_command_mask = (self.leg_switch_command.squeeze(dim=-1) == 0)

        # Set translations for left leg visualizer
        left_viz_translations = torch.where(is_left_command_mask.unsqueeze(-1), self.pos_command_w, far_away_translations)
        self.left_goal_position_visualizer.visualize(
            translations=left_viz_translations, orientations=default_orientations_quat
        )
        
        # Set translations for right leg visualizer
        right_viz_translations = torch.where(~is_left_command_mask.unsqueeze(-1), self.pos_command_w, far_away_translations)
        self.right_goal_position_visualizer.visualize(
            translations=right_viz_translations, orientations=default_orientations_quat
        )

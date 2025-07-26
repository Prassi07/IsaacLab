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


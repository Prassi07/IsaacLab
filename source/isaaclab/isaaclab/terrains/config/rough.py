# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg
from ..terrain_generator_cfg import FlatPatchSamplingCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.25),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)


FLAT_SAMPLING_CFG = FlatPatchSamplingCfg(
        num_patches=10,
        patch_radius=0.5,
        max_height_diff = 0.7,
)

FLAT_SAMPLING_CFG_2 = FlatPatchSamplingCfg(
        num_patches=15,
        patch_radius=0.7,
        max_height_diff = 0.7,
)

ROUGH_TERRAINS_CFG_2 = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=15,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_w_holes": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=1.25,
            border_width=0.0,
            holes=True,
            flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG}
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.25,
            platform_width=0.25,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG}
        ),
        "pyramid_stairs_inv_w_holes": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.28,
            platform_width=1.25,
            border_width=0.0,
            holes=True,
            flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG}
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.28,
            platform_width=0.25,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG}
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.05, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=0.15, flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG}
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.05, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25, flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG}
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=0.15, border_width=0.25, flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG}
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=.15, border_width=0.25, flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG}
        ),
        "hf_discrete_terrain": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.1, platform_width=1.0, border_width=0.25, num_obstacles = 15, obstacle_width_range = (0.2, 0.8), 
            obstacle_height_range = (-0.2, 0.6), flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG}
        ),
    },
)


ROUGH_TERRAINS_CFG_3 = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=20,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs1": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.075,
            step_height_range=(0.08, 0.23),
            step_width=0.3,
            platform_width=1.5,
            border_width=0.0,
            holes=False,
            flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
        "pyramid_stairs2": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.075,
            step_height_range=(0.08, 0.23),
            step_width=0.25,
            platform_width=1.25,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.15), platform_width=1.0, flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25, flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=1.0, border_width=0.25, flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.5), platform_width=1.0, border_width=0.25, flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
        "pyramid_stairs_inv1": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.075,
            step_height_range=(0.08, 0.23),
            step_width=0.28,
            platform_width=1.5,
            border_width=0.0,
            holes=False,
            flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.075,
            step_height_range=(0.08, 0.23),
            step_width=0.28,
            platform_width=1.25,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
        # "hf_discrete_terrain": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        #     proportion=0.1, platform_width=1.0, border_width=0.25, num_obstacles = 20, obstacle_width_range = (0.2, 0.8), 
        #     obstacle_height_range = (-0.2, 0.6), flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        # ),
    },
)

ROUGH_STAIR_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=20,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs1": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.08, 0.23),
            step_width=0.3,
            platform_width=2.0,
            border_width=0.0,
            holes=False,
            flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
        "pyramid_stairs2": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.08, 0.23),
            step_width=0.25,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
        "pyramid_stairs_inv1": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.08, 0.23),
            step_width=0.28,
            platform_width=1.5,
            border_width=0.0,
            holes=False,
            flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.08, 0.23),
            step_width=0.28,
            platform_width=1.25,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.15), platform_width=1.0, flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25, flat_patch_sampling={'init_pos': FLAT_SAMPLING_CFG_2}
        ),
    },
)
"""Rough terrains configuration."""

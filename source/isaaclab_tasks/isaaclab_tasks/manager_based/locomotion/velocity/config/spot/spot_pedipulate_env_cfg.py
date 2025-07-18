# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, MySceneCfg, PedipulationEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(4.0, 4.0),
    border_width=10.0,
    num_rows=50,
    num_cols=50,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.6, noise_range=(-0.08, 0.08), noise_step=0.02, border_width=0.25
        ),
    },
)

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG, ROUGH_TERRAINS_CFG_2

@configclass
class SpotActionsPedipulateCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True)


@configclass
class SpotCommandsPedipulateCfg:
    """Command specifications for the MDP."""

    foot_position = mdp.UniformPosition3dCommandCfg(
        asset_name="robot",
        left_leg_name="fl_foot",
        right_leg_name="fr_foot",
        resampling_time_range=(9.0, 18.0),
        standing_ratio = 0.25, 
        debug_vis=True,
        ranges=mdp.UniformPosition3dCommandCfg.Ranges(
            pos_x=(0.25, 0.75), pos_y=(0, 0.25), pos_z = (-0.5, 0.0)), 
        max_ranges=mdp.UniformPosition3dCommandCfg.Ranges(
            pos_x=(0.0, 1.75), pos_y=(-0.15, 0.75), pos_z = (-0.65, 0.25)),
        # ranges=mdp.UniformPosition3dCommandCfg.Ranges(
        #     pos_x=(0.25, 1.25), pos_y=(0, 0.75), pos_z = (0, 0.75)),
        # max_ranges=mdp.UniformPosition3dCommandCfg.Ranges(
        #     pos_x=(0.0, 1.75), pos_y=(-0.15, 1.0), pos_z = (0, 0.75))
        )
        
    

@configclass
class SpotObservationsPedipulateCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # `` observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        foot_position_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "foot_position"})
        
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)
        
        height_scan =ObsTerm(
            func = mdp.height_scan,
            params = {"sensor_cfg":SceneEntityCfg("height_scanner")},
            noise = Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class SpotEventPedipulateCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "mass_distribution_params": (-2.5, 2.5),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_from_terrain,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"yaw" : (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7),
                "pitch": (-0.7, 0.7),
                "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(9.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )
    
    # interval
    apply_force_left_leg = EventTerm(
        func=mdp.apply_external_force_torque_left_leg,
        mode="interval",
        interval_range_s=(13.0, 13.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="fl_foot"),
            "force_range": (0.0, 12.0),
            "torque_range": (0.0, 0.0),
        }
    )
    
    apply_force_right_leg = EventTerm(
        func=mdp.apply_external_force_torque_right_leg,
        mode="interval",
        interval_range_s=(13.0, 13.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="fr_foot"),
            "force_range": (0.0, 12.0),
            "torque_range": (0.0, 0.0),
        }
    )


@configclass
class SpotRewardsPedipulateCfg:
    # -- task              
    goal_reward = RewardTermCfg(spot_mdp.multileg_pedipulation_reward,
                                weight=15.0,
                                params={
                                    "robot_cfg": SceneEntityCfg("robot"),
                                    "left_leg_asset_cfg" : SceneEntityCfg("robot", body_names="fl_foot"),
                                    "right_leg_asset_cfg" : SceneEntityCfg("robot", body_names="fr_foot"),
                                    "std": 0.8}
    ) # Weight From Paper
    
    # Standing Rewards when no pedipulation command
    zero_vel_reward = RewardTermCfg(spot_mdp.zero_velocity_reward,
                                    weight = 2.5,
                                    params={
                                        "robot_cfg": SceneEntityCfg("robot"),
                                        "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                                        "force_threshold" : 1.0,
                                        "velocity_std" : 1.0,
                                    }
    )
    
    zero_orientation_reward = RewardTermCfg(spot_mdp.body_terrain_alignment_reward,
                                    weight = 5.0,
                                    params={
                                        "robot_cfg": SceneEntityCfg("robot"),
                                        "leg_asset_cfg":  SceneEntityCfg("robot", body_names=".*_foot"),
                                        "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                                        "force_threshold" : 1.0,
                                    }
    )
    
    zero_joint_pos_reward = RewardTermCfg(spot_mdp.default_joint_pos_reward,
                                    weight = 2.5,
                                    params={
                                        "robot_cfg": SceneEntityCfg("robot"),
                                        "joint_pos_std" : 0.5,
                                    }
    )
    
    # -- penalties
    
    base_linear_z_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_z_velocity_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_angular_xy_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_xy_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_square_penalty,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    ) # Weight From Paper
    
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_square_penalty,
        weight=-5.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    ) # Weight From Paper

    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_square_penalty,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    ) # Weight From Paper
    
    action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_square_penalty, weight=-0.02) # Weight From Paper
    
    robot_link_collision =  RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*uleg"]), "threshold": 1.0},
    )
    
    termination_penalty = RewardTermCfg(
        func=mdp.is_terminated,
        weight=-200.0,
    )


@configclass
class SpotTerminationsPedipulateCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0},
    )
    # terrain_out_of_bounds = DoneTerm(
    #     func=mdp.terrain_out_of_bounds,
    #     params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
    #     time_out=True,
    # )


@configclass
class SpotCurriculumPedipulateCfg:
    """Curriculum terms for the MDP."""
    
    pedipulation_range = CurrTerm(func=mdp.pedipulation_multileg_levels_size)


@configclass
class SpotPedipulationTaskCfg(PedipulationEnvCfg):

    # Basic settings'
    scene: MySceneCfg = MySceneCfg(num_envs = 4096, env_spacing = 2.5)
    observations: SpotObservationsPedipulateCfg = SpotObservationsPedipulateCfg()
    actions: SpotActionsPedipulateCfg = SpotActionsPedipulateCfg()
    commands: SpotCommandsPedipulateCfg = SpotCommandsPedipulateCfg()
    
    # MDP setting
    rewards: SpotRewardsPedipulateCfg = SpotRewardsPedipulateCfg()
    terminations: SpotTerminationsPedipulateCfg = SpotTerminationsPedipulateCfg()
    events: SpotEventPedipulateCfg = SpotEventPedipulateCfg()
    curriculum: SpotCurriculumPedipulateCfg = SpotCurriculumPedipulateCfg()


    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # self.scene.contact_forces.update_period = self.sim.dt

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
            
        # terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG_2,
            max_init_terrain_level=ROUGH_TERRAINS_CFG_2.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=True,
        )

        # terrain
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
        
        # switch robot to Spot-d
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # no height scan
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/body"



class SpotPedipulationTaskCfg_PLAY(SpotPedipulationTaskCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 10
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        
        self.commands.foot_position.ranges = mdp.UniformPosition3dCommandCfg.Ranges(
            pos_x=(0.0, 1.75), pos_y=(-0.15, 0.75), pos_z = (-0.6, 0.25))
        
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Crazyflie steering tracking task configurations."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Track-Steering-Vision-Crazyflie-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_steering_env_cfg:TrackSteeringEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Track-Steering-Vision-Crazyflie-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_steering_env_cfg:TrackSteeringEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

# CTBR body-rate tracking variants (full 3-axis [wx, wy, wz, thrust])
gym.register(
    id="Isaac-Track-BodyRate-Crazyflie-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.body_rate_env_cfg:TrackBodyRateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

# RL velocity controller (Agent B Track 1c-b): learned nav-interface controller
# (yaw_rate + forward_speed + cruise altitude -> thrust/moments), the RL upgrade to warehouse_mlp_control.
gym.register(
    id="Isaac-Track-VelocityCtrl-Crazyflie-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_control_env_cfg:TrackVelocityCtrlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Track-VelocityCtrl-Crazyflie-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_control_env_cfg:TrackVelocityCtrlEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Track-VelocityCtrl-DR-Crazyflie-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_control_env_cfg:TrackVelocityCtrlEnvCfg_DR",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Track-VelocityCtrlRateDR-Crazyflie-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_control_env_cfg:TrackVelocityCtrlRateDREnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Track-BodyRate-Crazyflie-Easy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.body_rate_env_cfg:TrackBodyRateEnvCfg_EASY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Track-BodyRate-Crazyflie-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.body_rate_env_cfg:TrackBodyRateEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

# Domain-randomized body-rate env (anti-overfit / sim2real)
gym.register(
    id="Isaac-Track-BodyRate-DR-Crazyflie-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.body_rate_env_cfg:TrackBodyRateEnvCfg_DR",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

# Harsh reward variants
gym.register(
    id="Isaac-Track-Steering-Vision-Crazyflie-Harsh-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_steering_env_cfg_harsh:TrackSteeringEnvCfg_Harsh",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Track-Steering-Vision-Crazyflie-Harsh-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_steering_env_cfg_harsh:TrackSteeringEnvCfg_Harsh_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SteeringTrackingPPORunnerCfg",
    },
)

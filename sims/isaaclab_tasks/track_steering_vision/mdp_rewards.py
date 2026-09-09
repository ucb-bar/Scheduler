# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP reward functions for steering angle tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


def steering_angle_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
    command_name: str = "steering_command",
) -> torch.Tensor:
    """Reward for tracking a target steering angle (yaw rate).

    Computes exp(-(yaw_rate - target_yaw_rate)^2 / std^2).

    Args:
        env: The RL environment.
        asset_cfg: The robot asset config.
        std: Standard deviation for exponential kernel.
        command_name: Name of the steering command.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Command is [target_yaw_rate, target_forward_velocity]
    target_yaw_rate = command[:, 0]

    # Current yaw rate (angular velocity around z-axis in body frame)
    current_yaw_rate = asset.data.root_ang_vel_b[:, 2]

    # Compute tracking error
    error = target_yaw_rate - current_yaw_rate

    return torch.exp(-(error**2) / std**2)


def body_rate_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
    command_name: str = "body_rate_command",
) -> torch.Tensor:
    """Reward for tracking a full 3-axis body-rate command (CTBR).

    Computes ``exp(-||command[:, :3] - root_ang_vel_b[:, :3]||^2 / std^2)`` over the
    roll/pitch/yaw rates. Generalizes :func:`steering_angle_tracking` (yaw only) to
    all three body axes.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_rates = command[:, :3]
    current_rates = asset.data.root_ang_vel_b[:, :3]
    error = torch.sum((target_rates - current_rates) ** 2, dim=1)
    return torch.exp(-error / std**2)


def collective_thrust_tracking(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    command_name: str = "body_rate_command",
) -> torch.Tensor:
    """Reward for producing the commanded collective thrust (channel 3 of the
    body-rate command).

    The applied normalized thrust is ``(action[:, 0].clamp(-1, 1) + 1) / 2`` — the
    same [-1,1]->[0,1] map the :class:`DirectThrustMomentAction` term uses (thrust is
    action channel 0). Rewarding the match keeps the policy honouring the thrust
    channel (so it holds altitude rather than cutting throttle to make body-rate
    tracking trivial), while staying a light term so it never swamps the rate objective.
    """
    command = env.command_manager.get_command(command_name)
    target_thrust = command[:, 3]
    action = env.action_manager.action  # concatenated raw policy action; ch0 = thrust
    applied_thrust_norm = (action[:, 0].clamp(-1.0, 1.0) + 1.0) / 2.0
    error = (target_thrust - applied_thrust_norm) ** 2
    return torch.exp(-error / std**2)


def forward_velocity_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
    command_name: str = "steering_command",
) -> torch.Tensor:
    """Reward for maintaining forward velocity.

    Computes exp(-(vel_x - target_vel_x)^2 / std^2).

    Args:
        env: The RL environment.
        asset_cfg: The robot asset config.
        std: Standard deviation for exponential kernel.
        command_name: Name of the steering command.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Command is [target_yaw_rate, target_forward_velocity]
    target_vel = command[:, 1]

    # Current forward velocity (x-axis in body frame)
    current_vel = asset.data.root_lin_vel_b[:, 0]

    # Compute tracking error
    error = target_vel - current_vel

    return torch.exp(-(error**2) / std**2)


def upright_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.3,
) -> torch.Tensor:
    """Reward for keeping the drone upright (roll and pitch near zero).

    Args:
        env: The RL environment.
        asset_cfg: The robot asset config.
        std: Standard deviation for exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Extract roll and pitch from quaternion
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)

    # Wrap to [-pi, pi]
    roll = math_utils.wrap_to_pi(roll)
    pitch = math_utils.wrap_to_pi(pitch)

    # Penalize deviations from zero
    orientation_error = roll**2 + pitch**2

    return torch.exp(-orientation_error / std**2)


def height_tracking(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 1.0,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward for maintaining a target height.

    Args:
        env: The RL environment.
        asset_cfg: The robot asset config.
        target_height: Desired height above ground.
        std: Standard deviation for exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Current height (z position relative to env origin)
    current_height = (asset.data.root_pos_w - env.scene.env_origins)[:, 2]

    # Height error
    height_error = (current_height - target_height) ** 2

    return torch.exp(-height_error / std**2)


def lateral_drift_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.0,
) -> torch.Tensor:
    """Penalize lateral drift (sideways velocity).

    Args:
        env: The RL environment.
        asset_cfg: The robot asset config.
        std: Standard deviation for exponential kernel.

    Returns:
        Reward tensor of shape (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # Lateral velocity (y-axis in body frame)
    lateral_vel = asset.data.root_lin_vel_b[:, 1]

    return torch.exp(-(lateral_vel**2) / std**2)


def action_smoothness(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalize rapid changes in actions (encourages smooth control).

    Returns:
        Negative L2 norm of action rate.
    """
    return -torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

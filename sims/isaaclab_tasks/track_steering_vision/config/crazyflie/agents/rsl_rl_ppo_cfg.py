# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for steering angle tracking."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlMLPModelCfg,
)


@configclass
class SteeringTrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for steering tracking task."""

    num_steps_per_env = 24
    # 2000 was too short: the run silently stops before the policy is trained (Dima's
    # reference checkpoint is model_6998).
    max_iterations = 7000
    save_interval = 50
    experiment_name = "crazyflie_steering_tracking"

    # Map observation groups to actor/critic
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    # Actor network (policy)
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            # "scalar" makes the action std a RAW learnable parameter, so PPO can drive it
            # to <= 0 -- which is exactly how a run dies with
            #   RuntimeError: normal expects all elements of std >= 0.0
            # (it also explains the "entropy collapse" we saw: std decaying 0.94 -> 0.22 -> 0).
            # "log" parameterizes std = exp(log_std), so it is positive by construction.
            std_type="log",
        ),
    )

    # Critic network (value function)
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )

    # PPO Algorithm
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

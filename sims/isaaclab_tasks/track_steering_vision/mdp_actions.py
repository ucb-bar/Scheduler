"""Custom action terms for steering tracking task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DirectThrustMomentAction(ActionTerm):
    """Direct thrust and moment control action term.

    This action term applies direct thrust (vertical force) and moments (roll, pitch, yaw torques)
    to the drone body. This is simpler and more direct than controlling individual propellers.

    Action space (4D):
        - action[0]: Total vertical thrust (normalized to [-1, 1])
        - action[1]: Roll moment (torque around x-axis)
        - action[2]: Pitch moment (torque around y-axis)
        - action[3]: Yaw moment (torque around z-axis)
    """

    cfg: DirectThrustMomentActionCfg
    _asset: Articulation
    _body_id: torch.Tensor

    def __init__(self, cfg: DirectThrustMomentActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Get the robot asset
        self._asset: Articulation = env.scene[cfg.asset_name]

        # Get body IDs for applying wrench (returns tensor of body indices)
        self._body_id = self._asset.find_bodies(cfg.body_name)[0]

        # Compute robot weight for thrust scaling
        robot_mass = self._asset.root_physx_view.get_masses()[0].sum()
        gravity = torch.tensor(env.sim.cfg.gravity, device=env.device).norm()
        self._robot_weight = (robot_mass * gravity).item()

        # Storage for forces and torques (applied) + targets (for optional motor lag)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust_target = torch.zeros_like(self._thrust)
        self._moment_target = torch.zeros_like(self._moment)

        # Storage for raw and processed actions (required by ActionTerm base class)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # --- Domain-randomization state (defaults are no-ops => identical to the
        # non-DR steering env). Per-env actuation gains + first-order motor lag. ---
        self._t2w = torch.full((self.num_envs,), cfg.thrust_to_weight, device=self.device)
        self._mscale = torch.full((self.num_envs,), cfg.moment_scale, device=self.device)
        self._motor_tau = float(cfg.motor_tau)
        self._step_dt = env.step_dt

        # --- Command-hold (rate) DR: a per-env zero-order hold so the policy trains against a
        # RANDOMIZED effective command period (=> rate robustness). hold=1 refreshes every control
        # step (no-op default). The effective period is exposed to the policy via
        # `effective_period_obs` so a period-aware policy can scale its moment appropriately. ---
        self._hold = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self._hold_ctr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._held_action = torch.zeros(self.num_envs, 4, device=self.device)
        self._max_hold = int(cfg.randomize_hold_steps[1]) if cfg.randomize_hold_steps else 1

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """Resample per-env actuation gains (DR) and clear the lag state on reset."""
        extras = super().reset(env_ids)
        idx = slice(None) if env_ids is None else env_ids
        if self.cfg.randomize_thrust_to_weight is not None:
            lo, hi = self.cfg.randomize_thrust_to_weight
            self._t2w[idx] = torch.empty_like(self._t2w[idx]).uniform_(lo, hi)
        if self.cfg.randomize_moment_scale is not None:
            lo, hi = self.cfg.randomize_moment_scale
            self._mscale[idx] = torch.empty_like(self._mscale[idx]).uniform_(lo, hi)
        if self.cfg.randomize_hold_steps is not None:
            lo, hi = self.cfg.randomize_hold_steps
            self._hold[idx] = torch.randint(int(lo), int(hi) + 1, self._hold[idx].shape, device=self.device)
        self._hold_ctr[idx] = 0
        self._held_action[idx] = 0.0
        # clear applied wrench so motor-lag doesn't carry across episodes
        self._thrust[idx] = 0.0
        self._moment[idx] = 0.0
        return {}

    @property
    def effective_period_obs(self) -> torch.Tensor:
        """Per-env effective command period (s) = hold * step_dt, as a [num_envs,1] obs so the
        policy is aware of how stale its command will be and can size its moment accordingly."""
        return (self._hold.float() * self._step_dt).unsqueeze(1)

    def process_actions(self, actions: torch.Tensor):
        """Process actions and prepare wrench to apply.

        Args:
            actions: Actions from policy [num_envs, 4]
                action[:, 0]: Thrust (normalized [-1, 1])
                action[:, 1:4]: Moments [roll, pitch, yaw]
        """
        # Store raw actions (required by ActionTerm base class)
        self._raw_actions[:] = actions

        # Clamp actions to valid range
        actions = actions.clone().clamp(-1.0, 1.0)

        # Command-hold (rate) DR: zero-order hold per env — only refresh the applied command when
        # this env's counter reaches its hold length, otherwise reuse the last one (a staler command).
        if self.cfg.randomize_hold_steps is not None:
            fresh = (self._hold_ctr % self._hold) == 0
            self._held_action = torch.where(fresh.unsqueeze(1), actions, self._held_action)
            self._hold_ctr += 1
            actions = self._held_action

        # Convert thrust action from [-1, 1] to [0, thrust_to_weight * weight]
        # action=-1 → 0% thrust, action=+1 → 100% thrust. Uses PER-ENV gains
        # (equal to the cfg scalar unless DR is enabled) into TARGET buffers; the
        # applied wrench follows the target via optional first-order motor lag.
        thrust_normalized = (actions[:, 0] + 1.0) / 2.0  # Map [-1, 1] → [0, 1]
        self._thrust_target[:, 0, 2] = self._t2w * self._robot_weight * thrust_normalized
        self._moment_target[:, 0, :] = self._mscale.unsqueeze(1) * actions[:, 1:]

        # Store processed actions (required by ActionTerm base class)
        self._processed_actions[:] = actions

    def apply_actions(self):
        """Apply the computed wrench to the robot body (with optional motor lag)."""
        if self._motor_tau > 0.0:
            # first-order lag: applied += (dt/(tau+dt)) * (target - applied)
            a = self._step_dt / (self._motor_tau + self._step_dt)
            self._thrust += a * (self._thrust_target - self._thrust)
            self._moment += a * (self._moment_target - self._moment)
        else:
            self._thrust.copy_(self._thrust_target)
            self._moment.copy_(self._moment_target)
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            body_ids=self._body_id,
            forces=self._thrust,
            torques=self._moment,
        )

    """
    Properties
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (4: thrust + 3 moments)."""
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw actions received from the policy."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed actions after clamping."""
        return self._processed_actions


@configclass
class DirectThrustMomentActionCfg(ActionTermCfg):
    """Configuration for direct thrust and moment action term."""

    class_type: type[ActionTerm] = DirectThrustMomentAction

    asset_name: str = "robot"
    """Name of the robot asset in the scene."""

    body_name: str = "body"
    """Name of the body to apply forces/torques to."""

    thrust_to_weight: float = 1.9
    """Thrust-to-weight ratio. Maximum thrust = thrust_to_weight * robot_weight."""

    moment_scale: float = 0.01
    """Scaling factor for moments. Final moment = moment_scale * action[1:4]."""

    # --- Domain randomization (anti-overfit / sim2real). All default to no-op so
    # the steering env and model_6998 are unaffected. Set on the body-rate DR env. ---
    randomize_thrust_to_weight: tuple[float, float] | None = None
    """If set, per-env thrust-to-weight is resampled U(lo,hi) each reset (actuation authority DR)."""
    randomize_moment_scale: tuple[float, float] | None = None
    """If set, per-env moment scale is resampled U(lo,hi) each reset (rotational authority DR)."""
    motor_tau: float = 0.0
    """First-order motor-lag time constant (s) on the applied wrench. 0 = instantaneous (default)."""
    randomize_hold_steps: tuple[int, int] | None = None
    """If set, per-env command-hold length (control steps) is resampled randint(lo,hi) each reset:
    a zero-order hold that makes the EFFECTIVE command period random (rate-robustness DR). The
    effective period is surfaced to the policy via `effective_period_obs`. None = fresh every step."""

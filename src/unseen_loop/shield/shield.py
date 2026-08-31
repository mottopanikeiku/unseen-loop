"""Deterministic clear reference for the two-step counterfactual safety shield.

The core deliberately has no cryptographic dependency and makes no encrypted-
execution claim.  Client-local certificates use state-derived margins, while
serialized receipts redact those numeric values.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from unseen_loop.shield.certificate import (
    CandidateCertificate,
    ErrorBuffer,
    HorizonMargins,
    SafetyMargins,
    certify_candidate,
)
from unseen_loop.shield.environment import polynomial_step
from unseen_loop.shield.types import Action, DynamicsConfig, SafetyLimits, ShieldState


class ShieldMode(StrEnum):
    """Reference and ablation modes with explicit safety semantics."""

    ROBUST = "robust"
    CLEAR_BASELINE = "clear_baseline"
    ONE_STEP_ABLATION = "one_step_ablation"
    NO_SHIELD_ABLATION = "no_shield_ablation"


@dataclass(frozen=True, slots=True)
class ShieldConfig:
    """Public shield configuration with explicit reference/ablation semantics."""

    mode: ShieldMode = ShieldMode.ROBUST
    error_buffer: ErrorBuffer = field(default_factory=ErrorBuffer)
    emergency_action: Action = Action.BRAKE

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ShieldMode(self.mode))
        object.__setattr__(self, "emergency_action", Action(self.emergency_action))
        if self.emergency_action is not Action.BRAKE:
            raise ValueError("the shield protocol emergency action is BRAKE")

    @property
    def effective_buffer(self) -> ErrorBuffer:
        if self.mode is ShieldMode.CLEAR_BASELINE:
            return ErrorBuffer()
        return self.error_buffer

    @property
    def evaluated_horizons(self) -> int:
        return 1 if self.mode is ShieldMode.ONE_STEP_ABLATION else 2


@dataclass(frozen=True, slots=True)
class CandidateRollout:
    """Two future states under one constant public candidate action."""

    action: Action
    states: tuple[ShieldState, ShieldState]


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Deterministic result of exact client-side candidate selection."""

    action: Action
    reason: str
    emergency_fallback: bool
    selected_certified: bool


@dataclass(frozen=True, slots=True)
class DecisionReceipt:
    """Client-local replay evidence with privacy-redacted serialization."""

    step: int
    requested_action: Action
    selected_action: Action
    mode: ShieldMode
    emergency_action: Action
    reason: str
    emergency_fallback: bool
    selected_certified: bool
    candidates: tuple[CandidateCertificate, ...] = field(repr=False)
    receipt_digest: str

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        actions = tuple(candidate.action for candidate in self.candidates)
        if actions != tuple(Action):
            raise ValueError("receipt candidates must appear once in public enum order")
        if len(self.receipt_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.receipt_digest
        ):
            raise ValueError("receipt_digest must be a lowercase SHA-256 digest")

    def verify(self) -> None:
        """Replay selection from evidence and authenticate the canonical receipt."""

        expected = select_action(
            self.candidates,
            self.requested_action,
            emergency_action=self.emergency_action,
            enabled=self.mode is not ShieldMode.NO_SHIELD_ABLATION,
        )
        observed = SelectionResult(
            action=self.selected_action,
            reason=self.reason,
            emergency_fallback=self.emergency_fallback,
            selected_certified=self.selected_certified,
        )
        if observed != expected:
            raise ValueError("receipt selection does not replay from its candidate evidence")
        if self.receipt_digest != _receipt_digest(self, include_digest=False):
            raise ValueError("receipt digest does not match its canonical evidence")
        for candidate in self.candidates:
            candidate.assert_sound()

    def to_dict(self) -> dict[str, object]:
        """Serialize redacted client evidence; do not send it to server/evaluator logs."""

        return _receipt_payload(self, include_digest=True)


@dataclass(frozen=True, slots=True)
class ShieldMetrics:
    """Aggregate certificate coverage and externally observed false-safe outcomes."""

    decisions: int
    covered_decisions: int
    certified_candidates: int
    candidate_evaluations: int
    false_safes: int
    certified_outcomes: int
    emergency_fallbacks: int

    @property
    def coverage(self) -> float:
        return self.covered_decisions / self.decisions if self.decisions else 0.0

    @property
    def candidate_coverage(self) -> float:
        if not self.candidate_evaluations:
            return 0.0
        return self.certified_candidates / self.candidate_evaluations

    @property
    def false_safe_rate(self) -> float:
        return self.false_safes / self.certified_outcomes if self.certified_outcomes else 0.0

    @property
    def emergency_fallback_rate(self) -> float:
        return self.emergency_fallbacks / self.decisions if self.decisions else 0.0


def rollout_candidates(
    state: ShieldState,
    dynamics: DynamicsConfig,
) -> tuple[CandidateRollout, ...]:
    """Roll every candidate through exactly two public polynomial transitions."""

    rollouts: list[CandidateRollout] = []
    for action in Action:
        first = polynomial_step(state, action, dynamics)
        second = polynomial_step(first, action, dynamics)
        rollouts.append(CandidateRollout(action=action, states=(first, second)))
    return tuple(rollouts)


def _obstacle_margin(state: ShieldState, limits: SafetyLimits) -> float:
    """Return the minimum signed spatial margin.

    Circular-obstacle margins use squared distance and workspace margins use
    signed distance.  All are polynomial expressions and share the invariant
    that strict positivity means spatial safety.  The public error buffer must
    be stated in the units of this minimum reference expression.
    """

    clearance = limits.vehicle_radius + limits.obstacle_clearance
    margins = [
        state.x - (limits.x_bounds[0] + clearance),
        (limits.x_bounds[1] - clearance) - state.x,
        state.y - (limits.y_bounds[0] + clearance),
        (limits.y_bounds[1] - clearance) - state.y,
    ]
    for obstacle in limits.obstacles:
        required = obstacle.radius + clearance
        dx = state.x - obstacle.x
        dy = state.y - obstacle.y
        margins.append(dx * dx + dy * dy - required * required)
    return min(margins)


def state_safety_margins(state: ShieldState, limits: SafetyLimits) -> SafetyMargins:
    """Evaluate the frozen four-family clear margin specification."""

    speed_squared = state.vx * state.vx + state.vy * state.vy
    return SafetyMargins(
        obstacle=_obstacle_margin(state, limits),
        speed=limits.max_speed * limits.max_speed - speed_squared,
        tilt=limits.max_abs_tilt * limits.max_abs_tilt - state.tilt * state.tilt,
        battery=state.battery - limits.min_battery,
    )


def compute_safety_margins(
    rollouts: Sequence[CandidateRollout],
    limits: SafetyLimits,
) -> dict[Action, tuple[HorizonMargins, HorizonMargins]]:
    """Compute every candidate/horizon margin without logging the input state."""

    actions = tuple(rollout.action for rollout in rollouts)
    if actions != tuple(Action):
        raise ValueError("rollouts must contain every candidate once in public enum order")
    return {
        rollout.action: (
            HorizonMargins(horizon=1, margins=state_safety_margins(rollout.states[0], limits)),
            HorizonMargins(horizon=2, margins=state_safety_margins(rollout.states[1], limits)),
        )
        for rollout in rollouts
    }


def certify_rollouts(
    margins: Mapping[Action, tuple[HorizonMargins, HorizonMargins]],
    config: ShieldConfig,
) -> tuple[CandidateCertificate, ...]:
    """Certify all five candidates under the selected baseline/ablation mode."""

    if set(margins) != set(Action):
        raise ValueError("margins must contain exactly the five public candidate actions")
    count = config.evaluated_horizons
    return tuple(
        certify_candidate(
            action,
            margins[action][:count],
            error_buffer=config.effective_buffer,
        )
        for action in Action
    )


def select_action(
    certificates: Sequence[CandidateCertificate],
    requested_action: Action,
    *,
    emergency_action: Action = Action.BRAKE,
    enabled: bool = True,
) -> SelectionResult:
    """Select exactly and stably; action enum order breaks equal-margin ties."""

    requested = Action(requested_action)
    emergency = Action(emergency_action)
    ordered = tuple(certificates)
    if tuple(certificate.action for certificate in ordered) != tuple(Action):
        raise ValueError("certificates must contain every action once in public enum order")
    if not enabled:
        return SelectionResult(
            action=requested,
            reason="shield_disabled",
            emergency_fallback=False,
            selected_certified=False,
        )

    by_action = {certificate.action: certificate for certificate in ordered}
    if by_action[requested].certified:
        return SelectionResult(
            action=requested,
            reason="requested_certified",
            emergency_fallback=False,
            selected_certified=True,
        )

    safe = tuple(certificate for certificate in ordered if certificate.certified)
    if safe:
        selected = max(
            safe,
            key=lambda certificate: (
                certificate.minimum_buffered_margin,
                -int(certificate.action),
            ),
        )
        return SelectionResult(
            action=selected.action,
            reason="safest_certified_alternative",
            emergency_fallback=False,
            selected_certified=True,
        )
    return SelectionResult(
        action=emergency,
        reason="uncertified_emergency_fallback",
        emergency_fallback=True,
        selected_certified=False,
    )


def shield_step(
    state: ShieldState,
    requested_action: Action,
    *,
    step: int,
    dynamics: DynamicsConfig,
    limits: SafetyLimits,
    config: ShieldConfig | None = None,
) -> DecisionReceipt:
    """Evaluate one private state and return state-free, replayable decision evidence."""
    config = config or ShieldConfig()

    if step < 0:
        raise ValueError("step must be non-negative")
    rollouts = rollout_candidates(state, dynamics)
    margins = compute_safety_margins(rollouts, limits)
    certificates = certify_rollouts(margins, config)
    selection = select_action(
        certificates,
        requested_action,
        emergency_action=config.emergency_action,
        enabled=config.mode is not ShieldMode.NO_SHIELD_ABLATION,
    )
    provisional = DecisionReceipt(
        step=step,
        requested_action=Action(requested_action),
        selected_action=selection.action,
        mode=config.mode,
        emergency_action=config.emergency_action,
        reason=selection.reason,
        emergency_fallback=selection.emergency_fallback,
        selected_certified=selection.selected_certified,
        candidates=certificates,
        receipt_digest="0" * 64,
    )
    receipt = DecisionReceipt(
        step=provisional.step,
        requested_action=provisional.requested_action,
        selected_action=provisional.selected_action,
        mode=provisional.mode,
        emergency_action=provisional.emergency_action,
        reason=provisional.reason,
        emergency_fallback=provisional.emergency_fallback,
        selected_certified=provisional.selected_certified,
        candidates=provisional.candidates,
        receipt_digest=_receipt_digest(provisional, include_digest=False),
    )
    receipt.verify()
    return receipt


@dataclass(frozen=True, slots=True)
class SafetyShield:
    """Configured facade for repeated deterministic clear shield decisions."""

    dynamics: DynamicsConfig
    limits: SafetyLimits
    config: ShieldConfig = field(default_factory=ShieldConfig)

    def decide(self, state: ShieldState, requested_action: Action, *, step: int) -> DecisionReceipt:
        return shield_step(
            state,
            requested_action,
            step=step,
            dynamics=self.dynamics,
            limits=self.limits,
            config=self.config,
        )


def compute_shield_metrics(
    receipts: Sequence[DecisionReceipt],
    realized_safe: Sequence[bool],
) -> ShieldMetrics:
    """Score receipts against state-free external safety outcomes.

    ``realized_safe`` is an independently observed boolean for the action that
    was executed.  A false safe is a certified selection whose observed outcome
    is unsafe; unshielded and emergency actions never receive a safety claim.
    """

    if len(receipts) != len(realized_safe):
        raise ValueError("receipts and realized_safe must have equal length")
    for receipt in receipts:
        receipt.verify()
    covered = sum(receipt.selected_certified for receipt in receipts)
    certified_candidates = sum(
        candidate.certified for receipt in receipts for candidate in receipt.candidates
    )
    false_safes = sum(
        receipt.selected_certified and not bool(outcome)
        for receipt, outcome in zip(receipts, realized_safe, strict=True)
    )
    return ShieldMetrics(
        decisions=len(receipts),
        covered_decisions=covered,
        certified_candidates=certified_candidates,
        candidate_evaluations=sum(len(receipt.candidates) for receipt in receipts),
        false_safes=false_safes,
        certified_outcomes=covered,
        emergency_fallbacks=sum(receipt.emergency_fallback for receipt in receipts),
    )


def _receipt_payload(
    receipt: DecisionReceipt,
    *,
    include_digest: bool,
) -> dict[str, object]:
    ranked = sorted(
        (candidate for candidate in receipt.candidates if candidate.certified),
        key=lambda candidate: (-candidate.minimum_buffered_margin, int(candidate.action)),
    )
    selection_ranks = {candidate.action: rank for rank, candidate in enumerate(ranked)}
    candidates: list[dict[str, object]] = [
        {
            "action": candidate.action.name,
            "certified": candidate.certified,
            "selection_rank": selection_ranks.get(candidate.action),
            "evaluated_horizons": [step.horizon for step in candidate.steps],
        }
        for candidate in receipt.candidates
    ]
    payload: dict[str, object] = {
        "step": receipt.step,
        "requested_action": receipt.requested_action.name,
        "selected_action": receipt.selected_action.name,
        "mode": receipt.mode.value,
        "emergency_action": receipt.emergency_action.name,
        "reason": receipt.reason,
        "emergency_fallback": receipt.emergency_fallback,
        "selected_certified": receipt.selected_certified,
        "candidates": candidates,
        "privacy_scope": (
            "client-local categorical evidence; do not include in server/evaluator logs; "
            "numeric state-derived margins redacted"
        ),
    }
    if include_digest:
        payload["receipt_digest"] = receipt.receipt_digest
    return payload


def _receipt_digest(receipt: DecisionReceipt, *, include_digest: bool) -> str:
    """Hash the redacted decision summary, not the client-private margin evidence."""

    payload = _receipt_payload(receipt, include_digest=include_digest)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()

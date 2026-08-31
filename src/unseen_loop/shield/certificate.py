"""Clear safety-margin certificates for two-step counterfactual rollouts.

This module contains deterministic arithmetic and evidence objects only.  It does
not perform, or claim to perform, encrypted execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from unseen_loop.shield.types import Action


class MarginFamily(StrEnum):
    """The four independently certified safety obligations."""

    OBSTACLE = "obstacle"
    SPEED = "speed"
    TILT = "tilt"
    BATTERY = "battery"


MARGIN_FAMILIES = tuple(MarginFamily)


@dataclass(frozen=True, slots=True)
class SafetyMargins:
    """Signed polynomial margins; a value is safe exactly when it is positive."""

    obstacle: float
    speed: float
    tilt: float
    battery: float

    def __post_init__(self) -> None:
        for family in MARGIN_FAMILIES:
            value = self.for_family(family)
            if not isfinite(value):
                raise ValueError(f"{family.value} margin must be finite")

    def for_family(self, family: MarginFamily) -> float:
        return float(getattr(self, family.value))

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.obstacle, self.speed, self.tilt, self.battery)

    def as_dict(self) -> dict[str, float]:
        return {family.value: self.for_family(family) for family in MARGIN_FAMILIES}


@dataclass(frozen=True, slots=True)
class ErrorBuffer:
    """Conservative absolute error bounds in the units of each margin family.

    Bounds must cover every source of approximation between the reference
    margins and the margins being certified.  One conservative bound is applied
    at each horizon; callers may instead certify already horizon-specific bounds
    by constructing separate certificates.
    """

    obstacle: float = 0.0
    speed: float = 0.0
    tilt: float = 0.0
    battery: float = 0.0

    def __post_init__(self) -> None:
        for family in MARGIN_FAMILIES:
            value = self.for_family(family)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{family.value} error buffer must be finite and non-negative")

    def for_family(self, family: MarginFamily) -> float:
        return float(getattr(self, family.value))

    def as_margins(self) -> SafetyMargins:
        return SafetyMargins(self.obstacle, self.speed, self.tilt, self.battery)

    def as_dict(self) -> dict[str, float]:
        return {family.value: self.for_family(family) for family in MARGIN_FAMILIES}


@dataclass(frozen=True, slots=True)
class HorizonMargins:
    """The four margins for one future state in the candidate rollout."""

    horizon: int
    margins: SafetyMargins

    def __post_init__(self) -> None:
        if self.horizon not in (1, 2):
            raise ValueError("shield horizons must be 1 or 2")


@dataclass(frozen=True, slots=True)
class BufferedHorizonMargins:
    """Raw and robustly buffered margins for one candidate and horizon."""

    horizon: int
    raw: SafetyMargins
    buffer: ErrorBuffer
    buffered: SafetyMargins

    def __post_init__(self) -> None:
        if self.horizon not in (1, 2):
            raise ValueError("shield horizons must be 1 or 2")
        for family in MARGIN_FAMILIES:
            expected = self.raw.for_family(family) - self.buffer.for_family(family)
            if self.buffered.for_family(family) != expected:
                raise ValueError("buffered margins must equal raw margins minus the error buffer")


@dataclass(frozen=True, slots=True)
class CandidateCertificate:
    """Proof obligations for one public candidate action."""

    action: Action
    steps: tuple[BufferedHorizonMargins, ...]
    active_families: tuple[MarginFamily, ...]
    certified: bool

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("a candidate certificate requires at least one horizon")
        horizons = tuple(step.horizon for step in self.steps)
        if horizons not in ((1,), (1, 2)):
            raise ValueError("candidate horizons must be ordered and contiguous from one")
        if self.active_families != MARGIN_FAMILIES:
            raise ValueError("a safety certificate must cover all four margin families")
        expected = all(
            step.buffered.for_family(family) > 0
            for step in self.steps
            for family in self.active_families
        )
        if self.certified != expected:
            raise ValueError("certified must exactly match positivity of every buffered margin")

    @property
    def minimum_buffered_margin(self) -> float:
        return min(
            step.buffered.for_family(family)
            for step in self.steps
            for family in self.active_families
        )

    @property
    def failed_obligations(self) -> tuple[tuple[int, MarginFamily], ...]:
        return tuple(
            (step.horizon, family)
            for step in self.steps
            for family in self.active_families
            if step.buffered.for_family(family) <= 0
        )

    def assert_sound(self) -> None:
        """Fail if a claimed certificate lacks a strictly positive obligation."""

        if self.certified and self.failed_obligations:
            raise AssertionError("certified-safe candidate has a non-positive buffered margin")


def certify_candidate(
    action: Action,
    margins: tuple[HorizonMargins, ...],
    *,
    error_buffer: ErrorBuffer | None = None,
) -> CandidateCertificate:
    """Apply conservative buffers and certify all four strict inequalities."""
    error_buffer = error_buffer or ErrorBuffer()

    horizons = tuple(item.horizon for item in margins)
    if horizons not in ((1,), (1, 2)):
        raise ValueError("margins must contain ordered horizon one, or horizons one and two")
    buffer_values = error_buffer.as_margins()
    steps = tuple(
        BufferedHorizonMargins(
            horizon=item.horizon,
            raw=item.margins,
            buffer=error_buffer,
            buffered=SafetyMargins(
                *(
                    raw - bound
                    for raw, bound in zip(
                        item.margins.as_tuple(), buffer_values.as_tuple(), strict=True
                    )
                )
            ),
        )
        for item in margins
    )
    certified = all(
        step.buffered.for_family(family) > 0 for step in steps for family in MARGIN_FAMILIES
    )
    result = CandidateCertificate(
        action=Action(action),
        steps=steps,
        active_families=MARGIN_FAMILIES,
        certified=certified,
    )
    result.assert_sound()
    return result

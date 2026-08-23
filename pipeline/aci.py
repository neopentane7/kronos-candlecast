"""Adaptive conformal inference for the served bands.

**Why ACI and not split conformal.** Report §17d establishes that regime-stratified split
conformal at a 30-session horizon needs 18 independent forecast dates at 80% and 38 at
90%; twelve exist. That is infeasible, not underpowered, and it is a property of the
horizon and the calendar rather than of this corpus. ACI's guarantee is asymptotic in the
number of update steps and needs neither exchangeability nor a minimum calibration set, so
it is the only method available at the levels split conformal cannot reach. The design
decision is forced, not preferred.

**State is per (ticker, level, horizon step).** This is the part worth arguing for. A
30-step forecast is not one prediction problem: Fact of Record F2 found the cone nearly
correctly sized at h=1 and less than half wide enough by h=30. A single alpha per ticker
would average a near-correct short horizon against a badly broken long one and fix
neither. Per-step alphas let the correction grow with horizon, which is the shape of the
defect.

The update is the standard one. After observing whether step h of a forecast made h
sessions ago actually landed inside its band:

    alpha <- alpha + gamma * (alpha_target - err)      err = 1 if it missed, else 0

A miss pushes alpha down, which widens the next band; a hit lets it drift back. Long-run
coverage converges to the target for any bounded sequence of outcomes, adversarial
included -- which is the property that matters when the outcome is a market.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Provisional until the A5 validation grid lands and supplies a fitted value. Recorded in
# every response's metadata as provisional so nothing downstream mistakes it for tuned.
DEFAULT_GAMMA = 0.005
PROVISIONAL = True

# Alpha is a probability and the update is unbounded, so a long adverse run could drive it
# negative or past 1. Clamping keeps the band finite; hitting a clamp is a signal the
# forecaster is badly wrong, and the run summary reports it.
# 0.60 rather than 0.50 on purpose: the 50% band's target alpha *is* 0.50, so a bound at
# 0.50 sits exactly on it and clips every hit that nudges alpha upward -- which showed up
# as 1252 spurious clamps in the first seeding run while 80% and 90% were untouched.
ALPHA_MIN = 0.001
ALPHA_MAX = 0.600

LEVELS = (0.50, 0.80, 0.90)


def _key(ticker: str, level: float, step: int) -> str:
    return f"{ticker}|{level:.2f}|{step:d}"


@dataclass
class ACIState:
    """Per (ticker, level, step) alphas, persisted as data and committed by the job."""

    gamma: float = DEFAULT_GAMMA
    provisional: bool = PROVISIONAL
    alphas: dict[str, float] = field(default_factory=dict)
    updates: int = 0
    clamped: int = 0

    def alpha(self, ticker: str, level: float, step: int) -> float:
        """Current alpha, defaulting to the nominal miss rate before any observation."""
        return self.alphas.get(_key(ticker, level, step), round(1.0 - level, 10))

    def effective_level(self, ticker: str, level: float, step: int) -> float:
        """The level the band should actually be drawn at to hit ``level`` long-run."""
        return 1.0 - self.alpha(ticker, level, step)

    def update(self, ticker: str, level: float, step: int, covered: bool) -> float:
        """Fold one realized outcome into the state and return the new alpha."""
        target = 1.0 - level
        current = self.alpha(ticker, level, step)
        err = 0.0 if covered else 1.0
        new = current + self.gamma * (target - err)

        clipped = min(max(new, ALPHA_MIN), ALPHA_MAX)
        if clipped != new:
            self.clamped += 1
        self.alphas[_key(ticker, level, step)] = clipped
        self.updates += 1
        return clipped

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "provisional": self.provisional,
            "updates": self.updates,
            "clamped": self.clamped,
            "alpha_min": ALPHA_MIN,
            "alpha_max": ALPHA_MAX,
            "alphas": dict(sorted(self.alphas.items())),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ACIState:
        return cls(
            gamma=float(d.get("gamma", DEFAULT_GAMMA)),
            provisional=bool(d.get("provisional", PROVISIONAL)),
            alphas={str(k): float(v) for k, v in (d.get("alphas") or {}).items()},
            updates=int(d.get("updates", 0)),
            clamped=int(d.get("clamped", 0)),
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> ACIState:
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

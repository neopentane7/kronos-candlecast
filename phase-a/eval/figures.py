"""Figures for the evaluation harness.

A3's acceptance names a reliability diagram and a PIT histogram; both live here, along
with the regime and horizon views the analysis actually turns on.

Two conventions applied throughout:

* **Coverage is drawn with its uncertainty.** A reliability point without an interval
  invites the reader to over-read a difference the panel cannot resolve — our own sweep
  had 95% intervals 0.37 wide. Every coverage marker carries a block-bootstrap interval.
* **Every figure carries the research-only disclaimer**, because figures travel
  separately from the report that made them.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from eval.metrics import (  # noqa: E402
    NOMINAL_LEVELS,
    block_bootstrap_ci,
    coverage_indicator,
    crps,
    pit_values,
)

from common.results import DISCLAIMER  # noqa: E402

# Colourblind-safe, and deliberately not red/green — this is a finance surface where
# those carry an unintended meaning.
PALETTE = {
    "kronos_zeroshot": "#0B6E7F",
    "kronos_finetuned": "#7A3E9D",
    "random_walk_drift": "#B26B00",
    "last_value": "#6C8288",
    "conformal": "#2E6F4E",
}
FALLBACK = ["#0B6E7F", "#B26B00", "#7A3E9D", "#2E6F4E", "#9E3A31", "#6C8288"]


def _colour(name: str, i: int) -> str:
    return PALETTE.get(name, FALLBACK[i % len(FALLBACK)])


def _finish(fig, out_path: Path, subtitle: str = "") -> Path:
    # Two lines rather than one: concatenated, the provenance string and the disclaimer
    # overflow the canvas and both get clipped at the edges.
    if subtitle:
        fig.text(0.5, 0.032, subtitle, ha="center", fontsize=7, color="#3D4E54")
        fig.text(0.5, 0.008, DISCLAIMER, ha="center", fontsize=6.5, color="#6C8288")
        rect = (0, 0.062, 1, 1)
    else:
        fig.text(0.5, 0.010, DISCLAIMER, ha="center", fontsize=6.5, color="#6C8288")
        rect = (0, 0.038, 1, 1)
    fig.tight_layout(rect=rect)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def reliability_diagram(
    obs: np.ndarray,
    ensembles: dict[str, np.ndarray],
    block_ids: np.ndarray,
    out_path: Path,
    levels: tuple[float, ...] = NOMINAL_LEVELS,
    seed: int = 0,
    subtitle: str = "",
) -> Path:
    """Nominal against empirical coverage, one line per model, with bootstrap intervals.

    The diagonal is perfect calibration. Below it is over-confidence — bands too narrow —
    which is the failure mode this project exists to measure.
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#8FA1A6", zorder=1, label="perfect calibration")

    for i, (name, ens) in enumerate(ensembles.items()):
        emp, lo, hi = [], [], []
        for level in levels:
            ind = coverage_indicator(obs, ens, level)
            a, b = block_bootstrap_ci(ind, block_ids, n_boot=1000, seed=seed)
            emp.append(float(ind.mean()))
            lo.append(a)
            hi.append(b)
        emp, lo, hi = np.array(emp), np.array(lo), np.array(hi)
        ax.errorbar(
            levels,
            emp,
            yerr=[emp - lo, hi - emp],
            marker="o",
            ms=5,
            lw=1.6,
            capsize=3,
            color=_colour(name, i),
            label=name,
            zorder=3,
        )

    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Reliability: does the stated coverage happen?")
    ax.set_xlim(0.4, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    return _finish(fig, out_path, subtitle or "error bars are 95% block-bootstrap intervals")


def pit_histogram(
    obs: np.ndarray,
    ens: np.ndarray,
    out_path: Path,
    bins: int = 10,
    seed: int = 0,
    subtitle: str = "",
    title: str = "PIT histogram",
) -> Path:
    """Randomized PIT values; flat means calibrated.

    Shape reads as diagnosis: a U means bands too narrow, a dome means too wide, and a
    tilt means the forecast is biased in the direction of the heavy end.

    The reference band assumes independent observations, which a market panel is not, so
    it is drawn as guidance rather than a test.
    """
    pit = pit_values(obs, ens, rng=np.random.default_rng(seed)).reshape(-1)
    n = pit.size
    counts, edges = np.histogram(pit, bins=bins, range=(0, 1))
    frac = counts / n

    expected = 1.0 / bins
    sd = np.sqrt(expected * (1 - expected) / n)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(
        edges[:-1],
        frac,
        width=np.diff(edges),
        align="edge",
        color="#0B6E7F",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(expected, color="#3D4E54", ls="--", lw=1, label="uniform")
    ax.axhspan(
        expected - 2 * sd,
        expected + 2 * sd,
        color="#8FA1A6",
        alpha=0.20,
        label="±2σ if independent",
    )

    ax.set_xlabel("PIT value")
    ax.set_ylabel("fraction of observations")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", fontsize=8, frameon=False, ncol=2)
    return _finish(
        fig,
        out_path,
        subtitle or "U-shape = bands too narrow · dome = too wide · tilt = biased",
    )


def coverage_by_regime(
    obs: np.ndarray,
    ensembles: dict[str, np.ndarray],
    terciles: np.ndarray,
    out_path: Path,
    level: float = 0.80,
    subtitle: str = "",
) -> Path:
    """Coverage per volatility regime — the view marginal coverage hides.

    A model can sit on the diagonal in the reliability diagram while every stratum here is
    wrong, which is why A5 treats this as the headline rather than the appendix.
    """
    names = ["calm", "mid", "volatile"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    width = 0.8 / max(len(ensembles), 1)

    for i, (model, ens) in enumerate(ensembles.items()):
        vals = []
        for t in range(3):
            idx = np.flatnonzero(terciles == t)
            vals.append(
                float(coverage_indicator(obs[idx], ens[idx], level).mean()) if len(idx) else np.nan
            )
        ax.bar(
            np.arange(3) + i * width - 0.4 + width / 2,
            vals,
            width * 0.92,
            color=_colour(model, i),
            label=model,
        )

    ax.axhline(level, color="#3D4E54", ls="--", lw=1.2, label=f"nominal {level:.0%}")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{n}\n(lookback ATR)" for n in names])
    ax.set_ylabel(f"empirical coverage at {level:.0%}")
    ax.set_title("Coverage by volatility regime")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    return _finish(
        fig, out_path, subtitle or "marginal coverage can be right while every bar is wrong"
    )


def horizon_curve(
    obs: np.ndarray,
    ensembles: dict[str, np.ndarray],
    out_path: Path,
    subtitle: str = "",
) -> Path:
    """CRPS against horizon step, log-log, with the fitted exponent.

    Pure diffusion gives a slope of 0.5. A steeper slope means drift error compounding
    rather than variance merely widening.
    """
    horizon = obs.shape[1]
    steps = np.arange(1, horizon + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for i, (name, ens) in enumerate(ensembles.items()):
        vals = np.array(
            [float(np.mean(crps(obs[:, h : h + 1], ens[:, h : h + 1, :]))) for h in range(horizon)]
        )
        k = np.polyfit(np.log(steps), np.log(vals), 1)[0]
        ax.plot(
            steps,
            vals,
            marker="o",
            ms=3,
            lw=1.5,
            color=_colour(name, i),
            label=f"{name}  (h^{k:.3f})",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("horizon step")
    ax.set_ylabel("CRPS (fair)")
    ax.set_title("Error growth with horizon")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, frameon=False)
    return _finish(
        fig, out_path, subtitle or "slope 0.5 is pure diffusion; steeper is compounding drift"
    )


def spread_ratio_sweep(
    rows: list[dict],
    out_path: Path,
    target: float = 0.85,
    subtitle: str = "",
) -> Path:
    """Spread ratio against horizon, one curve per temperature arm.

    The pre-registered target is drawn on: reaching it at h = max means the horizon
    compression is substantially a sampler artifact rather than a tokenizer limitation.
    A ratio of 1.0 is a correctly sized cone.
    """
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for i, row in enumerate(rows):
        ratios = row["spread_ratio_by_step"]
        steps = np.arange(1, len(ratios) + 1)
        label = f"T = {row['temperature']}  (CRPS {row['crps_fair']:.1f})"
        ax.plot(steps, ratios, marker="o", ms=2.6, lw=1.6, color=_colour(label, i), label=label)

    ax.axhline(1.0, color="#39525A", lw=1.0, ls="-", alpha=0.5)
    ax.text(
        len(steps), 1.005, "correctly sized", ha="right", va="bottom", fontsize=7.5, color="#39525A"
    )
    ax.axhline(target, color="#A6402E", lw=1.2, ls="--")
    # Below the line and hard right: the curves start high on the left and fall to the
    # right, so this is the one corner none of them passes through.
    ax.text(
        len(steps),
        target - 0.015,
        f"pre-registered target {target:.2f}",
        ha="right",
        va="top",
        fontsize=7.5,
        color="#A6402E",
    )

    ax.set_xlabel("horizon step")
    ax.set_ylabel("predicted spread / realized spread")
    ax.set_title("Does temperature repair horizon under-propagation?")
    ax.set_ylim(0.0, max(1.08, max(max(r["spread_ratio_by_step"]) for r in rows) + 0.06))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    return _finish(
        fig, out_path, subtitle or "identical windows across arms; ratio 1.0 is correct width"
    )


def write_all(
    obs: np.ndarray,
    ensembles: dict[str, np.ndarray],
    block_ids: np.ndarray,
    run_dir: Path,
    terciles: np.ndarray | None = None,
    seed: int = 0,
    subtitle: str = "",
) -> list[Path]:
    """Every figure for one run, written into its results directory."""
    written = [
        reliability_diagram(
            obs, ensembles, block_ids, run_dir / "reliability.png", seed=seed, subtitle=subtitle
        ),
        horizon_curve(obs, ensembles, run_dir / "horizon_crps.png", subtitle=subtitle),
    ]
    for name, ens in ensembles.items():
        written.append(
            pit_histogram(
                obs,
                ens,
                run_dir / f"pit_{name}.png",
                seed=seed,
                title=f"PIT histogram — {name}",
                subtitle=subtitle,
            )
        )
    if terciles is not None:
        written.append(
            coverage_by_regime(
                obs, ensembles, terciles, run_dir / "coverage_by_regime.png", subtitle=subtitle
            )
        )
    return written

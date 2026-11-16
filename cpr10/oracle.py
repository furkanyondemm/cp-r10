"""Predeclared grid + local-refinement CP targets. No model inference.

Search and tie-break match the prior CIFAR-100 coverage diagnostic. The
implementation uses stable log probabilities, cached probabilities clipped at
1e-12, and the finite-sample plug-in level. This is a finite-grid diagnostic,
not a global optimizer or a distribution-free conformal guarantee.
"""

from __future__ import annotations
import numpy as np
from .core import ALPHAS, validate_probs
from .numerics import (
    temperature_log_probs,
    plugin_log_thresholds,
    metrics_from_log_threshold,
)


def coarse_grid():
    return np.unique(
        np.concatenate(
            [
                np.array([0.0]),
                np.linspace(0.05, 0.30, 6),
                np.linspace(0.325, 1.50, 48),
                np.linspace(1.60, 3.00, 15),
                np.linspace(3.25, 5.00, 8),
                np.linspace(6.0, 20.0, 15),
            ]
        )
    )


def local_grid(coarse, beta, points=41):
    i = int(np.argmin(np.abs(coarse - beta)))
    return np.linspace(
        coarse[max(0, i - 1)], coarse[min(len(coarse) - 1, i + 1)], points
    )


def scan_betas(p, y, betas, device="cpu"):
    p = validate_probs(p, y)
    y = np.asarray(y, int)
    rows = []
    for beta in np.asarray(betas, float):
        b = float(beta)
        lp = temperature_log_probs(p, b)
        thresholds = plugin_log_thresholds(lp, ALPHAS)
        for alpha, t in zip(ALPHAS, thresholds):
            m = metrics_from_log_threshold(lp, y, t)
            coverage = m["coverage"]
            rows.append(
                {
                    "beta": b,
                    "alpha": alpha,
                    "q_plugin": float(-np.expm1(t)),
                    "log_threshold": str(t),
                    **m,
                    "abs_coverage_error": abs(coverage - (1 - alpha)),
                    "signed_coverage_error": coverage - (1 - alpha),
                }
            )
    return rows


def best_alpha(rows, alpha, objective="coverage"):
    z = [r for r in rows if r["alpha"] == alpha]
    if objective == "coverage":
        return min(
            z, key=lambda r: (r["abs_coverage_error"], r["mean_set_size"], r["beta"])
        )
    if objective == "efficient_feasible":
        feasible = [r for r in z if r["coverage"] >= 1 - alpha - 1e-12]
        return (
            min(
                feasible,
                key=lambda r: (r["mean_set_size"], r["abs_coverage_error"], r["beta"]),
            )
            if feasible
            else best_alpha(rows, alpha)
        )
    raise ValueError(objective)


def best_shared(rows):
    by = {}
    for r in rows:
        by.setdefault(r["beta"], {})[r["alpha"]] = r
    candidates = []
    for beta, z in by.items():
        if set(z) != set(ALPHAS):
            continue
        es = [z[a]["abs_coverage_error"] for a in ALPHAS]
        ss = [z[a]["mean_set_size"] for a in ALPHAS]
        candidates.append((max(es), np.mean(es), np.mean(ss), beta))
    return float(min(candidates)[3])


def refined_scan(p, y, device="cpu", extra_betas=()):
    coarse = coarse_grid()
    initial = scan_betas(p, y, coarse, device)
    anchors = [
        best_alpha(initial, a, obj)["beta"]
        for a in ALPHAS
        for obj in ("coverage", "efficient_feasible")
    ]
    anchors.append(best_shared(initial))
    extra = [float(x) for x in extra_betas if np.isfinite(x) and 0 <= x <= 20]
    anchors += extra
    grid = np.unique(
        np.round(
            np.concatenate(
                [
                    coarse,
                    *[local_grid(coarse, b) for b in anchors],
                    np.asarray(extra, float),
                ]
            ),
            10,
        )
    )
    # Reuse identical coarse evaluations; retaining the full sorted grid preserves
    # the declared search and tie rules while halving redundant work.
    present = set(float(b) for b in coarse)
    added = scan_betas(p, y, [b for b in grid if float(b) not in present], device)
    final_grid = set(float(b) for b in grid)
    retained = [r for r in initial if r["beta"] in final_grid]
    return sorted(retained + added, key=lambda r: (r["beta"], r["alpha"]))


def select_targets(p, y, device="cpu", extra_betas=()):
    scan = refined_scan(p, y, device, extra_betas)
    out = {a: best_alpha(scan, a, "coverage") for a in ALPHAS}
    return out, scan

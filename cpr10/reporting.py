"""Draw-level errors, equal-weight corruption means and paired type bootstrap."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluation import ALPHAS, CORRUPTIONS, SYSTEM_CLASSES

METRICS = (
    "coverage",
    "abs_coverage_error",
    "undercoverage",
    "overcoverage",
    "mean_set_size",
)
STRATUM = ("system", "dataset", "regime", "arm", "source_seed", "alpha")
CELL = (*STRATUM, "method", "corruption", "severity")
CI_METRICS = METRICS[1:]


def csv_write(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _groups(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def _mean(rows, keys):
    output = []
    for values, group in sorted(_groups(rows, keys).items()):
        output.append(
            {
                **dict(zip(keys, values)),
                **{m: float(np.mean([row[m] for row in group])) for m in METRICS},
            }
        )
    return output


def validate_rows(rows):
    if not rows:
        raise ValueError("No evaluation rows")
    identities = set()
    for row in rows:
        identity = tuple(row[key] for key in (*CELL, "repeat"))
        if identity in identities:
            raise ValueError(f"Duplicate evaluation row: {identity}")
        identities.add(identity)
        values = [row[m] for m in METRICS]
        if (
            not np.isfinite(values).all()
            or not 0 <= row["coverage"] <= 1
            or not 0 <= row["mean_set_size"] <= SYSTEM_CLASSES[row["system"]]
        ):
            raise ValueError("Nonfinite or out-of-range evaluation metrics")
        error = row["coverage"] - (1 - row["alpha"])
        if (
            abs(row["abs_coverage_error"] - abs(error)) > 2e-15
            or abs(row["undercoverage"] - max(-error, 0)) > 2e-15
            or abs(row["overcoverage"] - max(error, 0)) > 2e-15
        ):
            raise ValueError("A/U/O must be computed before averaging draws")
    method_sets = {
        "main": {"CP", "NLL", "NLL-beta", "NLL-alpha", "QTC"},
        "clean": {"CP", "NLL", "NLL-beta", "NLL-alpha"},
        "grid": {"CP-grid", "NLL-grid", "Shared-CP"},
        "equal_coverage": {"CP", "NLL", "NLL-matched"},
        "pool": {"CP-grid"},
        "source_repeats": {"CP", "NLL"},
    }
    for identity, group in _groups(rows, STRATUM[:-1]).items():
        if {row["alpha"] for row in group} != set(ALPHAS):
            raise ValueError(f"Missing nominal coverage level: {identity}")
    for identity, group in _groups(rows, STRATUM).items():
        expected = set(method_sets[identity[2]])
        if identity[0].startswith("imagenet_") and identity[2] == "main":
            expected.add("QTC-cap")
        if {row["method"] for row in group} != expected:
            raise ValueError(f"Missing paper comparator: {identity}")
    for identity, group in _groups(rows, CELL).items():
        regime, arm = identity[2], identity[3]
        expected = (
            {-1}
            if arm == "full"
            else set(range(50 if regime in ("main", "clean") else 20))
        )
        if {row["repeat"] for row in group} != expected:
            raise ValueError(f"Missing paired repetitions: {identity}")
    for identity, group in _groups(rows, (*STRATUM, "method")).items():
        if identity[1] == "corruption":
            expected = {(c, s) for c in CORRUPTIONS for s in range(1, 6)}
            if {(row["corruption"], row["severity"]) for row in group} != expected:
                raise ValueError(f"Incomplete 15 x 5 corruption grid: {identity}")


def _bootstrap(stratum, left, right):
    """Retain the original paper analyses' deterministic cluster draws."""
    system, dataset, regime, arm, _, alpha = stratum
    imagenet = system.startswith("imagenet_")
    if not imagenet:
        if regime != "main" or (left, right) != ("CP", "NLL"):
            return None, sorted(CORRUPTIONS)
        dataset_key = "cifar10c" if system == "c10_resnet18" else "cifar100c"
        payload = json.dumps(
            [20261021, dataset_key, format(alpha, ".8f")], separators=(",", ":")
        )
        seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")
        return np.random.default_rng(seed).integers(0, 15, (2000, 15)), sorted(
            CORRUPTIONS
        )
    if dataset != "corruption" or regime not in ("main", "grid"):
        return None, sorted(CORRUPTIONS)
    if regime == "main" and right in ("NLL", "QTC"):
        return np.random.default_rng(20260914).integers(0, 15, (2000, 15)), CORRUPTIONS
    architecture = system.removeprefix("imagenet_")
    legacy_regime = "original_matched1500" if regime == "main" else "disjoint_test4"
    legacy_arm = "matched1500" if arm == "1500" else arm
    legacy = [architecture, "imagenet_c", legacy_regime, legacy_arm, alpha]
    payload = json.dumps(
        ["imagenet-controls-cluster-v1", legacy], separators=(",", ":")
    )
    seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).integers(0, 15, (2000, 15)), sorted(CORRUPTIONS)


def paired_intervals(corruptions):
    output = []
    comparisons = (
        ("CP", "NLL"),
        ("CP", "QTC"),
        ("QTC-cap", "QTC"),
        ("CP", "NLL-beta"),
        ("CP", "NLL-alpha"),
        ("CP-grid", "NLL-grid"),
        ("Shared-CP", "NLL-grid"),
        ("CP-grid", "Shared-CP"),
    )
    for stratum, rows in sorted(_groups(corruptions, STRATUM).items()):
        available = {r["method"] for r in rows}
        for left, right in comparisons:
            if not {left, right} <= available:
                continue
            a = {r["corruption"]: r for r in rows if r["method"] == left}
            b = {r["corruption"]: r for r in rows if r["method"] == right}
            if set(a) != set(b):
                raise ValueError("Unpaired corruption clusters")
            draws, order = _bootstrap(stratum, left, right)
            if set(a) != set(CORRUPTIONS):
                draws, order = None, sorted(a)
            for metric in CI_METRICS:
                scale = 1 if metric == "mean_set_size" else 100
                delta = np.asarray([a[c][metric] - b[c][metric] for c in order]) * scale
                low = high = None
                if draws is not None:
                    low, high = map(
                        float, np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
                    )
                output.append(
                    {
                        **dict(zip(STRATUM, stratum)),
                        "comparison": f"{left} - {right}",
                        "metric": metric,
                        "estimate": float(delta.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "units": "classes" if scale == 1 else "pp",
                        "bootstrap_resamples": 2000 if draws is not None else 0,
                        "interval_status": (
                            "conditional_unadjusted_type_bootstrap"
                            if draws is not None
                            else "descriptive_no_interval"
                        ),
                    }
                )
    return output


def _display(row):
    output = {k: v for k, v in row.items() if k not in METRICS}
    output.update(
        coverage_percent=100 * row["coverage"],
        A_pp=100 * row["abs_coverage_error"],
        U_pp=100 * row["undercoverage"],
        O_pp=100 * row["overcoverage"],
        S=row["mean_set_size"],
    )
    return output


def write_report(output_dir, rows):
    """CSV values name their units; no intervals over overlapping target draws."""
    validate_rows(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    conditions = _mean(rows, CELL)
    corruptions = _mean(conditions, (*STRATUM, "method", "corruption"))
    macro = _mean(corruptions, (*STRATUM, "method"))
    csv_write(output / "metrics.csv", rows)
    csv_write(output / "condition_means.csv", map(_display, conditions))
    csv_write(output / "macro.csv", map(_display, macro))
    csv_write(output / "paired_intervals.csv", paired_intervals(corruptions))
    matched = []
    for stratum, group in _groups(
        [r for r in macro if r["regime"] == "equal_coverage"], STRATUM
    ).items():
        methods = {r["method"]: r for r in group}
        cp, nll = methods["CP"], methods["NLL-matched"]
        source_rows = [
            r
            for r in rows
            if tuple(r[k] for k in STRATUM) == stratum and r["method"] == "NLL-matched"
        ]
        matched.append(
            {
                **dict(zip(STRATUM, stratum)),
                "CP_S": cp["mean_set_size"],
                "matched_NLL_S": nll["mean_set_size"],
                "CP_extra_size_percent": 100
                * (cp["mean_set_size"] / nll["mean_set_size"] - 1),
                "CP_coverage_percent": 100 * cp["coverage"],
                "matched_NLL_coverage_percent": 100 * nll["coverage"],
                "tie_overshoot_draws": sum(
                    r["coverage_overshoot_count"] > 0 for r in source_rows
                ),
                "max_tie_overshoot_count": max(
                    r["coverage_overshoot_count"] for r in source_rows
                ),
                "target_labels_used": True,
            }
        )
    if matched:
        csv_write(output / "equal_coverage.csv", matched)
    text = (
        "# Reproduction results\n\n"
        "`macro.csv` reports equal-weight corruption means after averaging each draw's "
        "metrics over draws and severities. A/U/O are percentage points; S counts classes.\n\n"
        "`paired_intervals.csv` uses 2,000 paired corruption-type bootstrap resamples "
        "where specified in the paper. These intervals condition on the fitted classifier "
        "and source model, have no multiplicity correction, and do not measure training "
        "variance. CIFAR grid and clean/natural results are descriptive.\n\n"
        "`equal_coverage.csv`, when present, is a target-label-assisted diagnostic, "
        "not a deployable method. Inclusive-score ties and any overshoot are recorded. "
        "Source-repeat rows identify each source fit separately.\n"
    )
    (output / "README.md").write_text(text)
    return macro

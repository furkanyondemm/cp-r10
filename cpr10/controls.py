"""Matched-grid targets and scalar source corrections; no target-data access."""

from __future__ import annotations
import numpy as np
from . import core as cp
from .numerics import (
    temperature_log_probs,
    plugin_log_thresholds,
    metrics_from_log_threshold,
)

ALPHAS = (0.05, 0.1)


METHODS = ("nll_grid", "cp_grid", "cp_shared_grid")


SHARED_METHODS = ("nll_grid", "cp_shared_grid")


FLOOR = 1e-12


TIE_ATOL = 1e-12


def coarse_grid():
    """Exactly the CIFAR controlled comparison's 93-point beta grid."""
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


def d_coarse():
    return np.round(np.arange(-0.5, 0.5 + 1e-9, 0.01), 10)


def d_refine(center):
    """CIFAR's local grid, including extension to .51 at a coarse boundary."""
    if not np.isfinite(center) or float(center) not in d_coarse():
        raise ValueError("Refinement center must belong to the coarse offset grid")
    return np.round(np.linspace(float(center) - 0.01, float(center) + 0.01, 41), 10)


def alpha_levels(alpha):
    if alpha not in ALPHAS:
        raise ValueError("Unexpected nominal alpha")
    return np.round(np.arange(0.1 * alpha, 1.9 * alpha + 1e-12, 0.0002), 10)


def choose(offsets, mean_err, mean_size):
    """CIFAR's exact (mean absolute error, mean size, |offset|) ordering."""
    values = np.asarray([offsets, mean_err, mean_size], dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] != 3
        or not values.shape[1]
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected aligned finite nonempty offset objectives")
    return min(
        range(len(offsets)),
        key=lambda i: (float(mean_err[i]), float(mean_size[i]), abs(float(offsets[i]))),
    )


def _features(values):
    if isinstance(values, dict):
        values = [values[name] for name in cp.FEATURES]
    values = np.asarray(values, dtype=float)
    if values.shape != (10,) or not np.isfinite(values).all():
        raise ValueError("Expected ten finite source features")
    return values


def _probabilities(values):
    # The authenticated loader already applies the original ImageNet row
    # normalization once. Repeating it here would change cached numerics.
    p = cp.validate_probs(values)
    if p.shape[1] < 2 or np.any(p < 0):
        raise ValueError("Expected nonnegative multiclass probabilities")
    return p


def _labels(values, n, k):
    y = np.asarray(values)
    if (
        y.shape != (n,)
        or not np.issubdtype(y.dtype, np.integer)
        or np.any(y < 0)
        or np.any(y >= k)
    ):
        raise ValueError("Expected one integer class per source design image")
    return y.astype(np.int64, copy=False)


def _identity(spec):
    arch = str(spec.get("architecture", spec.get("source_id", "")))
    result = {
        "architecture": arch,
        "source_id": str(spec.get("source_id", arch)),
        "condition_id": str(spec["condition_id"]),
        "family": str(spec["family"]),
    }
    if not all(result.values()) or result["source_id"] != arch:
        raise ValueError("Expected consistent nonempty source identifiers")
    return result


def _metrics(logp, y, threshold, alpha, cache=None):
    threshold = float(threshold)
    # Nearby alpha levels often select the identical threshold. Reuse the
    # original mask calculation, without introducing a new numerical engine.
    if cache is None:
        m = metrics_from_log_threshold(logp, y, threshold)
    else:
        if threshold not in cache:
            cache[threshold] = metrics_from_log_threshold(logp, y, threshold)
        m = cache[threshold]
    signed = m["coverage"] - (1 - alpha)
    return {
        **m,
        "n_images": int(len(y)),
        "covered_count": int(round(m["coverage"] * len(y))),
        "signed_coverage_error": signed,
        "abs_coverage_error": abs(signed),
        "undercoverage": max(-signed, 0.0),
        "overcoverage": max(signed, 0.0),
    }


def scan_betas(probabilities, labels, grid=None):
    """Common source trace, including both coverage risks and shared NLL."""
    p = _probabilities(probabilities)
    y = _labels(labels, len(p), p.shape[1])
    grid = coarse_grid() if grid is None else np.asarray(grid, dtype=float)
    if (
        grid.ndim != 1
        or not len(grid)
        or not np.isfinite(grid).all()
        or np.any(grid < 0)
        or np.any(grid > 20)
        or len(np.unique(grid)) != len(grid)
    ):
        raise ValueError("Expected a unique nonempty beta grid within [0,20]")
    rows = []
    for beta in np.sort(grid):
        logp = temperature_log_probs(p, float(beta), log_floor=FLOOR)
        nll = float(-np.mean(logp[np.arange(len(y)), y]))
        for alpha, threshold in zip(ALPHAS, plugin_log_thresholds(logp, ALPHAS)):
            rows.append(
                {
                    "beta": float(beta),
                    "alpha": alpha,
                    "nll": nll,
                    "log_threshold": float(threshold),
                    **_metrics(logp, y, threshold, alpha),
                }
            )
    return rows


def _minimum_ties(rows, key):
    minimum = min(float(row[key]) for row in rows)
    return [row for row in rows if abs(float(row[key]) - minimum) <= TIE_ATOL]


def _diagnostic(method, head, ties, chosen, primary_count):
    betas = sorted(float(row["beta"]) for row in ties)
    return {
        "method": method,
        "head": head,
        "selected_beta": float(chosen),
        "primary_loss_tie_count": int(primary_count),
        "tie_count": len(betas),
        "tied_betas": betas,
        "tied_beta_min": min(betas),
        "tied_beta_max": max(betas),
        "tied_beta_range": max(betas) - min(betas),
        "tied_mean_set_size_min": min(float(row["mean_set_size"]) for row in ties),
        "tied_mean_set_size_max": max(float(row["mean_set_size"]) for row in ties),
        "loss_tie_atol": TIE_ATOL,
        "tie_break": "smallest_beta_no_size_tiebreak",
    }


def select_targets(scan):
    """Four heads: NLL shared, CP per alpha, and minimax CP shared."""
    if not scan:
        raise ValueError("Cannot select targets from an empty source scan")
    by_beta = {}
    for row in scan:
        beta, alpha = float(row["beta"]), float(row["alpha"])
        required = [
            beta,
            alpha,
            row["nll"],
            row["abs_coverage_error"],
            row["mean_set_size"],
        ]
        if not np.isfinite(required).all() or not 0 <= beta <= 20:
            raise ValueError("Invalid source scan values")
        if alpha not in ALPHAS or alpha in by_beta.setdefault(beta, {}):
            raise ValueError("Duplicate or unexpected alpha in source scan")
        by_beta[beta][alpha] = row
    if any(set(rows) != set(ALPHAS) for rows in by_beta.values()):
        raise ValueError("Every source beta must have both alphas")
    joint = []
    for beta, rows in by_beta.items():
        errors = [float(rows[a]["abs_coverage_error"]) for a in ALPHAS]
        nlls = [float(rows[a]["nll"]) for a in ALPHAS]
        if max(nlls) - min(nlls) > TIE_ATOL:
            raise ValueError("NLL must be the same at both alphas")
        joint.append(
            {
                "beta": beta,
                "nll": nlls[0],
                "max_error": max(errors),
                "mean_error": float(np.mean(errors)),
                "mean_set_size": float(
                    np.mean([rows[a]["mean_set_size"] for a in ALPHAS])
                ),
            }
        )
    targets, diagnostics = {method: {} for method in METHODS}, []
    for method in SHARED_METHODS:
        ties = _minimum_ties(joint, "nll" if method == "nll_grid" else "max_error")
        primary_count = len(ties)
        if method == "cp_shared_grid":
            ties = _minimum_ties(ties, "mean_error")
        beta = min(float(row["beta"]) for row in ties)
        targets[method]["shared"] = beta
        diagnostics.append(_diagnostic(method, "shared", ties, beta, primary_count))
    for alpha in ALPHAS:
        ties = _minimum_ties(
            [rows[alpha] for rows in by_beta.values()], "abs_coverage_error"
        )
        beta = min(float(row["beta"]) for row in ties)
        targets["cp_grid"][str(alpha)] = beta
        diagnostics.append(_diagnostic("cp_grid", str(alpha), ties, beta, len(ties)))
    return targets, diagnostics


def beta_offset_scan(probabilities, labels, beta_nll, d_values):
    p = _probabilities(probabilities)
    y = _labels(labels, len(p), p.shape[1])
    ds = np.asarray(d_values, dtype=float)
    if (
        ds.ndim != 1
        or not len(ds)
        or not np.isfinite(ds).all()
        or len(np.unique(ds)) != len(ds)
        or np.any(np.abs(ds) > 0.51)
        or not np.isfinite(beta_nll)
        or not 0 <= beta_nll <= 20
    ):
        raise ValueError("Invalid beta offset grid or frozen NLL beta")
    rows, cache = [], {}
    for d in np.sort(ds):
        beta = float(np.clip(float(beta_nll) + float(d), 0.0, 20.0))
        if beta not in cache:
            logp = temperature_log_probs(p, beta, log_floor=FLOOR)
            cache[beta] = [
                (float(t), _metrics(logp, y, t, a))
                for a, t in zip(ALPHAS, plugin_log_thresholds(logp, ALPHAS))
            ]
        for alpha, (threshold, metrics) in zip(ALPHAS, cache[beta]):
            rows.append(
                {
                    "alpha": alpha,
                    "d": float(d),
                    "beta": beta,
                    "beta_clip_effect": beta - (float(beta_nll) + float(d)),
                    "log_threshold": threshold,
                    **metrics,
                }
            )
    return rows


def alpha_offset_scan(probabilities, labels, beta_nll):
    p = _probabilities(probabilities)
    y = _labels(labels, len(p), p.shape[1])
    if not np.isfinite(beta_nll) or not 0 <= beta_nll <= 20:
        raise ValueError("Invalid frozen NLL beta")
    logp = temperature_log_probs(p, float(beta_nll), log_floor=FLOOR)
    levels = np.unique(np.concatenate([alpha_levels(a) for a in ALPHAS]))
    thresholds = dict(zip(levels, plugin_log_thresholds(logp, levels)))
    rows, cache = [], {}
    for alpha in ALPHAS:
        for level in alpha_levels(alpha):
            threshold = float(thresholds[level])
            rows.append(
                {
                    "alpha": alpha,
                    "level": float(level),
                    "c": float(level - alpha),
                    "beta": float(beta_nll),
                    "log_threshold": threshold,
                    **_metrics(logp, y, threshold, alpha, cache),
                }
            )
    return rows


def _records(records):
    rows = sorted(records, key=lambda row: str(row["condition_id"]))
    ids = [str(row["condition_id"]) for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Expected nonempty unique source condition IDs")
    identities = [_identity(row) for row in rows]
    if len({row["architecture"] for row in identities}) != 1:
        raise ValueError("Source fitting cannot mix architectures")
    return rows


def _scan_map(record, key, alpha, field):
    rows = [row for row in record[key] if float(row["alpha"]) == alpha]
    table = {float(row[field]): row for row in rows}
    if len(table) != len(rows):
        raise ValueError("Duplicate offset candidates in a source scan")
    return table


def _aggregate(tables, grid, alpha, *, columnwise=False):
    # Average the absolute error per condition. The absolute macro coverage
    # error would cancel under- and overcoverage and select a different rule.
    cov = np.asarray([[table[float(v)]["coverage"] for v in grid] for table in tables])
    size = np.asarray(
        [[table[float(v)]["mean_set_size"] for v in grid] for table in tables]
    )
    if not np.isfinite(cov).all() or not np.isfinite(size).all():
        raise ValueError("Nonfinite source offset objective")
    signed = cov - (1 - alpha)

    def mean(values):
        if columnwise:
            # Original CIFAR final beta selection reduced separate contiguous
            # candidate vectors. Preserve its summation order at exact ties.
            return np.array(
                [
                    np.mean(np.ascontiguousarray(values[:, i]))
                    for i in range(values.shape[1])
                ]
            )
        return values.mean(axis=0)

    return {
        "source_mean_A": mean(np.abs(signed)),
        "source_mean_U": mean(np.maximum(-signed, 0.0)),
        "source_mean_O": mean(np.maximum(signed, 0.0)),
        "source_mean_coverage": mean(cov),
        "source_mean_S": mean(size),
    }


def refinement_plan(records):
    rows = _records(records)
    centers = {}
    for alpha in ALPHAS:
        tables = [_scan_map(row, "beta_offset_coarse", alpha, "d") for row in rows]
        if any(set(table) != set(d_coarse()) for table in tables):
            raise ValueError("Incomplete coarse offset grid")
        stats = _aggregate(tables, d_coarse(), alpha)
        centers[str(alpha)] = float(
            d_coarse()[
                choose(d_coarse(), stats["source_mean_A"], stats["source_mean_S"])
            ]
        )
    return {
        "coarse_centers": centers,
        "d_values": np.unique(
            np.concatenate([d_refine(c) for c in centers.values()])
        ).tolist(),
    }


def select_offsets(records, refine_records, *, cifar_final_reduction=False):
    rows, refined = _records(records), _records(refine_records)
    if [_identity(row) for row in rows] != [_identity(row) for row in refined]:
        raise ValueError("Source passes differ in condition or architecture identity")
    for first, second in zip(rows, refined):
        if first["original_nll_beta"] != second[
            "original_nll_beta"
        ] or not np.array_equal(
            _features(first["features"]), _features(second["features"])
        ):
            raise ValueError("Source passes differ in frozen beta or features")
    plan = refinement_plan(rows)
    offsets, curves = {}, []
    for alpha in ALPHAS:
        center = plan["coarse_centers"][str(alpha)]
        # Only this alpha's own local refinement enters its candidate set.
        grid = np.unique(np.concatenate([d_coarse(), d_refine(center)]))
        tables, level_tables = [], []
        for first, second in zip(rows, refined):
            table = _scan_map(first, "beta_offset_coarse", alpha, "d")
            refinement = _scan_map(second, "beta_offset_refine", alpha, "d")
            if set(refinement) != set(plan["d_values"]):
                raise ValueError("Incomplete or unexpected refinement candidates")
            for d, entry in refinement.items():
                if d in table and any(
                    table[d][key] != entry[key]
                    for key in (
                        "coverage",
                        "mean_set_size",
                        "empty_rate",
                        "log_threshold",
                        "beta",
                    )
                ):
                    raise ValueError("Source passes disagree at a repeated offset")
                table[d] = entry
            level_table = _scan_map(first, "alpha_offset", alpha, "level")
            if set(level_table) != set(alpha_levels(alpha)):
                raise ValueError("Incomplete alpha-offset candidates")
            for key in ("coverage", "mean_set_size", "empty_rate", "log_threshold"):
                if table[0.0][key] != level_table[alpha][key]:
                    raise ValueError("Source zero-offset controls disagree")
            tables.append(table)
            level_tables.append(level_table)
        stats = _aggregate(tables, grid, alpha, columnwise=cifar_final_reduction)
        levels = alpha_levels(alpha)
        lstats = _aggregate(level_tables, levels, alpha)
        i = choose(grid, stats["source_mean_A"], stats["source_mean_S"])
        k = choose(levels - alpha, lstats["source_mean_A"], lstats["source_mean_S"])
        i0 = int(np.flatnonzero(grid == 0.0)[0])
        offsets[str(alpha)] = {
            "d": float(grid[i]),
            "level": float(levels[k]),
            "c": float(levels[k] - alpha),
            "coarse_center": center,
            "coarse_center_boundary": bool(abs(center) == 0.5),
            "d_at_candidate_boundary": bool(i in (0, len(grid) - 1)),
            "d_outside_coarse_range": bool(abs(grid[i]) > 0.5),
            "d_search_min": float(grid[0]),
            "d_search_max": float(grid[-1]),
            "c_boundary": bool(k in (0, len(levels) - 1)),
            "source_A_at_d": float(stats["source_mean_A"][i]),
            "source_S_at_d": float(stats["source_mean_S"][i]),
            "source_A_at_level": float(lstats["source_mean_A"][k]),
            "source_S_at_level": float(lstats["source_mean_S"][k]),
            "source_A_frozen_R10": float(stats["source_mean_A"][i0]),
        }
        for kind, values, statistics in (
            ("beta", grid, stats),
            ("alpha", levels - alpha, lstats),
        ):
            for j, value in enumerate(values):
                curves.append(
                    {
                        "architecture": rows[0]["architecture"],
                        "alpha": alpha,
                        "offset": kind,
                        "value": float(value),
                        **{key: float(array[j]) for key, array in statistics.items()},
                    }
                )
    return {
        "offsets": offsets,
        "curves": curves,
        **plan,
        "architecture": rows[0]["architecture"],
        "n_conditions": len(rows),
        "training_condition_ids": [row["condition_id"] for row in rows],
        "original_frozen_nll_head": True,
        "target_labels_used_for_fitting": False,
        "source_feature_labels_read": False,
        "zero_offset_parity": "EXACT",
        "selection": "exact_mean_absolute_error_then_mean_size_then_absolute_offset",
        "error_aggregation": "mean_of_per_condition_absolute_errors",
    }

"""Paper protocols. Labels enter only scoring and the equal-coverage diagnostic."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
from time import perf_counter

import numpy as np

from . import core
from .io import (
    read_json as _read,
    sha256 as _sha,
    write_json as _write,
    verified_cache_hash,
)
from .numerics import (
    metrics_from_log_threshold,
    plugin_log_thresholds,
    temperature_log_probs,
)

ALPHAS = (0.05, 0.1)
CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)
EXPERIMENTS = ("main", "grid", "equal_coverage", "pool", "source_repeats", "clean")
EXTRA_SOURCE_SEEDS = (20260928, 20260929, 20260930)
SYSTEM_CLASSES = {
    "c10_resnet18": 10,
    "c100_vit_b_16": 100,
    "imagenet_resnet50": 1000,
    "imagenet_vit_b_16": 1000,
}


def _rows_hash(rows):
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def normalize(p):
    """Normalize raw cache rows once; floors belong to the subsequent methods."""
    p = core.validate_probs(p)
    if np.any(p < 0):
        raise ValueError("Negative cached probabilities")
    return p / p.sum(axis=1, keepdims=True)


def main_draws(n, system):
    if n < 1500:
        raise ValueError("The main protocol requires at least 1500 target images")
    if system.startswith("imagenet_"):
        return [
            np.random.default_rng(np.random.SeedSequence([20260912, r])).choice(
                n, 1500, replace=False
            )
            for r in range(50)
        ]
    seed = 20260908 if system == "c10_resnet18" else 20260911
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(n, 1500, replace=False)) for _ in range(50)]


def disjoint_pools(n, system, *, source_repeats=False, eligible=None):
    """Return the fixed evaluation pool, full threshold pool and 20 feature draws."""
    if system.startswith("imagenet_"):
        prefix = [20260928, 3 if source_repeats else 4, 0]
    else:
        prefix = [20260924, 0 if system == "c10_resnet18" else 1]
    eligible = (
        np.arange(n) if eligible is None else np.asarray(eligible, dtype=np.int64)
    )
    n_eval = 2000
    if len(eligible) < n_eval + 1500:
        raise ValueError("Insufficient disjoint adaptation/evaluation images")
    perm = np.random.default_rng(np.random.SeedSequence([*prefix, 0])).permutation(
        eligible
    )
    evaluation, full = np.sort(perm[:n_eval]), np.sort(perm[n_eval:])
    anchors = [
        np.sort(
            np.random.default_rng(np.random.SeedSequence([*prefix, 1, r])).permutation(
                full
            )[:1500]
        )
        for r in range(20)
    ]
    return evaluation, full, anchors


def _head(models, method, alpha):
    if method in ("CP", "CP-grid"):
        return models["cp" if method == "CP" else "cp_grid"][str(alpha)]
    return models[
        {
            "NLL": "nll",
            "NLL-beta": "nll",
            "NLL-alpha": "nll",
            "NLL-grid": "nll_grid",
            "Shared-CP": "shared_cp",
        }[method]
    ]


def requests(p_feature, models, methods, *, vector=False):
    features = core.r10_features(p_feature)
    x = np.asarray([features[name] for name in core.FEATURES])
    if not vector:
        x = x[None, :]
    rows = []
    for method in methods:
        for alpha in ALPHAS:
            beta = float(
                np.asarray(core.ridge_predict(_head(models, method, alpha), x)).item()
            )
            level = alpha
            if method == "NLL-beta":
                beta = float(
                    np.clip(beta + models["offsets"][str(alpha)]["beta"], 0, 20)
                )
            if method == "NLL-alpha":
                level = alpha + float(models["offsets"][str(alpha)]["alpha"])
            if not math.isfinite(beta) or not 0 <= beta <= 20 or not 0 < level < 1:
                raise ValueError("Invalid frozen prediction or source-selected offset")
            rows.append(
                dict(
                    method=method,
                    alpha=alpha,
                    beta=beta,
                    threshold_level=level,
                    n_feature=len(p_feature),
                )
            )
    return rows


def _errors(metrics, alpha):
    error = metrics["coverage"] - (1 - alpha)
    return {
        **metrics,
        "abs_coverage_error": abs(error),
        "undercoverage": max(-error, 0.0),
        "overcoverage": max(error, 0.0),
    }


def score_requests(rows, p_threshold, p_eval, y_eval):
    """Freeze all cutoffs before reading evaluation labels; reuse equal-beta sorts."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["beta"]].append(row)
    for beta, group in groups.items():
        levels = sorted({r["threshold_level"] for r in group})
        logs = temperature_log_probs(p_threshold, beta, log_floor=1e-12)
        cutoffs = dict(zip(levels, plugin_log_thresholds(logs, levels)))
        for row in group:
            row.update(
                log_threshold=float(cutoffs[row["threshold_level"]]),
                n_threshold=len(p_threshold),
            )
        del logs
    for beta, group in groups.items():
        logs = temperature_log_probs(p_eval, beta, log_floor=1e-12)
        for row in group:
            metrics = metrics_from_log_threshold(logs, y_eval, row["log_threshold"])
            row.update(_errors(metrics, row["alpha"]), n_eval=len(y_eval))
        del logs
    return rows


def empirical_threshold(logp, labels, count):
    """Label-assisted diagnostic: highest inclusive threshold attaining count."""
    if not isinstance(count, (int, np.integer)) or not 0 <= count <= len(labels):
        raise ValueError("Invalid requested coverage count")
    if count == 0:
        return math.inf
    true = logp[np.arange(len(labels)), labels]
    return float(-np.partition(-true, count - 1)[count - 1])


def equal_coverage(p, y, evaluation, anchors, models, system):
    """Retarget NLL to each paired CP coverage; explicitly retain boundary ties."""
    p_eval, labels = normalize(p[evaluation]), y[evaluation]
    batches = [normalize(p[ids]) for ids in anchors]
    betas = None
    if not system.startswith("imagenet_"):
        # The published CIFAR diagnostic predicts its 20 rows in one matrix call.
        features = [core.r10_features(batch) for batch in batches]
        x = np.asarray([[f[name] for name in core.FEATURES] for f in features])
        betas = {
            (method, a): core.ridge_predict(_head(models, method, a), x)
            for method in ("CP", "NLL")
            for a in ALPHAS
        }
    output = []
    for repeat, batch in enumerate(batches):
        rows = requests(batch, models, ("CP", "NLL"))
        if betas is not None:
            for row in rows:
                row["beta"] = float(betas[row["method"], row["alpha"]][repeat])
        score_requests(rows, batch, p_eval, labels)
        for alpha in ALPHAS:
            cp = next(r for r in rows if r["method"] == "CP" and r["alpha"] == alpha)
            nll = next(r for r in rows if r["method"] == "NLL" and r["alpha"] == alpha)
            cp_logs = temperature_log_probs(p_eval, cp["beta"], log_floor=1e-12)
            count = int(
                np.count_nonzero(
                    cp_logs[np.arange(len(labels)), labels] >= cp["log_threshold"]
                )
            )
            del cp_logs
            logs = temperature_log_probs(p_eval, nll["beta"], log_floor=1e-12)
            threshold = empirical_threshold(logs, labels, count)
            metrics = metrics_from_log_threshold(logs, labels, threshold)
            actual = int(
                np.count_nonzero(logs[np.arange(len(labels)), labels] >= threshold)
            )
            if actual < count:
                raise AssertionError(
                    "Inclusive retargeting did not attain requested coverage"
                )
            matched = {
                **nll,
                **_errors(metrics, alpha),
                "method": "NLL-matched",
                "log_threshold": threshold if math.isfinite(threshold) else None,
                "requested_count": count,
                "coverage_overshoot_count": actual - count,
                "match_status": "exact" if actual == count else "tie_overshoot",
                "target_labels_used_for_threshold": True,
            }
            output.extend({**r, "repeat": repeat} for r in (cp, nll, matched))
    return output


def _metadata(condition, system):
    if condition in ("clean", "imagenetv2"):
        return dict(
            condition=condition,
            corruption="none",
            severity=0,
            dataset="imagenetv2" if condition == "imagenetv2" else "clean",
        )
    corruption, severity = condition.rsplit("_", 1)
    if corruption not in CORRUPTIONS or int(severity) not in range(1, 6):
        raise ValueError(f"Unknown paper condition: {condition}")
    return dict(
        condition=condition,
        corruption=corruption,
        severity=int(severity),
        dataset="corruption",
    )


def _excluded(manifest, *, additional=False):
    result = {str(x) for x in manifest["excluded_target_ids"]}
    if additional:
        for seed in (manifest["main_source_seed"], *EXTRA_SOURCE_SEEDS):
            roles = manifest["sources"][str(seed)]
            for name in ("design_ids", "feature_ids", "calibration_ids"):
                result.update(str(x) for x in roles[name])
    return result


def _worker(task):
    cache, fits, output, system, condition, experiments, upstream = task
    cache, fits, output = Path(cache), Path(fits), Path(output)
    manifest = _read(cache / "manifest.json")
    seed = str(manifest["main_source_seed"])
    model_paths = {seed: fits / seed / "models.json"}
    if "source_repeats" in experiments and condition not in ("clean", "imagenetv2"):
        model_paths.update(
            {str(s): fits / str(s) / "models.json" for s in EXTRA_SOURCE_SEEDS}
        )
    cache_path = cache / "targets" / (condition + ".npz")
    calibration = cache / "source" / seed / "calibration.npz"
    files = [cache / "manifest.json", cache_path, *model_paths.values()]
    if "main" in experiments and condition not in ("clean", "imagenetv2"):
        files.append(calibration)
        from .qtc import QTCSource
    signatures = {
        str(path.resolve()): (
            verified_cache_hash(path) if path.suffix == ".npz" else _sha(path)
        )
        for path in files
    }
    signatures.update(
        {"code/" + p.name: _sha(p) for p in Path(__file__).parent.glob("*.py")}
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            [signatures, system, condition, experiments], sort_keys=True
        ).encode()
    ).hexdigest()
    destination = output / "conditions" / (condition + ".json")
    if destination.exists():
        previous = _read(destination)
        if previous["fingerprint"] != fingerprint:
            raise RuntimeError(
                f"Inputs/code changed; use a new output directory: {destination}"
            )
        if previous["rows_sha256"] != _rows_hash(previous["rows"]):
            raise RuntimeError(f"Completed condition was modified: {destination}")
        return previous["rows"]
    started = perf_counter()
    with np.load(cache_path, allow_pickle=False) as data:
        raw, labels, ids = data["p"], data["y"], data["ids"]
    if (
        raw.ndim != 2
        or labels.shape != (len(raw),)
        or ids.shape != labels.shape
        or raw.shape[1] != SYSTEM_CLASSES[system]
        or not np.issubdtype(labels.dtype, np.integer)
        or len(np.unique(ids)) != len(ids)
        or np.any(labels < 0)
        or np.any(labels >= raw.shape[1])
    ):
        raise ValueError(f"Malformed target cache: {cache_path}")
    order = np.argsort(ids)
    raw, labels, ids = raw[order], labels[order], ids[order]
    if condition != "imagenetv2":
        excluded = _excluded(manifest)
        keep = np.asarray([str(value) not in excluded for value in ids])
        raw, labels, ids = raw[keep], labels[keep], ids[keep]
    expected_n = (
        10000
        if condition == "imagenetv2"
        else (
            45000
            if system.startswith("imagenet_")
            else 9744 if system == "c10_resnet18" else 10000
        )
    )
    if len(raw) != expected_n:
        raise ValueError(
            f"Expected {expected_n} eligible images, found {len(raw)}: {condition}"
        )
    models = {key: _read(path) for key, path in model_paths.items()}
    model = models[seed]
    metadata = dict(system=system, **_metadata(condition, system))
    rows = []

    def append(batch, regime, repeat, *, source_seed=seed, arm="1500"):
        rows.extend(
            {
                **metadata,
                **r,
                "regime": regime,
                "repeat": repeat,
                "source_seed": source_seed,
                "arm": arm,
            }
            for r in batch
        )

    is_corruption = metadata["dataset"] == "corruption"
    if ("main" in experiments and is_corruption) or (
        "clean" in experiments and not is_corruption
    ):
        qtc = cap = None
        if is_corruption:
            with np.load(calibration, allow_pickle=False) as data:
                pcal, ycal = normalize(data["p"]), data["y"]
            if pcal.shape != (2500, SYSTEM_CLASSES[system]):
                raise ValueError(
                    "QTC requires the disjoint 2500-image source calibration role"
                )
            qtc = QTCSource(pcal, ycal, upstream_root=upstream, boundary="infinity")
            if system.startswith("imagenet_"):
                cap = QTCSource(pcal, ycal, upstream_root=upstream, boundary="cap")
        for repeat, selected in enumerate(main_draws(len(raw), system)):
            batch, y = normalize(raw[selected]), labels[selected]
            req = requests(batch, model, ("CP", "NLL", "NLL-beta", "NLL-alpha"))
            score_requests(req, batch, batch, y)
            for wrapper in (qtc, cap):
                if wrapper is None:
                    continue
                for result in wrapper.evaluate(batch, alphas=ALPHAS):
                    mask = result.pop("mask")
                    metrics = dict(
                        coverage=float(mask[np.arange(len(y)), y].mean()),
                        mean_set_size=float(mask.sum(axis=1).mean()),
                        empty_rate=float((mask.sum(axis=1) == 0).mean()),
                    )
                    req.append(
                        dict(
                            method="QTC-cap" if wrapper is cap else "QTC",
                            alpha=float(result["alpha"]),
                            **_errors(metrics, result["alpha"]),
                            boundary_applied=bool(result["boundary_applied"]),
                            n_feature=len(batch),
                            n_threshold=len(pcal),
                            n_eval=len(y),
                        )
                    )
            append(req, "main" if is_corruption else "clean", repeat)

    if is_corruption and any(
        e in experiments for e in ("grid", "equal_coverage", "pool")
    ):
        ev, full, anchors = disjoint_pools(len(raw), system)
        if "grid" in experiments or "pool" in experiments:
            req = []
            for repeat, selected in enumerate(anchors):
                methods = (
                    ("CP-grid", "NLL-grid", "Shared-CP")
                    if "grid" in experiments
                    else ("CP-grid",)
                )
                batch = requests(
                    normalize(raw[selected]),
                    model,
                    methods,
                    vector=not system.startswith("imagenet_"),
                )
                for row in batch:
                    row.update(repeat=repeat, arm="1500")
                req.extend(batch)
            if "pool" in experiments:
                batch = requests(normalize(raw[full]), model, ("CP-grid",))
                for row in batch:
                    row.update(repeat=-1, arm="full")
                req.extend(batch)
            score_requests(req, normalize(raw[full]), normalize(raw[ev]), labels[ev])
            for row in req:
                if row["arm"] == "1500" and "grid" in experiments:
                    append([row], "grid", row["repeat"])
                if row["method"] == "CP-grid" and "pool" in experiments:
                    append([row], "pool", row["repeat"], arm=row["arm"])
        if "equal_coverage" in experiments:
            for row in equal_coverage(raw, labels, ev, anchors, model, system):
                append([row], "equal_coverage", row["repeat"])

    if is_corruption and "source_repeats" in experiments:
        excluded = _excluded(manifest, additional=True)
        eligible = np.asarray(
            [i for i, value in enumerate(ids) if str(value) not in excluded]
        )
        ev, _, anchors = disjoint_pools(
            len(raw), system, source_repeats=True, eligible=eligible
        )
        for source_seed, fitted in models.items():
            for repeat, selected in enumerate(anchors):
                batch = normalize(raw[selected])
                req = requests(batch, fitted, ("CP", "NLL"))
                score_requests(req, batch, normalize(raw[ev]), labels[ev])
                append(req, "source_repeats", repeat, source_seed=source_seed)
    if not rows:
        raise ValueError(f"No requested experiment applies to {condition}")
    if any(_sha(path) != signatures[str(path.resolve())] for path in files):
        raise RuntimeError(
            "An input changed during evaluation; no condition result was saved"
        )
    _write(
        destination,
        dict(
            fingerprint=fingerprint,
            input_sha256=signatures,
            elapsed_seconds=perf_counter() - started,
            rows_sha256=_rows_hash(rows),
            rows=rows,
        ),
    )
    return rows


def evaluate(
    cache_dir,
    fit_dir,
    output_dir,
    system,
    experiments=None,
    workers=1,
    upstream_root=None,
):
    """Run only final-paper experiments, resuming complete unchanged conditions."""
    is_imagenet = system.startswith("imagenet_")
    if system not in SYSTEM_CLASSES:
        raise ValueError(f"Unknown paper system: {system}")
    if experiments is None:
        experiments = EXPERIMENTS if is_imagenet else ("main", "grid", "equal_coverage")
    experiments = tuple(experiments)
    if not experiments or not set(experiments) <= set(EXPERIMENTS):
        raise ValueError(f"Choose experiments from {EXPERIMENTS}")
    if not is_imagenet and set(experiments) & {"pool", "source_repeats", "clean"}:
        raise ValueError(
            "The final-paper pool/source-selection/clean checks are ImageNet experiments"
        )
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    manifest = _read(Path(cache_dir) / "manifest.json")
    if manifest["system"] != system:
        raise ValueError("Requested system differs from cache manifest")
    conditions = [
        c
        for c in manifest["conditions"]
        if (c not in ("clean", "imagenetv2") and set(experiments) - {"clean"})
        or (c in ("clean", "imagenetv2") and "clean" in experiments)
    ]
    required = {f"{c}_{s}" for c in CORRUPTIONS for s in range(1, 6)}
    if set(experiments) - {"clean"} and not required <= set(conditions):
        raise ValueError("All 75 corruption conditions are required for a paper run")
    if "clean" in experiments and not {"clean", "imagenetv2"} <= set(conditions):
        raise ValueError(
            "The clean/natural check requires clean ImageNet and ImageNetV2"
        )
    tasks = [
        (
            str(cache_dir),
            str(fit_dir),
            str(output_dir),
            system,
            c,
            experiments,
            upstream_root,
        )
        for c in conditions
    ]
    rows = []
    if workers == 1:
        results = map(_worker, tasks)
        for condition, result in zip(conditions, results):
            rows.extend(result)
            print(f"{system}: {condition} complete", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as pool:
            for condition, result in zip(conditions, pool.map(_worker, tasks)):
                rows.extend(result)
                print(f"{system}: {condition} complete", flush=True)
    from .reporting import write_report

    write_report(output_dir, rows)
    return rows

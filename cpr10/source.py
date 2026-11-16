"""Fit the reported heads using only synthetic design labels and feature pools."""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import numpy as np
from . import controls, core, oracle
from .io import (
    code_hashes,
    fingerprint,
    lock_run,
    probabilities,
    read_json,
    sha256,
    verified_cache_hash,
    write_json,
)


def _load(path, system):
    with np.load(path, allow_pickle=False) as data:
        p = probabilities(data["p_design"], normalize=system.startswith("imagenet"))
        pf = probabilities(data["p_feature"], normalize=system.startswith("imagenet"))
        y = data["y_design"]
    core.validate_probs(p, y)
    if (len(p), len(pf)) != (1000, 1500):
        raise ValueError(
            "Paper source roles must contain 1000 design and 1500 feature images"
        )
    classes = (
        10 if system == "c10_resnet18" else (100 if system == "c100_vit_b_16" else 1000)
    )
    if p.shape[1] != classes or pf.shape[1] != classes:
        raise ValueError("Source class count does not match the classifier system")
    if not np.issubdtype(y.dtype, np.integer) or y.shape != (len(p),):
        raise ValueError("Design labels must be a one-dimensional integer array")
    return p, y, pf


def _initial(job):
    path, system, destination, common, family = job
    path, destination = Path(path), Path(destination)
    signature = verified_cache_hash(path)
    if destination.exists():
        row = read_json(destination)
        if row["cache_sha256"] != signature:
            raise ValueError("Synthetic cache changed after target selection")
        return row
    p, y, pf = _load(path, system)
    beta_nll = core.original_nll_target(p, y, system)
    # The original CP search uses stable-NLL beta as a refinement anchor.
    anchor = core.fit_beta(p, y)
    targets, _ = oracle.select_targets(p, y, extra_betas=(anchor,))
    summary = core.r10_features(pf)
    features = [summary[name] for name in core.FEATURES]
    row = dict(
        condition_id=path.stem,
        family=family,
        architecture=system,
        source_id=system,
        features=features,
        nll=beta_nll,
        cp={str(a): float(targets[a]["beta"]) for a in core.ALPHAS},
        cache_sha256=signature,
    )
    if common:
        row["grid"], row["grid_ties"] = controls.select_targets(
            controls.scan_betas(p, y)
        )
    if sha256(path) != signature:
        raise ValueError("Synthetic cache changed during fitting")
    write_json(destination, row)
    return row


def _correction(job):
    path, system, destination, row, model, refinement = job
    destination = Path(destination)
    signature = fingerprint([row, model, refinement])
    if destination.exists():
        cached = read_json(destination)
        if cached["signature"] != signature:
            raise ValueError("Correction inputs changed")
        return cached["record"]
    if sha256(path) != row["cache_sha256"]:
        raise ValueError("Source cache changed between fitting and offset selection")
    p, y, _ = _load(path, system)
    x = np.asarray(row["features"])
    # Original scalar-control fits use a single-row matrix in both datasets.
    beta = float(core.ridge_predict(model, x[None, :])[0])
    record = {
        key: row[key]
        for key in ("condition_id", "family", "architecture", "source_id", "features")
    }
    record["original_nll_beta"] = beta
    if refinement is None:
        record["beta_offset_coarse"] = controls.beta_offset_scan(
            p, y, beta, controls.d_coarse()
        )
        record["alpha_offset"] = controls.alpha_offset_scan(p, y, beta)
    else:
        record["beta_offset_refine"] = controls.beta_offset_scan(p, y, beta, refinement)
    write_json(destination, dict(signature=signature, record=record))
    return record


def _map(function, jobs, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        results = []
        for i, job in enumerate(jobs, 1):
            results.append(function(job))
            if i % 10 == 0 or i == len(jobs):
                print(f"{function.__name__}: {i}/{len(jobs)}", flush=True)
        return results
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as pool:
        results = []
        for i, result in enumerate(pool.map(function, jobs), 1):
            results.append(result)
            if i % 10 == 0 or i == len(jobs):
                print(f"{function.__name__}: {i}/{len(jobs)}", flush=True)
        return results


def fit(cache_dir, fit_dir, system, *, workers=1, source_seeds=None):
    """All original-fit controls; repeated ImageNet sources use CP and NLL only."""
    cache_dir, fit_dir = Path(cache_dir), Path(fit_dir)
    manifest = read_json(cache_dir / "manifest.json")
    if manifest["system"] != system:
        raise ValueError("Cache system mismatch")
    original = str(manifest["main_source_seed"])
    seeds = (
        list(map(str, source_seeds))
        if source_seeds is not None
        else list(manifest["sources"])
    )
    for seed in seeds:
        source_dir, out = cache_dir / "source" / seed, fit_dir / seed
        paths = sorted(source_dir.glob("syn_*.npz"))
        if len(paths) != 320:
            raise ValueError(
                f"Expected 320 source conditions, found {len(paths)} in {source_dir}"
            )
        common = seed == original
        # The preparation step records the exact 320 transform specifications.
        specs = read_json(cache_dir / "specs.json")
        from .transforms import make_specs

        if specs != make_specs() or specs != manifest["source_specifications"]:
            raise ValueError(
                "Source transformations differ from the paper specification"
            )
        family = {spec["condition_id"]: spec["family"] for spec in specs}
        if set(family) != {p.stem for p in paths}:
            raise ValueError(
                "Source conditions differ from the transform specification"
            )
        lock_run(
            out / "run.json",
            dict(
                system=system,
                seed=seed,
                common=common,
                code=code_hashes(),
                cache_manifest=sha256(cache_dir / "manifest.json"),
                caches={p.name: verified_cache_hash(p) for p in paths},
            ),
        )
        records = _map(
            _initial,
            [
                (
                    str(p),
                    system,
                    str(out / "targets" / f"{p.stem}.json"),
                    common,
                    family[p.stem],
                )
                for p in paths
            ],
            workers,
        )
        X = np.asarray([row["features"] for row in records])
        models = dict(
            cp={
                str(a): core.ridge_fit(X, [row["cp"][str(a)] for row in records])
                for a in core.ALPHAS
            },
            nll=core.fit_original_nll(X, [row["nll"] for row in records], system),
        )
        if common:
            models["cp_grid"] = {
                str(a): core.ridge_fit(
                    X, [row["grid"]["cp_grid"][str(a)] for row in records]
                )
                for a in core.ALPHAS
            }
            models["nll_grid"] = core.ridge_fit(
                X, [r["grid"]["nll_grid"]["shared"] for r in records]
            )
            models["shared_cp"] = core.ridge_fit(
                X, [r["grid"]["cp_shared_grid"]["shared"] for r in records]
            )
            first = _map(
                _correction,
                [
                    (
                        str(p),
                        system,
                        str(out / "offset_coarse" / f"{p.stem}.json"),
                        row,
                        models["nll"],
                        None,
                    )
                    for p, row in zip(paths, records)
                ],
                workers,
            )
            refinement = controls.refinement_plan(first)["d_values"]
            second = _map(
                _correction,
                [
                    (
                        str(p),
                        system,
                        str(out / "offset_refine" / f"{p.stem}.json"),
                        row,
                        models["nll"],
                        refinement,
                    )
                    for p, row in zip(paths, records)
                ],
                workers,
            )
            offsets = controls.select_offsets(
                first, second, cifar_final_reduction=not system.startswith("imagenet")
            )
            write_json(out / "offset_selection.json", offsets)
            models["offsets"] = {
                a: {"beta": v["d"], "alpha": v["c"]}
                for a, v in offsets["offsets"].items()
            }
        models["metadata"] = dict(
            system=system,
            source_seed=seed,
            n_conditions=320,
            probability_floor=1e-12,
            features=core.FEATURES,
            nll_post_temperature_floor=(
                1e-300
                if system == "c10_resnet18"
                else 1e-15 if system == "c100_vit_b_16" else None
            ),
            target_labels_used_for_fitting=False,
            fit_signature=sha256(out / "run.json"),
        )
        write_json(out / "models.json", models)
        print(f'Fitted {system}, source {seed}: {out / "models.json"}', flush=True)

"""Command line entry point; no inference or training occurs on import."""

from __future__ import annotations
import os

# One BLAS thread per process; --workers controls independent numerical jobs.
for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from pathlib import Path

SYSTEMS = ("c10_resnet18", "c100_vit_b_16", "imagenet_resnet50", "imagenet_vit_b_16")


def parser():
    ap = argparse.ArgumentParser(
        description="Reproduce the final-paper CP-R10 experiments."
    )
    commands = ap.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check", help="Check dependencies, packaged splits and pinned QTC."
    )
    check.add_argument("--qtc-root", type=Path)
    run = commands.add_parser(
        "run", help="Train/prepare/fit/evaluate; each completed cache can be reused."
    )
    run.add_argument("--system", choices=SYSTEMS, required=True)
    run.add_argument(
        "--stage", choices=("all", "train", "prepare", "fit", "evaluate"), default="all"
    )
    run.add_argument(
        "--work",
        type=Path,
        required=True,
        help="Separate output directory for this system.",
    )
    run.add_argument(
        "--data",
        type=Path,
        help="CIFAR download directory or ImageNet validation root.",
    )
    run.add_argument(
        "--corruptions", type=Path, help="CIFAR-C arrays or ImageNet-C directories."
    )
    run.add_argument(
        "--v2", type=Path, help="ImageNetV2 matched-frequency numeric class folders."
    )
    run.add_argument(
        "--ground-truth",
        type=Path,
        help="Required for flat ImageNet validation images.",
    )
    run.add_argument(
        "--meta-mat",
        type=Path,
        help="Devkit meta.mat for flat ImageNet validation images.",
    )
    run.add_argument(
        "--checkpoint", type=Path, help="Existing CIFAR checkpoint with model_state."
    )
    run.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    run.add_argument(
        "--workers", type=int, default=1, help="Parallel CPU fitting/evaluation jobs."
    )
    run.add_argument(
        "--loader-workers",
        type=int,
        default=None,
        help="Override dataset loader workers.",
    )
    run.add_argument(
        "--download", action="store_true", help="Allow torchvision to download CIFAR."
    )
    run.add_argument(
        "--qtc-root",
        type=Path,
        help="Alternative pinned QTC checkout (file hashes checked).",
    )
    run.add_argument(
        "--experiments",
        nargs="+",
        choices=("main", "grid", "equal_coverage", "pool", "source_repeats", "clean"),
        help="Default: every experiment reported for this system.",
    )
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    if args.command == "check":
        import numpy, scipy, sklearn, PIL, torch, torchvision
        from .datasets import cifar_splits
        from .qtc import verify_upstream

        for system in SYSTEMS[:2]:
            splits = cifar_splits(system)
            roles = [
                set(splits[f"{r}_positions"].tolist())
                for r in ("design", "feature", "calibration")
            ]
            if [len(r) for r in roles] != [1000, 1500, 2500] or len(
                set.union(*roles)
            ) != 5000:
                raise ValueError("Invalid packaged source split")
        result = verify_upstream(args.qtc_root)
        print(
            f"Python dependencies OK; torch={torch.__version__}, torchvision={torchvision.__version__}"
        )
        print(
            f'Fixed source splits OK; pinned QTC={result["commit"]}; CUDA={torch.cuda.is_available()}'
        )
        return
    if args.work.exists() and not args.work.is_dir():
        ap.error("--work must name a directory")
    if args.work.resolve() == Path(__file__).resolve().parents[1]:
        ap.error("Use a separate work directory, not the repository root")
    if args.workers < 1 or (
        args.loader_workers is not None and args.loader_workers < 0
    ):
        ap.error("workers must be positive; loader-workers must be nonnegative")
    imagenet = args.system.startswith("imagenet_")
    experiments = args.experiments or (
        ["main", "grid", "equal_coverage", "pool", "source_repeats", "clean"]
        if imagenet
        else ["main", "grid", "equal_coverage"]
    )
    if not imagenet and set(experiments) & {"pool", "source_repeats", "clean"}:
        ap.error("pool, source_repeats and clean are ImageNet-only in the final paper")
    if args.stage in ("all", "prepare", "train") and args.data is None:
        ap.error("--data is required for training/preparation")
    if args.stage in ("all", "prepare"):
        if args.corruptions is None or not args.corruptions.is_dir():
            ap.error("--corruptions must point to the extracted corruption dataset")
        if (
            imagenet
            and "clean" in experiments
            and (args.v2 is None or not args.v2.is_dir())
        ):
            ap.error(
                "The full paper run requires --v2; otherwise select experiments without clean"
            )
    if imagenet and args.checkpoint is not None:
        ap.error("ImageNet uses frozen torchvision V1 weights, not --checkpoint")
    if imagenet and args.stage == "train":
        ap.error("No classifier training is required for ImageNet")
    if args.stage in ("all", "evaluate") and "main" in experiments:
        from .qtc import verify_upstream

        verify_upstream(args.qtc_root)
    cache, fits, results = (args.work / name for name in ("cache", "fits", "results"))
    checkpoint = args.checkpoint or args.work / "classifier" / "best.pt"
    if not imagenet and args.stage in ("all", "train"):
        if (
            args.checkpoint is None
            and not (checkpoint.parent / "complete.json").exists()
        ):
            if checkpoint.exists():
                ap.error(
                    "Training is incomplete. Use a fresh --work to retrain, or explicitly select a checkpoint."
                )
            from .train import train_classifier

            checkpoint = train_classifier(
                args.system,
                args.data,
                checkpoint.parent,
                device=args.device,
                workers=args.loader_workers,
                download=args.download,
            )
        elif not checkpoint.is_file():
            ap.error(f"Checkpoint is missing: {checkpoint}")
    if args.stage in ("all", "prepare"):
        from .datasets import prepare_cifar, prepare_imagenet

        if imagenet:
            seeds = (
                (20260911, 20260928, 20260929, 20260930)
                if "source_repeats" in experiments
                else (20260911,)
            )
            prepare_imagenet(
                args.system,
                args.data,
                args.corruptions,
                cache,
                device=args.device,
                ground_truth=args.ground_truth,
                meta_mat=args.meta_mat,
                v2_root=args.v2,
                source_seeds=seeds,
                workers=4 if args.loader_workers is None else args.loader_workers,
            )
        else:
            if not checkpoint.is_file():
                ap.error(
                    f"CIFAR checkpoint missing: {checkpoint}. Run --stage train first."
                )
            prepare_cifar(
                args.system,
                args.data,
                args.corruptions,
                checkpoint,
                cache,
                device=args.device,
                download=args.download,
            )
    if args.stage in ("all", "fit"):
        from .source import fit

        fit(cache, fits, args.system, workers=args.workers)
    if args.stage in ("all", "evaluate"):
        from .evaluation import evaluate

        evaluate(
            cache,
            fits,
            results,
            args.system,
            experiments=experiments,
            workers=args.workers,
            upstream_root=args.qtc_root,
        )
    print(f"Completed {args.stage}: {args.work}", flush=True)


if __name__ == "__main__":
    main()

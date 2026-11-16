"""Dataset indexing, paper splits, and probability-cache generation."""

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Iterable, Sequence
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode, functional as TF
from .transforms import apply_synthetic, apply_spec, make_specs

IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}
VAL_NAME = re.compile(r"^(ILSVRC2012_val_\d{8})\.(?:jpe?g|png)$", re.IGNORECASE)
WNID_NAME = re.compile(r"^n\d{8}$")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CORRUPTIONS = [
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
]


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    image_id: str
    wnid: str | None = None


def canonical_val_id(path: str | Path) -> str:
    match = VAL_NAME.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"Not an official ImageNet validation filename: {path}")
    return "ILSVRC2012_val_" + match.group(1)[-8:]


def _images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _unique(records: Sequence[ImageRecord]) -> list[ImageRecord]:
    ids = [r.image_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate canonical image IDs in dataset")
    if not records:
        raise ValueError("No images found in dataset")
    return sorted(records, key=lambda record: record.image_id)


def devkit_class_mapping(
    meta_mat: str | Path, expected_classes: int = 1000
) -> dict[int, tuple[str, int]]:
    """Map ILSVRC integer IDs to (WNID, TorchVision index), using leaf synsets."""
    from scipy.io import loadmat

    synsets = np.atleast_1d(loadmat(meta_mat, squeeze_me=True)["synsets"])
    leaves = [(int(row[0]), str(row[1])) for row in synsets if int(row[4]) == 0]
    if (
        len(leaves) != expected_classes
        or len({wnid for _, wnid in leaves}) != expected_classes
    ):
        raise ValueError(
            f"Devkit must contain exactly {expected_classes} distinct leaf classes"
        )
    if len({ilsvrc_id for ilsvrc_id, _ in leaves}) != expected_classes:
        raise ValueError("Duplicate ILSVRC class IDs in devkit")
    canonical = {wnid: i for i, wnid in enumerate(sorted(wnid for _, wnid in leaves))}
    return {ilsvrc_id: (wnid, canonical[wnid]) for ilsvrc_id, wnid in leaves}


def index_imagenet_validation(
    root: str | Path,
    ground_truth: str | Path | None = None,
    meta_mat: str | Path | None = None,
    expected_classes: int = 1000,
) -> list[ImageRecord]:
    """Index WNID folders, or flat validation JPEGs plus both devkit files.

    For flat archives the ground-truth integers are *not* TorchVision indices;
    the official meta.mat conversion is mandatory. Filenames, not directory
    enumeration order, select the appropriate ground-truth row.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"ImageNet validation directory does not exist: {root}")
    class_dirs = sorted(
        p for p in root.iterdir() if p.is_dir() and WNID_NAME.fullmatch(p.name)
    )
    records = []
    if class_dirs:
        if len(class_dirs) != expected_classes:
            raise ValueError(
                f"Expected {expected_classes} WNID directories; found {len(class_dirs)}"
            )
        if any(
            p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in root.iterdir()
        ):
            raise ValueError("Mixed flat and WNID-organized validation images")
        if meta_mat is not None:
            official = {
                wnid
                for wnid, _ in devkit_class_mapping(meta_mat, expected_classes).values()
            }
            if {p.name for p in class_dirs} != official:
                raise ValueError(
                    "Validation class folders do not match the supplied official devkit"
                )
        for label, directory in enumerate(class_dirs):
            paths = _images(directory)
            if not paths:
                raise ValueError(f"Class directory contains no images: {directory}")
            records.extend(
                ImageRecord(p, label, canonical_val_id(p), directory.name)
                for p in paths
            )
    else:
        if ground_truth is None or meta_mat is None:
            raise ValueError(
                "Flat ImageNet validation requires ground_truth TXT and devkit meta.mat"
            )
        mapping = devkit_class_mapping(meta_mat, expected_classes)
        truth = [
            int(line.strip())
            for line in Path(ground_truth).read_text().splitlines()
            if line.strip()
        ]
        for path in _images(root):
            image_id = canonical_val_id(path)
            position = int(image_id[-8:]) - 1
            if not 0 <= position < len(truth):
                raise ValueError(f"Validation ID outside ground-truth file: {image_id}")
            if truth[position] not in mapping:
                raise ValueError(
                    f"Unknown ILSVRC class ID {truth[position]} for {image_id}"
                )
            wnid, label = mapping[truth[position]]
            records.append(ImageRecord(path, label, image_id, wnid))
    return _unique(records)


def index_imagenet_v2(
    root: str | Path, expected_classes: int = 1000
) -> list[ImageRecord]:
    """Index official numeric class folders without ImageFolder's lexical bug."""
    root = Path(root)
    dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    labels = [int(directory.name) for directory in dirs]
    if len(labels) != expected_classes or set(labels) != set(range(expected_classes)):
        raise ValueError(
            f"ImageNetV2 requires numeric class directories 0 through {expected_classes - 1}"
        )
    records = []
    for directory in sorted(dirs, key=lambda p: int(p.name)):
        label = int(directory.name)
        paths = _images(directory)
        if not paths:
            raise ValueError(f"Empty ImageNetV2 class directory: {directory}")
        for path in paths:
            relative = path.relative_to(directory).as_posix()
            records.append(ImageRecord(path, label, f"imagenetv2/{label}/{relative}"))
    return _unique(records)


def exclude_source_ids(
    records: Sequence[ImageRecord], exclude_ids: Iterable[str]
) -> list[ImageRecord]:
    """Remove every target record whose base image was used by a source role."""
    excluded = set(exclude_ids)
    return [record for record in records if record.image_id not in excluded]


def index_imagenet_c(
    root: str | Path,
    corruption: str,
    severity: int,
    reference_records: Sequence[ImageRecord],
    exclude_ids: Iterable[str] = (),
    expected_classes: int = 1000,
) -> list[ImageRecord]:
    """Validate a corruption against clean base identities, then exclude source.

    ``reference_records`` must describe the complete clean validation index,
    including source and held-out images. Unknown base IDs and mismatched class
    folders are errors even when the corresponding image would be excluded.
    """
    if not isinstance(severity, int) or not 1 <= severity <= 5:
        raise ValueError("ImageNet-C severity must be an integer in [1, 5]")
    if Path(corruption).name != corruption or corruption in {"", ".", ".."}:
        raise ValueError("Corruption must be a directory name")
    root = Path(root) / corruption / str(severity)
    reference = {record.image_id: record for record in reference_records}
    if len(reference) != len(reference_records):
        raise ValueError("Duplicate IDs in clean reference")
    excluded = set(exclude_ids)
    if not excluded.issubset(reference):
        raise ValueError("Excluded source IDs are absent from the clean reference")
    records = index_imagenet_validation(root, expected_classes=expected_classes)
    for record in records:
        clean = reference.get(record.image_id)
        if clean is None:
            raise ValueError(f"Unknown clean base ID in ImageNet-C: {record.image_id}")
        if clean.label != record.label or (
            clean.wnid is not None and clean.wnid != record.wnid
        ):
            raise ValueError(
                f"Corruption label disagrees with clean base: {record.image_id}"
            )
    return _unique(exclude_source_ids(records, excluded))


class ImageDataset(Dataset):
    """Return (normalized RGB tensor, class index, canonical image ID).

    The default resize/crop/interpolation/normalization matches the explicit
    TorchVision ResNet50 IMAGENET1K_V1 transform. Synthetic perturbations operate
    on the cropped [0, 1] tensor before normalization.
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        spec: dict | None = None,
        pool_salt: int = 0,
        image_size: int = 224,
        resize_size: int | None = 256,
        normalize: bool = True,
    ):
        if image_size < 16 or (resize_size is not None and resize_size < image_size):
            raise ValueError("Require resize_size >= image_size >= 16")
        self.records = list(records)
        self.spec = spec
        self.pool_salt = int(pool_salt)
        self.image_size, self.resize_size, self.normalize = (
            image_size,
            resize_size,
            normalize,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, position: int):
        record = self.records[position]
        with Image.open(record.path) as source:
            image = source.convert("RGB")
            # The official precomputed ImageNet-C protocol only center-crops;
            # pass resize_size=None for those already processed JPEGs.
            if self.resize_size is not None:
                image = TF.resize(
                    image,
                    self.resize_size,
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
            image = TF.center_crop(image, self.image_size)
            tensor = TF.to_tensor(image)
        if self.spec is not None:
            tensor = apply_synthetic(tensor, self.spec, record.image_id, self.pool_salt)
        if self.normalize:
            tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, int(record.label), record.image_id


def split_source(records, seed=20260911, *, classes=1000, per_class=50):
    """Disjoint IDs, exactly five source and 45 test images in every class."""
    if classes % 2 or per_class < 6:
        raise ValueError(
            "Expected an even class count and at least six images per class"
        )
    if len({r.image_id for r in records}) != len(records):
        raise ValueError("Duplicate validation ID")
    rng = np.random.default_rng(seed)
    extra_feature = set(rng.choice(classes, classes // 2, replace=False).tolist())
    roles = {key: [] for key in ("design", "calibration", "feature", "test")}
    labels = np.array([r.label for r in records])
    if set(labels.tolist()) != set(range(classes)):
        raise ValueError(
            "Validation class mapping must be the full contiguous label set"
        )
    for label in range(classes):
        positions = np.flatnonzero(labels == label)
        if len(positions) != per_class:
            raise ValueError(
                f"Class {label}: expected {per_class} validation images, got {len(positions)}"
            )
        rng.shuffle(positions)
        nf = 2 if label in extra_feature else 1
        roles["design"].extend(positions[:1])
        roles["feature"].extend(positions[1 : 1 + nf])
        roles["calibration"].extend(positions[1 + nf : 5])
        roles["test"].extend(positions[5:])
    return {
        role: np.sort(positions).astype(np.int64) for role, positions in roles.items()
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; use --device cpu explicitly."
        )
    return device


def cifar_splits(system):
    if system not in ("c10_resnet18", "c100_vit_b_16"):
        raise ValueError(f"Not a CIFAR system: {system}")
    with np.load(Path(__file__).parent / "assets" / f"{system}_splits.npz") as saved:
        return {key: saved[key] for key in saved.files}


def classifier(system, *, pretrained=False):
    from torchvision import models

    if system == "c10_resnet18":
        model = models.resnet18(weights=None, num_classes=10)
        model.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
    elif system == "c100_vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, 100)
    elif system == "imagenet_resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif system == "imagenet_vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown system: {system}")
    return model


def _write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    path.with_suffix(".sha256").write_text(sha256_file(path) + "\n")


def _begin_cache(output, manifest):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous != manifest:
            raise ValueError(
                "Cache provenance changed. Choose a new output directory; do not mix inference runs."
            )
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    specs = output / "specs.json"
    specs.write_text(json.dumps(manifest["source_specifications"], indent=2) + "\n")
    return output


def _cached(path, keys, *, expected_y=None, expected_ids=None):
    if not path.exists():
        return False
    checksum = path.with_suffix(".sha256")
    if not checksum.exists() or checksum.read_text().strip() != sha256_file(path):
        raise ValueError(f"Cache checksum missing or changed: {path}")
    with np.load(path, allow_pickle=False) as saved:
        if not set(keys).issubset(saved.files):
            raise ValueError(f"Incomplete cache: {path}")
        for key in keys:
            array = saved[key]
            if key.startswith("p") and (
                array.ndim != 2
                or not np.isfinite(array).all()
                or np.any(array < 0)
                or np.any(array > 1)
                or not np.allclose(array.sum(1), 1, atol=1e-5)
            ):
                raise ValueError(f"Invalid probabilities in {path}:{key}")
        label_key = "y_design" if "y_design" in keys else "y"
        probability_key = "p_design" if "p_design" in keys else "p"
        y, p = saved[label_key], saved[probability_key]
        if (
            y.ndim != 1
            or not np.issubdtype(y.dtype, np.integer)
            or len(y) != len(p)
            or np.any(y < 0)
            or np.any(y >= p.shape[1])
        ):
            raise ValueError(f"Invalid labels in {path}")
        if expected_y is not None and not np.array_equal(y, expected_y):
            raise ValueError(f"Cached labels differ from dataset: {path}")
        if "ids" in saved.files:
            ids = saved["ids"]
            if ids.ndim != 1 or len(ids) != len(y) or len(np.unique(ids)) != len(ids):
                raise ValueError(f"Invalid identities in {path}")
            if expected_ids is not None and not np.array_equal(ids, expected_ids):
                raise ValueError(f"Cached identities differ from dataset: {path}")
    return True


def _input_inventory(roots):
    inventory = []
    for value in roots:
        if value is None:
            continue
        root = Path(value).resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        digest = hashlib.sha256()

        def paths():
            if root.is_file():
                yield root
            else:
                for directory, folders, files in os.walk(root):
                    folders.sort()
                    for name in sorted(files):
                        yield Path(directory) / name

        count = 0
        for path in paths():
            count += 1
            stat = path.stat()
            relative = (
                path.name if root.is_file() else path.relative_to(root).as_posix()
            )
            digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        inventory.append(
            {"root": str(root), "files": count, "metadata_sha256": digest.hexdigest()}
        )
    return inventory


def _runtime(device):
    import torchvision

    return {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "code_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("datasets.py", "transforms.py")
        },
    }


def _raw(images):
    return (
        torch.from_numpy(np.array(images, copy=True)).permute(0, 3, 1, 2).float()
        / 255.0
    )


@torch.inference_mode()
def _predict_cifar(model, images, system, device, *, spec=None, salt=0, baseline=False):
    batch_size = (128 if baseline else 512) if system == "c10_resnet18" else 128
    results = []
    for start in range(0, len(images), batch_size):
        x = _raw(images[start : start + batch_size])
        if system == "c10_resnet18" and spec is None:
            x = TF.normalize(x.contiguous(), CIFAR10_MEAN, CIFAR10_STD).to(device)
        else:
            x = x.to(device)
            if spec is not None:
                x = apply_spec(x, spec, salt + start)
            if system == "c100_vit_b_16":
                x = torch.nn.functional.interpolate(
                    x, (224, 224), mode="bicubic", align_corners=False, antialias=True
                )
                mean, std = IMAGENET_MEAN, IMAGENET_STD
            else:
                mean, std = CIFAR10_MEAN, CIFAR10_STD
            mu = torch.tensor(mean, device=device).view(1, 3, 1, 1)
            sigma = torch.tensor(std, device=device).view(1, 3, 1, 1)
            x = (x - mu) / sigma
        logits = model(x)
        results.append(logits.cpu() if baseline else logits.softmax(1).cpu())
    result = torch.cat(results)
    if baseline:
        result = result.float().softmax(1)
    return result.numpy().astype(np.float32)


def prepare_cifar(
    system,
    data_root,
    corruption_root,
    checkpoint,
    output,
    *,
    device="cuda",
    download=False,
):
    """Generate all source/target caches using the fixed paper splits and batches."""
    from torchvision.datasets import CIFAR10, CIFAR100

    device = get_device(device)
    dataset = CIFAR10 if system == "c10_resnet18" else CIFAR100
    splits = cifar_splits(system)
    train = dataset(data_root, train=True, download=download)
    test = dataset(data_root, train=False, download=download)
    if len(train) != 50000 or len(test) != 10000:
        raise ValueError("Expected the complete official CIFAR train/test dataset.")
    y = np.asarray(train.targets, dtype=np.int64)
    ytest = np.asarray(test.targets, dtype=np.int64)
    role_ids = {
        role: splits["dev"][splits[f"{role}_positions"]]
        for role in ("design", "calibration", "feature")
    }
    source_seed = 20260908
    specs = make_specs()
    conditions = [
        f"{name}_{severity}" for name in CORRUPTIONS for severity in range(1, 6)
    ]
    manifest = {
        "system": system,
        "main_source_seed": source_seed,
        "sources": {
            str(source_seed): {
                f"{role}_ids": ids.tolist() for role, ids in role_ids.items()
            }
        },
        "excluded_target_ids": (
            np.linspace(0, 9999, 256, dtype=int).tolist()
            if system == "c10_resnet18"
            else []
        ),
        "conditions": conditions,
        "checkpoint_sha256": sha256_file(checkpoint),
        "split_sha256": sha256_file(
            Path(__file__).parent / "assets" / f"{system}_splits.npz"
        ),
        "runtime": _runtime(device),
        "synthetic_batch_size": 512 if system == "c10_resnet18" else 128,
        "source_specifications": specs,
        "input_inventory": _input_inventory(
            [Path(data_root) / dataset.base_folder, corruption_root]
        ),
    }
    output = _begin_cache(output, manifest)
    model = classifier(system)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint_data["model_state"])
    model.eval().requires_grad_(False).to(device)
    source = output / "source" / str(source_seed)
    cal_path = source / "calibration.npz"
    if not _cached(
        cal_path,
        ("p", "y"),
        expected_y=y[role_ids["calibration"]],
        expected_ids=role_ids["calibration"],
    ):
        ids = role_ids["calibration"]
        if system == "c10_resnet18":
            pdev = _predict_cifar(
                model, train.data[splits["dev"]], system, device, baseline=True
            )
            p = pdev[splits["calibration_positions"]]
        else:
            p = _predict_cifar(model, train.data[ids], system, device)
        _write_npz(cal_path, p=p, y=y[ids], ids=ids)
    for index, spec in enumerate(specs):
        path = source / f"{spec['condition_id']}.npz"
        if _cached(
            path,
            ("p_design", "y_design", "p_feature"),
            expected_y=y[role_ids["design"]],
        ):
            continue
        arrays = {}
        for role, salt in (("design", 1000000), ("feature", 2000000)):
            ids = role_ids[role]
            arrays[f"p_{role}"] = _predict_cifar(
                model,
                train.data[ids],
                system,
                device,
                spec=spec,
                salt=salt + index * 10000,
            )
        _write_npz(path, **arrays, y_design=y[role_ids["design"]])
        print(f"source {system}: {index + 1}/320", flush=True)
    corruption_root = Path(corruption_root)
    labels = np.load(corruption_root / "labels.npy", allow_pickle=False)
    if len(labels) not in (10000, 50000):
        raise ValueError("CIFAR-C labels must contain 10,000 or 50,000 entries.")
    for name in CORRUPTIONS:
        images = np.load(
            corruption_root / f"{name}.npy", mmap_mode="r", allow_pickle=False
        )
        if images.shape != (50000, 32, 32, 3) or images.dtype != np.uint8:
            raise ValueError(f"Invalid CIFAR-C image array: {name}")
        for severity in range(1, 6):
            sl = slice((severity - 1) * 10000, severity * 10000)
            ycondition = np.asarray(
                labels[sl] if len(labels) == 50000 else labels, dtype=np.int64
            )
            if not np.array_equal(ycondition, ytest):
                raise ValueError(f"Corruption label/order mismatch: {name}_{severity}")
            path = output / "targets" / f"{name}_{severity}.npz"
            if not _cached(
                path,
                ("p", "y", "ids"),
                expected_y=ycondition,
                expected_ids=np.arange(10000),
            ):
                p = _predict_cifar(model, images[sl], system, device)
                _write_npz(path, p=p, y=ycondition, ids=np.arange(10000))
            print(f"target {system}: {name}_{severity}", flush=True)
    (output / "complete.json").write_text(
        json.dumps({"conditions": len(conditions), "sources": 320}) + "\n"
    )
    return output


@torch.inference_mode()
def _predict_imagenet(
    model, records, device, batch_size, workers, *, spec=None, salt=0, corruption=False
):
    dataset = ImageDataset(
        records, spec=spec, pool_salt=salt, resize_size=None if corruption else 256
    )
    options = {"multiprocessing_context": "spawn"} if workers else {}
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        **options,
    )
    p = np.empty((len(records), 1000), dtype=np.float32)
    y = np.empty(len(records), dtype=np.int64)
    offset = 0
    for x, labels, ids in loader:
        count = len(x)
        if list(ids) != [
            record.image_id for record in records[offset : offset + count]
        ]:
            raise RuntimeError("Inference changed image order.")
        p[offset : offset + count] = (
            model(x.to(device, non_blocking=True)).softmax(1).cpu().numpy()
        )
        y[offset : offset + count] = labels.numpy()
        offset += count
    if offset != len(records) or not np.isfinite(p).all():
        raise RuntimeError("Incomplete or nonfinite inference.")
    return p, y


def prepare_imagenet(
    system,
    val_root,
    corruption_root,
    output,
    *,
    device="cuda",
    ground_truth=None,
    meta_mat=None,
    v2_root=None,
    source_seeds=(20260911, 20260928, 20260929, 20260930),
    workers=4,
):
    """Cache frozen V1 classifier predictions; validation images also supply sources."""
    if system not in ("imagenet_resnet50", "imagenet_vit_b_16"):
        raise ValueError(f"Not an ImageNet system: {system}")
    if 20260911 not in source_seeds or len(set(source_seeds)) != len(source_seeds):
        raise ValueError(
            "Include the main seed 20260911 once; source seeds must be distinct."
        )
    device = get_device(device)
    batch_size = 64 if system == "imagenet_resnet50" else 32
    records = index_imagenet_validation(val_root, ground_truth, meta_mat)
    splits = {str(seed): split_source(records, seed) for seed in source_seeds}
    sources = {
        seed: {
            f"{role}_ids": [records[i].image_id for i in roles[role]]
            for role in ("design", "calibration", "feature")
        }
        for seed, roles in splits.items()
    }
    specs = make_specs()
    conditions = ["clean"] + [
        f"{name}_{severity}" for name in CORRUPTIONS for severity in range(1, 6)
    ]
    if v2_root is not None:
        conditions.append("imagenetv2")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    model = classifier(system).eval().requires_grad_(False).to(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    model_digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        model_digest.update(name.encode())
        model_digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    manifest = {
        "system": system,
        "main_source_seed": 20260911,
        "sources": sources,
        "excluded_target_ids": sorted(sum(sources["20260911"].values(), [])),
        "conditions": conditions,
        "weights": "IMAGENET1K_V1",
        "model_state_sha256": model_digest.hexdigest(),
        "runtime": _runtime(device),
        "batch_size": batch_size,
        "source_specifications": specs,
        "input_inventory": _input_inventory(
            [val_root, corruption_root, v2_root, ground_truth, meta_mat]
        ),
    }
    output = _begin_cache(output, manifest)
    for seed, roles in splits.items():
        source = output / "source" / seed
        path = source / "calibration.npz"
        if not _cached(path, ("p", "y")):
            selected = [records[i] for i in roles["calibration"]]
            p, y = _predict_imagenet(model, selected, device, batch_size, workers)
            _write_npz(
                path, p=p, y=y, ids=np.asarray([record.image_id for record in selected])
            )
        for index, spec in enumerate(specs):
            path = source / f"{spec['condition_id']}.npz"
            if _cached(path, ("p_design", "y_design", "p_feature")):
                continue
            arrays = {}
            for role, salt in (("design", 1000000), ("feature", 2000000)):
                selected = [records[i] for i in roles[role]]
                p, y = _predict_imagenet(
                    model, selected, device, batch_size, workers, spec=spec, salt=salt
                )
                arrays[f"p_{role}"] = p
                if role == "design":
                    arrays["y_design"] = y
            _write_npz(path, **arrays)
            print(f"source {system}/{seed}: {index + 1}/320", flush=True)
    for condition in conditions:
        path = output / "targets" / f"{condition}.npz"
        corruption = condition not in ("clean", "imagenetv2")
        if condition == "clean":
            selected = exclude_source_ids(records, manifest["excluded_target_ids"])
        elif condition == "imagenetv2":
            selected = index_imagenet_v2(v2_root)
        else:
            name, severity = condition.rsplit("_", 1)
            selected = index_imagenet_c(
                corruption_root,
                name,
                int(severity),
                records,
                manifest["excluded_target_ids"],
            )
            expected = {record.image_id for record in records} - set(
                manifest["excluded_target_ids"]
            )
            if {record.image_id for record in selected} != expected:
                raise ValueError(f"Incomplete ImageNet-C condition: {condition}")
        if _cached(
            path,
            ("p", "y", "ids"),
            expected_y=np.array([record.label for record in selected]),
            expected_ids=np.array([record.image_id for record in selected]),
        ):
            continue
        p, y = _predict_imagenet(
            model, selected, device, batch_size, workers, corruption=corruption
        )
        _write_npz(
            path, p=p, y=y, ids=np.asarray([record.image_id for record in selected])
        )
        print(f"target {system}: {condition}", flush=True)
    (output / "complete.json").write_text(
        json.dumps({"conditions": len(conditions), "sources": len(splits) * 320}) + "\n"
    )
    return output

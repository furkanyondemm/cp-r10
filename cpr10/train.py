"""Paper classifier training; ImageNet uses frozen published weights instead."""

from __future__ import annotations
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from .datasets import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    cifar_splits,
    classifier,
    get_device,
    sha256_file,
)


def train_classifier(
    system, data_root, output, *, device="cuda", workers=None, download=False
):
    """Train with the fixed paper schedule and select by development loss."""
    if system not in ("c10_resnet18", "c100_vit_b_16"):
        raise ValueError(
            "Train only CIFAR classifiers. ImageNet classifiers remain pretrained and frozen."
        )
    is_c10 = system == "c10_resnet18"
    seed, epochs, batch_size = (42, 100, 128) if is_c10 else (20260908, 20, 64)
    workers = (2 if is_c10 else 4) if workers is None else int(workers)
    if workers < 0:
        raise ValueError("Worker count cannot be negative.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists() or (output / "config.json").exists():
        raise FileExistsError(
            "Training output already exists. Supply its best.pt to prepare, or use a new directory."
        )
    device = get_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if is_c10:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    splits = cifar_splits(system)
    augmentation = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    evaluation = [transforms.ToTensor()]
    if not is_c10:
        evaluation.append(
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            )
        )
    evaluation.append(
        transforms.Normalize(
            CIFAR10_MEAN if is_c10 else IMAGENET_MEAN,
            CIFAR10_STD if is_c10 else IMAGENET_STD,
        )
    )
    dataset = datasets.CIFAR10 if is_c10 else datasets.CIFAR100
    training = dataset(
        data_root,
        train=True,
        download=download,
        transform=transforms.Compose(augmentation + evaluation),
    )
    development = dataset(
        data_root, train=True, download=False, transform=transforms.Compose(evaluation)
    )
    if len(training) != 50000:
        raise ValueError("Expected the official 50,000-image CIFAR training set.")
    loader_options = dict(
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available() if is_c10 else True,
    )
    train_options = {}
    if not is_c10:
        generator = torch.Generator().manual_seed(seed)
        train_options["generator"] = generator
        loader_options["persistent_workers"] = workers > 0
    train_loader = DataLoader(
        Subset(training, splits["train"]),
        shuffle=True,
        **loader_options,
        **train_options,
    )
    dev_loader = DataLoader(
        Subset(development, splits["dev"]), shuffle=False, **loader_options
    )
    model = classifier(system, pretrained=not is_c10).to(device)
    if is_c10:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        head = list(model.heads.parameters())
        head_ids = {id(parameter) for parameter in head}
        backbone = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in head_ids
        ]
        optimizer = torch.optim.AdamW(
            [{"params": backbone, "lr": 3e-5}, {"params": head, "lr": 3e-4}],
            weight_decay=0.05,
        )
        total_steps, warmup = epochs * len(train_loader), 2 * len(train_loader)

        def multiplier(step):
            if step < warmup:
                return max(1e-3, (step + 1) / max(1, warmup))
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    mixed_precision = not is_c10 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0 if is_c10 else 0.1)
    config = {
        "system": system,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "workers": workers,
        "device": str(device),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "mixed_precision": mixed_precision,
        "split_sha256": sha256_file(
            Path(__file__).parent / "assets" / f"{system}_splits.npz"
        ),
        "code_sha256": sha256_file(__file__),
        "selection": "minimum development cross-entropy"
        + ("" if is_c10 else ", then maximum accuracy"),
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    history, best = [], None
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_correct = count = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=not is_c10), y.to(
                device, non_blocking=not is_c10
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=mixed_precision
            ):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if not is_c10:
                scheduler.step()
            train_loss += float(loss.detach()) * len(y)
            train_correct += int((logits.argmax(1) == y).sum())
            count += len(y)
        model.eval()
        dev_loss = dev_correct = dev_count = 0
        all_logits, all_labels = [], []
        with torch.inference_mode():
            for x, y in dev_loader:
                logits = model(x.to(device, non_blocking=not is_c10))
                if is_c10:
                    all_logits.append(logits.cpu())
                    all_labels.append(y)
                else:
                    y = y.to(device, non_blocking=True)
                    dev_loss += float(
                        torch.nn.functional.cross_entropy(logits, y)
                    ) * len(y)
                    dev_correct += int((logits.argmax(1) == y).sum())
                    dev_count += len(y)
        if is_c10:
            logits, y = torch.cat(all_logits).float(), torch.cat(all_labels).long()
            dev_loss = float(torch.nn.functional.cross_entropy(logits, y))
            dev_accuracy = float(
                (logits.softmax(1).numpy().argmax(1) == y.numpy()).mean()
            )
            scheduler.step()
        else:
            dev_loss, dev_accuracy = dev_loss / dev_count, dev_correct / dev_count
        row = {
            "epoch": epoch,
            "train_loss": train_loss / count,
            "train_accuracy": train_correct / count,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_accuracy,
        }
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        key = (dev_loss,) if is_c10 else (dev_loss, -dev_accuracy)
        if best is None or key < best:
            best = key
            temporary = output / "best.pt.tmp"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "metrics": row,
                    "system": system,
                    "seed": seed,
                },
                temporary,
            )
            temporary.replace(output / "best.pt")
        print(json.dumps(row), flush=True)
    (output / "complete.json").write_text(
        json.dumps(
            {
                "epochs": epochs,
                "wall_seconds": time.monotonic() - started,
                "checkpoint_sha256": sha256_file(output / "best.pt"),
            },
            indent=2,
        )
        + "\n"
    )
    return output / "best.pt"

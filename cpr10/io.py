"""Small atomic files and explicit cache validation."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def probabilities(p, normalize=False):
    from .core import validate_probs

    p = validate_probs(p)
    if np.any(p < 0):
        raise ValueError("Negative probability")
    return p / p.sum(axis=1, keepdims=True) if normalize else p


def verified_cache_hash(path):
    path = Path(path)
    digest = sha256(path)
    checksum = path.with_suffix(".sha256")
    if not checksum.is_file() or checksum.read_text().strip() != digest:
        raise ValueError(f"Prediction cache checksum missing or changed: {path}")
    return digest


def code_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def lock_run(path, settings):
    """A changed protocol/input must use a new output directory."""
    path = Path(path)
    if path.exists():
        if read_json(path) != settings:
            raise ValueError(
                f"Input, code or settings changed: use a new output directory ({path})"
            )
    else:
        write_json(path, settings)

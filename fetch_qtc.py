#!/usr/bin/env python3
"""Download the three pinned QTC author files used by the wrapper."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import urllib.request


def constants():
    # Installation should not require importing torch before dependencies exist.
    path = Path(__file__).resolve().parent / "cpr10/qtc.py"
    tree = ast.parse(path.read_text())
    wanted = {"UPSTREAM_REPOSITORY", "UPSTREAM_COMMIT", "SOURCE_SHA256"}
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in wanted
    }


def fetch(destination):
    config = constants()
    repository = config["UPSTREAM_REPOSITORY"].removeprefix("https://github.com/")
    for relative, expected in config["SOURCE_SHA256"].items():
        path = destination / relative
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            print(f"Verified {relative}")
            continue
        url = f"https://raw.githubusercontent.com/{repository}/{config['UPSTREAM_COMMIT']}/{relative}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(f"Downloaded QTC source hash mismatch: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        temporary.write_bytes(data)
        temporary.replace(path)
        print(f"Downloaded and verified {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parent / "third_party/qtc",
    )
    args = parser.parse_args()
    fetch(args.destination.resolve())


if __name__ == "__main__":
    main()

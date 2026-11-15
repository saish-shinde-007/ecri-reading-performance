#!/usr/bin/env python3
"""Download the ECRI CSVs from a pinned commit and verify every byte.

    python tools/fetch_data.py           fetch what is missing, verify all
    python tools/fetch_data.py --check   verify only; exit 1 if anything is off

The data is CC BY-NC-SA 4.0 (docs/data.md). It lands in data/raw, which git
ignores; it is never committed here.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

from ecri import paths, sources


def fetch(name: str, target: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        partial = Path(tmp.name)
    try:
        with urllib.request.urlopen(sources.url(name), timeout=60) as response:
            partial.write_bytes(response.read())
        got = sources.sha256(partial)
        if got != sources.SHA256[name]:
            raise SystemExit(
                f"{name}: downloaded bytes do not match the pinned checksum "
                f"(got {got[:12]}…). Refusing to use them."
            )
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify only, never download")
    args = parser.parse_args()

    paths.RAW.mkdir(parents=True, exist_ok=True)
    before = sources.status(paths.RAW)
    if args.check:
        for name, state in before.items():
            print(f"  {state:8s} {name}")
        return 0 if all(s == "ok" for s in before.values()) else 1

    for name, state in before.items():
        if state == "ok":
            print(f"  ok       {name} (already present)")
            continue
        if state == "changed":
            print(f"  changed  {name}: local copy differs from the pinned bytes; re-fetching")
        fetch(name, paths.RAW / name)
        print(f"  fetched  {name} @ {sources.COMMIT[:7]}, sha256 verified")
    print("ECRI data is CC BY-NC-SA 4.0 — cite Shangguan et al. (2024). See docs/data.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

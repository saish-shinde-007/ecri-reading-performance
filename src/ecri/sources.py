"""Where the data comes from, pinned to the byte.

Fetching from the dataset repo's `main` would let an upstream edit change every
result here without a trace. These URLs name a commit, and every file is
checked against its sha256 before anything uses it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO = "ecri-data/ECRI-Dataset"
# The repo's last commit (2024-08-19). Every published number used these bytes.
COMMIT = "7168d48d463ae92e58132d4c000611569cb59bba"

SHA256 = {
    "ECRIDataSet.csv": "1883c5796132cfb524cf691d9dc0461ad535239c0cfb720efa0da4c1254e58fc",
    "ECRIRawData.csv": "ca214ae9397b313b35824ae003ecd671618e3249fb9ed3d09c3f9f2b46245c5c",
    "MetaData.csv": "aae2184080c30b7b9ddad31e8f9853aaa63d021a3c2365818ce65e433399e722",
}


def url(name: str) -> str:
    return f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/ECRI/{name}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def status(directory: Path) -> dict[str, str]:
    """Per file: "ok", "missing", or "changed"."""
    result = {}
    for name, expected in SHA256.items():
        path = Path(directory) / name
        if not path.is_file():
            result[name] = "missing"
        else:
            result[name] = "ok" if sha256(path) == expected else "changed"
    return result

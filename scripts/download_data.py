from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

DATASETS = {
    "lotr.csv": "https://raw.githubusercontent.com/Raphtory/Data/main/lotr.csv",
    "lotr_properties.csv": "https://raw.githubusercontent.com/Raphtory/Data/main/lotr_properties.csv",
    "LOTRORDFXML.owl": "https://raw.githubusercontent.com/Lotro/lotro.github.io/master/LOTRORDFXML.owl",
}

MIN_BYTES = {
    "lotr.csv": 40_000,
    "lotr_properties.csv": 400,
    "LOTRORDFXML.owl": 60_000,
}


def download(url: str, target: Path, force: bool) -> None:
    if target.exists() and target.stat().st_size >= MIN_BYTES.get(target.name, 1) and not force:
        print(f"ok: {target.relative_to(ROOT)} ja existe")
        return

    print(f"baixando: {url}")
    try:
        with urllib.request.urlopen(url, timeout=45) as response:
            content = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"falha ao baixar {url}: {exc}") from exc

    min_size = MIN_BYTES.get(target.name, 1)
    if len(content) < min_size:
        raise RuntimeError(
            f"arquivo {target.name} parece incompleto: {len(content)} bytes < {min_size}"
        )

    target.write_bytes(content)
    print(f"salvo: {target.relative_to(ROOT)} ({len(content)} bytes)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa datasets LOTR para a demo.")
    parser.add_argument("--force", action="store_true", help="baixa novamente mesmo se ja existir")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        for filename, url in DATASETS.items():
            download(url, RAW_DIR / filename, args.force)
    except RuntimeError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1

    print("datasets prontos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


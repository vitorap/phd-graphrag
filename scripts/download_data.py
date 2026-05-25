from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

SNA_LOTR_BASE = "https://raw.githubusercontent.com/fonsofhervella/SNA_LOTR/main/dataset"

DATASETS = {
    "lotr.csv": "https://raw.githubusercontent.com/Raphtory/Data/main/lotr.csv",
    "lotr_properties.csv": "https://raw.githubusercontent.com/Raphtory/Data/main/lotr_properties.csv",
    "LOTRORDFXML.owl": "https://raw.githubusercontent.com/Lotro/lotro.github.io/master/LOTRORDFXML.owl",
    "sna_lotr/LOTR1_book_CLEAN.txt": f"{SNA_LOTR_BASE}/LOTR1_book_CLEAN.txt",
    "sna_lotr/LOTR2_book_CLEAN.txt": f"{SNA_LOTR_BASE}/LOTR2_book_CLEAN.txt",
    "sna_lotr/LOTR3_book_CLEAN.txt": f"{SNA_LOTR_BASE}/LOTR3_book_CLEAN.txt",
    "sna_lotr/cleaned_name.csv": f"{SNA_LOTR_BASE}/cleaned_name.csv",
    "sna_lotr/edges.csv": f"{SNA_LOTR_BASE}/edges.csv",
    "sna_lotr/edges_chapters.csv": f"{SNA_LOTR_BASE}/edges_chapters.csv",
    "sna_lotr/labMT1.txt": f"{SNA_LOTR_BASE}/labMT1.txt",
    "sna_lotr/lotr_scripts.csv": f"{SNA_LOTR_BASE}/lotr_scripts.csv",
    "sna_lotr/nodes.csv": f"{SNA_LOTR_BASE}/nodes.csv",
    "sna_lotr/nodes_chapter_sent.csv": f"{SNA_LOTR_BASE}/nodes_chapter_sent.csv",
    "sna_lotr/nodes_with_sentiment.csv": f"{SNA_LOTR_BASE}/nodes_with_sentiment.csv",
    "sna_lotr/prediction_1.csv": f"{SNA_LOTR_BASE}/prediction_1.csv",
    "sna_lotr/prediction_2.csv": f"{SNA_LOTR_BASE}/prediction_2.csv",
    "sna_lotr/prediction_3.csv": f"{SNA_LOTR_BASE}/prediction_3.csv",
    "sna_lotr/sentiment_per_character.csv": f"{SNA_LOTR_BASE}/sentiment_per_character.csv",
    "sna_lotr/weightededges.csv": f"{SNA_LOTR_BASE}/weightededges.csv",
}

MIN_BYTES = {
    "lotr.csv": 40_000,
    "lotr_properties.csv": 400,
    "LOTRORDFXML.owl": 60_000,
    "LOTR1_book_CLEAN.txt": 900_000,
    "LOTR2_book_CLEAN.txt": 750_000,
    "LOTR3_book_CLEAN.txt": 650_000,
    "cleaned_name.csv": 1_000,
    "edges.csv": 10_000,
    "edges_chapters.csv": 40_000,
    "labMT1.txt": 350_000,
    "lotr_scripts.csv": 200_000,
    "nodes.csv": 2_500,
    "nodes_chapter_sent.csv": 1_800,
    "nodes_with_sentiment.csv": 4_000,
    "prediction_1.csv": 35_000,
    "prediction_2.csv": 350_000,
    "prediction_3.csv": 350_000,
    "sentiment_per_character.csv": 2_000,
    "weightededges.csv": 18_000,
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

    target.parent.mkdir(parents=True, exist_ok=True)
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

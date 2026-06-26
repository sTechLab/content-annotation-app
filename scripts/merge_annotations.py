"""Merge per-coder annotation CSVs into one combined annotation file."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"

ANNOTATION_COLUMNS = [
    "annotation_id",
    "annotation_datetime",
    "coder_name",
    "clean_coding_mode",
    "item_id",
    "post_url",
    "skipped",
    "skip_datetime",
    "is_relevant",
    "tags_selected",
    "tags_added",
    "tags_final",
    "labels_selected",
    "labels_added",
    "labels_final",
    "coder_notes",
]


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        return {}

    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.where(pd.notna(df), "").astype(str)
    for column in ANNOTATION_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[ANNOTATION_COLUMNS]


def merge_annotations(raw_dir: Path) -> pd.DataFrame:
    annotation_files = sorted(raw_dir.glob("*_annotations.csv"))
    if not annotation_files:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    frames = [ensure_columns(pd.read_csv(path, dtype=str)) for path in annotation_files]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.where(pd.notna(merged), "").astype(str)
    merged = merged.sort_values("annotation_datetime")
    merged = merged.drop_duplicates(["coder_name", "item_id"], keep="last")

    return merged[ANNOTATION_COLUMNS]


def parse_args() -> argparse.Namespace:
    config = load_config()
    annotations_dir = config.get("annotations_dir", "annotations")

    parser = argparse.ArgumentParser(description="Merge raw annotation CSV files.")
    parser.add_argument(
        "--annotations-dir",
        default=annotations_dir,
        help="Annotation root directory. Defaults to config/config.json.",
    )
    parser.add_argument(
        "--output",
        help="Optional combined CSV path. Defaults to <annotations-dir>/combined/combined_annotations.csv.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations_dir = project_path(args.annotations_dir)
    raw_dir = annotations_dir / "raw"
    output_path = (
        project_path(args.output)
        if args.output
        else annotations_dir / "combined" / "combined_annotations.csv"
    )

    merged = merge_annotations(raw_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"Merged rows: {len(merged)}")
    print(f"Saved combined annotations to: {output_path}")


if __name__ == "__main__":
    main()

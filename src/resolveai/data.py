from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATA_PATH, PROJECT_ROOT, write_json


CLINC150_URL = (
    "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
)
EXPECTED_COUNTS = {
    ("train", False): 15_000,
    ("validation", False): 3_000,
    ("test", False): 4_500,
    ("train", True): 100,
    ("validation", True): 100,
    ("test", True): 1_000,
}
SOURCE_KEYS = {
    "train": ("train", False),
    "val": ("validation", False),
    "test": ("test", False),
    "oos_train": ("train", True),
    "oos_val": ("validation", True),
    "oos_test": ("test", True),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_clinc150(
    destination: str | Path = DEFAULT_DATA_PATH,
    force: bool = False,
) -> Path:
    destination = Path(destination)
    if destination.exists() and not force:
        load_clinc150(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        CLINC150_URL,
        headers={"User-Agent": "ResolveAI/0.1 dataset downloader"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response:
        temporary.write_bytes(response.read())
    temporary.replace(destination)
    load_clinc150(destination)
    return destination


def load_clinc150(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found at {path}; run `resolveai download-data`"
        )
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    missing = SOURCE_KEYS.keys() - payload.keys()
    if missing:
        raise ValueError(f"CLINC150 payload missing keys: {sorted(missing)}")

    rows: list[dict] = []
    for source_key, (split, is_oos) in SOURCE_KEYS.items():
        values = payload[source_key]
        expected = EXPECTED_COUNTS[(split, is_oos)]
        if len(values) != expected:
            raise ValueError(
                f"{source_key} expected {expected} rows, found {len(values)}"
            )
        for index, sample in enumerate(values):
            if not isinstance(sample, list) or len(sample) != 2:
                raise ValueError(f"invalid sample at {source_key}[{index}]")
            text, label = sample
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"empty or non-string text at {source_key}[{index}]")
            expected_label = "oos" if is_oos else label
            if not isinstance(label, str) or label != expected_label:
                raise ValueError(f"unexpected label at {source_key}[{index}]")
            rows.append(
                {
                    "sample_id": f"{source_key}:{index}",
                    "text": text.strip(),
                    "label": label,
                    "split": split,
                    "is_oos": is_oos,
                }
            )

    frame = pd.DataFrame.from_records(rows)
    _validate_frame(frame)
    return frame


def _validate_frame(frame: pd.DataFrame) -> None:
    required = {"sample_id", "text", "label", "split", "is_oos"}
    if set(frame.columns) != required:
        raise ValueError(f"unexpected dataframe columns: {list(frame.columns)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("sample IDs must be unique")
    if frame[["text", "label", "split"]].isna().any().any():
        raise ValueError("dataset contains null values")

    in_scope = frame.loc[~frame["is_oos"]]
    labels_by_split = {
        split: set(group["label"].unique())
        for split, group in in_scope.groupby("split", observed=True)
    }
    if any(len(labels) != 150 for labels in labels_by_split.values()):
        raise ValueError("each in-scope split must contain exactly 150 labels")
    if len({frozenset(labels) for labels in labels_by_split.values()}) != 1:
        raise ValueError("in-scope label sets differ across splits")

    per_label = in_scope.groupby(["split", "label"], observed=True).size()
    expected_per_class = {"train": 100, "validation": 20, "test": 30}
    for split, expected in expected_per_class.items():
        counts = per_label.loc[split]
        if not (counts == expected).all():
            raise ValueError(f"{split} does not have {expected} rows per intent")


def dataset_report(
    frame: pd.DataFrame,
    source_path: str | Path = DEFAULT_DATA_PATH,
) -> dict:
    source_path = Path(source_path)
    counts = (
        frame.groupby(["split", "is_oos"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    duplicated_text = int(frame.duplicated(subset=["text"], keep=False).sum())
    cross_split_duplicates = int(
        frame.groupby("text", observed=True)["split"].nunique().gt(1).sum()
    )
    return {
        "dataset": "CLINC150 / oos-eval data_full",
        "source_url": CLINC150_URL,
        "source_sha256": sha256_file(source_path),
        "license": "CC BY 3.0",
        "rows": int(len(frame)),
        "in_scope_labels": int(frame.loc[~frame["is_oos"], "label"].nunique()),
        "counts": [
            {
                "split": str(row.split),
                "is_oos": bool(row.is_oos),
                "count": int(row.count),
            }
            for row in counts.itertuples(index=False)
        ],
        "duplicate_rows_by_text": duplicated_text,
        "texts_present_in_multiple_splits": cross_split_duplicates,
    }


def write_dataset_report(
    frame: pd.DataFrame,
    source_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path = PROJECT_ROOT / "data" / "processed" / "dataset_report.json",
) -> dict:
    report = dataset_report(frame, source_path)
    write_json(output_path, report)
    return report


def select_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown split: {split}")
    return frame.loc[frame["split"].eq(split)].reset_index(drop=True).copy()

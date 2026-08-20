from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.json"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "clinc150.json"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "runtime"


@dataclass(frozen=True)
class Paths:
    project_root: Path = PROJECT_ROOT
    data_path: Path = DEFAULT_DATA_PATH
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR
    runtime_dir: Path = DEFAULT_RUNTIME_DIR


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        values = json.load(handle)
    required = {"seed", "vocabulary", "baseline", "neural", "retrieval"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"configuration missing sections: {sorted(missing)}")
    return values


def write_json(path: str | Path, values: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")

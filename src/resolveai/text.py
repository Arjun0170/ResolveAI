from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]
    max_length: int = 32

    def __post_init__(self) -> None:
        if self.token_to_id.get(PAD_TOKEN) != 0:
            raise ValueError("padding token must have ID 0")
        if self.token_to_id.get(UNK_TOKEN) != 1:
            raise ValueError("unknown token must have ID 1")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    @property
    def size(self) -> int:
        return len(self.token_to_id)

    @property
    def id_to_token(self) -> list[str]:
        values = [""] * self.size
        for token, index in self.token_to_id.items():
            values[index] = token
        return values

    @classmethod
    def build(
        cls,
        texts: Iterable[str],
        max_size: int = 12_000,
        min_frequency: int = 2,
        max_length: int = 32,
    ) -> "Vocabulary":
        if max_size < 2 or min_frequency <= 0:
            raise ValueError("invalid vocabulary settings")
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(tokenize(text))
        candidates = [
            (token, count)
            for token, count in counts.items()
            if count >= min_frequency
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for token, _ in candidates[: max_size - 2]:
            token_to_id[token] = len(token_to_id)
        return cls(token_to_id=token_to_id, max_length=max_length)

    def encode_tokens(self, tokens: Sequence[str], truncate: bool = True) -> list[int]:
        token_ids = [self.token_to_id.get(token, 1) for token in tokens]
        if truncate:
            token_ids = token_ids[: self.max_length]
        return token_ids or [1]

    def encode(self, text: str, truncate: bool = True) -> list[int]:
        return self.encode_tokens(tokenize(text), truncate=truncate)

    def encode_padded(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.zeros((len(texts), self.max_length), dtype=np.int64)
        lengths = np.ones(len(texts), dtype=np.int64)
        for row, text in enumerate(texts):
            token_ids = self.encode(text)
            length = min(len(token_ids), self.max_length)
            matrix[row, :length] = token_ids[:length]
            lengths[row] = length
        return matrix, lengths

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "max_length": self.max_length,
            "tokens": self.id_to_token,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported vocabulary schema")
        tokens = payload["tokens"]
        if len(tokens) != len(set(tokens)):
            raise ValueError("vocabulary contains duplicate tokens")
        return cls(
            token_to_id={token: index for index, token in enumerate(tokens)},
            max_length=int(payload["max_length"]),
        )


def build_label_mapping(labels: Iterable[str]) -> tuple[dict[str, int], list[str]]:
    unique = sorted(set(labels) - {"oos"}) + ["oos"]
    if len(unique) < 2:
        raise ValueError("at least one in-scope label and oos are required")
    return {label: index for index, label in enumerate(unique)}, unique

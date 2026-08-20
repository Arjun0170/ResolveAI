from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .artifacts import artifact_record, atomic_write_json, environment_manifest
from .config import DEFAULT_ARTIFACT_DIR, load_config
from .evaluation import latency_report, routing_report
from .metrics import fit_temperature, select_abstention_threshold, softmax
from .text import Vocabulary, build_label_mapping


class MultinomialNaiveBayes:
    """A compact NumPy implementation for a dependency-light benchmark."""

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.class_log_prior: np.ndarray | None = None
        self.feature_log_probability: np.ndarray | None = None

    def fit(
        self,
        token_rows: Sequence[Sequence[int]],
        labels: np.ndarray,
        num_classes: int,
        vocabulary_size: int,
    ) -> "MultinomialNaiveBayes":
        labels = np.asarray(labels, dtype=np.int64)
        if len(token_rows) != labels.size:
            raise ValueError("token rows and labels must have equal length")
        class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
        if np.any(class_counts == 0):
            raise ValueError("every configured class needs a training sample")
        token_counts = np.full(
            (num_classes, vocabulary_size),
            self.alpha,
            dtype=np.float64,
        )
        for token_ids, label in zip(token_rows, labels, strict=True):
            np.add.at(token_counts[int(label)], np.asarray(token_ids), 1.0)
        self.class_log_prior = np.log(class_counts / class_counts.sum())
        self.feature_log_probability = np.log(
            token_counts / token_counts.sum(axis=1, keepdims=True)
        )
        return self

    def predict_logits(self, token_rows: Sequence[Sequence[int]]) -> np.ndarray:
        if self.class_log_prior is None or self.feature_log_probability is None:
            raise RuntimeError("model must be fitted before prediction")
        output = np.repeat(
            self.class_log_prior[None, :],
            len(token_rows),
            axis=0,
        )
        for row, token_ids in enumerate(token_rows):
            output[row] += self.feature_log_probability[:, token_ids].sum(axis=1)
        return output

    def save(self, path: str | Path) -> None:
        if self.class_log_prior is None or self.feature_log_probability is None:
            raise RuntimeError("cannot save an unfitted model")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray([1]),
            alpha=np.asarray([self.alpha]),
            class_log_prior=self.class_log_prior,
            feature_log_probability=self.feature_log_probability,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MultinomialNaiveBayes":
        with np.load(path) as values:
            if int(values["schema_version"][0]) != 1:
                raise ValueError("unsupported baseline model schema")
            model = cls(alpha=float(values["alpha"][0]))
            model.class_log_prior = values["class_log_prior"].copy()
            model.feature_log_probability = values[
                "feature_log_probability"
            ].copy()
        return model


class BaselineRouter:
    def __init__(
        self,
        model: MultinomialNaiveBayes,
        vocabulary: Vocabulary,
        labels: list[str],
        temperature: float,
        threshold: float,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.labels = labels
        self.temperature = temperature
        self.threshold = threshold

    def predict_logits(self, texts: Sequence[str]) -> np.ndarray:
        rows = [self.vocabulary.encode(text, truncate=False) for text in texts]
        return self.model.predict_logits(rows)

    def predict(self, texts: Sequence[str]) -> list[dict]:
        probabilities = softmax(self.predict_logits(texts), self.temperature)
        results = []
        for row in probabilities:
            raw_id = int(row.argmax())
            confidence = float(row[raw_id])
            abstained = confidence < self.threshold
            label = "oos" if abstained else self.labels[raw_id]
            results.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "abstained": abstained or label == "oos",
                    "raw_label": self.labels[raw_id],
                }
            )
        return results

    @classmethod
    def load(cls, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> "BaselineRouter":
        artifact_dir = Path(artifact_dir)
        metadata = _load_json(artifact_dir / "baseline" / "metadata.json")
        vocabulary = Vocabulary.load(artifact_dir / "common" / "vocabulary.json")
        labels = _load_json(artifact_dir / "common" / "labels.json")["labels"]
        return cls(
            model=MultinomialNaiveBayes.load(
                artifact_dir / "baseline" / "model.npz"
            ),
            vocabulary=vocabulary,
            labels=labels,
            temperature=float(metadata["temperature"]),
            threshold=float(metadata["threshold"]),
        )


def train_baseline(
    frame: pd.DataFrame,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    config: dict | None = None,
) -> dict:
    config = config or load_config()
    artifact_dir = Path(artifact_dir)
    train = frame.loc[frame["split"].eq("train")].reset_index(drop=True)
    validation = frame.loc[frame["split"].eq("validation")].reset_index(drop=True)
    test = frame.loc[frame["split"].eq("test")].reset_index(drop=True)

    vocabulary = Vocabulary.build(train["text"], **config["vocabulary"])
    label_to_id, labels = build_label_mapping(train["label"])
    common_dir = artifact_dir / "common"
    common_dir.mkdir(parents=True, exist_ok=True)
    vocabulary.save(common_dir / "vocabulary.json")
    atomic_write_json(
        common_dir / "labels.json",
        {"schema_version": 1, "labels": labels},
    )

    def encode_texts(values: pd.Series) -> list[list[int]]:
        return [vocabulary.encode(text, truncate=False) for text in values]

    def encode_labels(values: pd.Series) -> np.ndarray:
        return values.map(label_to_id).to_numpy(dtype=np.int64)

    model = MultinomialNaiveBayes(config["baseline"]["alpha"])
    model.fit(
        encode_texts(train["text"]),
        encode_labels(train["label"]),
        len(labels),
        vocabulary.size,
    )
    validation_logits = model.predict_logits(encode_texts(validation["text"]))
    y_validation = encode_labels(validation["label"])
    temperature, validation_nll = fit_temperature(validation_logits, y_validation)
    validation_probabilities = softmax(validation_logits, temperature)
    threshold_result = select_abstention_threshold(
        y_validation,
        validation_probabilities.argmax(axis=1),
        validation_probabilities.max(axis=1),
        len(labels),
        label_to_id["oos"],
    )

    model_dir = artifact_dir / "baseline"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.npz"
    model.save(model_path)
    metadata = {
        "schema_version": 1,
        "model_type": "multinomial_naive_bayes",
        "alpha": model.alpha,
        "temperature": temperature,
        "threshold": threshold_result.threshold,
        "validation_nll": validation_nll,
        "threshold_selection": asdict(threshold_result),
        "training_samples": int(len(train)),
        "vocabulary_size": vocabulary.size,
        "classes": len(labels),
    }
    atomic_write_json(model_dir / "metadata.json", metadata)

    router = BaselineRouter(
        model,
        vocabulary,
        labels,
        temperature,
        threshold_result.threshold,
    )
    y_test = encode_labels(test["label"])
    report = routing_report(
        router.predict_logits(test["text"].tolist()),
        y_test,
        labels,
        temperature,
        threshold_result.threshold,
    )
    report["latency"] = latency_report(
        router.predict_logits,
        test["text"].tolist(),
    )
    report["model"] = metadata
    report["environment"] = environment_manifest()
    report["artifacts"] = [artifact_record(model_path, "numpy_model")]
    atomic_write_json(model_dir / "metrics.json", report)
    return report


def _load_json(path: str | Path) -> dict:
    import json

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)

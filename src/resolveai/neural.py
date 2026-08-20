from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import artifact_record, atomic_write_json, environment_manifest
from .config import DEFAULT_ARTIFACT_DIR, load_config
from .evaluation import latency_report, routing_report
from .metrics import classification_report, fit_temperature, select_abstention_threshold, softmax
from .text import Vocabulary


class NeuralTextClassifier(nn.Module):
    """Embedding-bag style encoder with mean/max pooling and a small MLP."""

    def __init__(
        self,
        vocabulary_size: int,
        num_classes: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dim,
            padding_idx=0,
        )
        self.encoder = nn.Sequential(
            nn.LayerNorm(embedding_dim * 2),
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.architecture = {
            "vocabulary_size": vocabulary_size,
            "num_classes": num_classes,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = token_ids.ne(0)
        embedded = self.embedding(token_ids)
        float_mask = mask.unsqueeze(-1).to(embedded.dtype)
        mean_pool = (embedded * float_mask).sum(dim=1) / float_mask.sum(
            dim=1
        ).clamp_min(1.0)
        masked = embedded.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        max_pool = masked.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return self.encoder(torch.cat([mean_pool, max_pool], dim=1))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(token_ids))


class NeuralRouter:
    def __init__(
        self,
        model: NeuralTextClassifier,
        vocabulary: Vocabulary,
        labels: list[str],
        temperature: float,
        threshold: float,
        batch_size: int = 512,
    ) -> None:
        self.model = model.eval()
        self.vocabulary = vocabulary
        self.labels = labels
        self.temperature = temperature
        self.threshold = threshold
        self.batch_size = batch_size

    def _token_matrix(self, texts: Sequence[str]) -> torch.Tensor:
        values, _ = self.vocabulary.encode_padded(list(texts))
        return torch.from_numpy(values)

    @torch.inference_mode()
    def predict_logits(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, len(self.labels)), dtype=np.float32)
        tokens = self._token_matrix(texts)
        chunks = []
        for start in range(0, len(tokens), self.batch_size):
            chunks.append(self.model(tokens[start : start + self.batch_size]).numpy())
        return np.concatenate(chunks, axis=0)

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        if not texts:
            width = int(self.model.architecture["hidden_dim"])
            return np.empty((0, width), dtype=np.float32)
        tokens = self._token_matrix(texts)
        chunks = []
        for start in range(0, len(tokens), self.batch_size):
            chunks.append(self.model.encode(tokens[start : start + self.batch_size]).numpy())
        embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings /= np.clip(norms, 1e-12, None)
        return embeddings

    def predict(self, texts: Sequence[str], candidates: int = 3) -> list[dict]:
        probabilities = softmax(self.predict_logits(texts), self.temperature)
        results = []
        for row in probabilities:
            ranking = np.argsort(row)[::-1][:candidates]
            raw_id = int(ranking[0])
            confidence = float(row[raw_id])
            abstained = confidence < self.threshold
            label = "oos" if abstained else self.labels[raw_id]
            results.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "abstained": abstained or label == "oos",
                    "raw_label": self.labels[raw_id],
                    "candidates": [
                        {"label": self.labels[int(index)], "score": float(row[index])}
                        for index in ranking
                    ],
                }
            )
        return results

    @classmethod
    def load(cls, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> "NeuralRouter":
        artifact_dir = Path(artifact_dir)
        model_dir = artifact_dir / "neural"
        with (model_dir / "metadata.json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        with (artifact_dir / "common" / "labels.json").open(encoding="utf-8") as handle:
            labels = json.load(handle)["labels"]
        vocabulary = Vocabulary.load(artifact_dir / "common" / "vocabulary.json")
        checkpoint = torch.load(
            model_dir / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
        model = NeuralTextClassifier(**checkpoint["architecture"])
        model.load_state_dict(checkpoint["state_dict"])
        return cls(
            model,
            vocabulary,
            labels,
            float(metadata["temperature"]),
            float(metadata["threshold"]),
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _evaluate_epoch(
    model: NeuralTextClassifier,
    loader: DataLoader,
    loss_function: nn.Module,
    num_classes: int,
) -> tuple[float, float, float]:
    model.eval()
    losses: list[float] = []
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.inference_mode():
        for token_ids, targets in loader:
            output = model(token_ids)
            losses.append(float(loss_function(output, targets)))
            logits.append(output.numpy())
            labels.append(targets.numpy())
    combined_logits = np.concatenate(logits)
    combined_labels = np.concatenate(labels)
    report = classification_report(
        combined_labels,
        combined_logits.argmax(axis=1),
        num_classes,
    )
    return float(np.mean(losses)), report["accuracy"], report["macro_f1"]


def train_neural(
    frame: pd.DataFrame,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    config: dict | None = None,
) -> dict:
    config = config or load_config()
    artifact_dir = Path(artifact_dir)
    seed = int(config["seed"])
    _seed_everything(seed)
    torch.set_num_threads(int(config["neural"]["threads"]))

    vocabulary = Vocabulary.load(artifact_dir / "common" / "vocabulary.json")
    with (artifact_dir / "common" / "labels.json").open(encoding="utf-8") as handle:
        labels = json.load(handle)["labels"]
    label_to_id = {label: index for index, label in enumerate(labels)}

    def make_dataset(split: str) -> tuple[TensorDataset, pd.DataFrame]:
        subset = frame.loc[frame["split"].eq(split)].reset_index(drop=True)
        token_ids, _ = vocabulary.encode_padded(subset["text"].tolist())
        targets = subset["label"].map(label_to_id).to_numpy(dtype=np.int64)
        return (
            TensorDataset(torch.from_numpy(token_ids), torch.from_numpy(targets)),
            subset,
        )

    train_dataset, train_frame = make_dataset("train")
    validation_dataset, validation_frame = make_dataset("validation")
    _, test_frame = make_dataset("test")
    settings = config["neural"]
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(settings["batch_size"]),
    )
    model = NeuralTextClassifier(
        vocabulary_size=vocabulary.size,
        num_classes=len(labels),
        embedding_dim=int(settings["embedding_dim"]),
        hidden_dim=int(settings["hidden_dim"]),
        dropout=float(settings["dropout"]),
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    history = []
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        training_losses = []
        for token_ids, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(token_ids), targets)
            loss.backward()
            optimizer.step()
            training_losses.append(float(loss.detach()))
        validation_loss, validation_accuracy, validation_macro_f1 = _evaluate_epoch(
            model,
            validation_loader,
            loss_function,
            len(labels),
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(training_losses)),
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": validation_macro_f1,
            }
        )
        if validation_macro_f1 > best_score + 1e-5:
            best_score = validation_macro_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(settings["patience"]):
                break

    if best_state is None:
        raise RuntimeError("training completed without a model checkpoint")
    model.load_state_dict(best_state)
    router = NeuralRouter(model, vocabulary, labels, 1.0, 0.0)
    validation_logits = router.predict_logits(validation_frame["text"].tolist())
    y_validation = validation_frame["label"].map(label_to_id).to_numpy(dtype=np.int64)
    temperature, validation_nll = fit_temperature(validation_logits, y_validation)
    validation_probabilities = softmax(validation_logits, temperature)
    threshold_result = select_abstention_threshold(
        y_validation,
        validation_probabilities.argmax(axis=1),
        validation_probabilities.max(axis=1),
        len(labels),
        label_to_id["oos"],
    )
    router.temperature = temperature
    router.threshold = threshold_result.threshold

    model_dir = artifact_dir / "neural"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pt"
    torch.save(
        {
            "schema_version": 1,
            "architecture": model.architecture,
            "state_dict": model.state_dict(),
        },
        model_path,
    )
    metadata = {
        "schema_version": 1,
        "model_type": "mean_max_pooling_text_encoder",
        "architecture": model.architecture,
        "temperature": temperature,
        "threshold": threshold_result.threshold,
        "validation_nll": validation_nll,
        "threshold_selection": asdict(threshold_result),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_samples": len(train_frame),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seed": seed,
    }
    atomic_write_json(model_dir / "metadata.json", metadata)
    atomic_write_json(model_dir / "training_history.json", history)

    y_test = test_frame["label"].map(label_to_id).to_numpy(dtype=np.int64)
    test_logits = router.predict_logits(test_frame["text"].tolist())
    report = routing_report(
        test_logits,
        y_test,
        labels,
        temperature,
        threshold_result.threshold,
    )
    report["latency"] = latency_report(
        router.predict_logits,
        test_frame["text"].tolist(),
    )
    report["model"] = metadata
    report["environment"] = environment_manifest()
    report["artifacts"] = [artifact_record(model_path, "pytorch_checkpoint")]
    atomic_write_json(model_dir / "metrics.json", report)
    return report

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    values = np.asarray(logits, dtype=np.float64) / temperature
    if values.ndim != 2:
        raise ValueError("logits must have shape [samples, classes]")
    values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError("labels and predictions must be equally sized vectors")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if y_true.size and (
        y_true.min() < 0
        or y_pred.min() < 0
        or y_true.max() >= num_classes
        or y_pred.max() >= num_classes
    ):
        raise ValueError("class ID outside configured range")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    include_confusion: bool = False,
) -> dict:
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0)
    actual = matrix.sum(axis=1)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        actual,
        out=np.zeros_like(true_positive),
        where=actual > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    total = matrix.sum()
    report = {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": actual.astype(int).tolist(),
    }
    if include_confusion:
        report["confusion_matrix"] = matrix.tolist()
    return report


def oos_report(y_true: np.ndarray, y_pred: np.ndarray, oos_id: int) -> dict:
    true_oos = np.asarray(y_true) == oos_id
    predicted_oos = np.asarray(y_pred) == oos_id
    true_positive = int(np.sum(true_oos & predicted_oos))
    false_positive = int(np.sum(~true_oos & predicted_oos))
    false_negative = int(np.sum(true_oos & ~predicted_oos))
    true_negative = int(np.sum(~true_oos & ~predicted_oos))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def expected_calibration_error(
    probabilities: np.ndarray,
    y_true: np.ndarray,
    bins: int = 15,
) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    confidence = probabilities.max(axis=1)
    correctness = probabilities.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            error += float(mask.mean()) * abs(
                float(correctness[mask].mean()) - float(confidence[mask].mean())
            )
    return error


def negative_log_likelihood(probabilities: np.ndarray, y_true: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    selected = probabilities[np.arange(y_true.size), y_true]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    candidates = np.geomspace(0.25, 5.0, 240)
    losses = np.asarray(
        [negative_log_likelihood(softmax(logits, value), y_true) for value in candidates]
    )
    best = int(losses.argmin())
    return float(candidates[best]), float(losses[best])


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    macro_f1: float
    accuracy: float
    in_scope_coverage: float
    oos_f1: float


def apply_abstention(
    predictions: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    oos_id: int,
) -> np.ndarray:
    output = np.asarray(predictions, dtype=np.int64).copy()
    output[np.asarray(confidence) < threshold] = oos_id
    return output


def select_abstention_threshold(
    y_true: np.ndarray,
    predictions: np.ndarray,
    confidence: np.ndarray,
    num_classes: int,
    oos_id: int,
    minimum_in_scope_coverage: float = 0.85,
) -> ThresholdResult:
    y_true = np.asarray(y_true, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    confidence = np.asarray(confidence, dtype=np.float64)
    in_scope = y_true != oos_id
    best: ThresholdResult | None = None
    for threshold in np.linspace(0.0, 0.99, 200):
        routed = apply_abstention(predictions, confidence, float(threshold), oos_id)
        coverage = float(np.mean(routed[in_scope] != oos_id))
        if coverage < minimum_in_scope_coverage:
            continue
        report = classification_report(y_true, routed, num_classes)
        candidate = ThresholdResult(
            threshold=float(threshold),
            macro_f1=report["macro_f1"],
            accuracy=report["accuracy"],
            in_scope_coverage=coverage,
            oos_f1=oos_report(y_true, routed, oos_id)["f1"],
        )
        if best is None or (candidate.macro_f1, candidate.oos_f1) > (
            best.macro_f1,
            best.oos_f1,
        ):
            best = candidate
    if best is None:
        raise ValueError("no threshold satisfies the minimum coverage")
    return best


def ranking_report(rankings: np.ndarray, relevant_indices: np.ndarray) -> dict:
    rankings = np.asarray(rankings, dtype=np.int64)
    relevant_indices = np.asarray(relevant_indices, dtype=np.int64)
    if rankings.ndim != 2 or rankings.shape[0] != relevant_indices.shape[0]:
        raise ValueError("rankings must align with one relevant index per query")
    matches = rankings == relevant_indices[:, None]
    reciprocal_ranks = []
    for row in matches:
        positions = np.flatnonzero(row)
        reciprocal_ranks.append(1.0 / (int(positions[0]) + 1) if positions.size else 0.0)
    return {
        "queries": int(rankings.shape[0]),
        "recall_at_1": float(matches[:, :1].any(axis=1).mean()),
        "recall_at_3": float(matches[:, : min(3, rankings.shape[1])].any(axis=1).mean()),
        "mrr": float(np.mean(reciprocal_ranks)),
    }


def top_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    limit: int = 10,
) -> list[dict]:
    matrix = confusion_matrix(y_true, y_pred, len(labels))
    np.fill_diagonal(matrix, 0)
    results = []
    for flat_index in np.argsort(matrix, axis=None)[::-1]:
        actual, predicted = np.unravel_index(flat_index, matrix.shape)
        count = int(matrix[actual, predicted])
        if not count:
            break
        results.append(
            {"actual": labels[actual], "predicted": labels[predicted], "count": count}
        )
        if len(results) == limit:
            break
    return results

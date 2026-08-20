from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import numpy as np

from .metrics import (
    apply_abstention,
    classification_report,
    expected_calibration_error,
    negative_log_likelihood,
    oos_report,
    softmax,
    top_confusions,
)


def routing_report(
    logits: np.ndarray,
    y_true: np.ndarray,
    labels: list[str],
    temperature: float,
    threshold: float,
) -> dict:
    """Build the shared offline report used by both routing models."""
    probabilities = softmax(logits, temperature)
    raw_predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    oos_id = labels.index("oos")
    predictions = apply_abstention(
        raw_predictions,
        confidence,
        threshold,
        oos_id,
    )
    in_scope = np.asarray(y_true) != oos_id
    raw = classification_report(y_true, raw_predictions, len(labels))
    routed = classification_report(y_true, predictions, len(labels))
    return {
        "samples": int(len(y_true)),
        "accuracy": routed["accuracy"],
        "macro_f1": routed["macro_f1"],
        "in_scope_accuracy": float(
            np.mean(predictions[in_scope] == np.asarray(y_true)[in_scope])
        ),
        "in_scope_coverage": float(np.mean(predictions[in_scope] != oos_id)),
        "oos": oos_report(y_true, predictions, oos_id),
        "calibration": {
            "temperature": temperature,
            "threshold": threshold,
            "ece": expected_calibration_error(probabilities, y_true),
            "nll": negative_log_likelihood(probabilities, y_true),
        },
        "without_abstention": {
            "accuracy": raw["accuracy"],
            "macro_f1": raw["macro_f1"],
            "oos": oos_report(y_true, raw_predictions, oos_id),
        },
        "top_confusions": top_confusions(y_true, predictions, labels),
    }


def latency_report(
    predictor: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    batch_size: int = 256,
    warmup_rounds: int = 2,
    measured_rounds: int = 5,
) -> dict:
    if not texts:
        raise ValueError("at least one text is required for a latency benchmark")
    sample = list(texts[:batch_size])
    for _ in range(warmup_rounds):
        predictor(sample)
    timings = []
    for _ in range(measured_rounds):
        started = time.perf_counter()
        predictor(sample)
        timings.append(time.perf_counter() - started)
    per_request_ms = np.asarray(timings, dtype=np.float64) * 1000 / len(sample)
    return {
        "batch_size": len(sample),
        "rounds": measured_rounds,
        "mean_ms_per_request": float(per_request_ms.mean()),
        "p50_ms_per_request": float(np.percentile(per_request_ms, 50)),
        "p95_ms_per_request": float(np.percentile(per_request_ms, 95)),
        "throughput_requests_per_second": float(
            len(sample) / np.mean(timings)
        ),
    }

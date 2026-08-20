from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .artifacts import atomic_write_json, environment_manifest, file_sha256, short_version
from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_RUNTIME_DIR
from .rag import GroundedAnswerer
from .retrieval import HybridRetriever


class ServiceMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = 0
        self._abstentions = 0
        self._llm_requests = 0
        self._llm_fallbacks = 0
        self._latencies_ms: list[float] = []

    def observe(
        self,
        latency_ms: float,
        abstained: bool,
        llm_requested: bool,
        llm_fallback: bool,
    ) -> None:
        with self._lock:
            self._requests += 1
            self._abstentions += int(abstained)
            self._llm_requests += int(llm_requested)
            self._llm_fallbacks += int(llm_fallback)
            self._latencies_ms.append(latency_ms)
            if len(self._latencies_ms) > 10_000:
                self._latencies_ms = self._latencies_ms[-10_000:]

    def snapshot(self) -> dict:
        with self._lock:
            latencies = np.asarray(self._latencies_ms, dtype=np.float64)
            return {
                "requests": self._requests,
                "abstentions": self._abstentions,
                "llm_requests": self._llm_requests,
                "llm_fallbacks": self._llm_fallbacks,
                "latency_ms_p50": float(np.percentile(latencies, 50))
                if latencies.size
                else 0.0,
                "latency_ms_p95": float(np.percentile(latencies, 95))
                if latencies.size
                else 0.0,
            }

    def openmetrics(self) -> str:
        values = self.snapshot()
        lines = [
            "# HELP resolveai_requests_total Processed assist requests.",
            "# TYPE resolveai_requests_total counter",
            f"resolveai_requests_total {values['requests']}",
            "# HELP resolveai_abstentions_total Requests routed to human review.",
            "# TYPE resolveai_abstentions_total counter",
            f"resolveai_abstentions_total {values['abstentions']}",
            "# HELP resolveai_llm_fallbacks_total LLM requests using safe fallback.",
            "# TYPE resolveai_llm_fallbacks_total counter",
            f"resolveai_llm_fallbacks_total {values['llm_fallbacks']}",
            "# TYPE resolveai_llm_requests_total counter",
            f"resolveai_llm_requests_total {values['llm_requests']}",
            "# TYPE resolveai_latency_ms_p50 gauge",
            f"resolveai_latency_ms_p50 {values['latency_ms_p50']}",
            "# TYPE resolveai_latency_ms_p95 gauge",
            f"resolveai_latency_ms_p95 {values['latency_ms_p95']}",
        ]
        return "\n".join(lines) + "\n"


class FeedbackStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(
        self,
        trace_id: str,
        rating: Literal["helpful", "unhelpful"],
        correct_intent: str | None,
    ) -> None:
        record = {
            "trace_id": trace_id,
            "rating": rating,
            "correct_intent": correct_intent,
            "created_at": datetime.now(UTC).isoformat(),
        }
        serialized = json.dumps(record, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")


class SupportIntelligenceService:
    def __init__(
        self,
        retriever: HybridRetriever,
        answerer: GroundedAnswerer | None = None,
        feedback_store: FeedbackStore | None = None,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    ) -> None:
        self.retriever = retriever
        self.answerer = answerer or GroundedAnswerer()
        self.feedback_store = feedback_store or FeedbackStore(
            DEFAULT_RUNTIME_DIR / "feedback.jsonl"
        )
        self.artifact_dir = Path(artifact_dir)
        self.metrics = ServiceMetrics()

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        prefer_cpp: bool = False,
    ) -> "SupportIntelligenceService":
        return cls(
            HybridRetriever.load(artifact_dir, prefer_cpp=prefer_cpp),
            artifact_dir=artifact_dir,
        )

    def route(self, text: str) -> dict:
        return self.retriever.router.predict([text])[0]

    def assist(self, text: str, top_k: int = 3, use_llm: bool = False) -> dict:
        started = time.perf_counter()
        trace_id = str(uuid.uuid4())
        route = self.route(text)
        if route["abstained"]:
            latency_ms = (time.perf_counter() - started) * 1000
            self.metrics.observe(latency_ms, True, use_llm, False)
            return {
                "trace_id": trace_id,
                "status": "needs_human_review",
                "route": route,
                "answer": (
                    "I cannot route this request with enough confidence. "
                    "It has been marked for human review."
                ),
                "citations": [],
                "evidence": [],
                "generation": {"provider": "abstention_policy", "fallback_used": False},
                "latency_ms": latency_ms,
            }

        evidence = self.retriever.retrieve(text, route["label"], top_k)
        generation = self.answerer.answer(
            text,
            route["label"],
            evidence,
            use_llm=use_llm,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        self.metrics.observe(
            latency_ms,
            False,
            use_llm,
            generation.fallback_used,
        )
        return {
            "trace_id": trace_id,
            "status": "resolved",
            "route": route,
            "answer": generation.answer,
            "citations": generation.citations,
            "evidence": evidence,
            "generation": {
                "provider": generation.provider,
                "fallback_used": generation.fallback_used,
            },
            "latency_ms": latency_ms,
        }

    def record_feedback(
        self,
        trace_id: str,
        rating: Literal["helpful", "unhelpful"],
        correct_intent: str | None,
    ) -> None:
        if correct_intent is not None and correct_intent not in self.retriever.router.labels:
            raise ValueError("correct_intent is not in the model label set")
        self.feedback_store.append(trace_id, rating, correct_intent)

    def model_info(self) -> dict:
        model_path = self.artifact_dir / "neural" / "model.pt"
        index_path = self.artifact_dir / "retrieval" / "index.npz"
        return {
            "service": "ResolveAI",
            "version": "0.1.0",
            "model_version": short_version(model_path),
            "model_sha256": file_sha256(model_path),
            "index_version": short_version(index_path),
            "labels": len(self.retriever.router.labels),
            "knowledge_documents": len(self.retriever.documents),
            "native_top_k": self.retriever.cpp_backend is not None,
        }


def benchmark_service(
    frame: pd.DataFrame,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    seed: int = 17,
    in_scope_requests: int = 400,
    oos_requests: int = 100,
) -> dict:
    service = SupportIntelligenceService.load(artifact_dir, prefer_cpp=False)
    test = frame.loc[frame["split"].eq("test")]
    in_scope = test.loc[~test["is_oos"]].sample(
        n=in_scope_requests,
        random_state=seed,
    )
    oos = test.loc[test["is_oos"]].sample(
        n=oos_requests,
        random_state=seed,
    )
    requests = pd.concat([in_scope, oos]).sample(frac=1.0, random_state=seed)
    for text in requests["text"].head(10):
        service.assist(text)
    latencies = []
    resolved = 0
    cited = 0
    started = time.perf_counter()
    for text in requests["text"]:
        result = service.assist(text)
        latencies.append(float(result["latency_ms"]))
        resolved += int(result["status"] == "resolved")
        cited += int(bool(result["citations"]))
    elapsed = time.perf_counter() - started
    values = np.asarray(latencies)
    report = {
        "requests": len(requests),
        "in_scope_requests": in_scope_requests,
        "oos_requests": oos_requests,
        "latency_ms_p50": float(np.percentile(values, 50)),
        "latency_ms_p95": float(np.percentile(values, 95)),
        "latency_ms_p99": float(np.percentile(values, 99)),
        "throughput_requests_per_second": len(requests) / elapsed,
        "resolved": resolved,
        "abstained": len(requests) - resolved,
        "citation_coverage_on_resolved": cited / resolved if resolved else 0.0,
        "configuration": "single process, sequential requests, CPU, extractive generation",
        "environment": environment_manifest(),
    }
    output_path = Path(artifact_dir) / "service" / "benchmark.json"
    atomic_write_json(output_path, report)
    return report

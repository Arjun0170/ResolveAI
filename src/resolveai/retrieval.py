from __future__ import annotations

import ctypes
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .artifacts import artifact_record, atomic_write_json, environment_manifest
from .config import DEFAULT_ARTIFACT_DIR, PROJECT_ROOT, load_config
from .knowledge import build_knowledge_base
from .metrics import ranking_report
from .neural import NeuralRouter
from .text import Vocabulary


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def build_tfidf_matrix(
    texts: Sequence[str],
    vocabulary: Vocabulary,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((len(texts), vocabulary.size), dtype=np.float32)
    for row, text in enumerate(texts):
        token_ids = vocabulary.encode(text, truncate=False)
        np.add.at(counts[row], token_ids, 1.0)
    document_frequency = (counts > 0).sum(axis=0)
    inverse_document_frequency = (
        np.log((1.0 + len(texts)) / (1.0 + document_frequency)) + 1.0
    ).astype(np.float32)
    matrix = counts * inverse_document_frequency[None, :]
    return _normalize_rows(matrix), inverse_document_frequency


def tfidf_queries(
    texts: Sequence[str],
    vocabulary: Vocabulary,
    inverse_document_frequency: np.ndarray,
) -> np.ndarray:
    counts = np.zeros((len(texts), vocabulary.size), dtype=np.float32)
    for row, text in enumerate(texts):
        token_ids = vocabulary.encode(text, truncate=False)
        np.add.at(counts[row], token_ids, 1.0)
    return _normalize_rows(counts * inverse_document_frequency[None, :])


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError("scores must have shape [queries, documents]")
    if not 0 < k <= scores.shape[1]:
        raise ValueError("k must be between one and the number of documents")
    candidates = np.argpartition(scores, -k, axis=1)[:, -k:]
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    order = np.argsort(candidate_scores, axis=1)[:, ::-1]
    return np.take_along_axis(candidates, order, axis=1)


class CppTopK:
    def __init__(self, library_path: str | Path) -> None:
        self.library_path = Path(library_path)
        self.library = ctypes.CDLL(str(self.library_path))
        function = self.library.resolve_top_k
        float_pointer = ctypes.POINTER(ctypes.c_float)
        integer_pointer = ctypes.POINTER(ctypes.c_int)
        function.argtypes = [
            float_pointer,
            ctypes.c_int,
            ctypes.c_int,
            float_pointer,
            ctypes.c_int,
            integer_pointer,
            float_pointer,
        ]
        function.restype = ctypes.c_int
        self.function = function

    def search(
        self,
        matrix: np.ndarray,
        query: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        query = np.ascontiguousarray(query, dtype=np.float32)
        if matrix.ndim != 2 or query.shape != (matrix.shape[1],):
            raise ValueError("matrix and query dimensions do not align")
        if not 0 < k <= matrix.shape[0]:
            raise ValueError("invalid top-k value")
        indices = np.empty(k, dtype=np.int32)
        scores = np.empty(k, dtype=np.float32)
        code = self.function(
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            matrix.shape[0],
            matrix.shape[1],
            query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k,
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if code != 0:
            raise RuntimeError(f"C++ top-k backend returned error code {code}")
        return indices.astype(np.int64), scores


class HybridRetriever:
    def __init__(
        self,
        router: NeuralRouter,
        documents: list[dict],
        lexical_matrix: np.ndarray,
        inverse_document_frequency: np.ndarray,
        neural_matrix: np.ndarray,
        weights: dict[str, float],
        cpp_backend: CppTopK | None = None,
    ) -> None:
        self.router = router
        self.documents = documents
        self.lexical_matrix = lexical_matrix.astype(np.float32)
        self.inverse_document_frequency = inverse_document_frequency.astype(np.float32)
        self.neural_matrix = neural_matrix.astype(np.float32)
        self.weights = weights
        self.cpp_backend = cpp_backend
        self.intent_to_document = {
            document["intent"]: index for index, document in enumerate(documents)
        }
        self.hybrid_matrix = np.concatenate(
            [
                np.sqrt(float(weights["lexical_weight"])) * self.lexical_matrix,
                np.sqrt(float(weights["neural_weight"])) * self.neural_matrix,
                np.sqrt(float(weights["route_weight"]))
                * np.eye(len(documents), dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32)

    def _query_features(
        self,
        texts: Sequence[str],
        routes: Sequence[str] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lexical_queries = tfidf_queries(
            texts,
            self.router.vocabulary,
            self.inverse_document_frequency,
        )
        neural_queries = self.router.encode_texts(texts)
        route_queries = np.zeros((len(texts), len(self.documents)), dtype=np.float32)
        if routes is not None:
            for row, route in enumerate(routes):
                document_index = self.intent_to_document.get(route)
                if document_index is not None:
                    route_queries[row, document_index] = 1.0
        hybrid_queries = np.concatenate(
            [
                np.sqrt(float(self.weights["lexical_weight"])) * lexical_queries,
                np.sqrt(float(self.weights["neural_weight"])) * neural_queries,
                np.sqrt(float(self.weights["route_weight"])) * route_queries,
            ],
            axis=1,
        ).astype(np.float32)
        return lexical_queries, neural_queries, route_queries, hybrid_queries

    def score(
        self,
        texts: Sequence[str],
        routes: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        lexical_queries, neural_queries, route_queries, _ = self._query_features(
            texts,
            routes,
        )
        lexical = lexical_queries @ self.lexical_matrix.T
        neural = neural_queries @ self.neural_matrix.T
        route_scores = route_queries
        combined = (
            float(self.weights["lexical_weight"]) * lexical
            + float(self.weights["neural_weight"]) * neural
            + float(self.weights["route_weight"]) * route_scores
        )
        return combined, {
            "lexical": lexical,
            "neural": neural,
            "route": route_scores,
        }

    def retrieve(
        self,
        text: str,
        route: str | None,
        top_k: int = 3,
    ) -> list[dict]:
        combined, components = self.score([text], [route] if route else None)
        if self.cpp_backend is not None:
            _, _, _, hybrid_queries = self._query_features(
                [text],
                [route] if route else None,
            )
            indices, scores = self.cpp_backend.search(
                self.hybrid_matrix,
                hybrid_queries[0],
                top_k,
            )
        else:
            indices = top_k_indices(combined, top_k)[0]
            scores = combined[0, indices]
        results = []
        for index, score in zip(indices, scores, strict=True):
            document = self.documents[int(index)]
            results.append(
                {
                    "doc_id": document["doc_id"],
                    "intent": document["intent"],
                    "title": document["title"],
                    "summary": document["summary"],
                    "guidance": document["guidance"],
                    "score": float(score),
                    "score_components": {
                        name: float(values[0, index])
                        for name, values in components.items()
                    },
                }
            )
        return results

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
        prefer_cpp: bool = False,
    ) -> "HybridRetriever":
        artifact_dir = Path(artifact_dir)
        router = NeuralRouter.load(artifact_dir)
        with (artifact_dir / "retrieval" / "knowledge.json").open(
            encoding="utf-8"
        ) as handle:
            documents = json.load(handle)["documents"]
        with np.load(artifact_dir / "retrieval" / "index.npz") as values:
            lexical = values["lexical_matrix"].copy()
            inverse_document_frequency = values["inverse_document_frequency"].copy()
            neural = values["neural_matrix"].copy()
        settings = load_config()["retrieval"]
        library_path = PROJECT_ROOT / "build" / "libresolve_topk.so"
        cpp_backend = (
            CppTopK(library_path) if prefer_cpp and library_path.exists() else None
        )
        return cls(
            router,
            documents,
            lexical,
            inverse_document_frequency,
            neural,
            settings,
            cpp_backend,
        )


def build_retrieval_index(
    frame: pd.DataFrame,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> dict:
    artifact_dir = Path(artifact_dir)
    output_dir = artifact_dir / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    knowledge_path = output_dir / "knowledge.json"
    documents = build_knowledge_base(frame, knowledge_path)
    router = NeuralRouter.load(artifact_dir)
    index_texts = [document["index_text"] for document in documents]
    lexical_matrix, inverse_document_frequency = build_tfidf_matrix(
        index_texts,
        router.vocabulary,
    )
    neural_matrix = router.encode_texts(index_texts)
    index_path = output_dir / "index.npz"
    np.savez_compressed(
        index_path,
        schema_version=np.asarray([1]),
        lexical_matrix=lexical_matrix,
        inverse_document_frequency=inverse_document_frequency,
        neural_matrix=neural_matrix,
    )
    manifest = {
        "schema_version": 1,
        "documents": len(documents),
        "lexical_dimensions": list(lexical_matrix.shape),
        "neural_dimensions": list(neural_matrix.shape),
        "knowledge_provenance": "CLINC150 training split only",
        "environment": environment_manifest(),
        "artifacts": [
            artifact_record(knowledge_path, "knowledge_base"),
            artifact_record(index_path, "hybrid_index"),
        ],
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def evaluate_retrieval(
    frame: pd.DataFrame,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> dict:
    artifact_dir = Path(artifact_dir)
    retriever = HybridRetriever.load(artifact_dir, prefer_cpp=False)
    test = frame.loc[
        frame["split"].eq("test") & ~frame["is_oos"]
    ].reset_index(drop=True)
    texts = test["text"].tolist()
    route_results = retriever.router.predict(texts)
    routes = [result["raw_label"] for result in route_results]
    combined, components = retriever.score(texts, routes)
    relevant = test["label"].map(retriever.intent_to_document).to_numpy(dtype=np.int64)
    report = {
        "lexical": ranking_report(
            top_k_indices(components["lexical"], len(retriever.documents)),
            relevant,
        ),
        "neural": ranking_report(
            top_k_indices(components["neural"], len(retriever.documents)),
            relevant,
        ),
        "hybrid_without_route": ranking_report(
            top_k_indices(
                float(retriever.weights["lexical_weight"]) * components["lexical"]
                + float(retriever.weights["neural_weight"]) * components["neural"],
                len(retriever.documents),
            ),
            relevant,
        ),
        "orchestrated": ranking_report(
            top_k_indices(combined, len(retriever.documents)),
            relevant,
        ),
        "weights": retriever.weights,
        "evaluation_split": "CLINC150 in-scope test only",
        "data_leakage_control": "Knowledge and vocabulary use training split only.",
    }
    atomic_write_json(artifact_dir / "retrieval" / "metrics.json", report)
    return report


def benchmark_top_k(
    library_path: str | Path = PROJECT_ROOT / "build" / "libresolve_topk.so",
    seed: int = 17,
    rows: int = 10_000,
    dimensions: int = 256,
    queries: int = 100,
    k: int = 5,
) -> dict:
    generator = np.random.default_rng(seed)
    matrix = _normalize_rows(generator.normal(size=(rows, dimensions)).astype(np.float32))
    query_matrix = _normalize_rows(
        generator.normal(size=(queries, dimensions)).astype(np.float32)
    )
    backend = CppTopK(library_path)
    numpy_results = []
    started = time.perf_counter()
    for query in query_matrix:
        scores = matrix @ query
        numpy_results.append(top_k_indices(scores[None, :], k)[0])
    numpy_seconds = time.perf_counter() - started

    cpp_results = []
    started = time.perf_counter()
    for query in query_matrix:
        indices, _ = backend.search(matrix, query, k)
        cpp_results.append(indices)
    cpp_seconds = time.perf_counter() - started
    exact_match = all(
        np.array_equal(left, right)
        for left, right in zip(numpy_results, cpp_results, strict=True)
    )
    report = {
        "rows": rows,
        "dimensions": dimensions,
        "queries": queries,
        "k": k,
        "numpy_total_ms": numpy_seconds * 1000,
        "cpp_total_ms": cpp_seconds * 1000,
        "speedup_vs_numpy": numpy_seconds / cpp_seconds,
        "exact_top_k_parity": exact_match,
        "default_backend": "numpy",
        "selection_reason": (
            "NumPy/BLAS is faster for the measured workload; C++ remains optional."
        ),
    }
    if not exact_match:
        raise AssertionError("C++ and NumPy top-k results differ")
    return report

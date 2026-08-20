from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import atomic_write_json
from .baseline import BaselineRouter, train_baseline
from .config import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_PATH
from .data import download_clinc150, load_clinc150, write_dataset_report
from .neural import NeuralRouter, train_neural
from .retrieval import (
    benchmark_top_k,
    build_retrieval_index,
    evaluate_retrieval,
)
from .service import SupportIntelligenceService, benchmark_service


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def command_download_data(args: argparse.Namespace) -> None:
    path = download_clinc150(args.output, force=args.force)
    report = write_dataset_report(load_clinc150(path), path)
    _print_json(report)


def command_inspect_data(args: argparse.Namespace) -> None:
    frame = load_clinc150(args.data)
    _print_json(write_dataset_report(frame, args.data))


def command_train_baseline(args: argparse.Namespace) -> None:
    report = train_baseline(load_clinc150(args.data), args.artifacts)
    _print_json(report)


def command_predict_baseline(args: argparse.Namespace) -> None:
    router = BaselineRouter.load(args.artifacts)
    _print_json(router.predict([args.text])[0])


def command_train_neural(args: argparse.Namespace) -> None:
    report = train_neural(load_clinc150(args.data), args.artifacts)
    _print_json(report)


def command_predict_neural(args: argparse.Namespace) -> None:
    router = NeuralRouter.load(args.artifacts)
    _print_json(router.predict([args.text])[0])


def command_build_index(args: argparse.Namespace) -> None:
    report = build_retrieval_index(load_clinc150(args.data), args.artifacts)
    _print_json(report)


def command_evaluate_rag(args: argparse.Namespace) -> None:
    report = evaluate_retrieval(load_clinc150(args.data), args.artifacts)
    _print_json(report)


def command_benchmark(args: argparse.Namespace) -> None:
    report = benchmark_top_k(args.library)
    atomic_write_json(DEFAULT_ARTIFACT_DIR / "retrieval" / "native_benchmark.json", report)
    _print_json(report)


def command_benchmark_service(args: argparse.Namespace) -> None:
    _print_json(
        benchmark_service(
            load_clinc150(args.data),
            args.artifacts,
            in_scope_requests=args.in_scope,
            oos_requests=args.oos,
        )
    )


def command_demo(args: argparse.Namespace) -> None:
    service = SupportIntelligenceService.load(
        args.artifacts,
        prefer_cpp=args.use_cpp,
    )
    _print_json(service.assist(args.text, args.top_k, args.use_llm))


def command_train_all(args: argparse.Namespace) -> None:
    frame = load_clinc150(args.data)
    results = {
        "baseline": train_baseline(frame, args.artifacts),
        "neural": train_neural(frame, args.artifacts),
        "index": build_retrieval_index(frame, args.artifacts),
        "retrieval": evaluate_retrieval(frame, args.artifacts),
        "service": benchmark_service(frame, args.artifacts),
    }
    _print_json(results)


def command_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "resolveai.api:app",
        host=args.host,
        port=args.port,
        reload=False,
        workers=1,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="resolveai",
        description="Train, evaluate, and serve the ResolveAI support platform.",
    )
    parser.add_argument("--version", action="version", version="ResolveAI 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="download CLINC150")
    download.add_argument("--output", type=Path, default=DEFAULT_DATA_PATH)
    download.add_argument("--force", action="store_true")
    download.set_defaults(handler=command_download_data)

    inspect = subparsers.add_parser("inspect-data", help="validate dataset contract")
    inspect.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    inspect.set_defaults(handler=command_inspect_data)

    baseline = subparsers.add_parser(
        "train-baseline",
        help="train and evaluate the NumPy classical baseline",
    )
    baseline.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    baseline.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    baseline.set_defaults(handler=command_train_baseline)

    predict = subparsers.add_parser(
        "predict-baseline",
        help="run a request through the classical router",
    )
    predict.add_argument("--text", required=True)
    predict.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    predict.set_defaults(handler=command_predict_baseline)

    neural = subparsers.add_parser(
        "train-neural",
        help="train, calibrate, and evaluate the PyTorch router",
    )
    neural.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    neural.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    neural.set_defaults(handler=command_train_neural)

    neural_predict = subparsers.add_parser(
        "predict-neural",
        help="run a request through the calibrated PyTorch router",
    )
    neural_predict.add_argument("--text", required=True)
    neural_predict.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
    )
    neural_predict.set_defaults(handler=command_predict_neural)

    index = subparsers.add_parser(
        "build-index",
        help="build the training-only hybrid knowledge index",
    )
    index.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    index.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    index.set_defaults(handler=command_build_index)

    evaluate = subparsers.add_parser(
        "evaluate-rag",
        help="evaluate lexical, neural, and orchestrated retrieval",
    )
    evaluate.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    evaluate.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    evaluate.set_defaults(handler=command_evaluate_rag)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="benchmark and verify the C++ top-k backend",
    )
    benchmark.add_argument(
        "--library",
        type=Path,
        default=Path("build/libresolve_topk.so"),
    )
    benchmark.set_defaults(handler=command_benchmark)

    service_benchmark = subparsers.add_parser(
        "benchmark-service",
        help="benchmark the end-to-end CPU inference pipeline",
    )
    service_benchmark.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    service_benchmark.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
    )
    service_benchmark.add_argument("--in-scope", type=int, default=400)
    service_benchmark.add_argument("--oos", type=int, default=100)
    service_benchmark.set_defaults(handler=command_benchmark_service)

    demo = subparsers.add_parser(
        "demo",
        help="run the complete routing and grounded-assistance workflow",
    )
    demo.add_argument("--text", required=True)
    demo.add_argument("--top-k", type=int, default=3, choices=range(1, 6))
    demo.add_argument("--use-llm", action="store_true")
    demo.add_argument(
        "--use-cpp",
        action="store_true",
        help="use the optional native top-k backend",
    )
    demo.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    demo.set_defaults(handler=command_demo)

    train_all = subparsers.add_parser(
        "train-all",
        help="reproduce all model and retrieval artifacts",
    )
    train_all.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    train_all.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    train_all.set_defaults(handler=command_train_all)

    serve = subparsers.add_parser("serve", help="start the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(handler=command_serve)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)

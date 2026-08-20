import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from resolveai.api import create_app
from resolveai.service import FeedbackStore, ServiceMetrics


class FakeService:
    def __init__(self, feedback_path: Path) -> None:
        self.metrics = ServiceMetrics()
        self.feedback_store = FeedbackStore(feedback_path)

    def model_info(self) -> dict:
        return {"model_version": "abc123", "labels": 151}

    def route(self, text: str) -> dict:
        return {
            "label": "find_phone",
            "confidence": 0.99,
            "abstained": False,
            "raw_label": "find_phone",
            "candidates": [{"label": "find_phone", "score": 0.99}],
        }

    def assist(self, text: str, top_k: int, use_llm: bool) -> dict:
        self.metrics.observe(2.0, False, use_llm, False)
        return {
            "trace_id": "00000000-0000-0000-0000-000000000000",
            "status": "resolved",
            "route": self.route(text),
            "answer": "Use the verified workflow [KB-001]",
            "citations": ["KB-001"],
            "evidence": [
                {
                    "doc_id": "KB-001",
                    "intent": "find_phone",
                    "title": "Find Phone",
                    "summary": "Find a phone.",
                    "guidance": "Use the verified workflow.",
                    "score": 0.9,
                    "score_components": {
                        "lexical": 0.8,
                        "neural": 0.9,
                        "route": 1.0,
                    },
                }
            ],
            "generation": {"provider": "extractive", "fallback_used": False},
            "latency_ms": 2.0,
        }

    def record_feedback(self, trace_id: str, rating: str, correct_intent: str | None) -> None:
        self.feedback_store.append(trace_id, rating, correct_intent)


class ApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        service = FakeService(Path(self.temporary.name) / "feedback.jsonl")
        self.client_context = TestClient(
            create_app(service),
            backend_options={"use_uvloop": True},
        )
        self.client = self.client_context.__enter__()

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.temporary.cleanup()

    def test_health_route_and_assist(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 200)
        route = self.client.post("/v1/route", json={"text": "find my phone"})
        self.assertEqual(route.json()["label"], "find_phone")
        assist = self.client.post("/v1/assist", json={"text": "find my phone"})
        self.assertEqual(assist.status_code, 200)
        self.assertEqual(assist.json()["citations"], ["KB-001"])

    def test_input_contract_and_metrics(self) -> None:
        invalid = self.client.post("/v1/assist", json={"text": "", "top_k": 20})
        self.assertEqual(invalid.status_code, 422)
        self.client.post("/v1/assist", json={"text": "find my phone"})
        metrics = self.client.get("/metrics")
        self.assertIn("resolveai_requests_total 1", metrics.text)

    def test_feedback_is_accepted(self) -> None:
        response = self.client.post(
            "/v1/feedback",
            json={
                "trace_id": "00000000-0000-0000-0000-000000000000",
                "rating": "helpful",
            },
        )
        self.assertEqual(response.status_code, 202)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from resolveai.api import create_app
from resolveai.service import SupportIntelligenceService


class ArtifactIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not Path("artifacts/neural/model.pt").exists():
            raise unittest.SkipTest("trained artifacts are unavailable")
        cls.service = SupportIntelligenceService.load()
        cls.client_context = TestClient(
            create_app(cls.service),
            backend_options={"use_uvloop": True},
        )
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_context.__exit__(None, None, None)

    def test_resolved_request_contains_known_citations(self) -> None:
        response = self.client.post(
            "/v1/assist",
            json={"text": "please help me find the phone i lost"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "resolved")
        known = {document["doc_id"] for document in self.service.retriever.documents}
        self.assertTrue(set(payload["citations"]) <= known)
        self.assertTrue(payload["citations"])

    def test_known_oos_request_abstains_without_evidence(self) -> None:
        response = self.client.post(
            "/v1/assist",
            json={"text": "how much has the dow changed today"},
        )
        payload = response.json()
        self.assertEqual(payload["status"], "needs_human_review")
        self.assertEqual(payload["evidence"], [])


if __name__ == "__main__":
    unittest.main()

import unittest

from resolveai.rag import GroundedAnswerer, extract_citations, validate_citations


DOCUMENTS = [
    {
        "doc_id": "KB-001",
        "title": "Card Help",
        "guidance": "Verify the account before changing card settings.",
    }
]


class StaticClient:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, request_text: str, route: str, documents: list[dict]) -> str:
        return self.answer


class RagTest(unittest.TestCase):
    def test_extract_and_validate_citations(self) -> None:
        answer = "Follow the verified workflow [KB-001]."
        self.assertEqual(extract_citations(answer), ["KB-001"])
        self.assertEqual(validate_citations(answer, DOCUMENTS), ["KB-001"])

    def test_unknown_citation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_citations("Unsupported [KB-999]", DOCUMENTS)

    def test_invalid_llm_output_uses_extractive_fallback(self) -> None:
        result = GroundedAnswerer(StaticClient("An uncited claim")).answer(
            "help",
            "card_help",
            DOCUMENTS,
            use_llm=True,
        )
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.provider, "extractive")
        self.assertEqual(result.citations, ["KB-001"])


if __name__ == "__main__":
    unittest.main()

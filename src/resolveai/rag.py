from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Sequence


CITATION_PATTERN = re.compile(r"\[(KB-\d{3})\]")


def extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(CITATION_PATTERN.findall(text)))


def validate_citations(answer: str, documents: Sequence[dict]) -> list[str]:
    citations = extract_citations(answer)
    allowed = {document["doc_id"] for document in documents}
    if not citations:
        raise ValueError("generated answer contains no knowledge-base citation")
    unknown = set(citations) - allowed
    if unknown:
        raise ValueError(f"generated answer cites unknown sources: {sorted(unknown)}")
    return citations


def extractive_answer(route: str, documents: Sequence[dict]) -> tuple[str, list[str]]:
    if not documents:
        raise ValueError("at least one document is required")
    primary = documents[0]
    answer = (
        f"I matched this request to {primary['title']} ({route}). "
        f"{primary['guidance']} [{primary['doc_id']}]"
    )
    return answer, [primary["doc_id"]]


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    citations: list[str]
    provider: str
    fallback_used: bool


class OpenAICompatibleClient:
    """Minimal chat-completions client using only the Python standard library."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls) -> "OpenAICompatibleClient | None":
        base_url = os.getenv("RESOLVEAI_LLM_BASE_URL")
        api_key = os.getenv("RESOLVEAI_LLM_API_KEY")
        model = os.getenv("RESOLVEAI_LLM_MODEL")
        if not (base_url and api_key and model):
            return None
        return cls(base_url, api_key, model)

    def generate(self, request_text: str, route: str, documents: Sequence[dict]) -> str:
        sources = "\n".join(
            (
                f"[{document['doc_id']}] {document['title']}: "
                f"{document['summary']} {document['guidance']}"
            )
            for document in documents
        )
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer the support request using only the supplied sources. "
                        "Every factual statement must cite a source as [KB-NNN]. "
                        "If the sources are insufficient, say that human review is needed."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Route: {route}\nRequest: {request_text}\n\nSources:\n{sources}"
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ResolveAI/0.1",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                result = json.load(response)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"LLM request failed: {error}") from error
        try:
            return str(result["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("LLM response does not match chat-completions schema") from error


class GroundedAnswerer:
    def __init__(self, client: OpenAICompatibleClient | None = None) -> None:
        self.client = client if client is not None else OpenAICompatibleClient.from_environment()

    def answer(
        self,
        request_text: str,
        route: str,
        documents: Sequence[dict],
        use_llm: bool = False,
    ) -> GenerationResult:
        fallback_answer, fallback_citations = extractive_answer(route, documents)
        if not use_llm or self.client is None:
            return GenerationResult(
                fallback_answer,
                fallback_citations,
                "extractive",
                fallback_used=use_llm,
            )
        try:
            answer = self.client.generate(request_text, route, documents)
            citations = validate_citations(answer, documents)
            return GenerationResult(answer, citations, "openai_compatible", False)
        except (RuntimeError, ValueError):
            return GenerationResult(
                fallback_answer,
                fallback_citations,
                "extractive",
                True,
            )

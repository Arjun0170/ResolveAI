from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from .artifacts import atomic_write_json
from .text import tokenize


FINANCE_TERMS = {
    "bank",
    "beneficiary",
    "cash",
    "card",
    "cash_withdrawal",
    "currency",
    "payment",
    "refund",
    "transfer",
}
ACCOUNT_TERMS = {
    "account",
    "contact",
    "email",
    "login",
    "password",
    "pin",
    "profile",
}
TRAVEL_TERMS = {
    "airline",
    "airport",
    "car_rental",
    "flight",
    "hotel",
    "travel",
    "visa",
}


def humanize_intent(intent: str) -> str:
    return intent.replace("_", " ").title()


def _safety_note(intent: str) -> str:
    tokens = set(intent.split("_")) | {intent}
    if tokens & FINANCE_TERMS:
        return (
            "Do not request full payment credentials. Verify transaction state and "
            "account ownership before changing or escalating a financial operation."
        )
    if tokens & ACCOUNT_TERMS:
        return (
            "Protect account data and use the approved identity-verification flow "
            "before changing credentials or personal information."
        )
    if tokens & TRAVEL_TERMS:
        return (
            "Confirm dates, provider, and booking identifier before changing a "
            "reservation or escalating to the relevant travel provider."
        )
    return (
        "Confirm the request details, provide the matching workflow, and escalate "
        "when the action requires account-specific access."
    )


def build_knowledge_base(
    frame: pd.DataFrame,
    output_path: str | Path,
    examples_per_intent: int = 8,
) -> list[dict]:
    train = frame.loc[
        frame["split"].eq("train") & ~frame["is_oos"]
    ].reset_index(drop=True)
    documents = []
    for index, (intent, group) in enumerate(sorted(train.groupby("label"))):
        examples = group["text"].head(examples_per_intent).tolist()
        counts: Counter[str] = Counter()
        for text in group["text"]:
            counts.update(tokenize(text))
        keywords = [token for token, _ in counts.most_common(12)]
        title = humanize_intent(intent)
        documents.append(
            {
                "doc_id": f"KB-{index + 1:03d}",
                "intent": intent,
                "title": title,
                "summary": f"Support workflow for requests about {title.lower()}.",
                "guidance": _safety_note(intent),
                "keywords": keywords,
                "training_examples": examples,
                "index_text": " ".join(
                    [title, intent.replace("_", " "), *keywords, *examples]
                ),
            }
        )
    if len(documents) != 150:
        raise ValueError(f"expected 150 knowledge documents, found {len(documents)}")
    atomic_write_json(
        output_path,
        {
            "schema_version": 1,
            "provenance": "Derived only from CLINC150 training examples.",
            "documents": documents,
        },
    )
    return documents

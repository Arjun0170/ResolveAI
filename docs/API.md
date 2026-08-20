# API Contract

Start the service with `.venv/bin/resolveai serve`. The interactive schema is at
`/docs`, the OpenAPI document at `/openapi.json`, and operational metrics at
`/metrics`.

## `POST /v1/route`

Request:

```json
{"text": "please help me find the phone i lost"}
```

Returns the accepted label, raw label, calibrated confidence, abstention state,
and top three candidate intents. Text must contain 1-1,000 characters.

## `POST /v1/assist`

Request:

```json
{
  "text": "please help me find the phone i lost",
  "top_k": 3,
  "use_llm": false
}
```

Accepted requests return `resolved`, evidence with lexical/neural/route score
components, an answer, validated citations, generation provider, trace ID, and
latency. Abstained requests return `needs_human_review`, empty evidence, and no
citations.

`use_llm` only attempts generation when all three `RESOLVEAI_LLM_*` variables
are configured. Provider failure or citation failure uses the offline fallback.

## `POST /v1/feedback`

```json
{
  "trace_id": "00000000-0000-0000-0000-000000000000",
  "rating": "helpful",
  "correct_intent": "find_phone"
}
```

Returns HTTP 202. The service stores no raw request text in the feedback file.

## Operations

- `GET /health`: readiness and loaded model version.
- `GET /v1/model-info`: model/index hashes, label and document counts, backend.
- `GET /metrics`: OpenMetrics-compatible request, abstention, fallback, p50, and
  p95 values.

Authentication and rate limiting are intentionally outside the local portfolio
scope and are required before internet-facing deployment.

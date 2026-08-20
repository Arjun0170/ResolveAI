# Security and Privacy Notes

ResolveAI is a local portfolio system, not an internet-ready support service.

Implemented controls:

- Input length validation on inference endpoints.
- Abstention before retrieval for OOS or low-confidence requests.
- Citation allow-list validation and deterministic fallback for optional LLMs.
- No raw request text in the feedback store.
- API-visible model and index hashes for traceability.
- No API secrets in source or artifacts.

Required before production:

- Authentication, authorization, TLS, rate limits, request-size limits at the
  proxy, and secrets management.
- PII classification, redaction, retention policy, encryption, and access audit.
- Prompt-injection defenses appropriate to the real knowledge source and tool
  permissions.
- Dependency and container scanning, signed model artifacts, least-privilege
  runtime identity, and a documented incident response process.
- Risk-tiered human approval for actions that affect identity, payments, or
  account state.

Do not report security vulnerabilities through a public issue containing user
data or credentials. Rotate any credential accidentally used with the optional
LLM client immediately.

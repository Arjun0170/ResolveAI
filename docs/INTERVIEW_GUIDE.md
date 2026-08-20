# Interview and Project Pitch Guide

## Thirty Seconds

ResolveAI is a support-intelligence platform that I built as three connected ML
systems. I first implemented a NumPy intent baseline, then a calibrated PyTorch
router with explicit out-of-scope abstention, and finally hybrid retrieval with
citation-checked generation behind FastAPI. On the held-out CLINC150 test set,
the neural model reached 86.25% macro F1 and 69.51% OOS F1, while route-aware
retrieval reached 94.84% Recall@3. The offline service runs at 4.06 ms p95 on CPU.

## Two Minutes

Start with the product problem: a support system must know both where a request
belongs and when not to guess. Explain the three stages in order:

1. IntentBench validates data and establishes a transparent classical benchmark.
2. NeuralRouter learns better representations, then calibrates and abstains using
   validation data rather than treating argmax as a business decision.
3. GroundedAssist reuses the encoder for retrieval, combines exact and semantic
   evidence, validates citations, and serves the workflow with model hashes,
   metrics, feedback, and tests.

Close with one result and one honest limitation: macro F1 improved by about six
percentage points, but CLINC150 is not real company traffic and the threshold
would need cost-sensitive tuning and drift monitoring in production.

## Deep-Dive Structure

Use this order for a 10-15 minute discussion:

1. Requirements and failure cost.
2. Data contract and leakage boundary.
3. Baseline and why it matters.
4. Neural architecture and training loop.
5. Calibration and abstention tradeoff.
6. Retrieval signals and held-out evaluation.
7. API, artifacts, observability, and feedback.
8. Tests, benchmark methodology, limitations, next production step.

## Likely Follow-ups

**Why not a Transformer?**

The objective was a complete, defensible ML system under a ten-day learning
constraint. The compact encoder trains on CPU, exposes every operation, reuses
representations for retrieval, and demonstrates a measurable baseline lift. A
Transformer is a controlled next experiment, not a substitute for the pipeline.

**Why train OOS as a class and also threshold confidence?**

The OOS class learns from known negative examples. The threshold also catches
uncertain in-scope-looking inputs that do not win the OOS logit. Together they
improve coverage of different failure modes.

**Is a confidence score a probability?**

Not automatically. Temperature scaling reduces validation NLL and calibration
error without changing the argmax order. Even calibrated probabilities are
distribution-dependent and must be monitored after deployment.

**How do you know retrieval is not just repeating the classifier?**

The report measures lexical and neural retrieval independently, hybrid without
the route, and the full route-aware ranker. The small route weight improves
Recall@3 from 92.80% to 94.84%, while lexical/neural signals still dominate.

**Does citation validation eliminate hallucination?**

No. It verifies that citations exist and belong to retrieved evidence. It cannot
prove every claim follows from a source. Production would add claim-level
entailment checks, stricter structured output, policy versioning, and review for
high-risk intents.

**Why include C++ if NumPy is faster?**

The extension is an interoperability and measurement exercise. Exact parity is
tested, and benchmarking showed optimized BLAS wins at this size. The correct
engineering decision is to keep NumPy as default instead of deploying a slower
component to justify its existence.

## Three Resume Project Angles

The repository can be presented as one flagship system or three connected
projects without inventing separate work:

- **IntentBench:** data quality, classical ML, from-scratch metrics, benchmarking.
- **NeuralRouter:** PyTorch, deep learning, calibration, OOD, artifact versioning.
- **GroundedAssist:** information retrieval, basic RAG, FastAPI, C++/Python,
  observability, integration testing.

For a general SWE role, lead with API contracts, testing, artifacts, C ABI, and
performance decisions. For an MLE role, lead with evaluation design,
calibration, OOS behavior, leakage controls, and shared representations. For an
AI startup, lead with the complete request-to-grounded-answer loop and safe
fallback.

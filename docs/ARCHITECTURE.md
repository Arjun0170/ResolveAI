# ResolveAI Architecture

## Objective

ResolveAI demonstrates the full lifecycle of a small text ML product: governed
data ingestion, a classical benchmark, deep model training, uncertainty-aware
decisions, retrieval, grounded generation, an API, observability, and native
interoperability. The design keeps the learning surface narrow by reusing one
tokenizer, label space, encoder, artifact format, and evaluation layer.

## Request Lifecycle

1. `NeuralRouter` tokenizes the request and produces calibrated probabilities
   across 150 intents plus OOS.
2. Explicit OOS predictions and scores below the validation-selected threshold
   return `needs_human_review`. Retrieval does not run.
3. Accepted requests are represented as TF-IDF vectors and 256-dimensional
   neural embeddings.
4. `HybridRetriever` scores 150 knowledge articles with 0.45 lexical, 0.45
   neural, and 0.10 predicted-route signals.
5. `GroundedAnswerer` returns an offline extractive answer or optionally calls a
   chat-completions-compatible model.
6. The citation guard requires at least one retrieved `[KB-NNN]` citation and
   rejects citations outside the retrieved evidence set.
7. FastAPI returns the trace ID, route, alternatives, confidence, evidence,
   decomposed retrieval scores, answer, citations, generation mode, and latency.

## Shared Contract

Both routing models expose:

```python
predict_logits(texts) -> ndarray[samples, 151]
predict(texts) -> list[route result]
```

The PyTorch router additionally exposes normalized encoder representations for
retrieval. The API depends on this contract rather than the training loop, so a
future model can replace the encoder without changing service responses.

## Data and Leakage Boundary

The official CLINC150 `data_full.json` is converted to one DataFrame with
`sample_id`, `text`, `label`, `split`, and `is_oos`. Loading fails if counts,
labels, schema, or per-class sizes differ from the expected dataset contract.

- Training: vocabulary, class parameters, neural weights, knowledge articles.
- Validation: temperature scaling, abstention threshold, early stopping.
- Test: one final routing report and retrieval evaluation only.

Knowledge articles concatenate intent names, training-only keywords, and eight
training examples. Each held-out test query has one relevant article: the
article for its true intent.

## Modeling Decisions

### Multinomial Naive Bayes

This is implemented directly in NumPy to make class priors, Laplace smoothing,
and token likelihoods visible. It provides a fast, meaningful reference point
and catches pipeline problems before deep learning is introduced.

### Mean/max pooling encoder

The network learns 128-dimensional token embeddings. Masked mean pooling
captures overall semantics, while max pooling preserves strong indicator words.
Their concatenation feeds a 256-unit GELU layer and a 151-class head. The same
hidden vector becomes the semantic retrieval representation.

This model was chosen over a Transformer because the goal is to learn and
explain the complete system in ten days. The architecture trains quickly on CPU,
has fewer than half a million parameters, and still delivers a measured lift.

### Calibration and abstention

Temperature is selected by validation negative log likelihood. The abstention
threshold maximizes validation macro F1 while retaining at least 85% in-scope
coverage. This separates prediction from the business decision to accept or
escalate a prediction.

### Hybrid retrieval

TF-IDF handles exact product terms; encoder cosine similarity handles paraphrase;
the predicted route supplies a weak prior. Component scores remain visible in
the response for debugging. The route weight is intentionally small, so the
ranker can recover when classification is wrong.

## Artifact Graph

```text
data/raw/clinc150.json
  -> artifacts/common/{vocabulary,labels}.json
  -> artifacts/baseline/{model,metadata,metrics}
  -> artifacts/neural/{model,metadata,history,metrics}
  -> artifacts/retrieval/{knowledge,index,manifest,metrics}
  -> artifacts/service/benchmark.json
```

Large binary artifacts and raw data are reproducible and gitignored. Small JSON
metadata and measured reports can be version controlled. `/v1/model-info`
returns hashes for the loaded neural checkpoint and retrieval index.

## Serving and Operations

The model and index load once in FastAPI's lifespan hook. Request validation is
performed by Pydantic. A bounded in-memory metrics window publishes request,
abstention, fallback, and latency signals in text format. Feedback is appended
under a lock to JSONL without raw text.

The current benchmark is a single-process CPU measurement. A production rollout
would add authenticated access, rate limiting, structured logs, concurrency and
load testing, a durable feedback sink, shadow evaluation, canary deployment,
and alerting on OOS rate, confidence, latency, and intent drift.

## C++ Backend Decision

`cpp/topk.cpp` implements matrix-vector scoring and partial top-k selection via a
stable C ABI. Python loads it with `ctypes`; parity is tested against NumPy. On
the 10,000 by 256 synthetic benchmark, optimized NumPy/BLAS is faster on this
machine. The service therefore defaults to NumPy. Keeping the measured but
optional path demonstrates responsible benchmarking and cross-language
integration without creating a false optimization claim.

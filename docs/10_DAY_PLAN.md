# ResolveAI Ten-Day Learning Plan

The project deliberately concentrates on four ideas: text representation,
classification, uncertainty, and retrieval. Spend roughly six focused hours per
day: three learning, two coding/explanation, and one interview drill.

## Day 1: Product and data

- Read `README.md`, `data.py`, and the CLINC150 dataset report.
- Learn train/validation/test roles, leakage, class balance, and macro F1.
- Run `resolveai inspect-data`; explain every validation check aloud.
- Interview drill: arrays, hash maps, string tokenization, big-O of vocabulary
  construction.

Deliverable: draw the system and explain why an OOS class is necessary.

## Day 2: NumPy baseline

- Learn Bayes rule, class priors, likelihood, log probabilities, and Laplace
  smoothing.
- Read `baseline.py`; derive one two-class prediction by hand.
- Run the baseline and inspect its top confusions.
- Interview drill: implement softmax and a confusion matrix without libraries.

Deliverable: explain why a baseline is part of engineering, not busywork.

## Day 3: Evaluation and uncertainty

- Learn precision, recall, F1, macro averaging, calibration, NLL, and coverage.
- Read `metrics.py` and compare raw versus abstained metrics.
- Change the threshold locally and describe the false-positive/false-negative
  tradeoff. Restore the configured value afterward.
- Interview drill: design metrics when an incorrect automatic action is costly.

Deliverable: whiteboard temperature scaling and the abstention decision.

## Day 4: PyTorch fundamentals

- Learn tensors, embeddings, forward pass, cross entropy, gradients, AdamW,
  train/eval modes, and batching.
- Read `NeuralTextClassifier` and trace shapes through every operation.
- Re-run training and chart the JSON history in a small pandas session.
- Interview drill: implement masked mean pooling and explain padding.

Deliverable: explain all 475,671 parameters by layer.

## Day 5: Training systems

- Learn deterministic seeds, early stopping, checkpoints, validation selection,
  underfitting, overfitting, and data drift.
- Explain why test data is not used to select temperature or threshold.
- Compare baseline and neural reports and calculate absolute improvements.
- Interview drill: diagnose train loss down, validation loss up.

Deliverable: a three-minute model training walkthrough without notes.

## Day 6: Retrieval

- Learn term frequency, inverse document frequency, cosine similarity, dense
  representations, Recall@k, and MRR.
- Read `knowledge.py` and `retrieval.py`; trace one query's three score parts.
- Run `resolveai evaluate-rag` and explain why hybrid Recall@3 matters.
- Interview drill: design search for exact identifiers plus semantic paraphrases.

Deliverable: derive TF-IDF for two tiny documents by hand.

## Day 7: Grounded generation and safety

- Learn the RAG sequence: retrieve, construct context, generate, validate.
- Read `rag.py`; force an invalid citation in the unit test and observe fallback.
- Explain why the LLM is optional and excluded from measured core results.
- Interview drill: threats from prompt injection, stale documents, and missing
  citations.

Deliverable: explain what this citation guard guarantees and what it does not.

## Day 8: API and software engineering

- Read `service.py` and `api.py`; learn request models, lifecycle loading, HTTP
  status codes, trace IDs, metrics, feedback, and thread safety.
- Use `/docs`, call every endpoint, and inspect `/metrics`.
- Run all tests and classify them as unit, contract, parity, or integration.
- Interview drill: design a model-serving API with versioning and rollback.

Deliverable: explain how you would deploy a new checkpoint without breaking API
clients.

## Day 9: Performance and system design

- Learn batching, p50/p95/p99, throughput versus latency, warmup, and benchmark
  validity.
- Read the C ABI bridge and run both benchmarks.
- Explain why NumPy wins here and when a native/approximate index might win.
- Interview drill: scale the service from one process to one million requests/day.

Deliverable: a system-design diagram with cache, queue, workers, monitoring,
artifact registry, canary, and feedback pipeline.

## Day 10: Presentation and mock interviews

- Rehearse the 30-second, two-minute, and deep-dive pitches in
  `INTERVIEW_GUIDE.md`.
- Start from a clean artifact directory and reproduce the project commands.
- Practice two coding questions, one ML fundamentals round, one project deep
  dive, and one serving-system design.
- Review every resume number against its JSON artifact. Never claim unmeasured
  scale, production users, or LLM quality.

Deliverable: record a ten-minute project presentation and remove every phrase
you cannot defend under follow-up questions.

## Minimum Interview Checklist

You are ready when you can answer these without memorized wording:

- Why macro F1 instead of only accuracy?
- Why does temperature change confidence but not class ranking?
- How was the abstention threshold chosen without test leakage?
- Why did the neural model improve OOS F1?
- What does mean/max pooling capture?
- Why combine lexical and neural retrieval?
- How is Recall@3 calculated?
- What makes an answer grounded here?
- What happens when the LLM fails or cites an unknown document?
- How are artifacts versioned and loaded?
- What would you monitor and how would you roll back?
- Why is the C++ backend optional instead of default?

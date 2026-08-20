# NeuralRouter Model Card

## Summary

NeuralRouter is a compact PyTorch classifier for short English support requests.
It predicts 150 CLINC150 intents plus an out-of-scope class, calibrates
probabilities with validation-only temperature scaling, and abstains below a
validation-selected confidence threshold.

## Intended Use

- Demonstrating end-to-end ML engineering and confidence-aware routing.
- Benchmarking intent classification and OOS detection on CLINC150.
- Supplying a weak routing prior and semantic representation to GroundedAssist.

It is not suitable for production customer support without company-specific
data, policy documents, risk analysis, security controls, and human review.

## Data

- Source: CLINC150 / `oos-eval` `data_full.json`.
- Training: 15,000 in-scope plus 100 OOS requests.
- Validation: 3,000 in-scope plus 100 OOS requests.
- Test: 4,500 in-scope plus 1,000 OOS requests.
- Language: English.
- License: CC BY 3.0.
- Source SHA-256: `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`.

Five normalized texts appear in multiple splits. The dataset report records this
property; no rows are moved because the project preserves the official split.

## Architecture and Training

- Vocabulary: 2,895 tokens, training split only.
- Maximum sequence length: 32 tokens.
- Encoder: 128-dimensional embeddings, masked mean/max pooling, 256-unit GELU.
- Parameters: 475,671.
- Loss and optimizer: cross entropy and AdamW.
- Batch size: 256.
- Seed: 17 with deterministic PyTorch algorithms.
- Selected checkpoint: epoch 24 of 24.
- Calibration temperature: 1.5981.
- Abstention threshold: 0.3681.

## Held-out Results

| Metric | Value |
|---|---:|
| Overall accuracy after abstention | 83.33% |
| In-scope intent accuracy | 86.80% |
| In-scope coverage | 93.98% |
| Macro F1 across 151 classes | 86.25% |
| OOS precision | 71.41% |
| OOS recall | 67.70% |
| OOS F1 | 69.51% |
| Expected calibration error | 1.37% |
| Negative log likelihood | 0.8899 |

The large OOS test set changes the overall accuracy denominator; in-scope
accuracy and OOS metrics are therefore reported separately.

## Risks and Limitations

- CLINC150 queries are short and synthetic compared with real support threads.
- The OOS training set has only 100 examples and cannot represent open-world
  traffic.
- A single confidence threshold does not encode different costs by intent.
- Calibration can drift when vocabulary, language, or request mix changes.
- English tokenization and labels may perform poorly on multilingual or noisy
  text.
- Confidence does not guarantee correctness. High-risk actions still require
  downstream authorization and policy checks.

## Monitoring Plan

Track intent distribution, OOS and abstention rates, confidence histograms,
latency percentiles, corrected-intent feedback, citation fallback rate, and
performance by traffic segment. Recalibrate or retrain only on versioned data,
and compare against the stored baseline before promotion.

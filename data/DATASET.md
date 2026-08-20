# Dataset: CLINC150 / OOS-Eval

ResolveAI downloads the official `data_full.json` file from the CLINC150
out-of-scope evaluation repository and validates the complete split and class
contract before training.

- Repository: <https://github.com/clinc/oos-eval>
- Paper: <https://aclanthology.org/D19-1131/>
- Source file: <https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json>
- License: Creative Commons Attribution 3.0 (CC BY 3.0)
- Retrieved source SHA-256:
  `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0`

The raw file is gitignored and reproducible with `resolveai download-data`.
`data/processed/dataset_report.json` records row counts, label counts,
duplicates, content hash, source, and license.

Expected split sizes:

| Split | In scope | OOS |
|---|---:|---:|
| Train | 15,000 | 100 |
| Validation | 3,000 | 100 |
| Test | 4,500 | 1,000 |

The 150 knowledge articles under `artifacts/retrieval/knowledge.json` are derived
from training examples and remain subject to the dataset's CC BY 3.0 terms. The
project MIT license applies to ResolveAI source code, not to CLINC150 data.

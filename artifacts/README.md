# Artifact Layout

Small JSON metadata, evaluation results, and training history are retained so
README and resume claims can be audited. Large/reproducible binaries are
gitignored and rebuilt locally.

| Directory | Contents |
|---|---|
| `common/` | shared labels and training-only vocabulary |
| `baseline/` | NumPy model metadata and held-out report |
| `neural/` | PyTorch architecture, calibration, history, and report |
| `retrieval/` | training-only knowledge, index manifest, evaluation, native benchmark |
| `service/` | end-to-end CPU serving benchmark |

Rebuild everything after downloading data:

```bash
.venv/bin/resolveai train-all
make cpp benchmark-native test
```

Generated binary files are `baseline/model.npz`, `neural/model.pt`,
`retrieval/index.npz`, and `build/libresolve_topk.so`.

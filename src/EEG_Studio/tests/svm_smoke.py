"""Verifica que el SVM admite varios kernels (sin GUI)."""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.core import classification
from eeg_studio.core.dataset import Dataset


def _ds(n=40, d=6):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (n // 2, d)), rng.normal(3, 1, (n // 2, d))])
    y = np.array(["a"] * (n // 2) + ["b"] * (n // 2), dtype=object)
    return Dataset(X=X, y=y, feature_names=[f"f{i}" for i in range(d)],
                   segment_ids=[str(i) for i in range(n)])


def main() -> int:
    ds = _ds()
    print("Kernels SVM disponibles:", list(classification.SVM_KERNELS))
    for kernel in classification.SVM_KERNELS:
        params = {"kernel": kernel, "C": 1.0, "gamma": "scale", "degree": 3}
        res = classification.train(ds, "svm", clf_params=params)
        used = res.model.named_steps["clf"].kernel
        assert used == kernel, f"esperaba kernel {kernel}, se usó {used}"
        pred = classification.predict(res, ds.X[:2])
        print(f"    kernel={kernel:8s} → entrenado, CV={res.cv_mean:.2f}, pred={pred.tolist()}")
        assert len(pred) == 2

    print("\nSVM (varios kernels) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

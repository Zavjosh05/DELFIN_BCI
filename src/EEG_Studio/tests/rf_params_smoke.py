"""Verifica que el Random Forest aplica los parámetros configurados (sin GUI)."""
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
    params = {"n_estimators": 50, "max_depth": 3, "min_samples_split": 4,
              "max_features": "log2", "criterion": "entropy"}

    print("[1] build_pipeline aplica los parámetros")
    pipe = classification.build_pipeline("random_forest", params)
    rf = pipe.named_steps["clf"]
    assert rf.n_estimators == 50, rf.n_estimators
    assert rf.max_depth == 3, rf.max_depth
    assert rf.min_samples_split == 4, rf.min_samples_split
    assert rf.max_features == "log2", rf.max_features
    assert rf.criterion == "entropy", rf.criterion
    print(f"    n_estimators={rf.n_estimators} max_depth={rf.max_depth} "
          f"criterio={rf.criterion} max_features={rf.max_features}")

    print("[2] max_depth=0 -> sin límite (None)")
    rf0 = classification.build_pipeline("random_forest", {"max_depth": 0}).named_steps["clf"]
    assert rf0.max_depth is None, "0 debería traducirse a None"

    print("[3] Entrena y predice con los parámetros")
    res = classification.train(_ds(), "random_forest", clf_params=params)
    assert res.model.named_steps["clf"].n_estimators == 50
    assert classification.predict(res, _ds().X[:3]).shape == (3,)
    print(f"    CV={res.cv_mean:.2f}, modelo entrenado")

    print("\nPARÁMETROS RANDOM FOREST OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

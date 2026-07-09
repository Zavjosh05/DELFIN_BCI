"""Datos de entrenamiento/evaluación por modelo (CV vs holdout) y su persistencia."""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.core import classification, neuralnet
from eeg_studio.core.dataset import Dataset


def _ds(n=24, feats=6):
    rng = np.random.default_rng(0)
    return Dataset(X=rng.normal(0, 1, (n, feats)).astype(float),
                   y=np.array(["a", "b"] * (n // 2), dtype=object),
                   feature_names=[f"f{i}" for i in range(feats)],
                   segment_ids=[str(i) for i in range(n)])


def main() -> int:
    ds = _ds(24)

    print("[1] Clásico (RF): validación cruzada")
    rf = classification.train(ds, "random_forest")
    assert rf.n_samples == 24 and rf.eval_method == "cross_val" and rf.cv_folds >= 2, (
        rf.n_samples, rf.eval_method, rf.cv_folds)
    assert "Validación cruzada" in rf.split_report(), rf.split_report()
    print(f"    {rf.split_report()}")

    print("[2] Red (MLP): holdout 75/25")
    if neuralnet.torch_available():
        cfg = neuralnet.default_config("mlp")
        cfg["epochs"] = 3
        nn = classification.train(ds, "nn_mlp", nn_config=cfg)
        assert nn.eval_method == "holdout", nn.eval_method
        assert nn.n_train + nn.n_eval == 24 and nn.n_eval == 6 and nn.n_train == 18, (
            nn.n_train, nn.n_eval)
        assert "holdout" in nn.split_report()
        print(f"    {nn.split_report()}")

        print("[3] Los campos se conservan al serializar (bundle/joblib)")
        r2 = classification.result_from_bytes(classification.result_to_bytes(nn))
        assert r2.eval_method == "holdout" and r2.n_train == 18 and r2.n_eval == 6
        print("    n_train/n_eval/eval_method persistidos ✓")
    else:
        print("    (sin torch; se omite)")

    print("\nDATOS DE ENTRENAMIENTO/EVALUACIÓN OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

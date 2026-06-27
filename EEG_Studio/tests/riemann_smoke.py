"""Prueba de los clasificadores de geometría de Riemann y CSP (sin GUI).

Usa datos separables **espacialmente** (distinta covarianza por clase), que es
lo que captan estos métodos, y comprueba que aprenden.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.core import classification
from eeg_studio.core.dataset import RawDataset


def _spatial_dataset(n=60, ch=14, T=256):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (n, ch, T))
    y = []
    for i in range(n):
        if i % 2 == 0:
            X[i, :7, :] *= 4.0      # clase A: más potencia en la mitad delantera
            y.append("A")
        else:
            X[i, 7:, :] *= 4.0      # clase B: más potencia en la mitad trasera
            y.append("B")
    return RawDataset(X=X, y=np.array(y, dtype=object), segment_ids=[str(i) for i in range(n)])


def main() -> int:
    assert classification.riemann_available(), "pyriemann no disponible"
    ds = _spatial_dataset()
    for name in ("riemann_mdm", "riemann_ts", "csp_lda"):
        res = classification.train_riemann(ds, name)
        pred = classification.predict(res, ds.X[:2])
        print(f"    {name:12s} CV={res.cv_mean:.3f}  input_kind={res.input_kind}  pred={pred.tolist()}")
        assert res.input_kind == "raw"
        assert res.cv_mean > 0.7, f"{name} no aprendió (CV={res.cv_mean:.3f})"

    print("\nRIEMANN / CSP OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

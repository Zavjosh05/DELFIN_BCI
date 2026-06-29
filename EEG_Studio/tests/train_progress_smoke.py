"""Progreso por épocas al entrenar una red y vaciado de segmentos."""
from __future__ import annotations

import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.core import classification, neuralnet
from eeg_studio.core.dataset import Dataset
from eeg_studio.core.project import Project


def main() -> int:
    if not neuralnet.torch_available():
        print("PyTorch no disponible; se omite la prueba de progreso.")
    else:
        print("[1] El entrenamiento de la MLP reporta progreso por época")
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (24, 8)).astype(np.float32)
        y = np.array(["a", "b"] * 12, dtype=object)
        ds = Dataset(X=X, y=y, feature_names=[f"f{i}" for i in range(8)],
                     segment_ids=[str(i) for i in range(24)])
        cfg = neuralnet.default_config("mlp")
        cfg["epochs"] = 5
        seen = []
        res = classification.train(ds, "nn_mlp", nn_config=cfg,
                                   progress=lambda d, t: seen.append((d, t)))
        assert seen, "no se reportó progreso"
        assert seen[-1] == (5, 5), seen[-1]
        assert [d for d, _ in seen] == [1, 2, 3, 4, 5], seen
        assert res.classifier_name == "nn_mlp"
        print(f"    progreso = {seen}")

    print("[2] clear_segments elimina todos de una vez")
    tmp = tempfile.mkdtemp()
    proj = Project.create(tmp, "vaciar")
    # Inyecta segmentos directamente en el estado y confirma.
    proj.state["segments"] = [
        {"id": "s1", "source_id": "x", "start": 0, "stop": 10, "label": "a", "channels": None},
        {"id": "s2", "source_id": "x", "start": 10, "stop": 20, "label": "b", "channels": None},
        {"id": "s3", "source_id": "x", "start": 20, "stop": 30, "label": "a", "channels": None},
    ]
    n = proj.clear_segments()
    assert n == 3, n
    assert proj.state["segments"] == [], proj.state["segments"]
    print(f"    {n} segmentos eliminados de golpe")

    print("[3] Se puede deshacer (Ctrl+Z) tras vaciar")
    assert proj.undo() is True
    assert len(proj.state["segments"]) == 3, len(proj.state["segments"])
    print("    deshacer restauró los 3 segmentos")

    print("\nPROGRESO DE ENTRENAMIENTO Y VACIADO OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

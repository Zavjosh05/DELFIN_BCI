"""Prueba de humo de las redes neuronales (sin GUI).

Entrena MLP (características), CNN 1D y LSTM (señal cruda) con datos sintéticos
separables y verifica que aprenden y predicen. Comprueba también guardar/cargar.
"""
from __future__ import annotations

import sys
import tempfile
import os

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.core import classification, neuralnet
from eeg_studio.core.dataset import Dataset, RawDataset


def _features(n=40, d=8):
    rng = np.random.default_rng(0)
    Xa = rng.normal(0, 1, (n // 2, d))
    Xb = rng.normal(3, 1, (n // 2, d))
    X = np.vstack([Xa, Xb])
    y = np.array(["a"] * (n // 2) + ["b"] * (n // 2), dtype=object)
    return Dataset(X=X, y=y, feature_names=[f"f{i}" for i in range(d)],
                   segment_ids=[str(i) for i in range(n)])


def _raw(n=40, ch=4, T=128):
    rng = np.random.default_rng(1)
    t = np.arange(T) / 128.0
    X = np.zeros((n, ch, T))
    y = []
    for i in range(n):
        freq = 6.0 if i % 2 == 0 else 18.0
        X[i] = np.sin(2 * np.pi * freq * t)[None, :] + rng.normal(0, 0.3, (ch, T))
        y.append("lenta" if i % 2 == 0 else "rapida")
    return RawDataset(X=X, y=np.array(y, dtype=object), segment_ids=[str(i) for i in range(n)])


def main() -> int:
    assert neuralnet.torch_available(), "PyTorch no disponible"

    print("[1] MLP sobre características")
    cfg = neuralnet.default_config("mlp"); cfg["epochs"] = 30
    res = classification.train(_features(), "nn_mlp", nn_config=cfg)
    acc = res.cv_mean
    print(f"    {res.score_label}: {acc:.3f}  clases={res.classes}")
    assert res.input_kind == "features"
    assert acc > 0.7, "el MLP no aprendió"

    print("[2] CNN 1D sobre señal cruda")
    cfg = neuralnet.default_config("cnn"); cfg["epochs"] = 25; cfg["window_samples"] = 128
    res_cnn = classification.train_raw(_raw(), "nn_cnn", nn_config=cfg)
    print(f"    {res_cnn.score_label}: {res_cnn.cv_mean:.3f}")
    assert res_cnn.input_kind == "raw"
    assert res_cnn.cv_mean > 0.7, "la CNN no aprendió"

    print("[3] LSTM sobre señal cruda")
    cfg = neuralnet.default_config("lstm"); cfg["epochs"] = 25; cfg["window_samples"] = 128
    res_lstm = classification.train_raw(_raw(), "nn_lstm", nn_config=cfg)
    print(f"    {res_lstm.score_label}: {res_lstm.cv_mean:.3f}")
    assert res_lstm.cv_mean > 0.6, "la LSTM no aprendió"

    print("[3b] EEGNet sobre señal cruda")
    cfg = neuralnet.default_config("eegnet"); cfg["epochs"] = 20; cfg["window_samples"] = 128
    res_eeg = classification.train_raw(_raw(T=128), "nn_eegnet", nn_config=cfg)
    print(f"    {res_eeg.score_label}: {res_eeg.cv_mean:.3f}")
    assert res_eeg.input_kind == "raw"
    assert res_eeg.cv_mean > 0.6, "EEGNet no aprendió"

    print("[4] Guardar y cargar el modelo CNN")

    class _P:  # proyecto mínimo para save_model
        path = tempfile.mkdtemp()
    os.makedirs(os.path.join(_P.path, "models"), exist_ok=True)
    path = classification.save_model(_P, res_cnn, "nn_cnn")
    loaded = classification.load_model(path)
    pred = classification.predict(loaded, _raw().X[:2])
    print(f"    cargado input_kind={loaded.input_kind} pred={pred.tolist()}")
    assert loaded.input_kind == "raw"

    print("\nREDES NEURONALES OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prueba del modo de control en línea (núcleo, sin GUI):
clasificación de ventana, suavizado y salidas de comandos (log y UDP).
"""
from __future__ import annotations

import os
import socket
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.core import classification
from eeg_studio.core.dataset import Dataset
from eeg_studio.core.processing import extract_feature_vector
from eeg_studio.core.project import Project
from eeg_studio.inference import LogSink, PredictionSmoother, UdpSink, classify_window, make_sink


def _window(kind, ch=14, T=256, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 128.0
    f = 8.0 if kind == "A" else 20.0
    return np.sin(2 * np.pi * f * t)[None, :] * np.ones((ch, 1)) + rng.normal(0, 0.3, (ch, T))


def main() -> int:
    print("[1] Modelo entrenado con características de ventanas de 14 canales")
    X, y = [], []
    for i in range(40):
        kind = "A" if i % 2 == 0 else "B"
        vec, _ = extract_feature_vector(_window(kind, seed=i), 128.0, True, True)
        X.append(vec); y.append(kind)
    ds = Dataset(np.vstack(X), np.array(y, dtype=object),
                 feature_names=[f"f{i}" for i in range(len(X[0]))],
                 segment_ids=[str(i) for i in range(40)])
    model = classification.train(ds, "random_forest")

    print("[2] classify_window clasifica ventanas nuevas")
    proj = Project.create(tempfile.mkdtemp(), "on")   # sin pipeline
    pred_a, conf_a = classify_window(model, proj, _window("A", seed=100), 128.0)
    pred_b, _ = classify_window(model, proj, _window("B", seed=101), 128.0)
    print(f"    A->{pred_a} ({conf_a}), B->{pred_b}")
    assert pred_a == "A" and pred_b == "B", "la clasificación en línea falla"

    print("[3] Suavizado: confirma una clase tras K iguales")
    sm = PredictionSmoother(k=3)
    assert sm.update("A") is None and sm.update("A") is None
    assert sm.update("A") == "A", "no confirmó tras 3 iguales"
    assert sm.update("A") is None, "no debe repetir la misma clase estable"
    assert sm.update("B") is None and sm.update("B") is None
    assert sm.update("B") == "B", "no confirmó el cambio de clase"
    print("    suavizado OK")

    print("[4] Salida LogSink y UDP")
    log = make_sink("log")
    assert isinstance(log, LogSink)
    log.send("L"); log.send("R")
    assert log.history == ["L", "R"], log.history

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(1.0)
    port = rx.getsockname()[1]
    udp = UdpSink("127.0.0.1", port)
    udp.send("adelante")
    data, _ = rx.recvfrom(64)
    assert data.decode().strip() == "adelante", data
    udp.close(); rx.close()
    print(f"    LogSink y UDP (puerto {port}) OK")

    print("\nCONTROL EN LÍNEA (núcleo) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

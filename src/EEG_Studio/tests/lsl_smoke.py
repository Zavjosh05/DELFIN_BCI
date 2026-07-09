"""Prueba de bucle LSL en proceso (sin OpenViBE).

Crea un outlet LSL llamado como el del Acquisition Server, empuja muestras y
comprueba que :class:`LSLSource` lo resuelve y recibe datos. Valida la ruta LSL.
"""
from __future__ import annotations

import sys
import threading
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
from pylsl import StreamInfo, StreamOutlet

from eeg_studio.acquisition import LSLSource, pylsl_available
from eeg_studio.config import EPOC_CHANNELS, LSL_SIGNAL_NAME


def _publisher(stop: threading.Event) -> None:
    info = StreamInfo(LSL_SIGNAL_NAME, "signal", 14, 128, "float32", "uid-test")
    chans = info.desc().append_child("channels")
    for name in EPOC_CHANNELS:
        chans.append_child("channel").append_child_value("label", name)
    outlet = StreamOutlet(info)
    rng = np.random.default_rng(0)
    while not stop.is_set():
        outlet.push_sample((4200 + rng.normal(0, 5, 14)).tolist())
        time.sleep(1 / 128)


def main() -> int:
    assert pylsl_available(), "pylsl no disponible"
    print("[1] Publicando outlet LSL de prueba")
    stop = threading.Event()
    pub = threading.Thread(target=_publisher, args=(stop,), daemon=True)
    pub.start()
    time.sleep(0.5)

    print("[2] Conectando LSLSource y leyendo ~1 s")
    src = LSLSource()
    src.start()
    total = 0
    t_end = time.time() + 1.2
    while time.time() < t_end:
        chunk = src.read()
        if chunk is not None:
            total += chunk.shape[1]
        time.sleep(0.03)
    src.stop()
    stop.set()

    assert src.error is None, f"error en LSLSource: {src.error}"
    print(f"    canales={src.n_channels} fs={src.sample_rate} muestras={total}")
    assert src.n_channels == 14, "nº de canales inesperado"
    assert total > 64, "se recibieron muy pocas muestras"
    assert src.channel_names[0] == EPOC_CHANNELS[0], "etiquetas de canal no leídas"

    print("\nLSL OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prueba de humo de la adquisición (sin GUI, sin hardware).

Arranca la fuente simulada, graba ~1 s a un CSV formato OpenViBE, lo recarga
con el loader del proyecto y comprueba que el formato es consistente.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from eeg_studio.acquisition import CSVRecorder, SimulatedSource
from eeg_studio.core.csv_loader import load_recording


def main() -> int:
    print("[1] Arrancando fuente simulada")
    src = SimulatedSource()
    src.start()

    tmp = tempfile.mkdtemp()
    csv_path = os.path.join(tmp, "live.csv")
    rec = CSVRecorder(csv_path, src.n_channels, src.sample_rate)

    print("[2] Capturando y grabando ~1 s")
    total = 0
    t_end = time.time() + 1.0
    while time.time() < t_end:
        chunk = src.read()
        if chunk is not None:
            if total == 0:  # un marcador al principio
                rec.add_marker("inicio")
            total += rec.write(chunk)
        time.sleep(0.03)
    chunk = src.read()
    if chunk is not None:
        total += rec.write(chunk)

    src.stop()
    rec.close()
    print(f"    muestras grabadas: {total} (esperado ~128)")
    assert src.error is None, f"error en la fuente: {src.error}"
    assert total > 64, "se grabaron muy pocas muestras"

    print("[3] Recargando el CSV grabado con el loader")
    loaded = load_recording(csv_path)
    print(f"    canales={loaded.n_channels} muestras={loaded.n_samples} "
          f"fs={loaded.sample_rate} eventos={len(loaded.events)}")
    assert loaded.n_channels == src.n_channels, "nº de canales no coincide"
    assert loaded.sample_rate == src.sample_rate, "fs no coincide"
    assert len(loaded.events) >= 1, "no se registró el marcador"

    print("\nADQUISICIÓN OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

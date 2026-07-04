"""Escala del visor en vivo: fija (µV) por defecto + auto (normalizada) como opción."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.ui.live_view import LiveSignalView  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841
    names = ["A", "B", "C", "D"]
    n = len(names)
    live = LiveSignalView()
    live.configure(names, 128.0, window_seconds=3.0)

    # Señal con offset DC grande (~4200) por canal + pequeña variación.
    rng = np.random.default_rng(0)
    buf = np.stack([4200 + i * 10 + rng.normal(0, 5, 300) for i in range(n)])

    print("[1] Por defecto: escala FIJA (µV), selector de µV activo")
    assert live.scale_box.currentIndex() == 0, "la escala fija debe ser la de por defecto"
    assert live.uv_box.isEnabled() and live.uv_box.currentText() == "200"

    print("[2] Fija: cada canal centrado en offset = (n-1-i)·µV (µV reales, sin offset DC)")
    live.append(buf)
    for i in range(n):
        _, y = live._curves[i].getData()
        assert abs(float(y.mean()) - (n - 1 - i) * 200) < 5, (i, float(y.mean()))
    print("    canales separados 200 µV, sin el offset DC de ~4200")

    print("[3] Cambiar a AUTO (normalizada): separación en unidades z, µV desactivado")
    live.scale_box.setCurrentIndex(1)
    assert not live.uv_box.isEnabled(), "en auto no se usa el selector de µV"
    for i in range(n):
        _, y = live._curves[i].getData()
        assert abs(float(y.mean()) - (n - 1 - i) * live._spacing) < 1, (i, float(y.mean()))
    print("    normalización por canal (amplitud uniforme)")

    print("[4] Volver a fija y cambiar la escala a 500 µV")
    live.scale_box.setCurrentIndex(0)
    live.uv_box.setCurrentText("500")
    live.append(buf)
    _, y0 = live._curves[0].getData()
    assert abs(float(y0.mean()) - (n - 1) * 500) < 10, float(y0.mean())
    print("    la escala fija es ajustable (200 → 500 µV)")

    print("[5] La opción auto sigue disponible (no se eliminó)")
    labels = [live.scale_box.itemText(i) for i in range(live.scale_box.count())]
    assert "Fija (µV)" in labels and "Auto (normalizada)" in labels, labels

    print("\nESCALA DEL VISOR EN VIVO OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

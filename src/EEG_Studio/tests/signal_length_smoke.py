"""Fijar la longitud (en tiempo) de la región de selección del visor. Offscreen."""
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

from eeg_studio.ui.signal_view import SignalView  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841
    view = SignalView()
    fs = 128.0
    data = np.random.randn(4, 1280)       # 4 canales, 10 s @128 Hz
    view.set_data(data, fs, ["A", "B", "C", "D"])

    print("[1] Al cargar datos, el campo de longitud se habilita y refleja la selección")
    assert view.len_spin.isEnabled()
    assert abs(view.len_spin.maximum() - 10.0) < 0.01, view.len_spin.maximum()
    assert view.len_spin.value() > 0

    print("[2] Fijar longitud=5 s → la región (y las muestras) miden 5 s")
    view.len_spin.setValue(5.0)
    s0, s1 = view.selection_samples()
    assert abs((s1 - s0) - int(5 * fs)) <= 2, (s0, s1)
    lo, hi = view.region.getRegion()
    assert abs((hi - lo) - 5.0) < 0.02, (lo, hi)

    print("[3] Arrastrar la región actualiza el campo (sincronización inversa)")
    view.region.setRegion((1.0, 3.5))     # simula un arrastre a 2.5 s
    assert abs(view.len_spin.value() - 2.5) < 0.05, view.len_spin.value()

    print("[4] Si la longitud no cabe al final, corre el inicio (mantiene la duración)")
    view.region.setRegion((8.0, 8.5))     # cerca del final (señal de 10 s)
    view.len_spin.setValue(5.0)           # 8+5=13 > 10 → debe correr el inicio a 5
    lo, hi = view.region.getRegion()
    assert hi <= 10.0 + 1e-6 and abs((hi - lo) - 5.0) < 0.05, (lo, hi)
    assert abs(lo - 5.0) < 0.05, lo

    print("[5] Sin datos, cambiar la longitud no rompe")
    empty = SignalView()
    assert not empty.len_spin.isEnabled()
    empty.len_spin.setValue(3.0)          # no debe lanzar

    print("\nLONGITUD DE LA SELECCIÓN (VISOR) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

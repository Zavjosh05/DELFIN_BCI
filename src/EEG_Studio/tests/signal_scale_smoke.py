"""Visor de CSV: se resta el offset DC para visualizar (escala centrada, no en 0)."""
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
    view.show()

    fs, ns = 128.0, 500
    t = np.arange(ns) / fs
    # Dos canales con OFFSET DC grande (~4200 y ~100 µV) + una señal de ±30 µV.
    d0 = 4200.0 + 30.0 * np.sin(2 * np.pi * 10 * t)
    d1 = 100.0 + 30.0 * np.sin(2 * np.pi * 10 * t)
    data = np.vstack([d0, d1])

    print("[1] Multicanal (sin normalizar): las curvas se centran (offset DC quitado)")
    view.norm_chk.setChecked(False)
    view.set_data(data, fs, ["A", "B"])
    curves = view.plot.listDataItems()
    assert len(curves) == 2, len(curves)
    means = [abs(float(c.getData()[1].mean())) for c in curves]
    # Si NO se quitara el DC, un canal se dibujaría en ~4200; con el fix, ninguno.
    assert max(means) < 300, means
    print(f"    medias de las curvas: {[round(m,1) for m in means]} (ninguna ~4200)")

    print("[2] Canal aislado: se centra en 0 (deviación en µV), no arranca en ~4200")
    view.channel_box.setCurrentIndex(1)      # aísla el canal A (DC ~4200)
    iso_curves = view.plot.listDataItems()
    assert len(iso_curves) == 1, len(iso_curves)
    y = iso_curves[0].getData()[1]
    assert abs(float(y.mean())) < 50, float(y.mean())
    assert float(y.max()) < 200 and float(y.min()) > -200, (float(y.min()), float(y.max()))
    # Las MEDIDAS de texto siguen sobre los datos REALES (media ~4200, con su offset).
    assert "4200" in view.stats_label.text(), view.stats_label.text()
    print(f"    canal aislado centrado (media≈{float(y.mean()):.1f}) · medidas reales OK")

    print("[3] Apartado de escalas: fijar rango X (tiempo) e Y (amplitud) manualmente")
    view.channel_box.setCurrentIndex(0)          # vuelve a multicanal
    view.xstart_spin.setValue(1.0); view.xwin_spin.setValue(3.0)   # ventana [1, 4] s
    (x0, x1), _ = view.plot.getViewBox().viewRange()
    assert abs(x0 - 1.0) < 0.05 and abs(x1 - 4.0) < 0.05, (x0, x1)
    view.ymin_spin.setValue(-10.0); view.ymax_spin.setValue(20.0)  # rango Y [-10, 20]
    _, (y0, y1) = view.plot.getViewBox().viewRange()
    assert abs(y0 + 10.0) < 0.1 and abs(y1 - 20.0) < 0.1, (y0, y1)
    view._autoscale_axes()                                        # auto: cubre toda la señal
    dur = view._data.shape[1] / fs                                # ~3.9 s
    assert view.xwin_spin.value() > dur * 0.9    # la ventana pasa a cubrir toda la señal
    print("    rango X/Y manual y auto-ajuste OK")

    print("\nESCALA DEL VISOR CSV (SIN OFFSET DC) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

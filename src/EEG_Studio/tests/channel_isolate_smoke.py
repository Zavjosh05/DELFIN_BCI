"""Aislar un canal en el visor (CSV) y en el visor en vivo, con sus medidas.

Sin pantalla (offscreen). Verifica que al elegir un canal se dibuja solo ese y
que la etiqueta de medidas (mín/máx/media/σ/rango) aparece.
"""
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
from eeg_studio.ui.signal_view import SignalView  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841
    names = ["AF3", "F7", "F3", "FC5"]
    rng = np.random.default_rng(0)
    data = rng.normal(0, 20, (len(names), 500)).astype(np.float64)

    print("[1] Visor CSV: 'Todos' dibuja todos los canales")
    view = SignalView()
    view.show()                            # isVisible() requiere estar mostrado
    view.set_data(data, 128.0, names)
    assert view.channel_box.count() == len(names) + 1, view.channel_box.count()
    assert len(view.plot.listDataItems()) == len(names), "no dibuja todos los canales"
    assert not view.stats_label.isVisible(), "no debería haber medidas en 'Todos'"

    print("[2] Aislar un canal: solo esa curva + medidas visibles")
    view.channel_box.setCurrentIndex(3)   # 'F3' (índice de canal 2)
    assert len(view.plot.listDataItems()) == 1, "debería dibujar un único canal"
    assert view.stats_label.isVisible(), "las medidas deben mostrarse"
    txt = view.stats_label.text()
    assert "F3" in txt and "µV" in txt and "pico-a-pico" in txt, txt
    print(f"    {txt}")

    print("[3] Volver a 'Todos' oculta las medidas y redibuja todo")
    view.channel_box.setCurrentIndex(0)
    assert len(view.plot.listDataItems()) == len(names)
    assert not view.stats_label.isVisible()

    print("[4] Visor en vivo: aislar canal muestra medidas en vivo")
    live = LiveSignalView()
    live.show()
    live.configure(names, 128.0, window_seconds=3.0)
    live.append(rng.normal(0, 15, (len(names), 200)).astype(np.float64))
    assert not live.stats_label.isVisible()
    live.channel_box.setCurrentIndex(2)   # 'F7'
    live.append(rng.normal(0, 15, (len(names), 64)).astype(np.float64))
    assert live.stats_label.isVisible(), "las medidas en vivo deben mostrarse"
    assert live._curves[1].isVisible() and not live._curves[0].isVisible()
    ltxt = live.stats_label.text()
    assert "F7" in ltxt and "µV" in ltxt, ltxt
    print(f"    {ltxt}")

    print("\nAISLAR CANAL (CSV + EN VIVO) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

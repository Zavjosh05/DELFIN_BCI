"""Barra de controles reacomodable (FlowLayout) del visor de señal y del visor en vivo.

El problema que corrige: con muchos botones/configuraciones, la barra superior del
visor se recortaba/desbordaba. Ahora los controles se reparten en varias filas
según el ancho (FlowLayout) y hay un botón para expandir/compactar la barra.

Cubre:
  * FlowLayout envuelve los ítems a varias filas al reducir el ancho (heightForWidth
    crece cuando el ancho baja).
  * SignalView y LiveSignalView se construyen, pueden encogerse (ancho mínimo 0) y
    su botón expandir/compactar cambia la altura de la barra.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from eeg_studio.ui.flow_layout import FlowLayout
from eeg_studio.ui.live_view import LiveSignalView
from eeg_studio.ui.signal_view import SignalView

EPOC = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
        "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def main() -> int:
    app = QApplication(sys.argv)

    print("[1] FlowLayout: al reducir el ancho, usa MÁS filas (más alto)")
    host = QWidget()
    flow = FlowLayout(h_spacing=6, v_spacing=4)
    host.setLayout(flow)
    for i in range(12):
        b = QPushButton(f"Botón {i}")
        b.setFixedSize(90, 26)
        flow.addWidget(b)
    h_wide = flow.heightForWidth(1200)     # cabe casi todo en una fila
    h_narrow = flow.heightForWidth(200)    # obliga a muchas filas
    print(f"    alto @1200px={h_wide}  ·  alto @200px={h_narrow}")
    assert h_narrow > h_wide, (h_wide, h_narrow)

    print("[2] SignalView: se construye, se puede encoger y expande/compacta")
    sv = SignalView()
    assert sv._controls_scroll.minimumWidth() == 0, "el visor no podría encogerse"
    compact = sv._controls_scroll.maximumHeight()
    sv.expand_btn.setChecked(True)
    expanded = sv._controls_scroll.maximumHeight()
    assert expanded > compact, (compact, expanded)
    sv.expand_btn.setChecked(False)
    assert sv._controls_scroll.maximumHeight() == compact
    data = np.random.default_rng(0).normal(0, 1, (14, 1024))
    sv.set_data(data, 128.0, EPOC)          # dibuja sin romperse con la nueva barra
    assert sv.channel_box.count() == 15
    print(f"    compacta={compact}px  expandida={expanded}px  ·  dibujo OK")

    print("[3] LiveSignalView: misma barra reacomodable + expandir/compactar")
    lv = LiveSignalView()
    assert lv._controls_scroll.minimumWidth() == 0
    c2 = lv._controls_scroll.maximumHeight()
    lv.expand_btn.setChecked(True)
    assert lv._controls_scroll.maximumHeight() > c2
    lv.configure(EPOC, 128.0, 5.0)
    lv.append(np.random.default_rng(1).normal(0, 1, (14, 64)))
    print("    visor en vivo OK (configura + append con la barra nueva)")

    print("[4] Cada chip mantiene juntos su etiqueta y su control")
    # Un chip es un QWidget con un QLabel + su control en un layout horizontal.
    chip_widgets = [sv.mode_box.parentWidget()]
    for w in chip_widgets:
        assert isinstance(w, QWidget)
        kids = w.findChildren(QLabel)
        assert kids, "el chip debería incluir su etiqueta"
    print("    etiquetas pegadas a su control (no se separan al envolver)")

    print("\nBARRA DE CONTROLES REACOMODABLE OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

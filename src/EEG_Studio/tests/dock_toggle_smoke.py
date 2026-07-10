"""El visor de señal se puede encoger y la barra de actividad (estilo PyCharm)
despliega/colapsa los paneles. Offscreen."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402
from eeg_studio.ui.signal_view import SignalView  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841

    print("[1] El visor de señal ya NO impone un ancho mínimo enorme (se encoge)")
    v = SignalView()
    v.set_data(np.random.randn(14, 1280), 128.0, [f"C{i}" for i in range(14)])
    assert v.minimumSizeHint().width() < 200, v.minimumSizeHint().width()
    v.resize(160, 300)
    assert v.width() == 160, v.width()      # se puede hacer pequeño

    print("[2] Barra de actividad: un botón por panel, que lo despliega/colapsa")
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "d")
    win.show(); app.processEvents()
    bar = win._activity_bar
    assert len(bar.actions()) == 3, len(bar.actions())
    dock = win.right_dock                    # panel «Herramientas»
    act = dock.toggleViewAction()
    before = dock.isVisible()
    act.trigger(); app.processEvents()
    assert dock.isVisible() != before, "el botón no cambió la visibilidad del panel"
    act.trigger(); app.processEvents()
    assert dock.isVisible() == before, "el botón no restauró la visibilidad"

    win.acq_panel.shutdown()
    print("\nVISOR ENCOGIBLE + BARRA DE ACTIVIDAD OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

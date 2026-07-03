"""Centro multi-fuente: varias señales como PESTAÑAS en una sola vista.

Offscreen. Verifica que abrir dos fuentes crea dos pestañas, que cambiar de
pestaña cambia la fuente activa, que se pueden cerrar y reabrir, y que sigue
existiendo la opción de desacoplar en ventana aparte.
"""
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

from eeg_studio.acquisition import CSVRecorder  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def _csv(path, val):
    rec = CSVRecorder(path, 14, 128.0)
    rec.add_marker("A")
    rec.write(np.full((14, 40), float(val)))
    rec.close()


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841
    tmp = tempfile.mkdtemp()
    a, b = os.path.join(tmp, "a.csv"), os.path.join(tmp, "b.csv")
    _csv(a, 0.0)
    _csv(b, 1.0)

    win = MainWindow()
    win.project = Project.create(tmp, "tabs")
    sid_a = win.project.add_source(a)["id"]
    sid_b = win.project.add_source(b)["id"]
    win.refresh_all()

    print("[1] Abrir la 1ª fuente crea una pestaña")
    win._on_source_selected(0)
    assert win._signal_tabs.count() == 1, win._signal_tabs.count()
    assert win.current_source_id == sid_a

    print("[2] Abrir la 2ª fuente crea una 2ª pestaña (no reemplaza)")
    win._on_source_selected(1)
    assert win._signal_tabs.count() == 2, win._signal_tabs.count()
    assert win.current_source_id == sid_b

    print("[3] Cambiar de pestaña cambia la fuente activa")
    win._signal_tabs.setCurrentIndex(0)
    assert win.current_source_id == sid_a, win.current_source_id
    assert win.signal_view is win._source_views[sid_a]

    print("[4] Reabrir una fuente ya abierta la enfoca (sin duplicar)")
    win._on_source_selected(1)
    assert win._signal_tabs.count() == 2
    assert win.current_source_id == sid_b

    print("[5] Cerrar una pestaña la quita; se puede reabrir")
    win._close_source_tab(win._signal_tabs.indexOf(win._source_views[sid_b]))
    assert win._signal_tabs.count() == 1 and sid_b not in win._source_views
    win._on_source_selected(1)
    assert win._signal_tabs.count() == 2 and sid_b in win._source_views

    print("[6] Sigue existiendo desacoplar en ventana aparte")
    win.open_source_window(sid_a)
    assert len(win._signal_windows) == 1
    for w in list(win._signal_windows):
        w.close()

    print("[7] Al abrir otro proyecto se cierran las pestañas de fuentes")
    win._reset_source_tabs()
    assert win._signal_tabs.count() == 0 and not win._source_views

    win.acq_panel.shutdown()
    print("\nCENTRO MULTI-FUENTE (PESTAÑAS) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

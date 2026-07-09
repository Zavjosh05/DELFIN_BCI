"""Verifica que la selección del pipeline sigue al paso al mover/eliminar (offscreen)."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from PyQt6.QtWidgets import QApplication

from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow


def _add(panel, key) -> None:
    panel.step_combo.setCurrentIndex(panel.step_combo.findData(key))
    panel._add_step()


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "sel")
    panel = win.preproc_panel

    print("[1] Al añadir, se selecciona el paso nuevo")
    _add(panel, "bandpass")
    _add(panel, "notch")
    _add(panel, "car")
    assert panel.steps_list.currentRow() == 2, "el último añadido debería estar seleccionado"

    print("[2] Mover hacia abajo: la selección sigue al paso")
    panel.steps_list.setCurrentRow(0)             # selecciona 'bandpass'
    panel._move(1)
    assert panel.steps_list.currentRow() == 1, "la selección no siguió al mover"
    assert win.project.state["pipeline"][1]["type"] == "bandpass", "no se movió el paso correcto"

    panel._move(1)                                # bandpass: 1 -> 2
    assert panel.steps_list.currentRow() == 2
    assert win.project.state["pipeline"][2]["type"] == "bandpass"
    print("    'bandpass' sigue seleccionado en su nueva posición")

    print("[3] Eliminar: queda un paso seleccionado")
    panel._remove()                               # elimina el de la fila 2
    assert len(win.project.state["pipeline"]) == 2, "no se eliminó"
    assert 0 <= panel.steps_list.currentRow() < 2, "no quedó nada seleccionado"
    assert panel.steps_list.selectedItems(), "ningún elemento resaltado"
    print(f"    fila seleccionada tras eliminar: {panel.steps_list.currentRow()}")

    print("\nSELECCIÓN DEL PIPELINE OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

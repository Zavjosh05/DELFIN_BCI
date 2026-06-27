"""Verifica el editor de parámetros de filtros (offscreen).

* Se puede tantear un parámetro (cambiar y aplicar) y queda en el pipeline.
* Un paso sin parámetros (CAR) muestra el aviso correspondiente.
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

from PyQt6.QtWidgets import QApplication, QDoubleSpinBox, QLabel  # noqa: E402

from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def _form_has_label(form, needle: str) -> bool:
    for i in range(form.rowCount()):
        for role in (form.ItemRole.FieldRole, form.ItemRole.SpanningRole, form.ItemRole.LabelRole):
            item = form.itemAt(i, role)
            if item and isinstance(item.widget(), QLabel) and needle in item.widget().text():
                return True
    return False


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "p")
    win.project.add_pipeline_step("bandpass", {"low": 1.0, "high": 45.0, "order": 4})
    win.project.add_pipeline_step("car")  # sin parámetros
    win.refresh_all()

    panel = win.preproc_panel

    print("[1] Tantear el parámetro 'low' del pasa-banda")
    panel.steps_list.setCurrentRow(0)  # dispara _show_params
    assert "low" in panel._param_widgets, "no se generó editor para 'low'"
    w = panel._param_widgets["low"]
    assert isinstance(w, QDoubleSpinBox)
    w.setValue(8.0)
    panel._apply_params(0)
    got = win.project.state["pipeline"][0]["params"]["low"]
    assert got == 8.0, f"el cambio no se aplicó (low={got})"
    print(f"    low aplicado correctamente: {got}")

    print("[2] CAR debe indicar que no tiene parámetros")
    panel.steps_list.setCurrentRow(1)
    assert panel._param_widgets == {}, "CAR no debería tener editores"
    assert _form_has_label(panel.params_form, "no tiene parámetros"), \
        "falta el aviso 'no tiene parámetros configurables'"
    print("    aviso de 'sin parámetros' mostrado")

    print("\nPARÁMETROS OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

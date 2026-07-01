"""Excluir EOG por defecto al importar .mat, conservando las etiquetas (offscreen)."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from eeg_studio.config import APP_NAME, IMPORTED_DIR, ORG_NAME
from eeg_studio.core.mat_loader import write_openvibe_csv
from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)

    print("[1] La opción está activada por defecto")
    s = QSettings(ORG_NAME, APP_NAME)
    backup = s.value("exclude_eog_on_mat", None)
    s.remove("exclude_eog_on_mat")            # estado limpio para ver el valor por defecto
    try:
        win = MainWindow()
        assert win.act_exclude_eog.isCheckable()
        assert win.act_exclude_eog.isChecked() is True, "debería venir activada por defecto"
        print("    casilla «Excluir EOG al importar .mat» = activada ✓")

        tmp = tempfile.mkdtemp()
        win.project = Project.create(tmp, "eog")
        imp = os.path.join(win.project.path, IMPORTED_DIR)
        os.makedirs(imp, exist_ok=True)
        csv = os.path.join(imp, "s.csv.gz")
        data = np.random.default_rng(0).normal(0, 1, (1000, 6)).astype(np.float32)
        names = ["C3", "Cz", "C4", "EOG-left", "EOG-central", "EOG-right"]
        write_openvibe_csv(csv, data, 250.0, names, [(100, "left_hand"), (600, "right_hand")])
        sid = win.project.add_source(csv)["id"]

        print("[2] _auto_exclude_eog excluye los 3 EOG (no los EEG)")
        n = win._auto_exclude_eog()
        assert n == 3, n
        rec = win.project.get_recording(sid)
        kept = win.project.kept_channel_names(rec)
        assert kept == ["C3", "Cz", "C4"], kept
        assert set(win.project.excluded_channels()) == {"EOG-left", "EOG-central", "EOG-right"}
        print(f"    activos = {kept}  ·  excluidos = EOG-*")

        print("[3] Las etiquetas/marcadores se conservan")
        assert len(rec.events) == 2, len(rec.events)
        assert {e["id"] for e in rec.events} == {"left_hand", "right_hand"}
        print(f"    {len(rec.events)} marcadores intactos: left_hand, right_hand")

        print("[4] Es idempotente (volver a llamar no añade nada)")
        assert win._auto_exclude_eog() == 0
        print("    segunda llamada excluye 0 ✓")

        win.acq_panel.shutdown()
    finally:
        if backup is None:
            s.remove("exclude_eog_on_mat")
        else:
            s.setValue("exclude_eog_on_mat", backup)

    print("\nEXCLUIR EOG AL IMPORTAR .MAT OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

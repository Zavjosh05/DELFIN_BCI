"""Activar/desactivar pasos, progreso del pipeline, ICA sin aviso y ventana de señal."""
from __future__ import annotations

import os
import sys
import tempfile
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
from PyQt6.QtWidgets import QApplication

from eeg_studio.config import IMPORTED_DIR
from eeg_studio.core import preprocessing
from eeg_studio.core.mat_loader import write_openvibe_csv
from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    tmp = tempfile.mkdtemp()
    proj = Project.create(tmp, "pipe")
    imp = os.path.join(proj.path, IMPORTED_DIR)
    os.makedirs(imp, exist_ok=True)
    csv = os.path.join(imp, "s.csv.gz")
    data = np.random.default_rng(0).normal(0, 1, (1500, 3)).astype(np.float32)
    write_openvibe_csv(csv, data, 250.0, ["C3", "Cz", "C4"], [])
    sid = proj.add_source(csv)["id"]

    print("[1] apply_pipeline informa progreso y respeta pasos desactivados")
    pipe = [{"type": "bandpass", "params": {"low": 1.0, "high": 40.0, "order": 4}},
            {"type": "detrend", "params": {"type": "linear"}, "enabled": False}]
    seen = []
    out = preprocessing.apply_pipeline(data.T.astype(float), 250.0, pipe,
                                       progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 1)], seen          # solo 1 paso activo de 2
    print(f"    progreso={seen} (el paso desactivado se omitió)")

    print("[2] set_step_enabled cambia el resultado procesado")
    proj.add_pipeline_step("bandpass")
    raw = proj.get_recording(sid).data[proj.kept_indices(proj.get_recording(sid))]
    proc_on = proj.get_processed(sid)
    assert not np.allclose(proc_on, raw), "el filtro no cambió nada"
    proj.set_step_enabled(0, False)
    proc_off = proj.get_processed(sid)
    assert np.allclose(proc_off, raw), "desactivar no devolvió la señal cruda"
    print("    activado≠crudo, desactivado=crudo ✓")
    proj.remove_pipeline_step(0)           # pipeline vacío -> ventanas en modo crudo (síncrono)

    print("[3] ICA no emite ConvergenceWarning")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        preprocessing.ica_artifact(np.random.default_rng(1).normal(0, 1, (3, 800)))
    conv = [x for x in w if "onvergence" in str(x.message) or "converge" in str(x.message)]
    assert not conv, [str(x.message) for x in conv]
    print(f"    sin avisos de convergencia ({len(w)} otros avisos)")

    print("[4] Se pueden abrir varias ventanas de señal a la vez")
    win = MainWindow()
    win.project = proj
    win.current_source_id = sid
    win.open_source_window(sid)
    win.open_source_window(sid)
    assert len(win._signal_windows) == 2, len(win._signal_windows)
    one = next(iter(win._signal_windows))
    one.close()
    assert len(win._signal_windows) == 1, len(win._signal_windows)
    print("    2 ventanas abiertas; cerrar una deja 1 ✓")

    for w in list(win._signal_windows):
        w.close()
    app.processEvents()
    win.acq_panel.shutdown()
    print("\nPIPELINE TOGGLE / PROGRESO / VENTANAS OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

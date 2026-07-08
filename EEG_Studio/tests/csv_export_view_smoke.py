"""Exportar CSV descomprimido + visor numérico de una grabación. Offscreen."""
from __future__ import annotations

import gzip
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

from eeg_studio.acquisition.recorder import CSVRecorder  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.core.recording import Recording  # noqa: E402
from eeg_studio.ui.csv_view import RecordingTableModel, build_data_dialog  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841
    tmp = tempfile.mkdtemp()

    # CSV plano y su versión comprimida .csv.gz (mismo contenido).
    csv = os.path.join(tmp, "orig.csv")
    rec = CSVRecorder(csv, 14, 128.0)
    rec.add_marker("a")
    rec.write(np.zeros((14, 50)))
    rec.close()
    with open(csv, "rb") as f:
        raw = f.read()
    gz = os.path.join(tmp, "orig.csv.gz")
    with gzip.open(gz, "wb") as f:
        f.write(raw)

    print("[1] Exportar un .csv.gz lo DESCOMPRIME (contenido idéntico al plano)")
    out = os.path.join(tmp, "exportado.csv")
    MainWindow._write_plain_csv(gz, out)
    with open(out, "rb") as f:
        assert f.read() == raw, "el CSV exportado no coincide con el original descomprimido"
    # y no queda comprimido (empieza con la cabecera de texto, no con magic gzip)
    with open(out, "rb") as f:
        head = f.read(4)
    assert head[:2] != b"\x1f\x8b", "el archivo exportado sigue comprimido"
    print(f"    {os.path.basename(gz)} -> {os.path.basename(out)} (texto plano)")

    print("[2] Exportar un .csv plano simplemente lo copia")
    out2 = os.path.join(tmp, "copia.csv")
    MainWindow._write_plain_csv(csv, out2)
    with open(out2, "rb") as f:
        assert f.read() == raw
    print("    copia directa OK")

    print("[3] Visor numérico: el modelo expone muestras × (#, t, canales, Event)")
    data = np.arange(14 * 20, dtype=float).reshape(14, 20)
    names = [f"Ch{i+1}" for i in range(14)]
    r = Recording(source_path=csv, channel_names=names, data=data,
                  time=np.arange(20) / 128.0, sample_rate=128.0,
                  events=[{"sample": 5, "id": "a"}])
    model = RecordingTableModel(r.data, names, r.sample_rate, r.events)
    assert model.rowCount() == 20, model.rowCount()
    assert model.columnCount() == 2 + 14 + 1, model.columnCount()   # #, t, canales, Event
    from PyQt6.QtCore import Qt
    # columna 0 = índice; columna 2 = primer canal; última = Event Id en la muestra 5
    assert model.data(model.index(5, 0), Qt.ItemDataRole.DisplayRole) == "5"
    assert model.data(model.index(0, 2), Qt.ItemDataRole.DisplayRole) == "0.000"
    assert model.data(model.index(5, model.columnCount() - 1),
                      Qt.ItemDataRole.DisplayRole) == "a"
    print(f"    modelo {model.rowCount()}×{model.columnCount()} con Event en la muestra 5")

    print("[4] El diálogo del visor se construye (sin ejecutarse)")
    dlg = build_data_dialog(None, r, names, "Datos — prueba")
    assert dlg is not None
    dlg.deleteLater()

    print("[5] Integración: fuente en el proyecto -> exportar por el controlador")
    win = MainWindow()
    win.project = Project.create(tmp, "csvexp")
    sid = win.project.add_source(csv)["id"]
    # helper accesible por el controlador
    out3 = os.path.join(tmp, "desde_ctrl.csv")
    win._write_plain_csv(win.project.get_source(sid)["path"], out3)
    assert os.path.isfile(out3)
    win.acq_panel.shutdown()
    print("    exportación vía controlador OK")

    print("\nEXPORTAR CSV (DESCOMPRIMIDO) + VISOR NUMÉRICO OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

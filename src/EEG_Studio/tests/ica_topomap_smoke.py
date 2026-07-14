"""Mapas topográficos de los componentes ICA (núcleo + montaje + vista + cableado).

Cubre:
  * ica_decompose devuelve la matriz de mezcla, kurtosis y flags de artefacto,
    con el MISMO ajuste que ica_artifact (lo que se ve = lo que se elimina).
  * montage.positions_2d ubica los 14 canales EPOC+ y devuelve None si no conoce.
  * la vista construye la figura (si hay matplotlib).
  * el botón «Ver mapas espaciales (ICA)…» aparece en el panel para el paso ICA y
    MainWindow.show_ica_topomaps calcula la descomposición (en un worker) y llama
    al diálogo (monkeypatch para no abrir la ventana modal).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
from PyQt6.QtWidgets import QApplication, QPushButton

from eeg_studio.core import montage
from eeg_studio.core import preprocessing as pp
from eeg_studio.core.mat_loader import write_openvibe_csv
from eeg_studio.core.project import Project
from eeg_studio.ui import ica_topomap_view as tv
from eeg_studio.ui.main_window import MainWindow

EPOC = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
        "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]


def _spike_data(n=1280, n_ch=14, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 1.0, (n_ch, n))
    spike = np.zeros(n)
    spike[::160] = 25.0                       # picos periódicos -> kurtosis alta
    data += np.outer(rng.uniform(0.4, 1.0, n_ch), spike)
    return data


def main() -> int:
    print("[1] ica_decompose: mezcla (n_ch, n_comp) + kurtosis + flags")
    data = _spike_data()
    res = pp.ica_decompose(data, 0, 5.0)
    assert res is not None
    assert res["mixing"].shape == (14, res["n_components"]), res["mixing"].shape
    assert res["kurtosis"].shape == (res["n_components"],)
    assert res["artifact"].dtype == bool and res["artifact"].any(), "no marcó ningún artefacto"
    # Coherencia con ica_artifact: mismo ajuste => mismos flags de kurtosis.
    fit = pp._fit_ica(data, 0)
    assert fit is not None
    from scipy.stats import kurtosis as _k
    k = _k(fit[1], axis=0, fisher=True)
    assert np.array_equal(np.abs(k) > 5.0, res["artifact"]), "flags no coinciden con ica_artifact"
    print(f"    {res['n_components']} componentes, {int(res['artifact'].sum())} artefacto(s)")

    print("[2] montage: 14/14 EPOC+ con posición; desconocido -> None")
    assert montage.known_count(EPOC) == 14
    assert montage.positions_2d(["Channel 1"])[0] is None
    print("    posiciones EPOC+ OK")

    print("[3] Vista: la figura se construye con un eje por componente")
    if tv.topomaps_available():
        fig = tv.build_topomap_figure(res["mixing"], EPOC, res["kurtosis"], res["artifact"])
        assert len(fig.axes) == res["n_components"], len(fig.axes)
        print(f"    figura con {len(fig.axes)} topomapas")
    else:
        print("    matplotlib no disponible: se omite el dibujo (esperado en ese caso)")

    print("[4] Cableado: botón en el panel + MainWindow.show_ica_topomaps")
    app = QApplication(sys.argv)
    win = MainWindow()
    tmp = tempfile.mkdtemp()
    win.project = Project.create(tmp, "ica")
    csv = os.path.join(tmp, "s.csv")
    write_openvibe_csv(csv, data.T, 128.0, EPOC, [])   # (muestras, canales)
    sid = win.project.add_source(csv)["id"]
    win.current_source_id = sid
    win.project.add_pipeline_step("ica", {"n_components": 0, "kurt_threshold": 5.0})
    win.refresh_all()

    # El panel muestra el botón de mapas espaciales para el paso ICA.
    panel = win.preproc_panel
    row = len(win.project.state["pipeline"]) - 1
    panel.steps_list.setCurrentRow(row)
    form = panel.params_form
    labels = [form.itemAt(i, form.ItemRole.SpanningRole).widget().text()
              for i in range(form.rowCount())
              if form.itemAt(i, form.ItemRole.SpanningRole)
              and isinstance(form.itemAt(i, form.ItemRole.SpanningRole).widget(), QPushButton)]
    assert any("mapas espaciales" in t.lower() for t in labels), labels
    print("    botón «Ver mapas espaciales (ICA)…» presente")

    # Monkeypatch del diálogo para no abrir la ventana modal; captura los datos.
    captured = {}
    tv.show_ica_topomaps_dialog = lambda *a, **k: captured.update(
        {"mixing": a[1], "names": a[2], "kurt": a[3], "artifact": a[4]})

    if tv.topomaps_available():
        win.show_ica_topomaps(row)
        t0 = time.time()
        while "mixing" not in captured and time.time() - t0 < 20:
            app.processEvents()
            time.sleep(0.02)
        assert "mixing" in captured, "el worker no terminó / no llamó al diálogo"
        assert np.asarray(captured["mixing"]).shape[0] == 14
        assert len(captured["names"]) == 14
        print("    show_ica_topomaps calculó la descomposición y llamó al diálogo")
    else:
        print("    (sin matplotlib) se omite la llamada al diálogo")

    win.acq_panel.shutdown()
    print("\nMAPAS TOPOGRÁFICOS ICA OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

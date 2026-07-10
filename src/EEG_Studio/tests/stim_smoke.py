"""Estimulación sincronizada: núcleo, config en el proyecto, editor de línea de
tiempo (estilo editor de video), exportar/importar y grabación con segmentos
exactos. General (no atado a Delfin). Offscreen."""
from __future__ import annotations

import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.core import stim as stim_core  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402


def _pump(app, ms):
    loop = QEventLoop(); QTimer.singleShot(ms, loop.quit); loop.exec()


def main() -> int:
    app = QApplication(sys.argv)

    print("[1] Núcleo: detección de clase, descubrir videos, segmentos EXACTOS")
    assert stim_core.class_from_filename("Estimulo_BCI_Arriba.mp4") == "arriba"
    videos = stim_core.discover_videos()
    assert videos and {v["label"] for v in videos} >= {"arriba", "abajo", "soltar"}
    events = [{"kind": "segment", "start": 2000, "stop": 6000, "label": "arriba"},
              {"kind": "marker", "t": 1000, "label": "arriba"}]
    assert stim_core.compute_segments(events, 128.0, base_sample=100) == \
        [(100 + 256, 100 + 768, "arriba")]

    print("[2] Varias clases por video (general): compute_segments respeta cada clase")
    multi = [{"kind": "segment", "start": 0, "stop": 1000, "label": "arriba"},
             {"kind": "segment", "start": 1000, "stop": 2000, "label": "abajo"}]
    segs = stim_core.compute_segments(multi, 100.0)
    assert [s[2] for s in segs] == ["arriba", "abajo"], segs

    print("[3] Clases desde el PROYECTO (no hardcodeadas)")
    proj = Project.create(tempfile.mkdtemp(), "stim")
    proj.save_stim_video({"path": videos[0]["path"], "name": videos[0]["name"],
                          "label": "arriba", "duration_ms": 60000, "events": multi})
    assert stim_core.project_classes(proj) == ["abajo", "arriba"], stim_core.project_classes(proj)

    print("[4] Reubicar video al importar (relocate_video)")
    vp = videos[0]["path"]
    assert stim_core.relocate_video(vp, None) == vp
    assert stim_core.relocate_video("/no/existe/" + videos[0]["name"],
                                    os.path.dirname(vp)) == vp
    assert stim_core.relocate_video("/no/existe/nada.mp4", "/tampoco") is None

    print("[5] Exportar/importar configuración (round-trip con reubicación)")
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "estimulos.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"stim_videos": proj.stim_videos()}, fh)
    proj2 = Project.create(tempfile.mkdtemp(), "imp")
    data = json.load(open(out, encoding="utf-8"))
    for cfg in data["stim_videos"]:
        c = dict(cfg); c["id"] = None
        c["path"] = stim_core.relocate_video(c["path"], None) or c["path"]
        proj2.save_stim_video(c)
    assert len(proj2.stim_videos()) == 1
    assert proj2.stim_videos()[0]["events"][1]["label"] == "abajo"

    print("[6] Editor de línea de tiempo: carga video, barra de tiempo, y CANCELAR limpia")
    from eeg_studio.ui.stim_timeline import StimTimelineDialog
    dlg = StimTimelineDialog(vp, "arriba", None, ["arriba", "abajo"])
    _pump(app, 1500)
    assert dlg.duration_ms() > 0
    dlg.timeline.resize(1000, 52); dlg.timeline.set_duration(60000)
    assert abs(dlg.timeline._ms_at(500) - 30000) < 100     # el instante bajo el cursor
    assert any(e["kind"] == "segment" for e in dlg.result_events())
    dlg.reject()                                            # CANCELAR (antes crasheaba)
    assert dlg._cleaned                                     # el player quedó limpio
    _pump(app, 150)

    print("[7] Panel: explorador de archivos general + lista con clases")
    from eeg_studio.ui.main_window import MainWindow
    win = MainWindow()
    win.project = proj
    win.acq_panel.refresh_stim()
    assert win.acq_panel.stim_list.count() == 1
    assert "arriba" in win.acq_panel.stim_list.item(0).text()
    assert "abajo" in win.acq_panel.stim_list.item(0).text()   # varias clases mostradas

    print("[8] Grabación automática: segmentos exactos colocados y guardados solos")
    panel = win.acq_panel
    panel.source_combo.setCurrentIndex(0)                  # Simulado
    panel._connect()
    for _ in range(120):
        _pump(app, 10)
        if panel._configured:
            break
    assert panel._configured and panel.stim_is_ready()
    assert panel.stim_start("estimulo_test")
    _pump(app, 300)
    fs = panel.source.sample_rate
    n = panel.stim_samples()
    ev = [{"kind": "segment", "start": 50, "stop": 200, "label": "arriba"}]
    panel.stim_finish(stim_core.compute_segments(ev, fs, base_sample=0, n_samples=n))
    _pump(app, 100)
    assert len(win.project.sources) == 1
    assert win.project.state["segments"][0]["label"] == "arriba"
    panel._disconnect(); panel.shutdown()

    print("\nESTIMULACIÓN SINCRONIZADA (GENERAL) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Estimulación sincronizada: núcleo, config en el proyecto, editor de línea de
tiempo y grabación automática con segmentos exactos. Offscreen."""
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
    assert stim_core.class_from_filename("algo_agarrar.mp4") == "agarre"
    videos = stim_core.discover_videos()
    assert videos, "no se descubrieron videos en data/videos"
    labels = {v["label"] for v in videos}
    assert {"arriba", "abajo", "izquierda", "derecha", "agarre", "soltar"} <= labels, labels
    # segmentos exactos: evento 2.0–6.0 s a 128 Hz, base 100 muestras
    events = [{"kind": "segment", "start": 2000, "stop": 6000, "label": "arriba"},
              {"kind": "marker", "t": 1000, "label": "arriba"}]
    segs = stim_core.compute_segments(events, 128.0, base_sample=100)
    assert segs == [(100 + 256, 100 + 768, "arriba")], segs
    assert stim_core.markers_in_order(events)[0]["t"] == 1000

    print("[2] Proyecto: guardar/listar/quitar estímulo (persiste al reabrir)")
    root = tempfile.mkdtemp()
    proj = Project.create(root, "stim")
    cfg = proj.save_stim_video({"path": videos[0]["path"], "name": videos[0]["name"],
                                "label": videos[0]["label"], "duration_ms": 60000,
                                "events": events})
    assert cfg["id"] and len(proj.stim_videos()) == 1
    cfg["label"] = "abajo"                        # actualizar por id (no duplica)
    proj.save_stim_video(cfg)
    assert len(proj.stim_videos()) == 1 and proj.stim_videos()[0]["label"] == "abajo"
    proj.save()
    assert Project.open(proj.path).stim_videos()[0]["events"][0]["stop"] == 6000
    proj.remove_stim_video(cfg["id"])
    assert proj.stim_videos() == []

    print("[3] Editor de línea de tiempo: carga el video y devuelve eventos")
    from eeg_studio.ui.stim_timeline import StimTimelineDialog
    dlg = StimTimelineDialog(videos[0]["path"], "arriba", None)
    _pump(app, 1500)                             # deja que cargue la duración
    assert dlg.duration_ms() > 0, "el editor no obtuvo la duración del video"
    # tras cargar, se prellenan eventos por defecto (una marca + un segmento)
    res = dlg.result_events()
    assert any(e["kind"] == "segment" for e in res), res
    dlg.close()

    print("[4] Panel: la sección de estímulos lista lo configurado")
    from eeg_studio.ui.main_window import MainWindow
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "cp")
    win.project.save_stim_video({"path": videos[0]["path"], "name": videos[0]["name"],
                                 "label": "arriba", "duration_ms": 60000, "events": events})
    win.acq_panel.refresh_stim()
    assert win.acq_panel.stim_list.count() == 1
    assert "arriba" in win.acq_panel.stim_list.item(0).text()

    print("[5] Grabación automática: segmentos exactos colocados y guardados solos")
    panel = win.acq_panel
    panel.source_combo.setCurrentIndex(0)        # Simulado (sin hardware)
    panel._connect()
    for _ in range(120):                         # espera a que se configure con muestras
        _pump(app, 10)
        if panel._configured:
            break
    assert panel._configured, "la fuente simulada no entregó muestras"
    assert panel.stim_is_ready()
    assert panel.stim_start("estimulo_test")
    _pump(app, 300)                              # graba ~0.3 s
    fs = panel.source.sample_rate
    n = panel.stim_samples()
    assert n > 0
    panel.stim_marker("arriba")
    ev = [{"kind": "segment", "start": 50, "stop": 200, "label": "arriba"}]  # 0.05–0.2 s
    segs = stim_core.compute_segments(ev, fs, base_sample=0, n_samples=n)
    panel.stim_finish(segs)                      # termina: guarda + añade como fuente
    _pump(app, 100)
    assert len(win.project.sources) == 1, "no se añadió la grabación como fuente"
    proj_segs = win.project.state["segments"]
    assert len(proj_segs) == 1 and proj_segs[0]["label"] == "arriba", proj_segs
    panel._disconnect()
    panel.shutdown()

    print("[6] Reproductor a pantalla completa: se construye y elige monitor")
    from eeg_studio.ui.stim_player import StimPlayerWindow, _best_screen
    screen, external = _best_screen()
    assert screen is not None
    w = StimPlayerWindow(videos[0]["path"])
    assert w is not None
    w.close()

    print("\nESTIMULACIÓN SINCRONIZADA OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

    print("[6] Editor: video nuevo SIN marcas auto, F6 segmento, repetir periódico, Ir a")
    import eeg_studio.ui.stim_timeline as stl
    from eeg_studio.ui.stim_timeline import StimTimelineDialog
    dlg = StimTimelineDialog(vp, "arriba", None, ["arriba", "abajo"])
    _pump(app, 1500)
    assert dlg.duration_ms() > 0
    assert dlg.result_events() == []                       # video nuevo empieza VACÍO
    dlg.timeline.resize(1000, 52); dlg.timeline.set_duration(60000)
    assert abs(dlg.timeline._ms_at(500) - 30000) < 100     # instante bajo el cursor
    # F6 = inicio/fin de segmento
    dlg.player.setPosition(2000); _pump(app, 200); dlg._segment_click()
    dlg.player.setPosition(6000); _pump(app, 200); dlg._segment_click()
    assert len([e for e in dlg._events if e["kind"] == "segment"]) == 1, dlg._events
    # repetir periódicamente (diálogos simulados): periodo 20 s, 5 reps — solo caben 2
    # (ejercita también la rama «no cabía en el video», que antes crasheaba).
    class _FakeInput:
        @staticmethod
        def getDouble(*a, **k): return (20.0, True)
        @staticmethod
        def getInt(*a, **k): return (5, True)
    orig = stl.QInputDialog; stl.QInputDialog = _FakeInput
    dlg.table.selectRow(0); dlg._repeat_segment()
    stl.QInputDialog = orig
    assert len([e for e in dlg._events if e["kind"] == "segment"]) == 3, dlg._events
    # campo «Ir a (s)»
    dlg.goto_spin.setValue(12.5); dlg._goto(); _pump(app, 200)
    assert abs(dlg.player.position() - 12500) < 1500
    dlg.reject(); assert dlg._cleaned; _pump(app, 150)     # cancelar limpia (no crashea)

    print("[7] Panel: explorador de archivos general + lista con clases")
    from eeg_studio.ui.main_window import MainWindow
    win = MainWindow()
    win.project = proj
    win.acq_panel.refresh_stim()
    assert win.acq_panel.stim_list.count() == 1
    assert "arriba" in win.acq_panel.stim_list.item(0).text()
    assert "abajo" in win.acq_panel.stim_list.item(0).text()   # varias clases mostradas
    # selector de monitor para el video
    assert win.acq_panel.monitor_combo.count() >= 1
    assert win.acq_panel._selected_screen() is not None

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

    print("[9] Indicador de segmento en el reproductor + detección por tiempo")
    from eeg_studio.ui.stim_player import StimPlayerWindow, StimSession
    pw = StimPlayerWindow(vp)
    pw.set_segment("arriba")
    assert not pw.seg_label.isHidden() and "arriba" in pw.seg_label.text()
    pw.set_segment(None); assert pw.seg_label.isHidden()
    pw.close(); app.processEvents()
    # la sesión detecta el segmento activo según la posición del video
    cfg = {"path": vp, "events": [{"kind": "segment", "start": 2000, "stop": 6000,
                                   "label": "abajo"}]}
    ses = StimSession(win, cfg, "x")
    assert next((l for a, b, l in ses._segments if a <= 4000 < b), None) == "abajo"
    assert next((l for a, b, l in ses._segments if a <= 8000 < b), None) is None

    print("[10] Importar estímulos: pregunta si SOBRESCRIBIR o IGNORAR los repetidos")
    import eeg_studio.ui.acquisition_panel as apmod
    ipanel = win.acq_panel
    win.project = proj                                     # ya tiene 1 estímulo ("arriba")
    assert len(proj.stim_videos()) == 1
    orig_open = apmod.QFileDialog.getOpenFileName
    apmod.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (out, "JSON (*.json)"))
    try:
        # (a) mismo estímulo, elijo IGNORAR -> no cambia nada, id intacto
        prev_id = proj.stim_videos()[0]["id"]
        ipanel._ask_stim_overwrite = lambda dups: False
        ipanel._import_stim()
        assert len(proj.stim_videos()) == 1, proj.stim_videos()
        assert proj.stim_videos()[0]["id"] == prev_id
        assert "ignorado" in ipanel.status.text().lower(), ipanel.status.text()

        # (b) mismo estímulo, elijo SOBRESCRIBIR -> sigue habiendo 1 (mismo id)
        ipanel._ask_stim_overwrite = lambda dups: True
        ipanel._import_stim()
        assert len(proj.stim_videos()) == 1, proj.stim_videos()
        assert proj.stim_videos()[0]["id"] == prev_id
        assert "sobrescrito" in ipanel.status.text().lower(), ipanel.status.text()

        # (c) estímulo NUEVO (otra etiqueta) -> se añade sin preguntar
        out2 = os.path.join(tmp, "estimulos2.json")
        with open(out2, "w", encoding="utf-8") as fh:
            json.dump({"stim_videos": [{"path": vp, "name": videos[0]["name"],
                                        "label": "soltar", "events": []}]}, fh)
        apmod.QFileDialog.getOpenFileName = staticmethod(
            lambda *a, **k: (out2, "JSON (*.json)"))
        ipanel._import_stim()
        assert len(proj.stim_videos()) == 2, proj.stim_videos()
        assert "nuevo" in ipanel.status.text().lower(), ipanel.status.text()
    finally:
        apmod.QFileDialog.getOpenFileName = orig_open

    print("\nESTIMULACIÓN SINCRONIZADA (GENERAL) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

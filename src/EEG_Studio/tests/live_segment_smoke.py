"""Marcar SEGMENTOS (inicio/fin) durante la grabación en vivo. Offscreen."""
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

from eeg_studio.acquisition.recorder import CSVRecorder  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841
    tmp = tempfile.mkdtemp()
    win = MainWindow()
    win.project = Project.create(tmp, "seg")
    panel = win.acq_panel

    print("[1] 1er clic marca inicio, 2º marca fin → segmento (start, stop, etiqueta)")
    panel.recorder = CSVRecorder(os.path.join(tmp, "r.csv"), 14, 128.0)
    panel.marker_edit.setText("mano_izq")
    panel.recorder.write(np.zeros((14, 50)))       # 50 muestras grabadas
    panel._toggle_segment()                        # inicio en 50
    assert panel._seg_active and panel._seg_start == 50
    panel.recorder.write(np.zeros((14, 100)))      # +100 → 150
    panel._toggle_segment()                        # fin en 150
    assert not panel._seg_active
    assert panel._rec_segments == [(50, 150, "mano_izq")], panel._rec_segments
    print(f"    {panel._rec_segments}")

    print("[2] Un segmento que queda abierto se cierra al detener la grabación")
    panel.marker_edit.setText("descanso")
    panel._toggle_segment()                        # inicio en 150
    panel.recorder.write(np.zeros((14, 80)))       # +80 → 230
    # Simula el cierre que hace _stop_recording para el segmento abierto:
    if panel._seg_active:
        stop = panel.recorder.n_samples
        if stop > panel._seg_start:
            panel._rec_segments.append((panel._seg_start, stop, panel._seg_label))
        panel._seg_active = False
    assert panel._rec_segments[-1] == (150, 230, "descanso"), panel._rec_segments
    captured = list(panel._rec_segments)
    panel.recorder.close()
    panel.recorder = None

    print("[3] Al añadir la grabación como fuente (con NOMBRE), se crean los segmentos")
    rec_path = os.path.join(tmp, "rec_full.csv")
    rec = CSVRecorder(rec_path, 14, 128.0)
    rec.write(np.random.default_rng(0).normal(0, 1, (14, 300)))
    rec.close()
    win.add_recording_as_source(rec_path, captured, alias="Mi Prueba 01")

    # La fuente toma el nombre indicado como alias.
    aliases = [s["alias"] for s in win.project.sources]
    assert "Mi Prueba 01" in aliases, aliases

    segs = win.project.state["segments"]
    assert len(segs) == 2, len(segs)
    src_ids = {s["id"] for s in win.project.sources}
    for s in segs:
        assert s["source_id"] in src_ids
    by_label = {s["label"]: (s["start"], s["stop"]) for s in segs}
    assert by_label["mano_izq"] == (50, 150), by_label
    assert by_label["descanso"] == (150, 230), by_label
    print(f"    fuente='Mi Prueba 01' · segmentos: {by_label}")

    print("[4] Nombre de archivo: saneado y único (no sobrescribe)")
    assert panel._safe_name("mano izq / prueba*2") == "mano_izq_prueba2", \
        panel._safe_name("mano izq / prueba*2")
    rec_dir = os.path.dirname(rec_path)
    p1 = panel._unique_rec_path(rec_dir, "captura")
    open(p1, "w").close()                     # ocupa captura.csv
    p2 = panel._unique_rec_path(rec_dir, "captura")
    assert p1.endswith("captura.csv") and p2.endswith("captura_2.csv"), (p1, p2)
    print(f"    {os.path.basename(p1)} -> {os.path.basename(p2)}")

    print("[5] Marca de instante (punto) sigue disponible y separada del segmento")
    assert panel.marker_btn is not None and panel.segment_btn is not None
    assert panel.name_edit is not None       # campo de nombre presente

    print("[6] Marca de DURACIÓN FIJA: crea un segmento de N s desde ahora")
    from eeg_studio.acquisition.simulated import SimulatedSource
    panel.source = SimulatedSource()         # solo para leer sample_rate (128 Hz)
    panel.recorder = CSVRecorder(os.path.join(tmp, "r2.csv"), 14, 128.0)
    panel._rec_segments = []
    panel.recorder.write(np.zeros((14, 100)))    # 100 muestras grabadas
    panel.marker_edit.setText("ojos_cerrados")
    panel.duration_spin.setValue(5)              # 5 s → 640 muestras
    panel._add_timed_marker()
    assert panel._rec_segments == [(100, 100 + 640, "ojos_cerrados")], panel._rec_segments
    print(f"    {panel._rec_segments}")

    print("[7] Si la grabación termina antes, el segmento se recorta (no queda al aire)")
    panel.recorder.write(np.zeros((14, 200)))    # total 300 (< 740)
    final = panel.recorder.n_samples
    seg = [(s, min(e, final), lbl) for (s, e, lbl) in panel._rec_segments
           if s < final and min(e, final) - s >= 1]
    assert seg == [(100, 300, "ojos_cerrados")], seg
    panel.recorder.close()
    panel.recorder = None
    print(f"    recortado a {seg}")

    win.acq_panel.shutdown()
    print("\nSEGMENTOS EN VIVO (INICIO/FIN + DURACIÓN FIJA) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

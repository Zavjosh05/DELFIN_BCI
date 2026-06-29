"""Verifica que los marcadores se convierten en segmentos etiquetados (clases)."""
from __future__ import annotations

import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.acquisition import CSVRecorder
from eeg_studio.core.project import Project


def main() -> int:
    tmp = tempfile.mkdtemp()
    csv = os.path.join(tmp, "marcado.csv")

    print("[1] Grabando un CSV con dos marcadores (izq, der)")
    rec = CSVRecorder(csv, n_channels=14, sample_rate=128.0)
    rec.add_marker("izq")
    rec.write(np.zeros((14, 10)))            # 'izq' marca la muestra 0
    rec.write(np.ones((14, 10)))
    rec.add_marker("der")
    rec.write(np.full((14, 10), 2.0))        # 'der' marca la muestra 20
    rec.close()

    print("[2] Creando proyecto y segmentando por marcadores")
    proj = Project.create(tmp, "mk")
    sid = proj.add_source(csv)["id"]
    n = proj.segments_from_markers(sid, window=0)   # 0 = hasta el siguiente marcador
    print(f"    segmentos creados: {n}, clases={proj.labels()}")
    assert n == 2, f"se esperaban 2 segmentos, hay {n}"
    assert set(proj.labels()) == {"izq", "der"}, f"clases inesperadas: {proj.labels()}"

    segs = sorted(proj.state["segments"], key=lambda s: s["start"])
    assert segs[0]["start"] == 0 and segs[0]["label"] == "izq"
    assert segs[1]["start"] == 20 and segs[1]["label"] == "der"
    print(f"    seg1={segs[0]['start']}-{segs[0]['stop']} «{segs[0]['label']}», "
          f"seg2={segs[1]['start']}-{segs[1]['stop']} «{segs[1]['label']}»")

    print("[3] Ventana fija: cada marcador genera un segmento corto")
    proj2 = Project.create(tempfile.mkdtemp(), "mk2")
    sid2 = proj2.add_source(csv)["id"]
    assert proj2.segments_from_markers(sid2, window=8) == 2
    assert all(s["stop"] - s["start"] == 8 for s in proj2.state["segments"])
    print("    ventana de 8 muestras aplicada")

    print("[4] Todas las fuentes a la vez")
    proj3 = Project.create(tempfile.mkdtemp(), "mk3")
    proj3.add_source(csv)          # misma grabación como 2 fuentes (2 marcadores c/u)
    proj3.add_source(csv)
    n_all = proj3.segments_from_markers_all(window=0)
    assert n_all == 4, f"esperaba 4 (2 fuentes × 2 marcadores), hay {n_all}"
    srcs = {s["source_id"] for s in proj3.state["segments"]}
    assert len(srcs) == 2, "deberían venir de las 2 fuentes"
    print(f"    {n_all} segmentos de {len(srcs)} fuentes")

    print("\nMARCADORES → CLASES OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

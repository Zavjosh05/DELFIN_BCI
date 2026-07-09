"""Verifica la exclusión de canales (núcleo): CAR/características/datasets usan
solo los canales activos."""
from __future__ import annotations

import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from eeg_studio.core.project import Project

from tests import data_dir
EEG_DIR = data_dir()


def main() -> int:
    csv = os.path.join(EEG_DIR, "Prueba_001.csv")
    proj = Project.create(tempfile.mkdtemp(), "ch")
    sid = proj.add_source(csv)["id"]
    rec = proj.get_recording(sid)

    print("[1] Sin exclusión: 14 canales")
    assert proj.get_processed(sid).shape[0] == 14
    proj.add_pipeline_step("car")
    assert proj.get_processed(sid).shape[0] == 14

    print("[2] Excluir 2 canales -> 12 activos")
    proj.edit("excluded_channels", ["Channel 13", "Channel 14"], "excluir")
    proc = proj.get_processed(sid)
    assert proc.shape[0] == 12, f"canales activos: {proc.shape[0]}"
    assert len(proj.kept_display_names(rec)) == 12
    assert "Channel 13" not in proj.kept_channel_names(rec)
    print(f"    activos={len(proj.kept_indices(rec))}  excluidos={proj.excluded_channels()}")

    print("[3] Los segmentos/datasets usan solo los activos")
    proj.add_segment(sid, 0, 256, "x")
    data, fs = proj.segment_data(proj.state["segments"][0])
    assert data.shape[0] == 12, f"segmento: {data.shape}"
    print(f"    segmento con {data.shape[0]} canales")

    print("[4] Deshacer restaura los 14 canales")
    proj.undo()                       # quita el segmento
    assert proj.get_processed(sid).shape[0] == 12   # exclusión sigue activa
    proj.undo()                       # quita la exclusión
    assert proj.get_processed(sid).shape[0] == 14, "no se restauraron los canales"
    print("    undo OK")

    print("\nEXCLUSIÓN DE CANALES OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

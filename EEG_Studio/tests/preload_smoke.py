"""add_source con grabación precargada: acierto de caché (carga fuera del hilo GUI)."""
from __future__ import annotations

import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.config import IMPORTED_DIR
from eeg_studio.core.csv_loader import load_recording
from eeg_studio.core.mat_loader import write_openvibe_csv
from eeg_studio.core.project import Project


def main() -> int:
    tmp = tempfile.mkdtemp()
    proj = Project.create(tmp, "preload")
    imp = os.path.join(proj.path, IMPORTED_DIR)
    os.makedirs(imp, exist_ok=True)
    csv = os.path.join(imp, "s.csv.gz")
    data = np.random.default_rng(0).normal(0, 1, (1500, 3)).astype(np.float32)
    write_openvibe_csv(csv, data, 250.0, ["C3", "Cz", "C4"], [(100, "a")])

    print("[1] add_source(recording=…) reutiliza la grabación (no recarga)")
    rec = load_recording(csv)                       # carga «en el hilo del worker»
    src = proj.add_source(csv, recording=rec)
    got = proj.get_recording(src["id"])
    assert got is rec, "no reutilizó la grabación precargada"
    print("    misma instancia en caché ✓")

    print("[2] add_source sin precarga sigue funcionando")
    csv2 = os.path.join(imp, "s2.csv.gz")
    write_openvibe_csv(csv2, data, 250.0, ["C3", "Cz", "C4"], [])
    src2 = proj.add_source(csv2)                    # sin recording -> carga sola
    r2 = proj.get_recording(src2["id"])
    assert r2.data.shape[0] == 3, r2.data.shape
    print(f"    cargada sola: {r2.data.shape}")

    print("\nPRECARGA DE FUENTE OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

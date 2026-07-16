"""Verifica la caché en disco de la señal procesada (sin GUI).

Comprueba que se escribe en cache/, que se reutiliza (sin recalcular) y que al
cambiar el pipeline se invalida y se sustituye por la nueva versión.
"""
from __future__ import annotations

import glob
import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.core import preprocessing
from eeg_studio.core.project import Project

from tests import sample_csv_path


def _cache_files(proj):
    return glob.glob(os.path.join(proj.path, "cache", "*.cache.npz"))


def main() -> int:
    csv = sample_csv_path()
    assert os.path.isfile(csv), "falta el CSV de ejemplo"
    proj = Project.create(tempfile.mkdtemp(), "dc")
    sid = proj.add_source(csv)["id"]
    proj.add_pipeline_step("bandpass", {"low": 1.0, "high": 45.0, "order": 4})

    print("[1] Primer cálculo escribe la caché en disco")
    out1 = proj.get_processed(sid)
    files = _cache_files(proj)
    assert len(files) == 1, f"esperaba 1 archivo de caché, hay {len(files)}"
    print(f"    {os.path.basename(files[0])}")

    print("[2] Tras vaciar la RAM, se carga de disco (sin recalcular)")
    proj._processed.clear()
    original = preprocessing.apply_pipeline
    preprocessing.apply_pipeline = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no debería recalcular: hay caché en disco"))
    try:
        out2 = proj.get_processed(sid)
    finally:
        preprocessing.apply_pipeline = original
    assert np.allclose(out1, out2), "la señal de la caché no coincide"
    print("    cargado de disco correctamente")

    print("[3] Cambiar el pipeline invalida y sustituye la caché")
    proj.add_pipeline_step("car")
    proj.get_processed(sid)
    files = _cache_files(proj)
    assert len(files) == 1, f"debería quedar 1 caché vigente, hay {len(files)}"
    print(f"    caché vigente: {os.path.basename(files[0])}")

    print("\nCACHÉ EN DISCO OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

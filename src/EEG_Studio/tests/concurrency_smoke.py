"""Verifica el pre-cálculo en paralelo y la seguridad entre hilos (sin GUI).

Comprueba que `Project.prewarm` calcula la señal procesada de varias fuentes en
paralelo y que el resultado coincide con el cálculo directo del pipeline.
"""
from __future__ import annotations

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

from tests import data_dir
EEG_DIR = data_dir()


def main() -> int:
    csvs = [os.path.join(EEG_DIR, "Prueba_001.csv"), os.path.join(EEG_DIR, "Prueba_002.csv")]
    proj = Project.create(tempfile.mkdtemp(), "conc")
    ids = [proj.add_source(c)["id"] for c in csvs if os.path.isfile(c)]
    assert ids, "no se encontraron CSV de ejemplo"

    proj.add_pipeline_step("bandpass", {"low": 1.0, "high": 45.0, "order": 4})
    proj.add_pipeline_step("car")

    print(f"[1] Pre-calculando {len(ids)} fuentes en paralelo")
    proj.prewarm(ids)

    print("[2] Verificando consistencia con el cálculo directo")
    for sid in ids:
        rec = proj.get_recording(sid)
        expected = preprocessing.apply_pipeline(rec.data, rec.sample_rate, proj.state["pipeline"])
        got = proj.get_processed(sid)            # debe venir de la caché
        assert np.allclose(got, expected), f"resultado inconsistente para {sid}"
    print(f"    {len(ids)} fuentes procesadas y verificadas")

    print("\nCONCURRENCIA OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

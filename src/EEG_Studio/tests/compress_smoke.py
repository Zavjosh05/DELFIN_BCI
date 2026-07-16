"""Verifica comprimir una fuente .csv a .csv.gz sin perder datos (núcleo)."""
from __future__ import annotations

import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.core.csv_loader import compress_csv, load_recording
from eeg_studio.core.project import Project

from tests import sample_csv


def main() -> int:
    tmp = tempfile.mkdtemp()
    src = sample_csv(os.path.join(tmp, "fuente.csv"))

    proj = Project.create(tmp, "cmp")
    sid = proj.add_source(src)["id"]
    rec0 = proj.get_recording(sid)

    print("[1] Comprimir a .csv.gz (más pequeño)")
    gz = compress_csv(src)
    assert gz.endswith(".csv.gz") and os.path.isfile(gz)
    s0, s1 = os.path.getsize(src), os.path.getsize(gz)
    assert s1 < s0, f"no comprimió ({s1} >= {s0})"
    print(f"    {s0/1e6:.2f} MB -> {s1/1e6:.2f} MB  ({s0/s1:.1f}×)")

    print("[2] Repuntar la fuente al .csv.gz y recargar")
    proj.set_source_path(sid, gz)
    assert proj.get_source(sid)["path"].endswith(".csv.gz")
    rec1 = proj.get_recording(sid)
    assert rec1.channel_names == rec0.channel_names
    assert np.array_equal(rec0.data, rec1.data), "los datos cambiaron al comprimir"
    print(f"    recargado: {rec1.n_channels} canales, datos idénticos ✓")

    print("[3] El .csv.gz se puede cargar directamente")
    assert load_recording(gz).n_samples == rec0.n_samples

    print("\nCOMPRIMIR FUENTE OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

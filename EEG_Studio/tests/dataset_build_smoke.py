"""Construir dataset: reconstrucción completa y robustez ante fuentes faltantes."""
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
from eeg_studio.core import dataset as dataset_mod
from eeg_studio.core.mat_loader import write_openvibe_csv
from eeg_studio.core.project import Project


def _make_source(path: str, markers) -> None:
    data = np.random.default_rng(1).normal(0, 1, (3000, 3)).astype(np.float32)
    write_openvibe_csv(path, data, 250.0, ["C3", "Cz", "C4"], markers)


def main() -> int:
    tmp = tempfile.mkdtemp()
    proj = Project.create(tmp, "build")
    imp = os.path.join(proj.path, IMPORTED_DIR)
    os.makedirs(imp, exist_ok=True)
    a = os.path.join(imp, "a.csv.gz")
    b = os.path.join(imp, "b.csv.gz")
    _make_source(a, [(100, "left"), (700, "right"), (1300, "feet"), (1900, "tongue")])
    _make_source(b, [(150, "left"), (750, "right"), (1350, "feet"), (1950, "tongue")])
    sid_a = proj.add_source(a)["id"]
    sid_b = proj.add_source(b)["id"]

    proj.segments_from_markers_all(window=400, offset=0)
    total = len(proj.state["segments"])
    print(f"[1] {total} segmentos de 2 fuentes")
    assert total >= 6, total

    ds = dataset_mod.build_dataset(proj)
    print(f"[2] Construcción completa: {ds.n_samples} muestras, {ds.n_features} feats, skipped={ds.skipped}")
    assert ds.n_samples == total and ds.skipped == 0

    print("[3] Añadir más segmentos y reconstruir incluye lo viejo + lo nuevo")
    proj.add_segment(sid_a, 2300, 2700, "left")
    ds2 = dataset_mod.build_dataset(proj)
    assert ds2.n_samples == total + 1, (ds2.n_samples, total)
    print(f"    ahora {ds2.n_samples} muestras (sin re-segmentar lo anterior)")

    print("[4] Si una fuente falta, se omite (no rompe)")
    proj._recordings.clear()
    proj._processed.clear()
    os.remove(b)
    ds3 = dataset_mod.build_dataset(proj)
    a_segs = sum(1 for s in proj.state["segments"] if s["source_id"] == sid_a)
    assert ds3.n_samples == a_segs and ds3.skipped > 0, (ds3.n_samples, a_segs, ds3.skipped)
    print(f"    construyó {ds3.n_samples} de A, omitió {ds3.skipped} de B ✓")

    print("[5] Si TODAS faltan, error claro (no traceback de E/S)")
    proj._recordings.clear(); proj._processed.clear()
    os.remove(a)
    try:
        dataset_mod.build_dataset(proj)
        raise AssertionError("debió lanzar ValueError")
    except ValueError as exc:
        assert "se pudo cargar" in str(exc), str(exc)
        print(f"    {exc}")

    print("\nCONSTRUIR DATASET OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

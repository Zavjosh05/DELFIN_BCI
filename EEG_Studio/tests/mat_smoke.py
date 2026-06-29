"""Verifica la conversión .mat (estructura BNCI 2a) a CSV y la adaptación de canales."""
from __future__ import annotations

import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
import scipy.io as sio

from eeg_studio.core.csv_loader import load_recording
from eeg_studio.core.mat_loader import BNCI_2A_CHANNELS, convert_bnci_mat
from eeg_studio.core.project import Project


def _make_fake_mat(path):
    classes = np.array(["left hand", "right hand", "feet", "tongue"], dtype=object)
    cal = {"X": np.random.randn(50, 25), "trial": np.array([]), "y": np.array([]),
           "fs": 250, "classes": classes}
    mi = {"X": np.random.randn(200, 25), "trial": np.array([10, 60, 110]),
          "y": np.array([1, 2, 3]), "fs": 250, "classes": classes}
    sio.savemat(path, {"data": np.array([cal, mi], dtype=object)})


def main() -> int:
    tmp = tempfile.mkdtemp()
    mat = os.path.join(tmp, "A01T.mat")
    _make_fake_mat(mat)

    print("[1] Convirtiendo .mat -> CSV (solo runs de MI)")
    csv = convert_bnci_mat(mat, only_mi=True)
    assert os.path.isfile(csv), "no se generó el CSV"

    print("[2] El CSV tiene 25 canales con nombres reales y los marcadores")
    rec = load_recording(csv)
    assert rec.n_channels == 25, f"canales: {rec.n_channels}"
    assert rec.channel_names == BNCI_2A_CHANNELS, "nombres de canal inesperados"
    assert rec.sample_rate == 250.0, rec.sample_rate
    assert rec.n_samples == 200, "solo debería incluir el run de MI"
    labels = [e["id"] for e in rec.events]
    assert labels == ["left_hand", "right_hand", "feet"], f"marcadores: {labels}"
    print(f"    canales={rec.n_channels} fs={rec.sample_rate} marcadores={labels}")

    print("[3] El proyecto respeta los nombres reales (no fuerza EPOC)")
    proj = Project.create(tmp, "mat")
    sid = proj.add_source(csv)["id"]
    names = proj.display_channel_names(proj.get_recording(sid))
    assert names[:3] == ["Fz", "FC3", "FC1"], f"alias: {names[:3]}"
    assert "AF3" not in names, "no debería renombrar a canales del EPOC+"

    print("[4] Segmentos desde marcadores con desfase y ventana")
    n = proj.segments_from_markers(sid, window=40, offset=5)
    assert n == 3, f"segmentos: {n}"
    segs = sorted(proj.state["segments"], key=lambda s: s["start"])
    assert segs[0]["start"] == 15 and segs[0]["stop"] == 55, segs[0]
    assert set(proj.labels()) == {"left_hand", "right_hand", "feet"}
    print(f"    {n} segmentos (offset 5, ventana 40), clases={proj.labels()}")

    print("\nIMPORTAR .MAT OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

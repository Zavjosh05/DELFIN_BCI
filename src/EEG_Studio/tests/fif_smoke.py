"""Verifica la conversión .fif (vía MNE) a CSV: canales, escala y marcadores."""
from __future__ import annotations

import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.core.csv_loader import load_recording
from eeg_studio.core.mne_loader import convert_with_mne, mne_available


def main() -> int:
    assert mne_available(), "MNE no está instalado"
    import mne

    tmp = tempfile.mkdtemp()
    fif = os.path.join(tmp, "demo_raw.fif")

    print("[1] Creando un .fif sintético (4 EEG + 1 stim, anotaciones)")
    info = mne.create_info(["C3", "C4", "Cz", "Pz", "STI"], 250.0,
                           ["eeg", "eeg", "eeg", "eeg", "stim"])
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-5, (5, 500))           # en Voltios (~10 µV)
    data[4] = 0.0
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_annotations(mne.Annotations(onset=[0.5, 1.5], duration=[0, 0],
                                        description=["left hand", "right hand"]))
    raw.save(fif, overwrite=True, verbose="ERROR")

    print("[2] Convirtiendo .fif -> CSV")
    csv = convert_with_mne(fif)
    rec = load_recording(csv)

    print("[3] Solo canales de datos (stim descartado) y escala en µV")
    assert rec.n_channels == 4, f"canales: {rec.n_channels} (¿se descartó el stim?)"
    assert rec.channel_names == ["C3", "C4", "Cz", "Pz"], rec.channel_names
    assert rec.sample_rate == 250.0
    assert 1.0 < float(np.std(rec.data)) < 100.0, "no se reescaló a µV"
    print(f"    canales={rec.channel_names} fs={rec.sample_rate} std≈{np.std(rec.data):.1f} µV")

    print("[4] Marcadores desde las anotaciones")
    labels = [(e["sample"], e["id"]) for e in rec.events]
    assert labels == [(125, "left_hand"), (375, "right_hand")], labels
    print(f"    marcadores={labels}")

    print("\nIMPORTAR .FIF OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

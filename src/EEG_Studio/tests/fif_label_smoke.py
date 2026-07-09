"""Verifica que se etiqueta un .fif con los marcadores del .mat sin tocar señales."""
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

from eeg_studio.core.mne_loader import label_fif_from_mat, mne_available


def main() -> int:
    assert mne_available(), "MNE no está instalado"
    import mne

    tmp = tempfile.mkdtemp()
    classes = np.array(["left hand", "right hand", "feet", "tongue"], dtype=object)
    # Run de calibración (len 30, sin trials) + run de MI (len 100, 2 trials).
    cal = {"X": np.zeros((30, 2)), "trial": np.array([]), "y": np.array([]),
           "fs": 250, "classes": classes}
    mi = {"X": np.ones((100, 2)), "trial": np.array([10, 60]), "y": np.array([1, 2]),
          "fs": 250, "classes": classes}
    mat = os.path.join(tmp, "S.mat")
    sio.savemat(mat, {"data": np.array([cal, mi], dtype=object)})

    # .fif = concatenación de los runs (len 130), señales arbitrarias.
    sig = np.random.default_rng(0).normal(0, 1e-5, (2, 130))
    raw = mne.io.RawArray(sig, mne.create_info(["C3", "C4"], 250.0, ["eeg", "eeg"]),
                          verbose="ERROR")
    fif = os.path.join(tmp, "S_PURO.fif")
    raw.save(fif, overwrite=True, verbose="ERROR")

    print("[1] Etiquetando el .fif con los marcadores del .mat")
    out, n = label_fif_from_mat(fif, mat)
    assert os.path.isfile(out) and n == 2, (out, n)
    print(f"    {n} etiquetas -> {os.path.basename(out)}")

    print("[2] Las anotaciones caen en las muestras globales correctas")
    labeled = mne.io.read_raw_fif(out, preload=True, verbose="ERROR")
    samples = [int(round(o * 250.0)) for o in labeled.annotations.onset]
    descr = list(labeled.annotations.description)
    assert samples == [40, 90], samples            # 30 (calib) + 10 / + 60
    assert descr == ["left_hand", "right_hand"], descr
    print(f"    muestras={samples} clases={descr}")

    print("[3] Las señales no se modificaron")
    assert np.allclose(labeled.get_data(), raw.get_data()), "¡las señales cambiaron!"
    print("    señales idénticas ✓")

    print("[4] Autodetección: .fif de solo runs de MI (sin calibración)")
    # .fif de longitud 100 = solo el run de MI -> el desfase NO cuenta la calibración.
    sig_mi = np.random.default_rng(1).normal(0, 1e-5, (2, 100))
    raw_mi = mne.io.RawArray(sig_mi, mne.create_info(["C3", "C4"], 250.0, ["eeg", "eeg"]),
                             verbose="ERROR")
    fif_mi = os.path.join(tmp, "S_MI.fif")
    raw_mi.save(fif_mi, overwrite=True, verbose="ERROR")
    out_mi, n_mi = label_fif_from_mat(fif_mi, mat)
    lab_mi = mne.io.read_raw_fif(out_mi, preload=False, verbose="ERROR")
    samples_mi = [int(round(o * 250.0)) for o in lab_mi.annotations.onset]
    assert samples_mi == [10, 60], samples_mi    # sin offset de calibración
    print(f"    alineado a MI-only: muestras={samples_mi}")

    print("\nETIQUETAR .FIF DESDE .MAT OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

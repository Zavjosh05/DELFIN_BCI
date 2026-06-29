"""Crea copias etiquetadas de los .fif «EEG PURO» usando los .mat originales.

Para cada ``A0xT_EEG_PURO.fif`` busca el ``A0xT.mat`` correspondiente y genera
``A0xT_EEG_PURO_etiquetado.fif`` con las clases de cada ensayo como anotaciones,
**sin modificar las señales** del .fif.

Uso (desde la carpeta EEG_Studio):
    python examples/label_fif_from_mat.py
"""
from __future__ import annotations

import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import mne  # noqa: E402

from eeg_studio.core.mne_loader import label_fif_from_mat  # noqa: E402

EEG = os.path.normpath(os.path.join(_HERE, "..", "..", "EEG"))
FIF_DIR = os.path.join(EEG, "EEG de prueba procesados")
MAT_DIR = os.path.join(EEG, "EEG de prueba")


def main() -> int:
    fifs = sorted(glob.glob(os.path.join(FIF_DIR, "*_EEG_PURO.fif")))
    if not fifs:
        print(f"No se encontraron .fif en {FIF_DIR}")
        return 1
    for fif in fifs:
        subject = os.path.basename(fif).split("_")[0]      # p. ej. A01T
        mat = os.path.join(MAT_DIR, subject + ".mat")
        if not os.path.isfile(mat):
            print(f"  · {os.path.basename(fif)}: no hay .mat ({subject}.mat) -> se omite")
            continue
        dst = os.path.splitext(fif)[0] + "_etiquetado.fif"
        if os.path.isfile(dst):
            print(f"  · {os.path.basename(dst)}: ya existe -> se omite")
            continue
        out, n = label_fif_from_mat(fif, mat)
        # Verificación rápida.
        raw = mne.io.read_raw_fif(out, preload=False, verbose="ERROR")
        from collections import Counter
        counts = dict(Counter(raw.annotations.description))
        print(f"  · {os.path.basename(out)}: {n} etiquetas {counts}")
    print("\nListo. Importa los .fif etiquetados con  Proyecto → Importar dataset…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

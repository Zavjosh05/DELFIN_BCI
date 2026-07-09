"""Conversión de datasets en ``.mat`` (BNCI Horizon 2020 / BCI Competition IV 2a)
al CSV en formato OpenViBE que lee la aplicación.

El dataset «001-2014 — Four class motor imagery» guarda en cada ``.mat`` una
variable ``data`` con varios *runs*; cada run tiene ``X`` (muestras × 25 canales),
``trial`` (índices de inicio de cada ensayo), ``y`` (clase 1–4), ``fs`` (250 Hz) y
``classes`` (nombres de las 4 clases). Los 3 últimos canales son EOG.

Se concatenan los runs de imaginación motora (los que tienen ensayos) en una señal
continua y se colocan **marcadores** en el inicio de cada ensayo con el nombre de
su clase, para etiquetar/segmentar después.
"""
from __future__ import annotations

import os

import numpy as np

# Montaje estándar del BCI Competition IV 2a: 22 canales EEG + 3 EOG.
BNCI_2A_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
    "EOG-left", "EOG-central", "EOG-right",
]


def converted_csv_path(src_path: str) -> str:
    """Ruta del CSV resultante de convertir un dataset (CSV **comprimido**).

    Los archivos convertidos son grandes; se guardan como ``.csv.gz`` (gzip), que
    pandas lee/escribe de forma transparente, ~3-4× más pequeños que el texto.
    """
    return os.path.splitext(src_path)[0] + ".csv.gz"


def _label_for(y, i, classes) -> str:
    try:
        v = int(np.atleast_1d(y)[i])
    except Exception:  # noqa: BLE001
        return "desconocido"
    if classes and 1 <= v <= len(classes):
        return str(classes[v - 1]).strip().replace(" ", "_")
    return f"clase_{v}"


def bnci_trial_markers(mat_path: str, count_all_runs: bool = True) -> tuple[list[tuple[int, str]], float, int]:
    """Marcadores de ensayo en muestras **globales** ``(sample, etiqueta)``.

    ``count_all_runs=True`` acumula el desfase sobre **todos** los runs (incl.
    calibración) → alinea con un ``.fif`` que concatena los 9 runs; ``False``
    acumula solo los runs de imaginación motora → alinea con un ``.fif`` que solo
    contiene esos runs. Devuelve ``(markers, fs, total_muestras)``.
    """
    import scipy.io as sio

    m = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    runs = np.atleast_1d(m["data"])
    markers: list[tuple[int, str]] = []
    classes = None
    fs = 250.0
    offset = 0
    for run in runs:
        X = getattr(run, "X", None)
        if X is None:
            continue
        X = np.asarray(X)
        if X.ndim != 2:
            continue
        trial = np.atleast_1d(getattr(run, "trial", np.array([])))
        y = getattr(run, "y", np.array([]))
        fs = float(getattr(run, "fs", fs) or fs)
        cls = getattr(run, "classes", None)
        if cls is not None:
            classes = [str(c) for c in np.atleast_1d(cls)]
        has_trials = trial.size > 0
        if has_trials:
            for i, t in enumerate(np.atleast_1d(trial)):
                markers.append((offset + int(t), _label_for(y, i, classes)))
        if count_all_runs or has_trials:
            offset += X.shape[0]
    return markers, fs, offset


def convert_bnci_mat(mat_path: str, csv_path: str | None = None,
                     only_mi: bool = True, progress=None) -> str:
    """Convierte un ``.mat`` del dataset 2a a CSV (formato OpenViBE).

    ``only_mi`` = solo los runs con ensayos (imaginación motora). Devuelve la ruta
    del CSV generado (por defecto, junto al ``.mat``).
    """
    import scipy.io as sio

    m = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    if "data" not in m:
        raise ValueError("El .mat no tiene la estructura esperada (falta 'data').")
    runs = np.atleast_1d(m["data"])

    blocks: list[np.ndarray] = []
    markers: list[tuple[int, str]] = []
    classes = None
    fs = 250.0
    offset = 0
    for run in runs:
        X = getattr(run, "X", None)
        if X is None:
            continue
        X = np.ascontiguousarray(np.nan_to_num(np.asarray(X, dtype=np.float64)))
        if X.ndim != 2:
            continue
        trial = np.atleast_1d(getattr(run, "trial", np.array([])))
        y = getattr(run, "y", np.array([]))
        fs = float(getattr(run, "fs", fs) or fs)
        cls = getattr(run, "classes", None)
        if cls is not None:
            classes = [str(c) for c in np.atleast_1d(cls)]
        has_trials = trial.size > 0
        if only_mi and not has_trials:
            continue
        if has_trials:
            for i, t in enumerate(np.atleast_1d(trial)):
                markers.append((offset + int(t), _label_for(y, i, classes)))
        blocks.append(X)
        offset += X.shape[0]

    if not blocks:
        raise ValueError("No se encontraron runs con señal en el .mat.")

    data = np.concatenate(blocks, axis=0)            # (N, n_canales)
    n_ch = data.shape[1]
    names = (BNCI_2A_CHANNELS if n_ch == len(BNCI_2A_CHANNELS)
             else [f"Ch{i + 1}" for i in range(n_ch)])

    csv_path = csv_path or converted_csv_path(mat_path)
    write_openvibe_csv(csv_path, data, fs, names, markers, progress)
    return csv_path


def write_openvibe_csv(csv_path, data, fs, names, markers, progress=None) -> None:
    """Escribe ``(N, n_canales)`` como CSV en formato OpenViBE con marcadores."""
    import pandas as pd

    n = data.shape[0]
    df = pd.DataFrame(data, columns=list(names))
    df.insert(0, f"Time:{int(round(fs))}Hz", np.arange(n) / fs)
    df.insert(1, "Epoch", 0)

    event_id = np.full(n, "", dtype=object)
    event_date = np.full(n, "", dtype=object)
    event_dur = np.full(n, "", dtype=object)
    for sample, label in markers:
        if 0 <= sample < n:
            event_id[sample] = label
            event_date[sample] = f"{sample / fs:.6f}"
            event_dur[sample] = "0"
    df["Event Id"] = event_id
    df["Event Date"] = event_date
    df["Event Duration"] = event_dur

    if progress:
        progress(1, 1)
    df.to_csv(csv_path, index=False, float_format="%.3f")

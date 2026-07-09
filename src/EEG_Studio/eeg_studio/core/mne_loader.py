"""Conversión de datasets en formatos profesionales (vía MNE-Python) a CSV.

Soporta ``.fif`` (formato nativo de MNE) y, de paso, otros formatos estándar que
MNE lee (``.edf``/``.bdf``, ``.gdf``, BrainVision ``.vhdr``, EEGLAB ``.set``).

Conserva los canales de datos (EEG/EOG/…), descarta los de estímulo/auxiliares,
escala a microvoltios y obtiene marcadores de las **anotaciones** o, si las hay,
de un **canal de estímulo** con códigos de evento discretos.

El import de MNE está protegido: si no está instalado, la app sigue funcionando y
solo se deshabilita la importación de estos formatos.
"""
from __future__ import annotations

import os

import numpy as np

from .mat_loader import converted_csv_path, write_openvibe_csv

try:
    import mne
    _MNE_OK = True
except Exception:  # noqa: BLE001
    mne = None
    _MNE_OK = False


def mne_available() -> bool:
    return _MNE_OK


# Extensión -> función lectora de MNE.
_READERS = {
    ".fif": "read_raw_fif", ".edf": "read_raw_edf", ".bdf": "read_raw_bdf",
    ".gdf": "read_raw_gdf", ".vhdr": "read_raw_brainvision", ".set": "read_raw_eeglab",
}

# Códigos de evento de clase del protocolo Graz / BCI IV 2a.
GRAZ_CLASS_CODES = {769: "left_hand", 770: "right_hand", 771: "feet", 772: "tongue"}


def supported_extensions() -> tuple[str, ...]:
    return tuple(_READERS)


def convert_with_mne(path: str, csv_path: str | None = None, progress=None) -> str:
    """Convierte un archivo legible por MNE a CSV en formato OpenViBE."""
    if not _MNE_OK:
        raise RuntimeError("MNE no está instalado (pip install mne).")
    ext = os.path.splitext(path)[1].lower()
    reader = _READERS.get(ext)
    if reader is None:
        raise ValueError(f"Formato no soportado: {ext}")

    raw = getattr(mne.io, reader)(path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])

    # Canales de datos (descarta stim/misc/auxiliares).
    picks = mne.pick_types(raw.info, eeg=True, eog=True, ecg=True, emg=True,
                           seeg=True, stim=False, misc=False)
    if len(picks) == 0:
        picks = mne.pick_types(raw.info, eeg=True, eog=True, exclude=[])
    names = [raw.info["ch_names"][i] for i in picks]

    data = raw.get_data(picks=picks)            # (n_canales, N)
    if np.nanmax(np.abs(data)) < 0.01:          # parece estar en Voltios -> µV
        data = data * 1e6
    data = np.nan_to_num(np.ascontiguousarray(data)).T   # (N, n_canales)

    markers = _extract_markers(raw, sfreq)
    csv_path = csv_path or converted_csv_path(path)
    write_openvibe_csv(csv_path, data, sfreq, names, markers, progress)
    return csv_path


def label_fif_from_mat(fif_path: str, mat_path: str, out_path: str | None = None) -> tuple[str, int]:
    """Crea un ``.fif`` nuevo = señales del ``.fif`` + etiquetas del ``.mat``.

    **No modifica las señales**: solo añade los marcadores de los ensayos (clase
    de imaginación motora) como *anotaciones*. Requiere que el ``.fif`` sea la
    concatenación de los runs del ``.mat`` (misma longitud). Devuelve la ruta del
    nuevo archivo y el número de etiquetas añadidas.
    """
    if not _MNE_OK:
        raise RuntimeError("MNE no está instalado (pip install mne).")
    from .mat_loader import bnci_trial_markers

    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    n = raw.n_times

    # Autodetecta la alineación: el .fif puede ser los 9 runs o solo los de MI.
    markers = None
    for count_all in (True, False):
        mk, _, total = bnci_trial_markers(mat_path, count_all_runs=count_all)
        if total == n:                      # longitud exacta -> alineación correcta
            markers = mk
            break
    if markers is None:                     # ninguna longitud coincide: probar la que encaje
        for count_all in (True, False):
            mk, _, _ = bnci_trial_markers(mat_path, count_all_runs=count_all)
            if mk and max(s for s, _ in mk) < n:
                markers = mk
                break
    if not markers:
        raise ValueError(
            f"No se pudo alinear el .mat con el .fif ({n} muestras): "
            "¿son del mismo sujeto/sesión?")

    onset = [s / sfreq for s, _ in markers]
    description = [lab for _, lab in markers]
    raw.set_annotations(mne.Annotations(onset=onset, duration=[0.0] * len(onset),
                                        description=description))
    out_path = out_path or (os.path.splitext(fif_path)[0] + "_etiquetado.fif")
    raw.save(out_path, overwrite=True, verbose="ERROR")
    return out_path, len(markers)


def _extract_markers(raw, sfreq: float) -> list[tuple[int, str]]:
    """Marcadores ``(muestra, etiqueta)`` desde anotaciones o canal de estímulo."""
    markers: list[tuple[int, str]] = []
    if len(raw.annotations):
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
            markers.append((int(round(onset * sfreq)), str(desc).strip().replace(" ", "_")))
        return markers

    # Eventos en canal de estímulo (solo si son códigos discretos válidos).
    if len(mne.pick_types(raw.info, stim=True)) == 0:
        return markers
    try:
        events = mne.find_events(raw, consecutive=True, min_duration=0,
                                 shortest_event=1, verbose="ERROR")
    except Exception:  # noqa: BLE001
        return markers
    # Un canal de estímulo limpio tiene pocos eventos; si cambia casi cada muestra
    # no es un canal de trigger (no aporta etiquetas).
    if events.shape[0] == 0 or events.shape[0] > raw.n_times * 0.05:
        return markers
    codes = set(int(c) for c in events[:, 2])
    class_codes = codes & set(GRAZ_CLASS_CODES)
    for sample, _prev, code in events:
        code = int(code)
        if class_codes and code not in GRAZ_CLASS_CODES:
            continue                            # con clases presentes, solo esas
        markers.append((int(sample), GRAZ_CLASS_CODES.get(code, f"evt_{code}")))
    return markers

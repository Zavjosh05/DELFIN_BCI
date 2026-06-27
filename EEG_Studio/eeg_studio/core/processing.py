"""Procesamiento y extracción de características de segmentos EEG.

Las funciones de este módulo son de nivel superior y *picklables*, de modo que
puedan ejecutarse en un ``ProcessPoolExecutor`` para acelerar la extracción de
características de muchos segmentos en paralelo (ver :mod:`eeg_studio.workers`).
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from ..config import FREQ_BANDS

# NumPy 2.0 renombró ``trapz`` a ``trapezoid``; soportamos ambas versiones.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


# --- Espectro --------------------------------------------------------------
def power_spectral_density(data: np.ndarray, fs: float, nperseg: int | None = None):
    """PSD por Welch. Devuelve ``(freqs, psd)`` con psd ``(n_canales, n_freqs)``."""
    n = data.shape[1]
    if nperseg is None:
        nperseg = int(min(n, max(64, fs * 2)))
    nperseg = min(nperseg, n)
    freqs, psd = sp_signal.welch(data, fs=fs, nperseg=nperseg, axis=1)
    return freqs, psd


def band_powers(data: np.ndarray, fs: float, bands: dict | None = None) -> dict[str, np.ndarray]:
    """Potencia absoluta por banda. Devuelve ``{banda: array(n_canales,)}``."""
    bands = bands or FREQ_BANDS
    freqs, psd = power_spectral_density(data, fs)
    out: dict[str, np.ndarray] = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            out[name] = np.zeros(data.shape[0])
        else:
            out[name] = _trapezoid(psd[:, mask], freqs[mask], axis=1)
    return out


# --- Características en el tiempo -------------------------------------------
def hjorth_parameters(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Actividad, movilidad y complejidad de Hjorth por canal."""
    d1 = np.diff(data, axis=1)
    d2 = np.diff(d1, axis=1)
    var0 = data.var(axis=1)
    var1 = d1.var(axis=1)
    var2 = d2.var(axis=1)
    var0_safe = np.where(var0 == 0, 1e-12, var0)
    var1_safe = np.where(var1 == 0, 1e-12, var1)
    mobility = np.sqrt(var1_safe / var0_safe)
    complexity = np.sqrt(var2 / var1_safe) / mobility
    return var0, mobility, complexity


def line_length(data: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(data, axis=1)).sum(axis=1)


def time_features(data: np.ndarray) -> dict[str, np.ndarray]:
    activity, mobility, complexity = hjorth_parameters(data)
    return {
        "mean": data.mean(axis=1),
        "std": data.std(axis=1),
        "rms": np.sqrt((data ** 2).mean(axis=1)),
        "ptp": np.ptp(data, axis=1),
        "line_length": line_length(data),
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
    }


# --- Vector de características completo -------------------------------------
def extract_feature_vector(
    data: np.ndarray,
    fs: float,
    use_bands: bool = True,
    use_time: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Aplana características por canal en un único vector.

    Devuelve ``(vector, nombres)`` donde ``nombres`` etiqueta cada componente
    como ``"<canal_idx>:<caracteristica>"``.
    """
    n_ch = data.shape[0]
    values: list[float] = []
    names: list[str] = []

    if use_bands:
        bp = band_powers(data, fs)
        for band, arr in bp.items():
            for ch in range(n_ch):
                values.append(float(arr[ch]))
                names.append(f"ch{ch}:{band}")

    if use_time:
        tf = time_features(data)
        for feat, arr in tf.items():
            for ch in range(n_ch):
                values.append(float(arr[ch]))
                names.append(f"ch{ch}:{feat}")

    return np.asarray(values, dtype=np.float64), names

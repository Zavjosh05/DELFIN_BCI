"""Técnicas de preprocesamiento de señales EEG.

Cada función recibe una matriz ``(n_canales, n_muestras)`` y devuelve una copia
transformada; nunca modifica la entrada in situ. El pipeline se describe como
una lista de pasos serializables (dict), lo que permite guardarlo en el proyecto
y reaplicarlo de forma reproducible sobre la fuente original.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis as _kurtosis


# --- Pasos individuales ----------------------------------------------------
def detrend(data: np.ndarray, type: str = "linear") -> np.ndarray:
    return sp_signal.detrend(data, axis=1, type=type)


def bandpass(data: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    low = max(low, 1e-6)
    high = min(high, nyq - 1e-6)
    sos = sp_signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sp_signal.sosfiltfilt(sos, data, axis=1)


def highpass(data: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    sos = sp_signal.butter(order, cutoff / nyq, btype="high", output="sos")
    return sp_signal.sosfiltfilt(sos, data, axis=1)


def lowpass(data: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    sos = sp_signal.butter(order, min(cutoff, nyq - 1e-6) / nyq, btype="low", output="sos")
    return sp_signal.sosfiltfilt(sos, data, axis=1)


def notch(data: np.ndarray, fs: float, freq: float = 60.0, q: float = 30.0) -> np.ndarray:
    nyq = fs / 2.0
    if freq >= nyq:
        return data.copy()
    b, a = sp_signal.iirnotch(freq / nyq, q)
    return sp_signal.filtfilt(b, a, data, axis=1)


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Re-referencia por promedio común (CAR)."""
    return data - data.mean(axis=0, keepdims=True)


def reference_to_channel(data: np.ndarray, channel: int) -> np.ndarray:
    return data - data[channel:channel + 1, :]


def normalize(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    if method == "zscore":
        mean = data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (data - mean) / std
    if method == "minmax":
        mn = data.min(axis=1, keepdims=True)
        mx = data.max(axis=1, keepdims=True)
        rng = mx - mn
        rng[rng == 0] = 1.0
        return (data - mn) / rng
    raise ValueError(f"Método de normalización desconocido: {method}")


def ica_artifact(data: np.ndarray, n_components: int = 0, kurt_threshold: float = 5.0) -> np.ndarray:
    """Elimina artefactos por ICA: rechaza componentes de kurtosis alta.

    Los parpadeos oculares y la actividad muscular producen componentes
    independientes muy "picudos" (kurtosis elevada). Se descompone la señal con
    ICA, se anulan esos componentes y se reconstruye. Enfoque clásico de
    eliminación de artefactos (revisión doi:10.18280/isi.290124).
    """
    import warnings

    from sklearn.decomposition import FastICA
    from sklearn.exceptions import ConvergenceWarning

    n_ch = data.shape[0]
    ncomp = n_ch if not n_components else min(int(n_components), n_ch)
    X = data.T  # (muestras, canales)
    # max_iter alto + tol algo más laxa reducen los avisos de no-convergencia;
    # aun sin converger del todo el resultado es utilizable, así que se silencia
    # el ConvergenceWarning (la no-convergencia ya se gestiona con el try/except).
    ica = FastICA(n_components=ncomp, random_state=0, max_iter=1000, tol=1e-3,
                  whiten="unit-variance")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            sources = ica.fit_transform(X)           # (muestras, componentes)
    except Exception:  # noqa: BLE001 - no converge: devolver la señal intacta
        return data.copy()
    k = _kurtosis(sources, axis=0, fisher=True)       # exceso de kurtosis por componente
    artifact = np.abs(k) > kurt_threshold
    if artifact.any() and not artifact.all():         # no anular toda la señal
        sources[:, artifact] = 0.0
    cleaned = ica.inverse_transform(sources)          # (muestras, canales)
    return np.ascontiguousarray(cleaned.T, dtype=np.float64)


# --- Registro de pasos para el pipeline ------------------------------------
# Cada paso del pipeline es un dict: {"type": <str>, "params": {...}}.
# Las funciones se invocan con (data, fs, **params); ignoran fs si no lo usan.
def _wrap(fn: Callable, use_fs: bool) -> Callable:
    def _apply(data, fs, **params):
        return fn(data, fs, **params) if use_fs else fn(data, **params)
    return _apply


STEP_REGISTRY: dict[str, Callable] = {
    "detrend": _wrap(detrend, use_fs=False),
    "bandpass": _wrap(bandpass, use_fs=True),
    "highpass": _wrap(highpass, use_fs=True),
    "lowpass": _wrap(lowpass, use_fs=True),
    "notch": _wrap(notch, use_fs=True),
    "car": _wrap(common_average_reference, use_fs=False),
    "reference": _wrap(reference_to_channel, use_fs=False),
    "normalize": _wrap(normalize, use_fs=False),
    "ica": _wrap(ica_artifact, use_fs=False),
}

# Etiquetas legibles para la interfaz.
STEP_LABELS = {
    "detrend": "Eliminar tendencia",
    "bandpass": "Filtro pasa-banda",
    "highpass": "Filtro pasa-altas",
    "lowpass": "Filtro pasa-bajas",
    "notch": "Filtro notch (red eléctrica)",
    "car": "Referencia promedio común (CAR)",
    "reference": "Referenciar a canal",
    "normalize": "Normalizar",
    "ica": "Eliminar artefactos (ICA)",
}

# Descripción de cada filtro/paso (qué hace) para mostrar en la interfaz.
STEP_DESCRIPTIONS = {
    "detrend": "Elimina la tendencia (deriva lenta) de cada canal restando una "
               "recta o la media ajustada a la señal.",
    "bandpass": "Deja pasar solo las frecuencias entre 'low' y 'high' y atenúa el "
                "resto: quita a la vez la deriva lenta y el ruido de alta frecuencia.",
    "highpass": "Atenúa las frecuencias por debajo de 'cutoff' (deriva, offset DC) "
                "y deja pasar las altas.",
    "lowpass": "Atenúa las frecuencias por encima de 'cutoff' (ruido rápido) y deja "
               "pasar las bajas.",
    "notch": "Elimina una banda muy estrecha en torno a 'freq': sirve para quitar la "
             "interferencia de la red eléctrica (50/60 Hz).",
    "car": "Referencia de promedio común (CAR). A cada canal le resta, en cada "
           "instante, el promedio de TODOS los canales activos. Así elimina lo que "
           "es común a todo el casco (interferencia de red, deriva global, la "
           "referencia física) y resalta la actividad local de cada electrodo. "
           "No tiene parámetros: usa todos los canales activos, por lo que si "
           "excluyes los EOG, esos no entran en el promedio. Cuidado con pocos "
           "canales o con un canal saturado: ese ruido se repartiría a todos.",
    "reference": "Re-referencia la señal restando un canal concreto (el de referencia) "
                 "a todos los demás. Útil si quieres una referencia física (p. ej. "
                 "una mastoides) en lugar del promedio común (CAR).",
    "normalize": "Reescala cada canal para homogeneizar amplitudes entre canales y "
                 "grabaciones. 'zscore' deja media 0 y desviación 1; 'minmax' lleva "
                 "cada canal al rango 0–1. Útil antes de modelos sensibles a la escala.",
    "ica": "Descompone la señal en componentes independientes (ICA) y elimina los de "
           "kurtosis alta (parpadeos, músculo), reconstruyendo sin esos artefactos.",
}

# Descripción de cada parámetro y el efecto de modificarlo.
PARAM_DESCRIPTIONS = {
    "low": "Frecuencia de corte inferior (Hz). Súbela para eliminar más deriva/ondas "
           "lentas; bájala para conservarlas.",
    "high": "Frecuencia de corte superior (Hz). Bájala para quitar más ruido rápido; "
            "súbela para conservar componentes de alta frecuencia.",
    "cutoff": "Frecuencia de corte (Hz) a partir de la cual el filtro empieza a atenuar.",
    "order": "Orden del filtro. Mayor orden = transición más abrupta entre lo que pasa "
             "y lo que se atenúa, pero más riesgo de distorsión/inestabilidad.",
    "freq": "Frecuencia central a eliminar (Hz). Normalmente 50 o 60 Hz (red eléctrica).",
    "q": "Factor de calidad. Mayor Q = muesca más estrecha, afecta menos a las "
         "frecuencias vecinas.",
    "type": "Tipo de tendencia a quitar: 'linear' (una recta) o 'constant' (solo la media).",
    "method": "Método de escalado: 'zscore' (media 0, desviación 1) o 'minmax' (rango 0–1).",
    "channel": "Índice del canal que se usa como referencia para restar a los demás.",
    "n_components": "Nº de componentes ICA (0 = tantos como canales). Menos componentes "
                    "= descomposición más gruesa y rápida.",
    "kurt_threshold": "Umbral de kurtosis para marcar un componente como artefacto. "
                      "Más bajo = elimina más componentes (más agresivo).",
}

# Parámetros por defecto al añadir un paso desde la interfaz.
STEP_DEFAULTS = {
    "detrend": {"type": "linear"},
    "bandpass": {"low": 1.0, "high": 45.0, "order": 4},
    "highpass": {"cutoff": 1.0, "order": 4},
    "lowpass": {"cutoff": 45.0, "order": 4},
    "notch": {"freq": 60.0, "q": 30.0},
    "car": {},
    "reference": {"channel": 0},
    "normalize": {"method": "zscore"},
    "ica": {"n_components": 0, "kurt_threshold": 5.0},
}


def apply_pipeline(data: np.ndarray, fs: float, pipeline: list[dict],
                   progress: Callable | None = None) -> np.ndarray:
    """Aplica la secuencia de pasos **activos** a una copia de ``data``.

    Los pasos con ``"enabled": False`` se omiten (se pueden activar/desactivar sin
    borrarlos). ``progress(hechos, total)`` informa del avance paso a paso.
    """
    out = np.ascontiguousarray(data, dtype=np.float64).copy()
    steps = [s for s in pipeline if s.get("enabled", True)]
    total = len(steps)
    for i, step in enumerate(steps):
        stype = step.get("type")
        if stype not in STEP_REGISTRY:
            raise ValueError(f"Paso de preprocesamiento desconocido: {stype}")
        params = dict(step.get("params", {}))
        out = STEP_REGISTRY[stype](out, fs, **params)
        if progress is not None:
            progress(i + 1, total)
    return np.ascontiguousarray(out, dtype=np.float64)

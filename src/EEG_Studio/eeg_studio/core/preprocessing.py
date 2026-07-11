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


def _odd_numtaps(numtaps: int, n_samples: int) -> int:
    """Nº de coeficientes FIR impar y lo bastante corto para ``filtfilt``.

    ``filtfilt`` necesita que la señal sea más larga que ~3·(numtaps-1); en
    segmentos cortos se recorta el filtro para no fallar.
    """
    numtaps = max(3, int(numtaps))
    if numtaps % 2 == 0:
        numtaps += 1
    max_taps = max(3, n_samples // 3)
    if max_taps % 2 == 0:
        max_taps -= 1
    return min(numtaps, max_taps)


def _sosfiltfilt(sos, data: np.ndarray) -> np.ndarray:
    """``sosfiltfilt`` robusto: en señales muy cortas reduce el ``padlen`` para no
    fallar (``The length of the input vector x must be greater than padlen``)."""
    n = int(data.shape[1])
    if n < 2:
        return np.asarray(data, dtype=float).copy()
    try:
        return sp_signal.sosfiltfilt(sos, data, axis=1)
    except ValueError:
        return sp_signal.sosfiltfilt(sos, data, axis=1, padlen=n - 1)


def _filtfilt(b, a, data: np.ndarray) -> np.ndarray:
    """``filtfilt`` (FIR/IIR) robusto, con la misma protección para señales cortas."""
    n = int(data.shape[1])
    if n < 2:
        return np.asarray(data, dtype=float).copy()
    try:
        return sp_signal.filtfilt(b, a, data, axis=1)
    except ValueError:
        return sp_signal.filtfilt(b, a, data, axis=1, padlen=n - 1)


def bandpass(data: np.ndarray, fs: float, low: float, high: float, order: int = 4,
             design: str = "butter", numtaps: int = 101) -> np.ndarray:
    """Pasa-banda en fase cero. ``design``: 'butter' (IIR) o 'fir' (FIR ventaneado).

    Tolera ``low``/``high`` desordenados y rangos inválidos (si no queda banda,
    devuelve la señal sin filtrar en vez de reventar)."""
    nyq = fs / 2.0
    low, high = sorted((float(low), float(high)))      # tolera low > high
    low = max(low, 1e-6)
    high = min(high, nyq - 1e-6)
    if low >= high:                                    # rango inválido: no filtra
        return np.asarray(data, dtype=float).copy()
    if design == "fir":
        taps = sp_signal.firwin(_odd_numtaps(numtaps, data.shape[1]), [low, high],
                                pass_zero=False, fs=fs)
        return _filtfilt(taps, [1.0], data)
    sos = sp_signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return _sosfiltfilt(sos, data)


def highpass(data: np.ndarray, fs: float, cutoff: float, order: int = 4,
             design: str = "butter", numtaps: int = 101) -> np.ndarray:
    nyq = fs / 2.0
    cutoff = min(max(float(cutoff), 1e-6), nyq - 1e-6)
    if design == "fir":
        taps = sp_signal.firwin(_odd_numtaps(numtaps, data.shape[1]), cutoff,
                                pass_zero=False, fs=fs)
        return _filtfilt(taps, [1.0], data)
    sos = sp_signal.butter(order, cutoff / nyq, btype="high", output="sos")
    return _sosfiltfilt(sos, data)


def lowpass(data: np.ndarray, fs: float, cutoff: float, order: int = 4,
            design: str = "butter", numtaps: int = 101) -> np.ndarray:
    nyq = fs / 2.0
    cutoff = min(max(float(cutoff), 1e-6), nyq - 1e-6)   # acota a (0, nyq)
    if design == "fir":
        taps = sp_signal.firwin(_odd_numtaps(numtaps, data.shape[1]), cutoff,
                                pass_zero=True, fs=fs)
        return _filtfilt(taps, [1.0], data)
    sos = sp_signal.butter(order, cutoff / nyq, btype="low", output="sos")
    return _sosfiltfilt(sos, data)


def notch(data: np.ndarray, fs: float, freq: float = 60.0, q: float = 30.0,
          design: str = "iir", numtaps: int = 257) -> np.ndarray:
    """Elimina una banda estrecha en torno a ``freq`` (interferencia de red).

    Parámetros:
      * ``freq``    Frecuencia central a eliminar en Hz (50 o 60 según el país).
      * ``q``       Factor de calidad: ancho de la muesca ≈ ``freq/q`` Hz. Mayor Q
                    = muesca más estrecha (afecta menos a las frecuencias vecinas).
      * ``design``  ``"iir"`` (recomendado) = notch IIR (``scipy.iirnotch``), muy
                    estrecho y barato; ``"fir"`` = band-stop FIR de **fase lineal**.
      * ``numtaps`` Solo para FIR: nº de coeficientes. Una muesca tan estrecha
                    necesita MUCHOS coeficientes y **segmentos largos**; si el
                    segmento es corto se recorta (la muesca se ensancha/atenúa).
    """
    nyq = fs / 2.0
    if freq >= nyq:
        return data.copy()
    if design == "fir":
        bw = max(freq / max(q, 1e-6), 0.5)         # ancho de banda de la muesca (Hz)
        lo = max(freq - bw / 2.0, 1e-6)
        hi = min(freq + bw / 2.0, nyq - 1e-6)
        taps = sp_signal.firwin(_odd_numtaps(numtaps, data.shape[1]), [lo, hi],
                                pass_zero=True, fs=fs)   # pass_zero=True => band-stop
        return _filtfilt(taps, [1.0], data)
    b, a = sp_signal.iirnotch(freq / nyq, q)
    return _filtfilt(b, a, data)


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


def asr_reconstruct(data: np.ndarray, fs: float, window_sec: float = 0.5,
                    cutoff: float = 5.0) -> np.ndarray:
    """Reconstrucción de Subespacios de Artefactos (ASR), versión simplificada.

    Inspirado en Mullen et al. 2013 (ASR de clean_rawdata/EEGLAB): en vez de
    corregir canal a canal, se descompone la señal en componentes espaciales
    (autovectores de la covarianza entre canales, vía PCA) y se estima, por
    ventanas, la energía (RMS) de cada componente. Un componente cuya energía en
    una ventana supera ``cutoff`` veces su nivel habitual (mediana robusta de sus
    RMS en todas las ventanas) se atenúa hasta ese límite antes de reconstruir la
    señal en el espacio original. Así se corrigen ráfagas de artefacto (saltos de
    electrodo, movimiento) sin descartar la ventana ni afectar a los componentes
    que sí llevan señal limpia.
    """
    n_ch, n_samples = data.shape
    x = np.asarray(data, dtype=np.float64)
    if n_ch < 2 or n_samples < 4:
        return x.copy()

    mean = x.mean(axis=1, keepdims=True)
    xc = x - mean
    cov = np.cov(xc)
    if not np.all(np.isfinite(cov)):
        return x.copy()

    eigvals, eigvecs = np.linalg.eigh(cov)        # covarianza simétrica: estable
    comps = eigvecs.T @ xc                        # (n_componentes, n_muestras)

    win = max(1, int(round(window_sec * fs)))
    n_win = int(np.ceil(n_samples / win))
    rms = np.zeros((comps.shape[0], n_win))
    for w in range(n_win):
        seg = comps[:, w * win:(w + 1) * win]
        if seg.shape[1]:
            rms[:, w] = np.sqrt(np.mean(seg ** 2, axis=1))

    baseline = np.median(rms, axis=1, keepdims=True)   # nivel habitual por componente
    baseline[baseline <= 0] = 1e-9
    limit = cutoff * baseline

    comps_clean = comps.copy()
    for w in range(n_win):
        sl = slice(w * win, min((w + 1) * win, n_samples))
        over = rms[:, w] > limit[:, 0]
        if np.any(over):
            scale = np.ones(comps.shape[0])
            scale[over] = limit[over, 0] / np.maximum(rms[over, w], 1e-9)
            comps_clean[:, sl] = comps_clean[:, sl] * scale[:, None]

    reconstructed = eigvecs @ comps_clean + mean
    return np.ascontiguousarray(reconstructed, dtype=np.float64)


def threshold_reject(data: np.ndarray, mode: str = "manual", threshold: float = 100.0,
                     k: float = 5.0) -> np.ndarray:
    """Rechazo por umbral: recorta (clip) las muestras que exceden un límite de
    amplitud, canal a canal, en vez de descartar la ventana entera.

    ``mode``:
      * ``"manual"``     usa ``threshold`` (µV) tal cual, simétrico en torno a 0.
      * ``"automatico"`` calcula un umbral propio por canal a partir de sus datos:
                         mediana ± ``k`` × MAD (desviación absoluta mediana,
                         escalada a una desviación estándar equivalente), así se
                         adapta a la amplitud típica de cada canal en vez de un
                         valor fijo para todos.
    """
    out = np.asarray(data, dtype=np.float64).copy()
    for ch in range(out.shape[0]):
        x = out[ch]
        if mode == "automatico":
            med = np.median(x)
            mad = np.median(np.abs(x - med))
            robust_std = mad * 1.4826 if mad > 0 else (float(x.std()) or 1.0)
            thr = max(float(k), 0.0) * robust_std
            lo, hi = med - thr, med + thr
        else:
            thr = abs(float(threshold))
            lo, hi = -thr, thr
        if hi > lo:
            np.clip(x, lo, hi, out=x)
    return out


def baseline_correction(data: np.ndarray, fs: float, baseline_sec: float = 0.2) -> np.ndarray:
    """Corrección de la línea base: resta a cada canal la media de una ventana de
    referencia al inicio del segmento (típico en análisis tipo ERP: alinear al
    nivel previo al evento). A diferencia de 'Eliminar tendencia' (que ajusta una
    recta/media a TODO el segmento), aquí la referencia es solo el tramo inicial
    de duración ``baseline_sec`` (segundos); si es mayor que el segmento, se usa
    el segmento completo como línea base.
    """
    x = np.asarray(data, dtype=np.float64)
    n = x.shape[1]
    if n <= 0:
        return x.copy()
    n_base = min(max(int(round(baseline_sec * fs)), 1), n)
    baseline_mean = x[:, :n_base].mean(axis=1, keepdims=True)
    return x - baseline_mean


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
    "asr": _wrap(asr_reconstruct, use_fs=True),
    "threshold": _wrap(threshold_reject, use_fs=False),
    "baseline": _wrap(baseline_correction, use_fs=True),
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
    "asr": "Reconstrucción de Subespacios de Artefactos (ASR)",
    "threshold": "Rechazo por umbral (Manual/Automático)",
    "baseline": "Corrección de la línea base",
}

# Descripción de cada filtro/paso (qué hace) para mostrar en la interfaz.
STEP_DESCRIPTIONS = {
    "detrend": "Elimina la tendencia (deriva lenta) de cada canal restando una "
               "recta o la media ajustada a la señal.",
    "bandpass": "Deja pasar solo las frecuencias entre 'low' y 'high' y atenúa el "
                "resto: quita a la vez la deriva lenta y el ruido de alta frecuencia. "
                "Elige el diseño con 'design': Butterworth (IIR) o FIR (fase lineal).",
    "highpass": "Atenúa las frecuencias por debajo de 'cutoff' (deriva, offset DC) "
                "y deja pasar las altas.",
    "lowpass": "Atenúa las frecuencias por encima de 'cutoff' (ruido rápido) y deja "
               "pasar las bajas.",
    "notch": "Elimina una banda muy estrecha en torno a 'freq': sirve para quitar la "
             "interferencia de la red eléctrica (50/60 Hz). 'design': 'iir' (notch "
             "estrecho recomendado) o 'fir' (band-stop de fase lineal).",
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
    "asr": "Reconstrucción de Subespacios de Artefactos: descompone la señal en "
           "componentes espaciales (PCA entre canales) y atenúa, ventana a ventana, "
           "los componentes cuya energía supera 'cutoff' veces su nivel habitual — "
           "corrige ráfagas de artefacto (saltos de electrodo, movimiento) sin "
           "descartar la ventana ni tocar los componentes con señal limpia.",
    "threshold": "Recorta (clip) las muestras que superan un límite de amplitud, canal "
                 "a canal, en vez de descartar la ventana. 'mode' = 'manual' usa un "
                 "valor fijo ('threshold' en µV); 'automatico' calcula un umbral propio "
                 "por canal a partir de sus datos (mediana ± k×MAD).",
    "baseline": "Resta a cada canal la media de una ventana de referencia al inicio del "
                "segmento (línea base), típico en análisis tipo ERP para alinear al "
                "nivel previo a un evento. A diferencia de 'Eliminar tendencia' (ajusta "
                "toda la señal), aquí solo se usa el tramo inicial de duración "
                "'baseline_sec' como referencia.",
}

# Descripción de cada parámetro y el efecto de modificarlo.
PARAM_DESCRIPTIONS = {
    "low": "Frecuencia de corte inferior (Hz). Súbela para eliminar más deriva/ondas "
           "lentas; bájala para conservarlas.",
    "high": "Frecuencia de corte superior (Hz). Bájala para quitar más ruido rápido; "
            "súbela para conservar componentes de alta frecuencia.",
    "cutoff": "Frecuencia de corte (Hz) a partir de la cual el filtro empieza a atenuar.",
    "order": "Orden del filtro Butterworth (IIR). Mayor orden = transición más abrupta "
             "entre lo que pasa y lo que se atenúa, pero más riesgo de inestabilidad. "
             "No aplica al diseño FIR (ahí manda 'numtaps').",
    "design": "Diseño del filtro. En pasa-banda/altas/bajas: 'butter' = Butterworth "
              "(IIR, recursivo, eficiente, orden bajo, fase cero con filtfilt) o "
              "'fir' = FIR ventaneado (fase lineal exacta, muy estable, más cómputo). "
              "En el notch: 'iir' (notch IIR estrecho, recomendado) o 'fir' "
              "(band-stop FIR de fase lineal; necesita muchos coeficientes).",
    "numtaps": "Nº de coeficientes del filtro FIR (solo si design='fir'). Más "
               "coeficientes = transición más abrupta (y muesca más estrecha en el "
               "notch), pero más cómputo y requiere segmentos más largos. Debe ser "
               "impar y caber en el segmento; se ajusta automáticamente.",
    "freq": "Frecuencia central a eliminar (Hz). Normalmente 50 Hz (Europa) o 60 Hz "
            "(América) por la red eléctrica. Debe ser menor que fs/2.",
    "q": "Factor de calidad del notch. El ancho de la muesca es ≈ freq/Q Hz "
         "(p. ej. 60/30 = 2 Hz). Mayor Q = muesca más estrecha y selectiva, afecta "
         "menos a las frecuencias vecinas. En FIR define el ancho del band-stop.",
    "type": "Tipo de tendencia a quitar: 'linear' (una recta) o 'constant' (solo la media).",
    "method": "Método de escalado: 'zscore' (media 0, desviación 1) o 'minmax' (rango 0–1).",
    "channel": "Índice del canal que se usa como referencia para restar a los demás.",
    "n_components": "Nº de componentes ICA (0 = tantos como canales). Menos componentes "
                    "= descomposición más gruesa y rápida.",
    "kurt_threshold": "Umbral de kurtosis para marcar un componente como artefacto. "
                      "Más bajo = elimina más componentes (más agresivo).",
    "window_sec": "Duración de la ventana (segundos) usada para medir la energía de "
                  "cada componente. Ventanas más cortas reaccionan más rápido a ráfagas "
                  "breves; más largas promedian mejor pero reaccionan más lento.",
    "cutoff": "Umbral de energía (en veces el nivel habitual del componente) a partir "
              "del cual se considera artefacto y se atenúa. Más bajo = más agresivo "
              "(corrige más, pero arriesga tocar señal limpia); más alto = más "
              "conservador.",
    "mode": "Cómo se fija el umbral de rechazo: 'manual' usa el valor de 'threshold' "
            "tal cual; 'automatico' lo calcula por canal a partir de sus propios datos "
            "(mediana ± k×MAD), adaptándose a la amplitud típica de cada canal.",
    "threshold": "Límite de amplitud (µV), simétrico en torno a 0. Las muestras que lo "
                 "superen se recortan a ese valor. Solo se usa si mode='manual'.",
    "k": "Multiplicador del umbral automático (mode='automatico'): límite = mediana ± "
         "k × MAD del canal. Más bajo = más agresivo (recorta más); más alto = más "
         "permisivo.",
    "baseline_sec": "Duración (segundos), desde el inicio del segmento, que se usa "
                    "como línea base de referencia. Si es mayor que el segmento, se usa "
                    "el segmento completo.",
}

# Parámetros por defecto al añadir un paso desde la interfaz.
STEP_DEFAULTS = {
    "detrend": {"type": "linear"},
    "bandpass": {"low": 1.0, "high": 45.0, "order": 4, "design": "butter", "numtaps": 101},
    "highpass": {"cutoff": 1.0, "order": 4, "design": "butter", "numtaps": 101},
    "lowpass": {"cutoff": 45.0, "order": 4, "design": "butter", "numtaps": 101},
    "notch": {"freq": 60.0, "q": 30.0, "design": "iir", "numtaps": 257},
    "car": {},
    "reference": {"channel": 0},
    "normalize": {"method": "zscore"},
    "ica": {"n_components": 0, "kurt_threshold": 5.0},
    "asr": {"window_sec": 0.5, "cutoff": 5.0},
    "threshold": {"mode": "manual", "threshold": 100.0, "k": 5.0},
    "baseline": {"baseline_sec": 0.2},
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

"""Clasificación de una ventana de señal en vivo con el modelo entrenado.

Aplica el **mismo preprocesamiento** del proyecto a la ventana entrante y, según
el tipo de modelo, extrae características (modelos sobre características) o ajusta
la ventana cruda (CNN/LSTM/EEGNet, Riemann/CSP), para que la inferencia sea
coherente con el entrenamiento.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..config import ONLINE_WINDOW_SAMPLES
from ..core import classification, preprocessing
from ..core.dataset import fit_window
from ..core.processing import extract_feature_vector


def classify_window(model, project, window: np.ndarray, fs: float) -> tuple[str, float | None]:
    """Clasifica una ventana ``(n_canales, n_muestras)``.

    Devuelve ``(clase, confianza)`` (confianza = ``None`` si el modelo no da
    probabilidades).
    """
    pipeline = project.state.get("pipeline", []) if project is not None else []
    proc = preprocessing.apply_pipeline(window, fs, pipeline) if pipeline else np.asarray(window, float)

    if getattr(model, "input_kind", "features") == "raw":
        T = int((model.nn_config or {}).get("window_samples", proc.shape[1])) \
            if model.nn_config else proc.shape[1]
        X = fit_window(np.ascontiguousarray(proc), T)[np.newaxis, ...]
    else:
        cfg = project.state.get("dataset", {}) if project is not None else {}
        vec, _ = extract_feature_vector(proc, fs, cfg.get("use_bands", True), cfg.get("use_time", True))
        X = vec.reshape(1, -1)

    pred = str(classification.predict(model, X)[0])
    proba = classification.predict_proba(model, X)
    confidence = float(np.max(proba)) if proba is not None else None
    return pred, confidence


def classify_recording(model, project, data: np.ndarray, fs: float,
                       window: int | None = None, step: int | None = None,
                       ground_truth=None) -> dict:
    """Clasifica una grabación **completa** por ventanas deslizantes y resume.

    Recorre la señal ``(n_canales, n_muestras)`` en ventanas de ``window`` muestras
    (paso ``step``, por defecto medio solape) clasificando cada una con
    :func:`classify_window`; el movimiento a ejecutar es la clase por **voto
    mayoritario**. Reutiliza el mismo preprocesamiento del proyecto, así que la
    inferencia es coherente con el entrenamiento.

    Devuelve un dict:
      * ``label``       clase ganadora (voto mayoritario) → el movimiento a hacer,
      * ``counts``      nº de ventanas por clase,
      * ``confidence``  confianza media de la clase ganadora (o ``None``),
      * ``n_windows``   nº de ventanas evaluadas,
      * ``per_window``  lista ``(muestra_central, clase, confianza)``,
      * ``accuracy``    aciertos/total frente a ``ground_truth`` (o ``None``).

    ``ground_truth`` (opcional): etiqueta verdadera **por muestra** (secuencia de
    longitud n_muestras) para medir la exactitud ventana a ventana; se compara con
    la etiqueta en la muestra central de cada ventana.
    """
    data = np.ascontiguousarray(data, dtype=np.float64)
    n = int(data.shape[1])
    win = int(window or ONLINE_WINDOW_SAMPLES)
    win = max(1, min(win, n))
    stp = int(step) if step else max(1, win // 2)

    per_window: list[tuple[int, str, float | None]] = []
    counts: dict[str, int] = {}
    conf_by_class: dict[str, list[float]] = {}
    correct = counted = 0

    i = 0
    while i + win <= n:
        cls, conf = classify_window(model, project, data[:, i:i + win], fs)
        center = i + win // 2
        per_window.append((center, cls, conf))
        counts[cls] = counts.get(cls, 0) + 1
        if conf is not None:
            conf_by_class.setdefault(cls, []).append(conf)
        if ground_truth is not None and len(ground_truth):
            truth = ground_truth[min(center, len(ground_truth) - 1)]
            if truth is not None:
                counted += 1
                correct += int(cls == truth)
        i += stp

    if not per_window:                      # ventana más larga que la grabación
        cls, conf = classify_window(model, project, data, fs)
        per_window.append((n // 2, cls, conf))
        counts[cls] = 1
        if conf is not None:
            conf_by_class.setdefault(cls, []).append(conf)

    label = max(counts, key=counts.get)
    confs = conf_by_class.get(label, [])
    return {
        "label": label,
        "counts": counts,
        "confidence": float(np.mean(confs)) if confs else None,
        "n_windows": len(per_window),
        "per_window": per_window,
        "accuracy": (correct / counted) if counted else None,
    }


class PredictionSmoother:
    """Confirma una clase solo tras ``k`` predicciones iguales seguidas.

    Evita que el ruido de clasificación haga oscilar al controlador. Notifica una
    clase **una sola vez** cuando cambia la clase estable (no la repite cada tick).
    """

    def __init__(self, k: int = 3) -> None:
        self.k = max(1, int(k))
        self._recent: deque[str] = deque(maxlen=self.k)
        self._stable: str | None = None

    def reset(self) -> None:
        self._recent.clear()
        self._stable = None

    def update(self, prediction: str) -> str | None:
        """Devuelve la clase si se confirma un **cambio** estable; si no, ``None``."""
        self._recent.append(prediction)
        if len(self._recent) == self.k and len(set(self._recent)) == 1:
            candidate = self._recent[0]
            if candidate != self._stable:
                self._stable = candidate
                return candidate
        return None

    @property
    def stable(self) -> str | None:
        return self._stable

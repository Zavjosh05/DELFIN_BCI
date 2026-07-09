"""Clasificación de una ventana de señal en vivo con el modelo entrenado.

Aplica el **mismo preprocesamiento** del proyecto a la ventana entrante y, según
el tipo de modelo, extrae características (modelos sobre características) o ajusta
la ventana cruda (CNN/LSTM/EEGNet, Riemann/CSP), para que la inferencia sea
coherente con el entrenamiento.
"""
from __future__ import annotations

from collections import deque

import numpy as np

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

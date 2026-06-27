"""Construcción de datasets a partir de segmentos etiquetados.

Reúne segmentos provenientes de varias grabaciones (varios CSV), los procesa con
el pipeline del proyecto y extrae un vector de características por segmento. La
extracción puede paralelizarse con ``ProcessPoolExecutor`` (ver
:func:`build_features_parallel`).

El dataset resultante se guarda **dentro del proyecto** (carpeta ``datasets/``),
nunca en los CSV de origen.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..config import DATASETS_DIR, N_WORKERS
from .processing import extract_feature_vector

# Por debajo de este nº de segmentos, la extracción es serie: en Windows crear
# procesos (spawn) cuesta más que el ahorro para conjuntos pequeños.
MIN_PARALLEL_SEGMENTS = 12


@dataclass
class Dataset:
    X: np.ndarray              # (n_segmentos, n_caracteristicas)
    y: np.ndarray              # (n_segmentos,) etiquetas (str)
    feature_names: list[str]
    segment_ids: list[str]

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1] if self.X.ndim == 2 else 0

    @property
    def classes(self) -> list[str]:
        return sorted(set(self.y.tolist()))


@dataclass
class RawDataset:
    """Dataset de señal cruda en ventanas fijas, para CNN/LSTM."""

    X: np.ndarray              # (n_segmentos, n_canales, T)
    y: np.ndarray              # (n_segmentos,)
    segment_ids: list[str]

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def window(self) -> int:
        return self.X.shape[2]

    @property
    def classes(self) -> list[str]:
        return sorted(set(self.y.tolist()))


def fit_window(data: np.ndarray, window: int) -> np.ndarray:
    """Ajusta ``data`` (n_canales, k) a una ventana fija de ``window`` muestras.

    Si sobra, recorta centrado; si falta, rellena con ceros centrado.
    """
    n_ch, k = data.shape
    out = np.zeros((n_ch, window), dtype=np.float64)
    if k >= window:
        start = (k - window) // 2
        out[:] = data[:, start:start + window]
    else:
        start = (window - k) // 2
        out[:, start:start + k] = data
    return out


# --- Función picklable para el pool de procesos ----------------------------
def _feature_job(args):
    data, fs, use_bands, use_time = args
    vec, names = extract_feature_vector(data, fs, use_bands=use_bands, use_time=use_time)
    return vec, names


def build_features_parallel(
    segment_arrays: list[tuple[np.ndarray, float]],
    use_bands: bool = True,
    use_time: bool = True,
    n_workers: int | None = None,
    progress=None,
) -> tuple[np.ndarray, list[str]]:
    """Extrae características de cada segmento, opcionalmente en paralelo.

    ``segment_arrays`` es una lista de ``(data, fs)``. Devuelve ``(X, nombres)``.
    """
    jobs = [(data, fs, use_bands, use_time) for data, fs in segment_arrays]
    if not jobs:
        return np.empty((0, 0)), []

    n_workers = n_workers if n_workers is not None else (N_WORKERS or os.cpu_count() or 1)
    n_workers = max(1, min(n_workers, len(jobs)))

    vectors: list[np.ndarray] = []
    names: list[str] = []

    if n_workers == 1 or len(jobs) < MIN_PARALLEL_SEGMENTS:
        for i, job in enumerate(jobs):
            vec, names = _feature_job(job)
            vectors.append(vec)
            if progress:
                progress(i + 1, len(jobs))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for i, (vec, nm) in enumerate(ex.map(_feature_job, jobs)):
                vectors.append(vec)
                names = nm
                if progress:
                    progress(i + 1, len(jobs))

    # Alinear longitudes (todos los segmentos deben usar los mismos canales).
    max_len = max(v.shape[0] for v in vectors)
    X = np.zeros((len(vectors), max_len))
    for i, v in enumerate(vectors):
        X[i, : v.shape[0]] = v
    return X, names


def build_dataset(project, progress=None) -> Dataset:
    """Construye un :class:`Dataset` a partir de los segmentos del proyecto."""
    segments = project.state["segments"]
    if not segments:
        raise ValueError("El proyecto no tiene segmentos etiquetados.")

    # Pre-calcula en paralelo la señal procesada de todas las fuentes implicadas.
    project.prewarm([seg["source_id"] for seg in segments])
    arrays = [project.segment_data(seg) for seg in segments]
    labels = np.array([seg["label"] for seg in segments], dtype=object)
    ids = [seg["id"] for seg in segments]

    cfg = project.state.get("dataset", {})
    X, names = build_features_parallel(
        arrays,
        use_bands=cfg.get("use_bands", True),
        use_time=cfg.get("use_time", True),
        progress=progress,
    )
    return Dataset(X=X, y=labels, feature_names=names, segment_ids=ids)


def build_raw_dataset(project, window_samples: int = 512, progress=None) -> RawDataset:
    """Dataset de señal cruda (n_segmentos, n_canales, T) para CNN/LSTM.

    Usa **todos** los canales (ignora subconjuntos por segmento) para que la
    dimensión de canal sea uniforme, y ajusta cada segmento a una ventana fija.
    """
    segments = project.state["segments"]
    if not segments:
        raise ValueError("El proyecto no tiene segmentos etiquetados.")

    # Pre-calcula en paralelo la señal procesada de todas las fuentes implicadas.
    project.prewarm([seg["source_id"] for seg in segments])

    windows: list[np.ndarray] = []
    labels: list[str] = []
    ids: list[str] = []
    n = len(segments)
    for i, seg in enumerate(segments):
        full = project.get_processed(seg["source_id"])      # (n_canales, n_muestras)
        data = full[:, seg["start"]:seg["stop"]]
        windows.append(fit_window(np.ascontiguousarray(data), window_samples))
        labels.append(seg["label"])
        ids.append(seg["id"])
        if progress:
            progress(i + 1, n)

    X = np.stack(windows, axis=0)
    return RawDataset(X=X, y=np.array(labels, dtype=object), segment_ids=ids)


def save_dataset(project, dataset: Dataset, name: str = "dataset") -> str:
    """Guarda el dataset como ``.npz`` dentro de ``<proyecto>/datasets``."""
    out_dir = os.path.join(project.path, DATASETS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.npz")
    np.savez_compressed(
        out_path,
        X=dataset.X,
        y=dataset.y,
        feature_names=np.array(dataset.feature_names, dtype=object),
        segment_ids=np.array(dataset.segment_ids, dtype=object),
    )
    return out_path

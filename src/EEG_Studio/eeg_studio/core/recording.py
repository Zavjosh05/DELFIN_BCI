"""Modelo de una grabación EEG cargada desde CSV.

Una :class:`Recording` representa la *fuente original* y es de solo lectura:
la interfaz nunca la modifica. Las transformaciones (filtros, referencia, etc.)
se aplican sobre copias y se describen en el proyecto, garantizando que el CSV
de origen permanezca intacto (control de cambios no destructivo).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Recording:
    """Contenedor inmutable de una grabación EEG.

    Atributos
    ---------
    source_path: ruta absoluta del CSV de origen (nunca se escribe).
    channel_names: nombres originales de los canales (p.ej. "Channel 1").
    data: matriz ``(n_canales, n_muestras)`` en microvoltios.
    time: vector de tiempo en segundos, longitud ``n_muestras``.
    sample_rate: frecuencia de muestreo en Hz.
    epochs: vector entero ``(n_muestras,)`` con el índice de época, o ``None``.
    events: lista de marcadores ``{sample, id, date, duration}``.
    """

    source_path: str
    channel_names: list[str]
    data: np.ndarray
    time: np.ndarray
    sample_rate: float
    epochs: np.ndarray | None = None
    events: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.data = np.ascontiguousarray(self.data, dtype=np.float64)
        self.data.setflags(write=False)  # refuerza la inmutabilidad de la fuente

    # --- Propiedades de conveniencia --------------------------------------
    @property
    def n_channels(self) -> int:
        return self.data.shape[0]

    @property
    def n_samples(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        """Duración en segundos."""
        return self.n_samples / self.sample_rate

    @property
    def epoch_ids(self) -> list[int]:
        if self.epochs is None:
            return []
        return sorted(set(int(e) for e in np.unique(self.epochs)))

    def channel_index(self, name: str) -> int:
        return self.channel_names.index(name)

    def get_channel(self, name: str) -> np.ndarray:
        return self.data[self.channel_index(name)]

    def epoch_range(self, epoch_id: int) -> tuple[int, int]:
        """Devuelve ``(inicio, fin)`` en muestras para una época dada."""
        if self.epochs is None:
            raise ValueError("La grabación no contiene información de épocas.")
        idx = np.flatnonzero(self.epochs == epoch_id)
        if idx.size == 0:
            raise ValueError(f"Época inexistente: {epoch_id}")
        return int(idx[0]), int(idx[-1]) + 1

    def slice(self, start: int, stop: int, channels: list[int] | None = None) -> np.ndarray:
        """Devuelve una *copia* escribible de un segmento de la señal."""
        start = max(0, start)
        stop = min(self.n_samples, stop)
        if channels is None:
            return self.data[:, start:stop].copy()
        return self.data[np.asarray(channels), start:stop].copy()

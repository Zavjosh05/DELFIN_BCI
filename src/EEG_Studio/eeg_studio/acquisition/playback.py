"""Fuente de reproducción de una grabación (CSV) como si fuera señal en vivo.

Permite alimentar el visor en vivo y el modo de Control con una grabación previa
(p. ej. «Sujeto001_Abajo.csv») **sin necesidad de la diadema**: la señal del
archivo se emite en bloques al ritmo real (1x), igual que haría el EPOC+. Así todo
lo que consume una fuente en vivo (preprocesamiento, clasificación, brazo) funciona
sin cambios. Pensado para demostraciones y para validar un modelo sobre
grabaciones ya registradas.
"""
from __future__ import annotations

import time

import numpy as np

from ..config import EPOC_CHANNELS
from .base import StreamSource


class FilePlaybackSource(StreamSource):
    """Reproduce una grabación como una fuente en vivo (una sola pasada, 1x).

    Se le puede dar la ruta de un CSV (se carga en el hilo productor, con import
    perezoso de ``core`` para no acoplar el módulo salvo cuando se usa) o los datos
    ya cargados. ``speed`` acelera la reproducción (1.0 = tiempo real; útil >1 en
    pruebas para no esperar en tiempo real)."""

    display_name = "Reproducir grabación (archivo)"

    def __init__(self, path: str | None = None, *, data: np.ndarray | None = None,
                 channel_names: list[str] | None = None, sample_rate: float = 128.0,
                 speed: float = 1.0, block_size: int = 16) -> None:
        # Si aún no hay datos, se ponen valores provisionales; los reales se fijan
        # al cargar (igual que la fuente LSL resuelve sus metadatos en _run).
        names = list(channel_names) if channel_names is not None else EPOC_CHANNELS
        super().__init__(names, sample_rate)
        self._path = path
        self._data = None if data is None else np.ascontiguousarray(data, dtype=np.float64)
        self._speed = max(1e-3, float(speed))
        self._block = max(1, int(block_size))
        self._finished = False

    @property
    def finished(self) -> bool:
        """True cuando se terminó de reproducir el archivo (una sola pasada)."""
        return self._finished

    def _load(self) -> None:
        """Carga la grabación del CSV. Import perezoso de ``core`` para que
        ``acquisition`` no dependa de ``core`` salvo al usar esta fuente."""
        if self._data is not None:
            return
        if not self._path:
            raise ValueError("No se indicó ningún archivo de grabación para reproducir.")
        from ..core.csv_loader import load_recording
        rec = load_recording(self._path)
        self._data = np.ascontiguousarray(rec.data, dtype=np.float64)
        self._channels = list(rec.channel_names)      # metadatos reales del archivo
        self._fs = float(rec.sample_rate)

    def _run(self) -> None:
        self._load()
        data = self._data
        n = int(data.shape[1])
        if n == 0:
            self._finished = True
            return
        period = self._block / (self._fs * self._speed)
        next_t = time.perf_counter()
        i = 0
        while self._running.is_set() and i < n:
            self._emit(np.ascontiguousarray(data[:, i:i + self._block]))
            i += self._block
            # Mantener el ritmo (1x) sin acumular deriva, como la fuente simulada.
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
        if i >= n:
            self._finished = True

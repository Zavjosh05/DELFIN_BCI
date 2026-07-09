"""Grabación de la señal entrante a un CSV con formato OpenViBE.

Escribe un archivo **nuevo** dentro del proyecto (carpeta ``recordings/``), de
modo que la captura en vivo respeta el principio de no destructividad y el
archivo resultante puede añadirse como una fuente más (mismo formato que los
CSV exportados por OpenViBE: ``Time:128Hz,Epoch,Channel 1..N,Event Id,...``).
"""
from __future__ import annotations

import threading

import numpy as np


class CSVRecorder:
    def __init__(self, path: str, n_channels: int, sample_rate: float,
                 epoch_samples: int = 512) -> None:
        self.path = path
        self._fs = float(sample_rate)
        self._n = n_channels
        self._epoch_samples = epoch_samples
        self._sample = 0
        self._pending_marker: str | None = None
        self._lock = threading.Lock()
        self._fh = open(path, "w", encoding="utf-8", newline="")
        self._write_header()

    def _write_header(self) -> None:
        channels = ",".join(f"Channel {i + 1}" for i in range(self._n))
        self._fh.write(
            f"Time:{int(self._fs)}Hz,Epoch,{channels},Event Id,Event Date,Event Duration\n"
        )

    def add_marker(self, event_id: str) -> None:
        """Marca la siguiente muestra escrita con un ``Event Id`` (etiqueta)."""
        with self._lock:
            self._pending_marker = str(event_id)

    def write(self, chunk: np.ndarray) -> int:
        """Escribe un bloque ``(n_canales, k)``. Devuelve nº de muestras escritas."""
        if chunk is None or chunk.size == 0:
            return 0
        k = chunk.shape[1]
        with self._lock:
            lines = []
            for j in range(k):
                t = self._sample / self._fs
                epoch = self._sample // self._epoch_samples
                values = ",".join(f"{chunk[c, j]:.10f}" for c in range(self._n))
                if self._pending_marker is not None:
                    ev = f"{self._pending_marker},{t:.10f},0"
                    self._pending_marker = None
                else:
                    ev = ",,"
                lines.append(f"{t:.10f},{epoch},{values},{ev}")
                self._sample += 1
            self._fh.write("\n".join(lines) + "\n")
        return k

    @property
    def n_samples(self) -> int:
        return self._sample

    def close(self) -> None:
        with self._lock:
            if self._fh and not self._fh.closed:
                self._fh.flush()
                self._fh.close()

"""Fuente simulada: permite desarrollar y probar la captura sin hardware.

Genera 14 canales con una línea base parecida a los CSV reales del EPOC+
(~4200 µV) más ritmos alfa (~10 Hz) y beta (~20 Hz) y ruido, en tiempo real.
"""
from __future__ import annotations

import time

import numpy as np

from ..config import EPOC_CHANNELS
from .base import StreamSource


class SimulatedSource(StreamSource):
    display_name = "Simulado (sin hardware)"

    def __init__(self, n_channels: int = 14, sample_rate: float = 128.0,
                 block_size: int = 16) -> None:
        names = EPOC_CHANNELS[:n_channels]
        super().__init__(names, sample_rate)
        self._block = block_size
        self._t0_sample = 0
        self._rng = np.random.default_rng()
        # Amplitudes y fases por canal para que no sean idénticos.
        self._alpha_gain = self._rng.uniform(5, 20, size=n_channels)
        self._beta_gain = self._rng.uniform(2, 8, size=n_channels)
        self._baseline = self._rng.uniform(4000, 4400, size=n_channels)

    def _run(self) -> None:
        n = self.n_channels
        period = self._block / self._fs
        next_t = time.perf_counter()
        while self._running.is_set():
            idx = self._t0_sample + np.arange(self._block)
            tvec = idx / self._fs
            alpha = self._alpha_gain[:, None] * np.sin(2 * np.pi * 10.0 * tvec)[None, :]
            beta = self._beta_gain[:, None] * np.sin(2 * np.pi * 20.0 * tvec)[None, :]
            noise = self._rng.normal(0, 3, size=(n, self._block))
            block = self._baseline[:, None] + alpha + beta + noise
            self._emit(block)
            self._t0_sample += self._block

            # Mantener el ritmo en tiempo real sin acumular deriva.
            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

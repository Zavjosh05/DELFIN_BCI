"""Fuente LSL: recibe la señal publicada por el OpenViBE Acquisition Server.

Requiere ``pylsl`` (y la librería nativa ``liblsl``). El import es perezoso para
que la app funcione aunque pylsl no esté instalado: solo falla si se intenta
usar esta fuente en concreto.

Para que el Acquisition Server publique LSL: Preferences → activar la salida LSL
(``LSL_EnableLSLOutput = true``), con el nombre de stream de señal indicado.
"""
from __future__ import annotations

import numpy as np

from ..config import EPOC_CHANNELS, LSL_SIGNAL_NAME
from .base import StreamSource


def pylsl_available() -> bool:
    try:
        import pylsl  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


class LSLSource(StreamSource):
    display_name = "OpenViBE Acquisition Server (LSL)"

    def __init__(self, stream_name: str = LSL_SIGNAL_NAME, timeout: float = 5.0) -> None:
        # Los metadatos reales (canales, fs) se leen al conectar; valores
        # provisionales hasta entonces.
        super().__init__(EPOC_CHANNELS, 128.0)
        self._stream_name = stream_name
        self._timeout = timeout

    def _resolve_inlet(self):
        import pylsl
        streams = pylsl.resolve_byprop("name", self._stream_name, timeout=self._timeout)
        if not streams:
            # Reintento por tipo genérico de señal.
            streams = pylsl.resolve_byprop("type", "signal", timeout=self._timeout)
        if not streams:
            raise RuntimeError(
                f"No se encontró ningún stream LSL «{self._stream_name}». "
                "¿Está el Acquisition Server en Play con la salida LSL activada?"
            )
        inlet = pylsl.StreamInlet(streams[0], max_buflen=360)
        info = inlet.info()
        self._fs = float(info.nominal_srate()) or 128.0
        self._channels = self._read_channel_labels(info) or EPOC_CHANNELS[: info.channel_count()]
        return inlet

    @staticmethod
    def _read_channel_labels(info) -> list[str]:
        labels: list[str] = []
        try:
            ch = info.desc().child("channels").child("channel")
            for _ in range(info.channel_count()):
                labels.append(ch.child_value("label"))
                ch = ch.next_sibling()
        except Exception:  # noqa: BLE001
            return []
        return [l for l in labels if l]

    def _run(self) -> None:
        inlet = self._resolve_inlet()
        while self._running.is_set():
            samples, _ts = inlet.pull_chunk(timeout=0.2, max_samples=64)
            if samples:
                # pull_chunk -> (n_muestras, n_canales); transponemos.
                self._emit(np.asarray(samples, dtype=np.float64).T)
        try:
            inlet.close_stream()
        except Exception:  # noqa: BLE001
            pass

"""Adquisición en tiempo real (opcional).

Fuentes intercambiables que entregan bloques ``(n_canales, k)`` mediante el
mismo interfaz :class:`StreamSource`. La interfaz no necesita ninguna de ellas
para funcionar: la captura es una función opcional.
"""
from .base import StreamSource
from .emotiv import EmotivDongleSource, emotiv_deps_available
from .lsl import LSLSource, pylsl_available
from .recorder import CSVRecorder
from .simulated import SimulatedSource
from .tcp import TCPSource

__all__ = [
    "StreamSource",
    "SimulatedSource",
    "LSLSource",
    "TCPSource",
    "EmotivDongleSource",
    "CSVRecorder",
    "pylsl_available",
    "emotiv_deps_available",
]

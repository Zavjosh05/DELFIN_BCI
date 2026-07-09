"""Fuente TCP genérica: pensada para CyKit (``CyKIT.py`` en modo ``generic``).

CyKit lee el EPOC+ directamente del dongle USB y emite líneas de texto con los
valores de los canales separados por comas, por un puerto TCP. Esta fuente se
conecta a ese socket, parsea cada línea y extrae los 14 canales.

Lanzar CyKit (en su propio Python 3.x, proceso aparte), por ejemplo::

    python .\\CyKIT.py 127.0.0.1 5555 6 generic+nocounter+noheader+nobattery

El número de canales y la columna de inicio son configurables porque el formato
exacto depende de las banderas de CyKit.
"""
from __future__ import annotations

import re
import socket

import numpy as np

from ..config import EPOC_CHANNELS
from .base import StreamSource

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


class TCPSource(StreamSource):
    display_name = "CyKit / TCP (líneas CSV)"

    def __init__(self, host: str, port: int, n_channels: int = 14,
                 channel_start: int = 2, sample_rate: float = 128.0,
                 connect_timeout: float = 5.0) -> None:
        super().__init__(EPOC_CHANNELS[:n_channels], sample_rate)
        self._host = host
        self._port = int(port)
        self._n = n_channels
        self._start = channel_start
        self._timeout = connect_timeout

    def _run(self) -> None:
        sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
        sock.settimeout(0.5)
        buffer = b""
        try:
            while self._running.is_set():
                try:
                    data = sock.recv(8192)
                except socket.timeout:
                    continue
                if not data:
                    break
                buffer += data
                *lines, buffer = buffer.split(b"\n")
                cols = self._parse_lines(lines)
                if cols is not None:
                    self._emit(cols)
        finally:
            try:
                sock.close()
            except Exception:  # noqa: BLE001
                pass

    def _parse_lines(self, lines: list[bytes]) -> np.ndarray | None:
        rows: list[list[float]] = []
        for raw in lines:
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            nums = _NUM_RE.findall(text)
            if len(nums) < self._start + self._n:
                continue
            sel = nums[self._start:self._start + self._n]
            try:
                rows.append([float(x) for x in sel])
            except ValueError:
                continue
        if not rows:
            return None
        # rows -> (n_muestras, n_canales); transponemos a (n_canales, k).
        return np.asarray(rows, dtype=np.float64).T

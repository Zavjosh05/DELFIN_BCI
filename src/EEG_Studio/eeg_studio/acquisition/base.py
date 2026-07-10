"""Interfaz común de fuentes de adquisición en tiempo real.

Patrón productor/consumidor seguro para Qt:

* Cada fuente lanza su **propio hilo** (``_run``) que lee del dispositivo,
  socket o generador y deja los bloques en una cola interna.
* La interfaz (hilo principal) llama periódicamente a :meth:`read` mediante un
  ``QTimer`` y obtiene todas las muestras nuevas, sin tocar widgets desde otros
  hilos.

Esto desacopla la fuente de la GUI y de Qt: las fuentes son Python puro y se
pueden probar sin interfaz.
"""
from __future__ import annotations

import queue
import threading

import numpy as np


class StreamSource:
    """Fuente de señal en streaming. Subclasear e implementar :meth:`_run`."""

    display_name = "Fuente genérica"

    def __init__(self, channel_names: list[str], sample_rate: float) -> None:
        self._channels = list(channel_names)
        self._fs = float(sample_rate)
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._error: str | None = None
        # «Tap» opcional llamado en el HILO PRODUCTOR con cada bloque emitido
        # (p. ej. la grabación): así capturar a disco NO depende del temporizador
        # de la GUI, que se estrangula cuando la app está en segundo plano.
        self._tap = None
        self._battery: int | None = None       # % de batería (si la fuente lo reporta)

    # --- Propiedades ------------------------------------------------------
    @property
    def channel_names(self) -> list[str]:
        return self._channels

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def n_channels(self) -> int:
        return len(self._channels)

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def battery(self) -> int | None:
        """Nivel de batería (0–100) si la fuente lo reporta; si no, ``None``."""
        return self._battery

    def set_tap(self, fn) -> None:
        """Registra (o quita, con ``None``) un consumidor llamado en el hilo
        productor con cada bloque emitido. Pensado para la grabación: que escribir
        a disco no dependa del temporizador de la GUI."""
        self._tap = fn

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # --- Ciclo de vida ----------------------------------------------------
    def start(self) -> None:
        if self.is_running():
            return
        self._error = None
        self._running.set()
        self._thread = threading.Thread(target=self._safe_run, daemon=True,
                                        name=f"acq-{self.display_name}")
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=1.5)
            self._thread = None

    # --- Comunicación productor -> consumidor -----------------------------
    def _emit(self, chunk: np.ndarray) -> None:
        """Encola un bloque ``(n_canales, k)`` de muestras nuevas.

        Antes de encolar (para la GUI), lo entrega al ``tap`` si hay uno (p. ej. la
        grabación), de modo que capturar a disco ocurra en este hilo productor y no
        se pierdan muestras si la GUI va lenta o está en segundo plano."""
        arr = np.ascontiguousarray(chunk, dtype=np.float64)
        tap = self._tap
        if tap is not None:
            try:
                tap(arr)
            except Exception:  # noqa: BLE001 - el tap nunca debe tumbar el productor
                pass
        self._queue.put(arr)

    def read(self) -> np.ndarray | None:
        """Devuelve todas las muestras acumuladas ``(n_canales, k)`` o ``None``."""
        chunks: list[np.ndarray] = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return None
        return np.concatenate(chunks, axis=1)

    # --- A implementar por las subclases ----------------------------------
    def _run(self) -> None:  # pragma: no cover - interfaz
        raise NotImplementedError

    def _safe_run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            self._running.clear()

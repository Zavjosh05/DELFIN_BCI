"""Worker genérico para ejecutar funciones en un hilo aparte.

Mantiene la interfaz fluida: el cómputo pesado (filtros sobre toda la señal,
extracción de características con multiproceso, entrenamiento del modelo) corre
en un ``QThread`` y comunica el resultado por señales.

La entrega de resultados se enruta a través de un *proxy* que vive en el hilo
principal (:class:`_MainThreadProxy`), garantizando que los callbacks que tocan
la GUI se ejecuten en el hilo de la interfaz (Qt no permite tocar widgets desde
otros hilos). La extracción de características usa además ``ProcessPoolExecutor``
(multiprocessing) dentro del hilo del worker, combinando ambos mecanismos.
"""
from __future__ import annotations

import traceback
from typing import Callable

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot


class Worker(QObject):
    """Ejecuta ``fn(*args, **kwargs)`` y emite el resultado o el error."""

    finished = pyqtSignal(object)        # resultado
    failed = pyqtSignal(str)             # mensaje de error
    progress = pyqtSignal(int, int)      # (hechos, total)

    def __init__(self, fn: Callable, *args, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    @pyqtSlot()
    def run(self) -> None:
        try:
            # Si la función declara un parámetro 'progress', le inyectamos el
            # emisor de señal para reportar avance sin acoplarla a Qt.
            code = getattr(self._fn, "__code__", None)
            if code is not None and "progress" in code.co_varnames:
                self._kwargs.setdefault("progress", lambda d, t: self.progress.emit(d, t))
            result = self._fn(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class _MainThreadProxy(QObject):
    """Reenvía las señales del worker a callbacks en el hilo principal.

    Al vivir en el hilo de la GUI, las conexiones desde el worker (otro hilo)
    son *queued* y los callbacks se ejecutan de forma segura en la interfaz.
    """

    def __init__(self, on_done, on_error, on_progress, parent=None) -> None:
        super().__init__(parent)
        self._on_done = on_done
        self._on_error = on_error
        self._on_progress = on_progress

    @pyqtSlot(object)
    def done(self, result) -> None:
        if self._on_done:
            self._on_done(result)

    @pyqtSlot(str)
    def error(self, msg) -> None:
        if self._on_error:
            self._on_error(msg)

    @pyqtSlot(int, int)
    def progress(self, done_n, total) -> None:
        if self._on_progress:
            self._on_progress(done_n, total)


def run_async(parent, fn, on_done=None, on_error=None, on_progress=None, *args, **kwargs):
    """Lanza ``fn`` en un hilo y conecta callbacks (ejecutados en el hilo GUI).

    Devuelve ``(thread, worker, proxy)``; el llamador debe conservar esa
    referencia para evitar que el recolector destruya los objetos a destiempo.
    """
    thread = QThread(parent)
    worker = Worker(fn, *args, **kwargs)
    worker.moveToThread(thread)

    proxy = _MainThreadProxy(on_done, on_error, on_progress, parent=parent)
    worker.finished.connect(proxy.done, Qt.ConnectionType.QueuedConnection)
    worker.failed.connect(proxy.error, Qt.ConnectionType.QueuedConnection)
    worker.progress.connect(proxy.progress, Qt.ConnectionType.QueuedConnection)

    # Arranque y ciclo de vida (sin wait() dentro de slots, para no bloquear).
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(proxy.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread, worker, proxy

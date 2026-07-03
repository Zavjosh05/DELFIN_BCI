"""Ventana independiente para ver una señal (permite abrir varias a la vez).

Reutiliza :class:`SignalView`. Es un visor (marcadores + segmentos + señal cruda
o procesada); el procesamiento se hace en un hilo para no congelar la interfaz.
La edición de segmentos sigue en la ventana principal.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow

from ..workers import run_async
from .signal_view import SignalView


class SignalWindow(QMainWindow):
    def __init__(self, controller, source_id: str) -> None:
        super().__init__()
        self.controller = controller
        self.source_id = source_id
        self._threads: list = []

        self.view = SignalView()
        self.view.mode_changed.connect(self.reload)
        self.setCentralWidget(self.view)

        src = controller.project.get_source(source_id) if controller.project else None
        alias = src["alias"] if src else source_id
        self.setWindowTitle(f"{alias} — señal")
        self.resize(900, 560)
        self.reload()

    def reload(self) -> None:
        proj = self.controller.project
        if proj is None:
            self.view.clear()
            return
        try:
            rec = proj.get_recording(self.source_id)
        except Exception:  # noqa: BLE001 — fuente no disponible
            self.view.clear()
            self.statusBar().showMessage("La fuente no está disponible en disco.")
            return

        names = proj.kept_display_names(rec)
        cuts = proj.cut_intervals(self.source_id)
        self.view.set_cuts(cuts)
        self.view.set_markers([(e["sample"], e["id"]) for e in rec.events
                               if not any(ca <= e["sample"] < cb for ca, cb in cuts)])
        self.view.set_segments([
            (s["start"], s["stop"], s["label"])
            for s in proj.state["segments"] if s["source_id"] == self.source_id
        ])

        if self.view.mode == "raw" or not proj.state["pipeline"]:
            self.view.set_data(rec.data[proj.kept_indices(rec)], rec.sample_rate, names)
            self.statusBar().showMessage("Señal cruda.", 1500)
            return

        # Si ya está en caché, dibujar al instante (sin hilo).
        cached = proj.processed_if_cached(self.source_id)
        if cached is not None:
            self.view.set_data(cached, rec.sample_rate, names)
            self.statusBar().showMessage("Procesada.", 1500)
            return

        # Procesamiento en segundo plano (hilo propio de esta ventana).
        self.statusBar().showMessage("Aplicando preprocesamiento…")
        sid = self.source_id

        def done(data):
            self.statusBar().showMessage("Procesada.", 1500)
            self.view.set_data(data, rec.sample_rate, names)

        def err(_msg):
            self.statusBar().showMessage("No se pudo procesar la señal.")

        handle = run_async(self, lambda: proj.get_processed(sid), on_done=done, on_error=err)
        self._threads.append(handle)
        handle[0].finished.connect(
            lambda: self._threads.remove(handle) if handle in self._threads else None)

    def closeEvent(self, event) -> None:  # noqa: N802 (API de Qt)
        self.controller._signal_windows.discard(self)
        super().closeEvent(event)

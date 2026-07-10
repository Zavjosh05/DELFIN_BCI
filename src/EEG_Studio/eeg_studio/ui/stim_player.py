"""Reproductor de estímulo a pantalla completa (multi-monitor) + orquestación de
la grabación sincronizada.

Al reproducir un estímulo configurado: se inicia la grabación EEG, el video se
lanza en **pantalla completa en un monitor externo** (o en la principal si no hay
otro), se registran las marcas en su instante y, al terminar, se colocan los
**segmentos exactos** (calculados desde la línea de tiempo) y se guarda todo.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from ..core.stim import compute_segments, markers_in_order


def _best_screen():
    """Pantalla donde lanzar el estímulo: una EXTERNA si hay; si no, la principal."""
    app = QApplication.instance()
    screens = app.screens() if app else []
    primary = app.primaryScreen() if app else None
    for s in screens:
        if s is not primary:
            return s, True
    return primary, False


class StimPlayerWindow(QWidget):
    """Ventana de video a pantalla completa que avisa cuando arranca y termina."""

    started = pyqtSignal()          # el video empezó a reproducirse de verdad
    finished = pyqtSignal()         # terminó (o se cerró)
    position = pyqtSignal(int)      # posición actual (ms)

    def __init__(self, video_path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Estímulo")
        self.setStyleSheet("background: #000;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.video = QVideoWidget()
        lay.addWidget(self.video)
        self._hint = QLabel("Esc para cancelar", self)
        self._hint.setStyleSheet("color: rgba(255,255,255,90); font-size: 12px;")
        self._hint.move(12, 8)
        # Indicador: se resalta cuando el instante actual cae dentro de un segmento.
        self.seg_label = QLabel("", self)
        self.seg_label.setStyleSheet(
            "background: rgba(220,60,60,210); color: white; font-size: 22px; "
            "font-weight: bold; padding: 6px 18px; border-radius: 10px;")
        self.seg_label.hide()

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        self.player.setSource(QUrl.fromLocalFile(video_path))
        self.player.positionChanged.connect(self._on_position)
        self.player.mediaStatusChanged.connect(self._on_status)
        self._started = False
        self._done = False

    def run(self, screen=None) -> bool:
        """Muestra a pantalla completa en el monitor indicado (o el mejor por
        defecto: uno externo si hay, si no el principal) y reproduce."""
        external = False
        if screen is None:
            screen, external = _best_screen()
        else:
            external = screen is not (QApplication.instance().primaryScreen()
                                      if QApplication.instance() else None)
        if screen is not None:
            self.setScreen(screen)
            self.setGeometry(screen.geometry())
        self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.player.play()
        return external

    def _on_position(self, p: int) -> None:
        if not self._started and p > 0:
            self._started = True
            self.started.emit()
        self.position.emit(p)

    def _on_status(self, status) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._finish()

    def set_segment(self, label) -> None:
        """Muestra/oculta el indicador de que el video está dentro de un segmento."""
        if label:
            self.seg_label.setText(f"●  SEGMENTO: {label}")
            self.seg_label.adjustSize()
            self._place_seg_label()
            self.seg_label.show()
            self.seg_label.raise_()
        else:
            self.seg_label.hide()

    def _place_seg_label(self) -> None:
        self.seg_label.move(max(0, (self.width() - self.seg_label.width()) // 2), 24)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._place_seg_label()

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self._finish()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):  # noqa: N802
        self._finish()
        super().closeEvent(event)

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        try:                                  # soltar el video antes de destruir (evita crash)
            self.player.stop()
            self.player.setVideoOutput(None)
            self.player.setSource(QUrl())
        except Exception:  # noqa: BLE001
            pass
        self.finished.emit()
        self.close()


class StimSession:
    """Coordina grabación + reproducción de UN estímulo. Mantener una referencia
    viva hasta que termine (``on_done`` se llama al final)."""

    def __init__(self, controller, config: dict, rec_name: str, screen=None,
                 on_done=None) -> None:
        self.controller = controller
        self.acq = controller.acq_panel
        self.config = config
        self.rec_name = rec_name
        self._screen = screen
        self._on_done = on_done
        self._events = config.get("events", [])
        self._markers = markers_in_order(self._events)
        self._segments = [(int(e["start"]), int(e["stop"]), str(e.get("label", "")))
                          for e in self._events if e.get("kind") == "segment"]
        self._fired = 0
        self._seg_active = 0                 # centinela distinto de None
        self._base: int | None = None
        self.window: StimPlayerWindow | None = None

    def start(self) -> bool:
        if not self.acq.stim_is_ready():
            self.controller.warn(
                "Sin señal en vivo",
                "Conecta una fuente en «Tiempo real» y espera muestras antes de "
                "reproducir un estímulo (la grabación necesita señal).")
            return False
        if not self.acq.stim_start(self.rec_name):
            self.controller.warn("No se pudo grabar", "Ya hay una grabación en curso.")
            return False
        path = self.config.get("path", "")
        self.window = StimPlayerWindow(path)
        self.window.started.connect(self._on_started)
        self.window.position.connect(self._on_position)
        self.window.finished.connect(self._on_finished)
        self.window.run(self._screen)
        return True

    def _on_started(self) -> None:
        # Muestra de la grabación que coincide con el inicio real del video.
        self._base = self.acq.stim_samples()

    def _on_position(self, pos: int) -> None:
        # Dispara las marcas (instantes) cuando el video las alcanza.
        while self._fired < len(self._markers) and self._markers[self._fired]["t"] <= pos:
            self.acq.stim_marker(self._markers[self._fired].get("label", ""))
            self._fired += 1
        # Indicador de segmento activo (video dentro de un lapso etiquetado).
        active = next((lbl for (a, b, lbl) in self._segments if a <= pos < b), None)
        if active != self._seg_active:
            self._seg_active = active
            if self.window is not None:
                self.window.set_segment(active)
            try:
                self.acq.status.setText(f"● Grabando segmento: {active}" if active
                                        else "Reproduciendo estímulo…")
            except Exception:  # noqa: BLE001
                pass

    def _on_finished(self) -> None:
        # Blindaje: pase lo que pase, se cierra la grabación (no se pierde nada).
        try:
            fs = self.acq.source.sample_rate if self.acq.source else 128.0
            base = self._base if self._base is not None else 0
            n = self.acq.stim_samples()
            segments = compute_segments(self._events, fs, base_sample=base, n_samples=n)
            self.acq.stim_finish(segments)   # guarda + añade como fuente con segmentos
        finally:
            if self._on_done is not None:
                self._on_done(True)

"""Editor de línea de tiempo de un video de estímulo (estilo editor de video).

Muestra el video con una **barra de tiempo** en la que puedes moverte a un punto
exacto (indica el instante bajo el cursor), y ahí fijar **marcas** (instantes) o
**segmentos** (lapsos), cada uno con su **clase** (varias clases por video). Las
clases se toman de las que ya existen en el proyecto. Devuelve los eventos en ms.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .signal_view import segment_color
from .theme import BORDER, MUTED, SURFACE, TEXT


def _fmt(ms: int) -> str:
    s = max(0, int(ms)) / 1000.0
    return f"{int(s // 60)}:{s % 60:06.3f}"


class _TimelineBar(QWidget):
    """Barra de tiempo: playhead, marcas/segmentos dibujados y hover con el instante."""

    seekRequested = pyqtSignal(int)   # ms

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(52)
        self.setMouseTracking(True)
        self._duration = 0
        self._position = 0
        self._events: list[dict] = []
        self._hover: int | None = None

    def set_duration(self, d: int) -> None:
        self._duration = max(0, int(d)); self.update()

    def set_position(self, p: int) -> None:
        self._position = max(0, int(p)); self.update()

    def set_events(self, events: list[dict]) -> None:
        self._events = events; self.update()

    def _ms_at(self, x: float) -> int:
        w = max(1, self.width())
        return int(max(0.0, min(1.0, x / w)) * self._duration)

    def _x_of(self, ms: float) -> int:
        if self._duration <= 0:
            return 0
        return int(self.width() * ms / self._duration)

    def mousePressEvent(self, e):  # noqa: N802
        self.seekRequested.emit(self._ms_at(e.position().x()))

    def mouseMoveEvent(self, e):  # noqa: N802
        self._hover = self._ms_at(e.position().x())
        if e.buttons() & Qt.MouseButton.LeftButton:
            self.seekRequested.emit(self._hover)
        self.update()

    def leaveEvent(self, e):  # noqa: N802
        self._hover = None; self.update()

    def paintEvent(self, e):  # noqa: N802
        p = QPainter(self)
        w, h = self.width(), self.height()
        top, th = 4, h - 22
        p.fillRect(0, top, w, th, QColor(SURFACE))
        p.setPen(QColor(BORDER)); p.drawRect(0, top, w - 1, th - 1)
        if self._duration <= 0:
            p.end(); return
        for ev in self._events:                       # segmentos (bloques)
            if ev.get("kind") == "segment":
                x0, x1 = self._x_of(ev["start"]), self._x_of(ev["stop"])
                c = QColor(segment_color(ev.get("label", ""))); c.setAlpha(90)
                p.fillRect(x0, top, max(2, x1 - x0), th, c)
        for ev in self._events:                       # marcas (líneas)
            if ev.get("kind") == "marker":
                x = self._x_of(ev["t"])
                p.setPen(QPen(QColor(segment_color(ev.get("label", ""))), 2))
                p.drawLine(x, top, x, top + th)
        xp = self._x_of(self._position)               # playhead
        p.setPen(QPen(QColor("#ffd54f"), 2))
        p.drawLine(xp, top - 3, xp, top + th + 3)
        if self._hover is not None:                   # hover + instante
            xh = self._x_of(self._hover)
            p.setPen(QPen(QColor(MUTED), 1, Qt.PenStyle.DashLine))
            p.drawLine(xh, top, xh, top + th)
            p.setPen(QColor(TEXT))
            tx = max(2, min(xh + 4, w - 74))
            p.drawText(tx, h - 5, _fmt(self._hover))
        p.end()


class StimTimelineDialog(QDialog):
    def __init__(self, video_path: str, label: str | None,
                 events: list[dict] | None = None,
                 classes: list[str] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Línea de tiempo del estímulo")
        self.resize(820, 640)
        self._duration = 0
        self._seg_start: int | None = None
        self._cleaned = False
        self._frame_shown = False
        self._events: list[dict] = [dict(e) for e in (events or [])]
        self._prefilled = bool(self._events)

        lay = QVBoxLayout(self)

        # --- Vista previa ---
        self.video = QVideoWidget()
        self.video.setMinimumHeight(300)
        lay.addWidget(self.video, 1)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        self.player.durationChanged.connect(self._on_duration)
        self.player.positionChanged.connect(self._on_position)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.player.setSource(QUrl.fromLocalFile(video_path))

        # --- Transporte + barra de tiempo (editor) ---
        row = QHBoxLayout()
        self.play_btn = QPushButton("▶"); self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self._toggle_play)
        row.addWidget(self.play_btn)
        self.timeline = _TimelineBar()
        self.timeline.set_events(self._events)
        self.timeline.seekRequested.connect(self._seek)   # busca Y muestra el frame
        row.addWidget(self.timeline, 1)
        self.time_lbl = QLabel("0:00.000 / 0:00.000")
        row.addWidget(self.time_lbl)
        lay.addLayout(row)

        # --- Ir a un instante exacto ---
        goto = QHBoxLayout()
        goto.addWidget(QLabel("Ir a (s):"))
        self.goto_spin = QDoubleSpinBox()
        self.goto_spin.setDecimals(3); self.goto_spin.setRange(0.0, 3600.0)
        self.goto_spin.setSingleStep(0.5)
        self.goto_spin.setToolTip("Escribe el segundo exacto y pulsa «Ir» (o Enter).")
        self.goto_spin.editingFinished.connect(self._goto)
        goto.addWidget(self.goto_spin)
        goto_btn = QPushButton("Ir"); goto_btn.setFixedWidth(40)
        goto_btn.clicked.connect(self._goto)
        goto.addWidget(goto_btn)
        goto.addStretch(1)
        lay.addLayout(goto)

        # --- Clase + captura ---
        cap = QHBoxLayout()
        cap.addWidget(QLabel("Clase:"))
        self.label_combo = QComboBox()
        self.label_combo.setEditable(True)
        for c in (classes or []):
            self.label_combo.addItem(c)
        default_label = label or (classes[0] if classes else "clase_1")
        if self.label_combo.findText(default_label) < 0:
            self.label_combo.addItem(default_label)
        self.label_combo.setCurrentText(default_label)
        self.label_combo.setToolTip("Clase del evento. Puedes usar varias en un mismo video.")
        cap.addWidget(self.label_combo, 1)
        mark_btn = QPushButton("＋ Marca aquí")
        mark_btn.clicked.connect(self._add_marker)
        cap.addWidget(mark_btn)
        self.seg_btn = QPushButton("Inicio de segmento aquí (F6)")
        self.seg_btn.clicked.connect(self._segment_click)
        cap.addWidget(self.seg_btn)
        lay.addLayout(cap)
        # F6 = inicio/fin de segmento (misma acción que el botón).
        self._sc_seg = QShortcut(QKeySequence("F6"), self)
        self._sc_seg.activated.connect(self._segment_click)

        # --- Tabla de eventos ---
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Tipo", "Inicio (s)", "Fin (s)", "Clase"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.setMaximumHeight(160)
        lay.addWidget(self.table)
        trow = QHBoxLayout()
        rm = QPushButton("Quitar seleccionado"); rm.clicked.connect(self._remove_selected)
        rep = QPushButton("Repetir periódicamente…"); rep.clicked.connect(self._repeat_segment)
        rep.setToolTip("Repite el segmento seleccionado cada N segundos, varias veces.")
        trow.addWidget(rm); trow.addWidget(rep); trow.addStretch(1)
        lay.addLayout(trow)

        self.hint = QLabel("Muévete por la barra (indica el instante bajo el cursor) y fija "
                           "ahí la marca o el segmento (F6 = inicio/fin). Se guardan en el proyecto.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        lay.addWidget(self.hint)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._rebuild_table()

    # --- Reproducción -----------------------------------------------------
    def _on_duration(self, d: int) -> None:
        self._duration = d
        self.timeline.set_duration(d)
        self.goto_spin.setMaximum(max(0.0, d / 1000.0))
        self.time_lbl.setText(f"{_fmt(0)} / {_fmt(d)}")
        # (Antes se prellenaban marcas/segmento automáticos; ahora un video nuevo
        # empieza VACÍO — el usuario coloca todo a mano.)

    def _on_position(self, p: int) -> None:
        self.timeline.set_position(p)
        self.time_lbl.setText(f"{_fmt(p)} / {_fmt(self._duration)}")
        if not self.goto_spin.hasFocus():          # refleja el instante actual (editable)
            self.goto_spin.blockSignals(True)
            self.goto_spin.setValue(p / 1000.0)
            self.goto_spin.blockSignals(False)

    def _on_media_status(self, status) -> None:
        # Al cargar, muestra el primer frame (play+pause breve) para que la vista
        # previa tenga imagen sin darle a reproducir.
        if not self._frame_shown and status in (
                QMediaPlayer.MediaStatus.LoadedMedia,
                QMediaPlayer.MediaStatus.BufferedMedia):
            self._frame_shown = True
            self.player.play()
            QTimer.singleShot(60, self._preview_pause)

    def _preview_pause(self) -> None:
        if self._cleaned:
            return
        try:
            self.player.pause(); self.play_btn.setText("▶")
        except Exception:  # noqa: BLE001
            pass

    def _seek(self, ms: int) -> None:
        """Busca a ``ms`` y deja el frame visible aunque el video esté pausado."""
        ms = max(0, int(ms))
        if self._duration:
            ms = min(ms, self._duration)
        if self.player.playbackState() == QMediaPlayer.PlaybackState.StoppedState:
            self.player.play(); self.player.pause(); self.play_btn.setText("▶")
        self.player.setPosition(ms)

    def _goto(self) -> None:
        self._seek(int(round(self.goto_spin.value() * 1000)))

    def _toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause(); self.play_btn.setText("▶")
        else:
            self.player.play(); self.play_btn.setText("⏸")

    # --- Eventos ----------------------------------------------------------
    def _add_marker(self) -> None:
        self._events.append({"kind": "marker", "t": self.player.position(),
                             "label": self.label_combo.currentText().strip() or "clase"})
        self._sync()

    def _segment_click(self) -> None:
        pos = self.player.position()
        if self._seg_start is None:
            self._seg_start = pos
            self.seg_btn.setText(f"Fin de segmento (inicio: {_fmt(pos)})")
        else:
            a, b = sorted((self._seg_start, pos))
            self._seg_start = None
            self.seg_btn.setText("Inicio de segmento aquí (F6)")
            if b - a >= 1:
                self._events.append({"kind": "segment", "start": a, "stop": b,
                                     "label": self.label_combo.currentText().strip() or "clase"})
                self._sync()

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self._events):
                self._events.pop(r)
        self._sync()

    def _repeat_segment(self) -> None:
        """Repite el segmento seleccionado cada N segundos, varias veces."""
        seg = None
        for r in sorted({i.row() for i in self.table.selectedItems()}):
            if 0 <= r < len(self._events) and self._events[r].get("kind") == "segment":
                seg = self._events[r]
                break
        if seg is None:
            QMessageBox.information(self, "Repetir segmento",
                                   "Selecciona un segmento en la tabla primero.")
            return
        period, ok = QInputDialog.getDouble(
            self, "Repetir periódicamente", "Periodo entre repeticiones (s):",
            (seg["stop"] - seg["start"]) / 1000.0, 0.1, 3600.0, 3)
        if not ok:
            return
        count, ok = QInputDialog.getInt(
            self, "Repetir periódicamente", "Nº de repeticiones (además del actual):",
            3, 1, 1000)
        if not ok:
            return
        step = int(round(period * 1000))
        dur = seg["stop"] - seg["start"]
        added = 0
        for i in range(1, count + 1):
            start = seg["start"] + i * step
            stop = start + dur
            if self._duration and stop > self._duration:
                break
            self._events.append({"kind": "segment", "start": start, "stop": stop,
                                 "label": seg.get("label", "")})
            added += 1
        if added:
            self._sync()
        if added < count:
            self.hint.setText(f"Se añadieron {added} repeticiones (el resto no cabía en el video).")

    def _sync(self) -> None:
        self._events.sort(key=lambda e: e.get("t", e.get("start", 0)))
        self._rebuild_table()
        self.timeline.set_events(self._events)

    def _rebuild_table(self) -> None:
        self.table.setRowCount(0)
        for e in self._events:
            r = self.table.rowCount(); self.table.insertRow(r)
            marker = e.get("kind") == "marker"
            start = e.get("t", 0) if marker else e.get("start", 0)
            stop = "" if marker else f"{e.get('stop', 0) / 1000.0:.3f}"
            self.table.setItem(r, 0, QTableWidgetItem("marca" if marker else "segmento"))
            self.table.setItem(r, 1, QTableWidgetItem(f"{start / 1000.0:.3f}"))
            self.table.setItem(r, 2, QTableWidgetItem(stop))
            self.table.setItem(r, 3, QTableWidgetItem(str(e.get("label", ""))))

    # --- Resultado / limpieza --------------------------------------------
    def result_events(self) -> list[dict]:
        return [dict(e) for e in self._events]

    def duration_ms(self) -> int:
        return int(self._duration)

    def _cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        try:                                  # detener y soltar el video ANTES de destruir
            self.player.stop()
            self.player.setVideoOutput(None)
            self.player.setSource(QUrl())
        except Exception:  # noqa: BLE001
            pass

    def done(self, result: int) -> None:  # noqa: N802
        # Se llama tanto en Aceptar como en Cancelar: limpia el player para no
        # dejarlo vivo al destruir el diálogo (causaba un crash al cancelar).
        self._cleanup()
        super().done(result)

    def closeEvent(self, event):  # noqa: N802
        self._cleanup()
        super().closeEvent(event)

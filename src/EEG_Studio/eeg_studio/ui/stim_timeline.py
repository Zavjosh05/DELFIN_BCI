"""Editor de línea de tiempo de un video de estímulo.

Reproduce el video en una vista previa y permite fijar, en el momento exacto, las
**marcas** (instantes) y los **segmentos** (lapsos) que se registrarán en la
grabación sincronizada. Devuelve la lista de eventos en milisegundos.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..core.stim import DELFIN_CLASSES, default_events


def _fmt(ms: int) -> str:
    s = max(0, ms) / 1000.0
    return f"{int(s // 60)}:{s % 60:06.3f}"


class StimTimelineDialog(QDialog):
    def __init__(self, video_path: str, label: str | None,
                 events: list[dict] | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Línea de tiempo del estímulo")
        self.resize(760, 620)
        self._label = label or (DELFIN_CLASSES[0])
        self._duration = 0
        self._seg_start: int | None = None

        lay = QVBoxLayout(self)

        # --- Vista previa del video ---
        self.video = QVideoWidget()
        self.video.setMinimumHeight(280)
        lay.addWidget(self.video, 1)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        self.player.setSource(QUrl.fromLocalFile(video_path))
        self.player.durationChanged.connect(self._on_duration)
        self.player.positionChanged.connect(self._on_position)

        # --- Transporte (play/seek) ---
        row = QHBoxLayout()
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self._toggle_play)
        row.addWidget(self.play_btn)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.player.setPosition)
        row.addWidget(self.slider, 1)
        self.time_lbl = QLabel("0:00.000 / 0:00.000")
        row.addWidget(self.time_lbl)
        lay.addLayout(row)

        # --- Etiqueta (clase) + captura de eventos ---
        cap = QHBoxLayout()
        cap.addWidget(QLabel("Clase:"))
        self.label_combo = QComboBox()
        self.label_combo.addItems(DELFIN_CLASSES)
        if self._label in DELFIN_CLASSES:
            self.label_combo.setCurrentText(self._label)
        self.label_combo.setEditable(True)
        cap.addWidget(self.label_combo)
        cap.addStretch(1)
        mark_btn = QPushButton("＋ Marca aquí")
        mark_btn.setToolTip("Registra una marca (instante) en la posición actual.")
        mark_btn.clicked.connect(self._add_marker)
        cap.addWidget(mark_btn)
        self.seg_btn = QPushButton("Inicio de segmento aquí")
        self.seg_btn.setToolTip("Marca el inicio; vuelve a pulsar en el fin para crear el segmento.")
        self.seg_btn.clicked.connect(self._segment_click)
        cap.addWidget(self.seg_btn)
        lay.addLayout(cap)

        # --- Tabla de eventos ---
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Tipo", "Inicio (s)", "Fin (s)", "Clase"])
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.table, 1)
        rm = QPushButton("Quitar seleccionado")
        rm.clicked.connect(self._remove_selected)
        lay.addWidget(rm)

        self.hint = QLabel("Coloca marcas/segmentos con el video; se guardan en el proyecto "
                           "y se registran automáticamente al grabar.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: #8a929b; font-size: 11px;")
        lay.addWidget(self.hint)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        for e in (events if events is not None else []):
            self._add_row(e)

    # --- Reproducción -----------------------------------------------------
    def _on_duration(self, d: int) -> None:
        self._duration = d
        self.slider.setRange(0, d)
        if not self.table.rowCount() and d > 0:      # config nueva: prellenar
            for e in default_events(self.label_combo.currentText(), d):
                self._add_row(e)

    def _on_position(self, p: int) -> None:
        self.slider.setValue(p)
        self.time_lbl.setText(f"{_fmt(p)} / {_fmt(self._duration)}")

    def _toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause(); self.play_btn.setText("▶")
        else:
            self.player.play(); self.play_btn.setText("⏸")

    # --- Eventos ----------------------------------------------------------
    def _add_marker(self) -> None:
        self._add_row({"kind": "marker", "t": self.player.position(),
                       "label": self.label_combo.currentText()})

    def _segment_click(self) -> None:
        pos = self.player.position()
        if self._seg_start is None:
            self._seg_start = pos
            self.seg_btn.setText(f"Fin de segmento (inicio: {_fmt(pos)})")
        else:
            a, b = sorted((self._seg_start, pos))
            self._seg_start = None
            self.seg_btn.setText("Inicio de segmento aquí")
            if b - a >= 1:
                self._add_row({"kind": "segment", "start": a, "stop": b,
                               "label": self.label_combo.currentText()})

    def _add_row(self, e: dict) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        kind = e.get("kind", "segment")
        if kind == "marker":
            start = e.get("t", 0); stop = None
        else:
            start = e.get("start", 0); stop = e.get("stop", 0)
        self.table.setItem(r, 0, QTableWidgetItem("marca" if kind == "marker" else "segmento"))
        si = QTableWidgetItem(f"{start / 1000.0:.3f}")
        self.table.setItem(r, 1, si)
        self.table.setItem(r, 2, QTableWidgetItem("" if stop is None else f"{stop / 1000.0:.3f}"))
        self.table.setItem(r, 3, QTableWidgetItem(str(e.get("label", ""))))

    def _remove_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedItems()}, reverse=True)
        for r in rows:
            self.table.removeRow(r)

    def _cell(self, r: int, c: int) -> str:
        it = self.table.item(r, c)
        return it.text().strip() if it else ""

    # --- Resultado --------------------------------------------------------
    def result_events(self) -> list[dict]:
        out = []
        for r in range(self.table.rowCount()):
            kind = "marker" if self._cell(r, 0).startswith("marca") else "segment"
            label = self._cell(r, 3) or self.label_combo.currentText()
            try:
                start = int(round(float(self._cell(r, 1)) * 1000))
            except ValueError:
                continue
            if kind == "marker":
                out.append({"kind": "marker", "t": start, "label": label})
            else:
                try:
                    stop = int(round(float(self._cell(r, 2)) * 1000))
                except ValueError:
                    continue
                a, b = sorted((start, stop))
                if b - a >= 1:
                    out.append({"kind": "segment", "start": a, "stop": b, "label": label})
        return out

    def duration_ms(self) -> int:
        return int(self._duration)

    def closeEvent(self, event):  # noqa: N802
        try:
            self.player.stop()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)

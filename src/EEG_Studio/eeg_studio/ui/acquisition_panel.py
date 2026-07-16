"""Panel de adquisición en tiempo real (opcional).

Selecciona una fuente (Simulado / OpenViBE-LSL / CyKit-TCP), la conecta, muestra
la señal en vivo y permite grabar a un CSV local del proyecto e insertar
marcadores. La interfaz funciona sin este panel: la captura es opcional.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..acquisition import (
    EmotivDongleSource,
    FilePlaybackSource,
    LSLSource,
    SimulatedSource,
    TCPSource,
    emotiv_deps_available,
    pylsl_available,
    quality,
)
from ..acquisition.emotiv import quick_diagnose
from ..acquisition.recorder import CSVRecorder
from ..core import marks_sidecar, stim as stim_core

try:                                    # la estimulación por video requiere QtMultimedia
    from .stim_player import StimSession
    from .stim_timeline import StimTimelineDialog
    _STIM_OK = True
except Exception:                       # noqa: BLE001
    StimSession = None
    StimTimelineDialog = None
    _STIM_OK = False
from ..config import (
    CYKIT_CHANNEL_START,
    CYKIT_HOST,
    CYKIT_PORT,
    LIVE_REFRESH_MS,
    LIVE_WINDOW_SECONDS,
    LSL_SIGNAL_NAME,
    ONLINE_BUFFER_SAMPLES,
    RECORDINGS_DIR,
)

# --- Detección de lag de la señal en vivo ---
_LAG_STALL_S = 2.0        # sin muestras nuevas por este tiempo => la señal se colgó
_LAG_WINDOW_S = 3.0       # ventana para medir la tasa efectiva de muestreo
_LAG_RATE_FRAC = 0.6      # tasa efectiva < 60% de la nominal => la señal se retrasa


def _stim_key(cfg: dict) -> tuple:
    """Identidad de una config de estímulo para detectar repetidos al importar:
    etiqueta + nombre del archivo de video (sin distinguir mayúsculas)."""
    label = str(cfg.get("label", "")).strip().lower()
    base = os.path.basename(str(cfg.get("path", ""))).strip().lower()
    return (label, base)


class AcquisitionPanel(QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self.source = None
        self.recorder: CSVRecorder | None = None
        self._configured = False
        self._keep_idx: list[int] | None = None
        self._n_samples = 0
        self._t_start = 0.0
        # Segmentos marcados durante la grabación (start, stop, etiqueta).
        self._rec_segments: list[tuple[int, int, str]] = []
        self._seg_active = False
        self._seg_start = 0
        self._seg_label = ""
        self._paused = False                     # grabación en pausa (no escribe)
        self._rec_path = None
        self._rec_alias = None
        self._roll: np.ndarray | None = None    # buffer circular para inferencia
        self._roll_filled = 0
        self._quality_tick = 0                   # para refrescar la calidad cada N ticks
        self._battery_warned = False             # aviso de batería baja (una sola vez)
        # Detección de LAG (la señal se retrasa / se pierden muestras).
        self._last_chunk_t = 0.0                 # instante del último bloque recibido
        self._rate_t0 = 0.0                      # ventana para medir la tasa efectiva
        self._rate_n0 = 0
        self._lagging = False                    # estado actual de lag (para el aviso)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)   # menos estrangulamiento en 2º plano
        self._timer.setInterval(LIVE_REFRESH_MS)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self._update_states()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Fuente de adquisición:"))
        self.source_combo = QComboBox()
        self.source_combo.addItem("Simulado (sin hardware)", "sim")
        self.source_combo.addItem("OpenViBE Acquisition Server (LSL)", "lsl")
        self.source_combo.addItem("Emotiv EPOC+ (lector integrado, sin CyKit)", "emotiv")
        self.source_combo.addItem("CyKit / TCP (respaldo)", "tcp")
        self.source_combo.addItem("Reproducir grabación (archivo)", "playback")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        layout.addWidget(self.source_combo)

        # Parámetros por fuente (apilados, en el mismo orden que el combo).
        self.params = QStackedWidget()
        self.params.addWidget(self._sim_params())
        self.params.addWidget(self._lsl_params())
        self.params.addWidget(self._emotiv_params())
        self.params.addWidget(self._tcp_params())
        self.params.addWidget(self._playback_params())
        layout.addWidget(self.params)

        conn = QHBoxLayout()
        self.connect_btn = QPushButton("Conectar")
        self.connect_btn.clicked.connect(self.toggle_connection)
        conn.addWidget(self.connect_btn)
        layout.addLayout(conn)

        # Estado + calidad (canales detectados): ARRIBA, para que quede siempre a la
        # vista aunque haya muchos botones de grabación debajo.
        self.status = QLabel("Desconectado.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.quality_label = QLabel("")
        self.quality_label.setWordWrap(True)
        layout.addWidget(self.quality_label)

        # Batería de la diadema (solo si la fuente la reporta, p. ej. Emotiv) +
        # umbral de advertencia configurable (por defecto 70%).
        self.battery_row = QWidget()
        bat_lay = QHBoxLayout(self.battery_row)
        bat_lay.setContentsMargins(0, 0, 0, 0)
        self.battery_label = QLabel("")
        bat_lay.addWidget(self.battery_label, 1)
        bat_lay.addWidget(QLabel("Avisar por debajo de:"))
        self.battery_thresh = QSpinBox()
        self.battery_thresh.setRange(0, 100)
        self.battery_thresh.setSuffix(" %")
        self.battery_thresh.setValue(
            int(self.controller._settings().value("battery_warn_pct", 70, type=int)))
        self.battery_thresh.setToolTip(
            "Avisa cuando la batería de la diadema baje de este porcentaje "
            "(la diadema vieja suele fallar por debajo).")
        self.battery_thresh.valueChanged.connect(self._on_battery_thresh)
        bat_lay.addWidget(self.battery_thresh)
        self.battery_row.setVisible(False)
        layout.addWidget(self.battery_row)

        # --- Grabación --------------------------------------------------------
        rec_box = QGroupBox("Grabación")
        rec_layout = QVBoxLayout(rec_box)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Nombre:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Nombre de la grabación (opcional)")
        self.name_edit.setToolTip("Nombre del CSV y alias de la fuente. Si se deja "
                                  "vacío, se usa la fecha/hora (rec_AAAAMMDD_HHMMSS).")
        name_row.addWidget(self.name_edit, 1)
        rec_layout.addLayout(name_row)

        # Controles: iniciar/detener · pausar · descartar.
        ctl = QHBoxLayout()
        self.record_btn = QPushButton("● Iniciar grabación")
        self.record_btn.clicked.connect(self.toggle_recording)
        self.pause_btn = QPushButton("⏸ Pausar")
        self.pause_btn.setToolTip("Pausa/reanuda la grabación (la señal en vivo sigue).")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.discard_btn = QPushButton("✕ Descartar")
        self.discard_btn.setToolTip("Detiene y BORRA la grabación actual (pide confirmación).")
        self.discard_btn.clicked.connect(self.discard_recording)
        ctl.addWidget(self.record_btn, 2)
        ctl.addWidget(self.pause_btn, 1)
        ctl.addWidget(self.discard_btn, 1)
        rec_layout.addLayout(ctl)

        # Etiqueta + botones de marca/segmento, compactos en rejilla.
        self.marker_edit = QLineEdit()
        self.marker_edit.setPlaceholderText("Etiqueta (p. ej. mano_izq / ojos_cerrados)")
        rec_layout.addWidget(self.marker_edit)

        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        self.marker_btn = QPushButton("Marca · F3")
        self.marker_btn.setToolTip("Marca un instante puntual en la grabación (F3).")
        self.marker_btn.clicked.connect(self._insert_marker)
        self.segment_btn = QPushButton("▶ Segmento · F4")
        self.segment_btn.setToolTip("Segmento con inicio/fin: 1er clic inicio, 2º fin (F4).")
        self.segment_btn.clicked.connect(self._toggle_segment)
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 600)
        self.duration_spin.setValue(5)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setToolTip("Duración de la marca de duración fija.")
        self.timed_btn = QPushButton("Marca fija · F5")
        self.timed_btn.setToolTip("Segmento de la duración indicada, desde ahora (F5).")
        self.timed_btn.clicked.connect(self._add_timed_marker)
        grid.addWidget(self.marker_btn, 0, 0)
        grid.addWidget(self.segment_btn, 0, 1)
        grid.addWidget(self.duration_spin, 1, 0)
        grid.addWidget(self.timed_btn, 1, 1)
        rec_layout.addLayout(grid)

        mk_hint = QLabel("Marca = un instante · Segmento = inicio/fin · "
                         "Marca fija = N s desde ahora.")
        mk_hint.setWordWrap(True)
        mk_hint.setStyleSheet("color: #8a929b; font-size: 11px;")
        rec_layout.addWidget(mk_hint)
        layout.addWidget(rec_box)

        layout.addWidget(self._stim_section())
        layout.addStretch(1)

        # Atajos de teclado para marcar sin soltar el ratón del casco/tarea.
        self._sc_marker = QShortcut(QKeySequence("F3"), self)
        self._sc_marker.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._sc_marker.activated.connect(self._insert_marker)
        self._sc_segment = QShortcut(QKeySequence("F4"), self)
        self._sc_segment.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._sc_segment.activated.connect(self._toggle_segment)
        self._sc_timed = QShortcut(QKeySequence("F5"), self)
        self._sc_timed.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._sc_timed.activated.connect(self._add_timed_marker)

    def _sim_params(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        lay.addRow(QLabel("Genera 14 canales sintéticos a 128 Hz.\n"
                          "Útil para probar sin el casco."))
        return w

    def _lsl_params(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        self.lsl_name = QLineEdit(LSL_SIGNAL_NAME)
        lay.addRow("Nombre del stream:", self.lsl_name)
        if pylsl_available():
            hint = "Activa la salida LSL en el Acquisition Server y ponlo en Play."
        else:
            hint = ("⚠ pylsl no está instalado. Instala 'pylsl' en el venv para usar "
                    "esta fuente (pip install pylsl).")
        note = QLabel(hint)
        note.setWordWrap(True)
        lay.addRow(note)
        return w

    def _emotiv_params(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        self.emotiv_mode = QComboBox()
        self.emotiv_mode.addItem("Auto (detectar 14/16-bit)", "auto")
        self.emotiv_mode.addItem("16 bits (EPOC+ Consumer)", "16bit")
        self.emotiv_mode.addItem("14 bits (EPOC+ modo 14-bit)", "14bit")
        self.emotiv_serial = QLineEdit()
        self.emotiv_serial.setPlaceholderText("(automático desde el dongle)")
        lay.addRow("Modo:", self.emotiv_mode)
        lay.addRow("Nº de serie:", self.emotiv_serial)
        if emotiv_deps_available():
            hint = ("Es CyKit integrado en la app (Python 3.13, sin programa aparte). "
                    "Conecta el receptor USB y empareja el casco; 'Auto' detecta el "
                    "modo 14/16-bit por ti.")
        else:
            hint = ("⚠ Faltan dependencias: instala 'hidapi' y 'pycryptodome' "
                    "en el venv para usar esta fuente.")
        note = QLabel(hint)
        note.setWordWrap(True)
        lay.addRow(note)
        self.emotiv_test_btn = QPushButton("Probar dongle Emotiv…")
        self.emotiv_test_btn.setToolTip("Comprueba detección, datos, modo y calidad de "
                                        "señal sin necesidad de conectar.")
        self.emotiv_test_btn.clicked.connect(self._test_emotiv)
        self.emotiv_test_btn.setEnabled(emotiv_deps_available())
        lay.addRow(self.emotiv_test_btn)
        return w

    def _test_emotiv(self) -> None:
        """Diagnóstico rápido del dongle (en un hilo, con su resultado en un diálogo)."""
        if not emotiv_deps_available():
            self.controller.warn("Faltan dependencias",
                                 "Instala 'hidapi' y 'pycryptodome' en el venv.")
            return
        if self.source is not None:
            self.controller.info("Ocupado",
                                 "Desconecta la fuente antes de probar el dongle.")
            return
        serial = self.emotiv_serial.text().strip() or None
        self.controller._busy("Probando dongle Emotiv…")
        self.controller.progress.setRange(0, 0)
        self.controller.progress.show()

        def done(report):
            self.controller._idle()
            title = "Dongle Emotiv — OK" if report.get("ok") else "Dongle Emotiv — revisar"
            self.controller.info(title, report["summary"])

        self.controller._spawn(lambda: quick_diagnose(serial=serial), done)

    def _tcp_params(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        self.tcp_host = QLineEdit(CYKIT_HOST)
        self.tcp_port = QSpinBox()
        self.tcp_port.setRange(1025, 65535)
        self.tcp_port.setValue(CYKIT_PORT)
        self.tcp_nch = QSpinBox()
        self.tcp_nch.setRange(1, 64)
        self.tcp_nch.setValue(14)
        self.tcp_start = QSpinBox()
        self.tcp_start.setRange(0, 32)
        self.tcp_start.setValue(CYKIT_CHANNEL_START)
        lay.addRow("Host:", self.tcp_host)
        lay.addRow("Puerto:", self.tcp_port)
        lay.addRow("Nº de canales:", self.tcp_nch)
        lay.addRow("Columna inicial:", self.tcp_start)
        launch_btn = QPushButton("Configurar / lanzar CyKit…")
        launch_btn.clicked.connect(self._open_cykit_launcher)
        lay.addRow(launch_btn)
        note = QLabel("Abre el configurador para activar banderas, ajustar cantidades "
                      "y lanzar CyKit. Con 'nocounter', la columna inicial es 0.")
        note.setWordWrap(True)
        lay.addRow(note)
        return w

    def _open_cykit_launcher(self) -> None:
        from .cykit_launcher import CyKitLauncherDialog
        dlg = CyKitLauncherDialog(self, self)
        dlg.show()

    def apply_cykit_settings(self, host: str, port: int, channel_start: int) -> None:
        """Recibe los ajustes del configurador de CyKit y selecciona la fuente."""
        self.tcp_host.setText(host)
        self.tcp_port.setValue(int(port))
        self.tcp_start.setValue(int(channel_start))
        idx = self.source_combo.findData("tcp")
        if idx >= 0:
            self.source_combo.setCurrentIndex(idx)

    def _playback_params(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        self.playback_path = QLineEdit()
        self.playback_path.setReadOnly(True)
        self.playback_path.setPlaceholderText("Ningún archivo elegido")
        pick = QPushButton("Elegir archivo…")
        pick.clicked.connect(self._choose_playback_file)
        row = QHBoxLayout()
        row.addWidget(self.playback_path, 1)
        row.addWidget(pick, 0)
        lay.addRow("Grabación:", self._wrap_row(row))
        note = QLabel("Reproduce un CSV grabado como si fuera la diadema (1x, una "
                      "pasada). Útil para ver la señal o controlar el brazo desde una "
                      "grabación previa sin el EPOC+.")
        note.setWordWrap(True)
        lay.addRow(note)
        return w

    @staticmethod
    def _wrap_row(row_layout) -> QWidget:
        """Envuelve un layout horizontal en un QWidget para meterlo en un QFormLayout."""
        holder = QWidget()
        holder.setLayout(row_layout)
        return holder

    def _choose_playback_file(self) -> None:
        start = ""
        proj = getattr(self.controller, "project", None)
        if proj is not None:
            rec_dir = os.path.join(proj.path, RECORDINGS_DIR)
            start = rec_dir if os.path.isdir(rec_dir) else proj.path
        path, _ = QFileDialog.getOpenFileName(
            self, "Elegir grabación para reproducir", start,
            "Grabaciones EEG (*.csv *.csv.gz);;Todos los archivos (*)")
        if path:
            self.playback_path.setText(path)

    # ------------------------------------------------------------------ #
    def _on_source_changed(self, idx: int) -> None:
        self.params.setCurrentIndex(idx)

    def _make_source(self):
        kind = self.source_combo.currentData()
        if kind == "sim":
            return SimulatedSource()
        if kind == "lsl":
            return LSLSource(self.lsl_name.text().strip() or LSL_SIGNAL_NAME)
        if kind == "emotiv":
            return EmotivDongleSource(
                mode=self.emotiv_mode.currentData(),
                serial=self.emotiv_serial.text().strip() or None,
            )
        if kind == "tcp":
            return TCPSource(
                self.tcp_host.text().strip() or CYKIT_HOST,
                self.tcp_port.value(),
                n_channels=self.tcp_nch.value(),
                channel_start=self.tcp_start.value(),
            )
        if kind == "playback":
            path = self.playback_path.text().strip()
            if not path:
                raise ValueError("Elige un archivo de grabación (CSV) para reproducir.")
            return FilePlaybackSource(path)
        return None

    # --- Conexión ---------------------------------------------------------
    def toggle_connection(self) -> None:
        if self.source is None:
            self._connect()
        else:
            self._disconnect()

    def _connect(self) -> None:
        try:
            self.source = self._make_source()
            self.source.start()
        except Exception as exc:  # noqa: BLE001
            self.controller.warn("No se pudo conectar", str(exc))
            self.source = None
            return
        self._configured = False
        self._keep_idx: list[int] | None = None
        self._n_samples = 0
        self._t_start = time.perf_counter()
        self._last_chunk_t = self._t_start           # reinicia la detección de lag
        self._rate_t0 = self._t_start
        self._rate_n0 = 0
        self._lagging = False
        self.controller.show_live_view()
        self._timer.start()
        self.status.setText("Conectando… esperando muestras.")
        self._update_states()

    def _disconnect(self) -> None:
        self._timer.stop()
        if self.recorder is not None:
            self._stop_recording()
        if self.source is not None:
            self.source.stop()
        self.source = None
        self._roll = None
        self._roll_filled = 0
        self._configured = False
        self._keep_idx: list[int] | None = None
        self.quality_label.setText("")
        self.status.setText("Desconectado.")
        self._update_states()

    # --- Bucle de adquisición (hilo principal vía QTimer) -----------------
    def _kept_indices(self) -> list[int] | None:
        """Índices de los canales ACTIVOS de la fuente (los NO excluidos del proyecto).

        «Análisis (CSV)» solo muestra y procesa los canales activos (p. ej. sin los
        EOG), así que el visor en vivo y la inferencia deben ver LOS MISMOS: si no, un
        modelo entrenado con N canales recibiría otro número y fallaría. La exclusión
        se guarda con el nombre **original** del canal (igual que ``project.kept_indices``).

        Devuelve ``None`` si no hay nada que filtrar (caso normal) o si no quedaría
        ningún canal. **No afecta a la grabación**: el CSV se escribe en el hilo
        productor (``_record_tap``) con la señal íntegra — excluir es cosa del
        análisis, no de la captura.
        """
        proj = getattr(self.controller, "project", None)
        if proj is None or self.source is None:
            return None
        excluded = set(proj.excluded_channels())
        if not excluded:
            return None
        # La exclusión se guarda con el nombre ORIGINAL del CSV («Channel 13»), pero
        # una fuente EN VIVO (Emotiv/LSL/Simulado) reporta el nombre clínico («F8»).
        # Sin comparar contra las dos formas, en vivo no coincidía ninguna y no se
        # excluía nada: el modelo entrenado con los canales activos recibía todos.
        aliases = proj.state.get("channel_aliases") or {}
        excluded |= {aliases.get(n, n) for n in excluded}
        names = list(self.source.channel_names)
        keep = [i for i, n in enumerate(names) if n not in excluded]
        return keep if 0 < len(keep) < len(names) else None

    def _display_channel_names(self, keep: list[int] | None = None) -> list[str]:
        """Nombres de canal a mostrar en el visor en vivo, con los alias del proyecto.

        Los CSV de OpenViBE nombran los canales «Channel 1».. y es el PROYECTO quien
        guarda el alias clínico (AF3, F7…) — por eso «Análisis (CSV)» los muestra con
        su nombre real y su color por región. Al reproducir una grabación como fuente
        en vivo llegaban los nombres crudos, así que el visor perdía los nombres y el
        código de colores (``channel_color`` los asigna POR NOMBRE). Traducirlos aquí
        deja ambas pestañas coherentes; las fuentes que ya reportan nombres reales
        (Emotiv, LSL) no cambian: el alias solo se aplica si existe.

        ``keep`` (de :meth:`_kept_indices`) deja fuera los canales excluidos.
        """
        names = list(self.source.channel_names) if self.source else []
        if keep is not None:
            names = [names[i] for i in keep]
        proj = getattr(self.controller, "project", None)
        aliases = (proj.state.get("channel_aliases") or {}) if proj is not None else {}
        return [aliases.get(n, n) for n in names]

    def _tick(self) -> None:
        if self.source is None:
            return
        if self.source.error:
            msg = self.source.error
            self._disconnect()               # guarda la grabación en curso, si la había
            self.status.setText(f"⚠ La fuente se detuvo: {msg}")
            self.controller.warn(
                "Error de adquisición",
                f"La fuente dejó de enviar datos:\n{msg}\n\n"
                "Vuelve a pulsar «Conectar» para seguir grabando.")
            return
        now = time.perf_counter()
        chunk = self.source.read()
        if chunk is None:
            self._check_lag(now, False)          # ¿lleva mucho sin muestras nuevas?
            return
        if not self._configured:
            # Los canales activos se fijan al conectar y no cambian mientras dure la
            # conexión (el buffer de inferencia se dimensiona con ellos).
            self._keep_idx = self._kept_indices()
            names = self._display_channel_names(self._keep_idx)
            self.controller.live_view.configure(
                names, self.source.sample_rate, LIVE_WINDOW_SECONDS
            )
            self._roll = np.zeros((len(names), ONLINE_BUFFER_SAMPLES))
            self._roll_filled = 0
            self._configured = True
        self._last_chunk_t = now
        # Visor E INFERENCIA ven solo los canales activos, como «Análisis (CSV)»: un
        # modelo entrenado con los activos debe recibir esos mismos. La grabación NO
        # se filtra: va por el tap del hilo productor con la señal íntegra.
        if self._keep_idx is not None:
            chunk = chunk[self._keep_idx, :]
        self.controller.live_view.append(chunk)
        self._push_buffer(chunk)
        # La grabación ya se escribe en el HILO PRODUCTOR (tap), NO aquí: así no se
        # pierde nada aunque este temporizador se estrangule en segundo plano (dos
        # monitores / app sin foco).
        self._n_samples += chunk.shape[1]
        self._check_lag(now, True)
        self._update_status()
        self._update_battery()
        self._quality_tick = (self._quality_tick + 1) % 8
        if self._quality_tick == 0:              # ~4 Hz: evita parpadeo
            self._update_quality()

    def _check_lag(self, now: float, got_chunk: bool) -> None:
        """Avisa (sin bloquear) cuando la señal en vivo se retrasa: o bien lleva un
        rato sin muestras nuevas, o su tasa efectiva cae muy por debajo de la nominal
        (típico del EPOC+ con la batería baja)."""
        if not self._configured or self.source is None:
            return
        fs = float(self.source.sample_rate or 1.0)
        lagging = (now - self._last_chunk_t) > _LAG_STALL_S      # se colgó
        dt = now - self._rate_t0
        if dt >= _LAG_WINDOW_S:                                  # tasa efectiva en la ventana
            eff = (self._n_samples - self._rate_n0) / dt
            if eff < _LAG_RATE_FRAC * fs:
                lagging = True
            self._rate_t0 = now
            self._rate_n0 = self._n_samples
        if lagging and not self._lagging:                       # entra en lag: avisar UNA vez
            self._lagging = True
            self.status.setText("⚠ La señal se está retrasando (lag / pérdida de muestras). "
                                "Suele pasar con la batería baja: conviene cargar la diadema.")
            self.status.setStyleSheet("color: #ff6b6b; font-weight: 600;")
            self.controller.statusBar().showMessage(
                "⚠ Señal EEG retrasada (lag). Revisa la batería/conexión de la diadema.", 8000)
        elif got_chunk and not lagging and self._lagging:        # recuperada
            self._lagging = False
            self.status.setStyleSheet("")

    def _record_tap(self, chunk) -> None:
        """Escribe un bloque a la grabación. Se llama en el HILO PRODUCTOR (no en la
        GUI), así capturar a disco no depende del temporizador. Respeta la pausa."""
        rec = self.recorder
        if rec is not None and not self._paused:
            rec.write(chunk)

    def _on_battery_thresh(self, value: int) -> None:
        self.controller._settings().setValue("battery_warn_pct", int(value))
        self._battery_warned = False             # reevaluar con el nuevo umbral
        self._update_battery()

    def _update_battery(self) -> None:
        pct = self.source.battery if self.source is not None else None
        self.battery_row.setVisible(pct is not None)
        if pct is None:
            return
        thr = self.battery_thresh.value()
        if pct < thr:
            self.battery_label.setText(f"🔋 Batería: {pct}%   ⚠ BAJA (por debajo de {thr}%)")
            self.battery_label.setStyleSheet("color: #ff6b6b; font-weight: 600;")
            if not self._battery_warned:
                self._battery_warned = True
                self.controller.warn(
                    "Batería baja de la diadema",
                    f"La batería del Emotiv está al {pct}% (por debajo del {thr}%).\n\n"
                    "Con esta diadema, por debajo de ese nivel la señal suele fallar; "
                    "conviene cargarla. Puedes ajustar el umbral en «Avisar por debajo de».")
        else:
            self.battery_label.setText(f"🔋 Batería: {pct}%")
            self.battery_label.setStyleSheet("color: #7fd1b9; font-weight: 600;")
            self._battery_warned = False

    def _push_buffer(self, chunk) -> None:
        """Mantiene el buffer circular con las últimas muestras (para inferencia)."""
        if self._roll is None:
            return
        k = chunk.shape[1]
        cap = self._roll.shape[1]
        if k >= cap:
            self._roll[:] = chunk[:, -cap:]
            self._roll_filled = cap
        else:
            self._roll = np.roll(self._roll, -k, axis=1)
            self._roll[:, -k:] = chunk
            self._roll_filled = min(self._roll_filled + k, cap)

    # --- API para el modo de control en tiempo real -----------------------
    def is_streaming(self) -> bool:
        return self.source is not None and self._configured

    def stream_fs(self) -> float | None:
        return self.source.sample_rate if self.source is not None else None

    def latest_window(self, n: int):
        """Últimas ``n`` muestras ``(n_canales, n)`` o ``None`` si aún no hay suficientes."""
        if self._roll is None or self._roll_filled < n:
            return None
        return self._roll[:, -n:].copy()

    def _update_status(self) -> None:
        if self._lagging:                    # no pisar el aviso de lag (rojo) mientras dure
            return
        elapsed = max(1e-6, time.perf_counter() - self._t_start)
        fs = self._n_samples / elapsed
        rec = f" | grabando: {self.recorder.n_samples} muestras" if self.recorder else ""
        info = getattr(self.source, "info", "") if self.source else ""
        prefix = f"{info}\n" if info else ""
        self.status.setText(
            f"{prefix}Conectado · {self._n_samples} muestras · ~{fs:.1f} Hz{rec}"
        )

    def _update_quality(self) -> None:
        """Refresca el indicador de calidad/ruido con la última ~1 s de señal."""
        if self.source is None or self._roll is None or self._roll_filled < 32:
            self.quality_label.setText("")
            return
        fs = int(self.source.sample_rate) or 128
        n = min(self._roll_filled, max(fs, 64))
        q = quality.assess(self._roll[:, -n:])
        if q["n"] == 0:
            self.quality_label.setText("")
            return
        if q["is_noise"]:
            color, txt = "#e06c6c", f"⚠ Señal RUIDOSA — {q['n_ok']}/{q['n']} canales OK"
        elif q["n_bad"]:
            color, txt = "#d6a23e", f"● Señal aceptable — {q['n_ok']}/{q['n']} OK · {q['n_bad']} con ruido"
        else:
            color, txt = "#7fd1b9", f"● Señal buena — {q['n']}/{q['n']} canales OK"
        self.quality_label.setText(txt)
        self.quality_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    # --- Grabación --------------------------------------------------------
    def toggle_recording(self) -> None:
        if self.recorder is None:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        if self.source is None or not self._configured:
            self.controller.info("Sin señal", "Conecta una fuente y espera muestras primero.")
            return
        if self.controller.project is None:
            self.controller.info("Sin proyecto", "Crea o abre un proyecto para grabar.")
            return
        rec_dir = os.path.join(self.controller.project.path, RECORDINGS_DIR)
        os.makedirs(rec_dir, exist_ok=True)
        raw_name = self.name_edit.text().strip()
        base = self._safe_name(raw_name) or time.strftime("rec_%Y%m%d_%H%M%S")
        path = self._unique_rec_path(rec_dir, base)
        # Alias mostrado: el nombre escrito, o el del archivo generado.
        self._rec_alias = raw_name or os.path.splitext(os.path.basename(path))[0]
        self.recorder = CSVRecorder(path, self.source.n_channels, self.source.sample_rate)
        self._rec_path = path
        self.source.set_tap(self._record_tap)   # graba en el hilo productor (blindaje)
        # Reinicia el estado de esta grabación.
        self._rec_segments = []
        self._seg_active = False
        self._paused = False
        self.segment_btn.setText("▶ Segmento · F4")
        self.pause_btn.setText("⏸ Pausar")
        self.record_btn.setText("■ Detener grabación")
        self._flush_sidecar()                # crea el lateral (vacío) desde ya
        self.status.setText(f"Grabando «{self._rec_alias}» en {os.path.basename(path)}…")
        self._update_states()

    def _flush_sidecar(self) -> None:
        """Escribe el archivo lateral con los segmentos actuales (blindaje anti-pérdida)."""
        if self.recorder is None or not self._rec_path:
            return
        fs = self.source.sample_rate if self.source else 128.0
        marks_sidecar.write_marks(self._rec_path, self._rec_segments, fs)

    def toggle_pause(self) -> None:
        if self.recorder is None:
            self.controller.info("Sin grabación", "No hay ninguna grabación en curso.")
            return
        self._paused = not self._paused
        self.pause_btn.setText("▶ Reanudar" if self._paused else "⏸ Pausar")
        if self._paused:
            self.status.setText(f"⏸ Grabación EN PAUSA en {self.recorder.n_samples} muestras…")
        else:
            self.status.setText(f"Grabando «{self._rec_alias}»…")

    def discard_recording(self) -> None:
        """Detiene y BORRA la grabación actual (archivo + lateral), tras confirmar."""
        if self.recorder is None:
            self.controller.info("Sin grabación", "No hay ninguna grabación en curso.")
            return
        alias = self._rec_alias or os.path.basename(self._rec_path or "")
        if QMessageBox.question(
                self, "Descartar grabación",
                f"¿Descartar la grabación «{alias}»?\n\nSe borrará el archivo y sus marcas. "
                "Esta acción no se puede deshacer.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        path = self._rec_path
        if self.source is not None:
            self.source.set_tap(None)        # deja de escribir en el hilo productor
        try:
            self.recorder.close()
        except Exception:  # noqa: BLE001
            pass
        self.recorder = None
        self._paused = False
        self._seg_active = False
        self._rec_segments = []
        self.record_btn.setText("● Iniciar grabación")
        self.pause_btn.setText("⏸ Pausar")
        self.segment_btn.setText("▶ Segmento · F4")
        try:
            if path and os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass
        if path:
            marks_sidecar.remove_marks(path)
        self.status.setText("Grabación descartada.")
        self._update_states()

    @staticmethod
    def _safe_name(name: str) -> str:
        """Nombre de archivo seguro (sin caracteres problemáticos)."""
        name = re.sub(r"[^\w\- ]", "", name, flags=re.UNICODE).strip()
        return re.sub(r"\s+", "_", name)

    @staticmethod
    def _unique_rec_path(rec_dir: str, base: str) -> str:
        """Ruta ``base.csv`` en ``rec_dir``, con sufijo _2, _3… si ya existe."""
        candidate = os.path.join(rec_dir, base + ".csv")
        i = 2
        while os.path.exists(candidate):
            candidate = os.path.join(rec_dir, f"{base}_{i}.csv")
            i += 1
        return candidate

    def _stop_recording(self) -> None:
        if self.recorder is None:
            return
        if self.source is not None:
            self.source.set_tap(None)        # deja de escribir en el hilo productor
        # Cierra un segmento que quedara abierto (fin = última muestra grabada).
        if self._seg_active:
            stop = self.recorder.n_samples
            if stop > self._seg_start:
                self._rec_segments.append((self._seg_start, stop, self._seg_label))
            self._seg_active = False
            self.segment_btn.setText("▶ Segmento · F4")
        # Recorta segmentos cuyo fin caiga más allá de lo grabado (marcas de
        # duración fija que no llegaron a completarse); descarta los vacíos.
        final = self.recorder.n_samples
        segments = [(s, min(e, final), lbl) for (s, e, lbl) in self._rec_segments
                    if s < final and min(e, final) - s >= 1]
        self._rec_segments = segments
        self._flush_sidecar()                # lateral final (por si el guardado fallara)
        self.recorder.close()
        path = self._rec_path
        alias = getattr(self, "_rec_alias", None)
        self.recorder = None
        self._paused = False
        self.pause_btn.setText("⏸ Pausar")
        self.record_btn.setText("● Iniciar grabación")
        self._update_states()
        # La grabación se añade como fuente del proyecto AUTOMÁTICAMENTE (sin preguntar),
        # con su nombre y segmentos. Se guarda enseguida (no se pierde al cerrar).
        self.controller.add_recording_as_source(path, segments, alias)

    # --- Estimulación sincronizada ---------------------------------------
    def _stim_section(self) -> QGroupBox:
        box = QGroupBox("Estimulación sincronizada")
        v = QVBoxLayout(box)
        if not _STIM_OK:
            note = QLabel("Requiere QtMultimedia (reproducción de video). No disponible "
                          "en esta instalación de PyQt6.")
            note.setWordWrap(True)
            note.setStyleSheet("color: #8a929b; font-size: 11px;")
            v.addWidget(note)
            self.stim_list = QListWidget()          # placeholder (evita atributos ausentes)
            self.stim_list.setVisible(False)
            v.addWidget(self.stim_list)
            self._stim_session = None
            return box
        v.addWidget(QLabel("Reproduce un video de estímulo: graba y coloca los "
                           "segmentos solo (sin error humano)."))
        self.stim_list = QListWidget()
        self.stim_list.setMaximumHeight(110)
        self.stim_list.itemDoubleClicked.connect(self._configure_stim_item)
        v.addWidget(self.stim_list)

        # Selector de monitor: en qué pantalla se despliega el VIDEO del estímulo
        # (la interfaz principal se queda en la pantalla actual).
        mon_row = QHBoxLayout()
        mon_row.addWidget(QLabel("Monitor del video:"))
        self.monitor_combo = QComboBox()
        self.monitor_combo.setToolTip(
            "Pantalla donde se muestra el video del estímulo (a pantalla completa). "
            "La interfaz de la app permanece en su pantalla actual.")
        self.monitor_combo.currentIndexChanged.connect(self._save_stim_monitor)
        mon_row.addWidget(self.monitor_combo, 1)
        refresh_mon = QPushButton("⟳")
        refresh_mon.setFixedWidth(32)
        refresh_mon.setToolTip("Actualizar la lista de monitores conectados.")
        refresh_mon.clicked.connect(self._refresh_monitors)
        mon_row.addWidget(refresh_mon)
        v.addLayout(mon_row)

        row = QHBoxLayout()
        add_btn = QPushButton("＋ Añadir / configurar…")
        add_btn.clicked.connect(self._add_stim)
        self.stim_play_btn = QPushButton("▶ Reproducir")
        self.stim_play_btn.clicked.connect(self._play_stim)
        rm_btn = QPushButton("Quitar")
        rm_btn.clicked.connect(self._remove_stim)
        row.addWidget(add_btn); row.addWidget(self.stim_play_btn); row.addWidget(rm_btn)
        v.addLayout(row)
        row2 = QHBoxLayout()
        exp_btn = QPushButton("Exportar configuración…"); exp_btn.clicked.connect(self._export_stim)
        imp_btn = QPushButton("Importar…"); imp_btn.clicked.connect(self._import_stim)
        row2.addWidget(exp_btn); row2.addWidget(imp_btn); row2.addStretch(1)
        v.addLayout(row2)
        self._stim_session = None
        self._refresh_monitors()
        self.refresh_stim()
        return box

    def _refresh_monitors(self) -> None:
        """Puebla el selector con los monitores conectados (por defecto uno externo)."""
        app = QApplication.instance()
        screens = list(app.screens()) if app else []
        primary = app.primaryScreen() if app else None
        self.monitor_combo.blockSignals(True)
        self.monitor_combo.clear()
        for i, s in enumerate(screens):
            g = s.geometry()
            tag = " (principal)" if s is primary else " (externo)"
            self.monitor_combo.addItem(f"Monitor {i + 1}: {g.width()}×{g.height()}{tag}", i)
        default = 0
        for i, s in enumerate(screens):                 # preferir un monitor externo
            if s is not primary:
                default = i
                break
        saved = self.controller._settings().value("stim_monitor", None)
        try:
            saved = int(saved) if saved is not None else None
        except (TypeError, ValueError):
            saved = None
        idx = self.monitor_combo.findData(saved) if saved is not None else -1
        self.monitor_combo.setCurrentIndex(idx if idx >= 0 else default)
        self.monitor_combo.blockSignals(False)

    def _save_stim_monitor(self, *_args) -> None:
        data = self.monitor_combo.currentData()
        if data is not None:
            self.controller._settings().setValue("stim_monitor", int(data))

    def _selected_screen(self):
        app = QApplication.instance()
        screens = list(app.screens()) if app else []
        idx = self.monitor_combo.currentData()
        if idx is not None and 0 <= int(idx) < len(screens):
            return screens[int(idx)]
        return None

    def refresh_stim(self) -> None:
        """Repuebla la lista de estímulos configurados desde el proyecto."""
        self.stim_list.clear()
        proj = self.controller.project
        if proj is None:
            return
        for cfg in proj.stim_videos():
            events = cfg.get("events", [])
            n_seg = sum(1 for e in events if e.get("kind") == "segment")
            cls = sorted({str(e.get("label", "")) for e in events if e.get("label")})
            cls_txt = ", ".join(cls) if cls else str(cfg.get("label", "?"))
            missing = "" if os.path.isfile(cfg.get("path", "")) else "  ⚠ video no encontrado"
            it = QListWidgetItem(f"🎬 {cfg.get('name', '')} — [{cls_txt}]"
                                 f"  ({n_seg} seg){missing}")
            it.setData(Qt.ItemDataRole.UserRole, cfg.get("id"))
            self.stim_list.addItem(it)

    def _add_stim(self) -> None:
        """Explora una carpeta para elegir el video y abre su línea de tiempo.

        Por defecto arranca en ``data/videos`` (Delfin), pero sirve para videos en
        cualquier ubicación (otros proyectos)."""
        if not self.controller._require_project():
            return
        start_dir = stim_core.find_videos_dir() or ""
        path, _f = QFileDialog.getOpenFileName(
            self, "Elegir video de estímulo", start_dir,
            "Videos (*.mp4 *.mov *.avi *.mkv *.webm *.m4v);;Todos los archivos (*)")
        if not path:
            return
        self._open_timeline(path, stim_core.class_from_filename(path),
                            None, os.path.basename(path))

    def _configure_stim_item(self, item) -> None:
        cfg = self._stim_config(item.data(Qt.ItemDataRole.UserRole))
        if cfg:
            self._open_timeline(cfg.get("path", ""), cfg.get("label"),
                                cfg, cfg.get("name", ""), cfg.get("id"))

    def _open_timeline(self, path, label, existing, name, vid_id=None) -> None:
        events = existing.get("events") if existing else None
        classes = stim_core.project_classes(self.controller.project)
        dlg = StimTimelineDialog(path, label, events, classes, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cfg = {"id": vid_id, "path": path, "name": name,
               "label": dlg.label_combo.currentText(),
               "duration_ms": dlg.duration_ms(), "events": dlg.result_events()}
        self.controller.project.save_stim_video(cfg)
        self.controller.request_autosave()
        self.refresh_stim()

    def _export_stim(self) -> None:
        proj = self.controller.project
        if proj is None or not proj.stim_videos():
            self.controller.info("Sin estímulos", "No hay estímulos configurados para exportar.")
            return
        path, _f = QFileDialog.getSaveFileName(self, "Exportar configuración de estímulos",
                                               "estimulos.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"stim_videos": proj.stim_videos()}, fh, indent=2, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            self.controller.warn("No se pudo exportar", str(exc)); return
        self.status.setText(f"Estímulos exportados a {os.path.basename(path)}.")

    def _import_stim(self) -> None:
        if not self.controller._require_project():
            return
        path, _f = QFileDialog.getOpenFileName(self, "Importar configuración de estímulos",
                                               "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            self.controller.warn("No se pudo importar", str(exc)); return
        configs = data.get("stim_videos", data) if isinstance(data, dict) else data
        configs = [c for c in (configs or []) if isinstance(c, dict)]
        if not configs:
            self.controller.warn("Nada que importar",
                                 "El archivo no contiene configuraciones de estímulo.")
            return

        # Índice de los existentes para detectar repetidos (por id, o por etiqueta+archivo).
        existing = self.controller.project.stim_videos()
        by_id = {c["id"]: c for c in existing if c.get("id")}
        by_key = {}
        for c in existing:
            by_key.setdefault(_stim_key(c), c)

        # Clasifica ANTES de pedir ubicar videos, para no molestar con los que se ignoren.
        pairs = [(cfg, by_id.get(cfg.get("id")) or by_key.get(_stim_key(cfg)))
                 for cfg in configs]
        dups = [p for p in pairs if p[1] is not None]

        overwrite = True
        if dups:
            overwrite = self._ask_stim_overwrite([d for _c, d in dups])
            if overwrite is None:
                return                                     # cancelar

        search_dir, added, updated, skipped = None, 0, 0, 0
        gave_up = False                                # el usuario canceló la búsqueda
        missing: list[str] = []
        for cfg, dup in pairs:
            if dup is not None and not overwrite:
                skipped += 1
                continue
            vp = cfg.get("path", "")
            # Busca la ruta original, luego la carpeta indicada y luego data/videos.
            resolved = stim_core.relocate_video(vp, search_dir)
            if resolved is None and not gave_up:   # no aparece: pregunta la ubicación
                folder = QFileDialog.getExistingDirectory(
                    self, f"¿Dónde están los videos? (falta «{os.path.basename(vp)}»)")
                if folder:
                    # Se recuerda para los demás; si alguno vive en otra carpeta, se
                    # vuelve a preguntar por ese.
                    search_dir = folder
                    resolved = stim_core.relocate_video(vp, search_dir)
                else:
                    gave_up = True             # canceló: no se insiste con los demás
            if resolved is None:
                missing.append(os.path.basename(vp) or str(cfg.get("label", "?")))
            new_cfg = dict(cfg)
            new_cfg["path"] = resolved or vp               # deja la original si no se ubicó
            if dup is not None:                            # sobrescribe el existente (mismo id)
                new_cfg["id"] = dup.get("id")
                updated += 1
            else:
                new_cfg["id"] = None                       # id nuevo (no pisa los existentes)
                added += 1
            self.controller.project.save_stim_video(new_cfg)
        self.controller.request_autosave()
        self.refresh_stim()
        parts = [f"{added} nuevo(s)"]
        if updated:
            parts.append(f"{updated} sobrescrito(s)")
        if skipped:
            parts.append(f"{skipped} ignorado(s)")
        if missing:
            parts.append(f"{len(missing)} sin video")
        self.status.setText("Estímulos importados: " + ", ".join(parts) + ".")
        if missing:
            # Antes se guardaba la ruta inexistente sin avisar y el estímulo quedaba
            # inservible sin que se notara.
            self.controller.warn(
                "Videos no encontrados",
                "La configuración se importó (marcas y segmentos), pero NO se "
                "localizaron estos videos:\n\n• " + "\n• ".join(missing[:12])
                + ("\n…" if len(missing) > 12 else "")
                + "\n\nNo podrás reproducir esos estímulos hasta que el video esté "
                  "disponible. Colócalos en «data/videos» (se buscan ahí solos) o "
                  "vuelve a configurar el estímulo eligiendo el archivo.")

    def _ask_stim_overwrite(self, dups: list) -> bool | None:
        """Pregunta qué hacer con estímulos repetidos al importar.

        Devuelve ``True`` (sobrescribir), ``False`` (ignorar los repetidos) o
        ``None`` (cancelar toda la importación)."""
        names = ", ".join(sorted({str(d.get("label", "")) or "(sin nombre)" for d in dups}))
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Estímulos repetidos")
        box.setText(f"Ya existen {len(dups)} configuración(es) igual(es): {names}.\n"
                    "¿Qué quieres hacer con las repetidas?")
        b_over = box.addButton("Sobrescribir", QMessageBox.ButtonRole.AcceptRole)
        b_skip = box.addButton("Ignorar", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked is b_over:
            return True
        if clicked is b_skip:
            return False
        return None                                        # cancelar

    def _stim_config(self, vid_id):
        proj = self.controller.project
        if proj is None:
            return None
        return next((c for c in proj.stim_videos() if c.get("id") == vid_id), None)

    def _remove_stim(self) -> None:
        item = self.stim_list.currentItem()
        if item is None or self.controller.project is None:
            return
        self.controller.project.remove_stim_video(item.data(Qt.ItemDataRole.UserRole))
        self.controller.request_autosave()
        self.refresh_stim()

    def _play_stim(self) -> None:
        item = self.stim_list.currentItem()
        if item is None:
            self.controller.info("Sin estímulo", "Selecciona un estímulo configurado.")
            return
        cfg = self._stim_config(item.data(Qt.ItemDataRole.UserRole))
        if cfg is None:
            return
        if not self.stim_is_ready():
            self.controller.info(
                "Sin señal en vivo",
                "Conecta una fuente en «Tiempo real» y espera muestras antes de "
                "reproducir el estímulo (la grabación necesita señal).")
            return
        self._refresh_monitors()                     # por si (des)conectaron monitores
        default_name = f"{cfg.get('label', 'estimulo')}_{time.strftime('%H%M%S')}"
        name, ok = QInputDialog.getText(self, "Nombre de la grabación",
                                        "Nombre:", text=default_name)
        if not ok:
            return
        self._stim_session = StimSession(self.controller, cfg, name.strip() or default_name,
                                         screen=self._selected_screen(),
                                         on_done=self._on_stim_done)
        self._stim_session.start()

    def _on_stim_done(self, ok: bool) -> None:
        self._stim_session = None
        self.refresh_stim()
        self.status.setText("Estímulo reproducido y grabación guardada." if ok
                            else "Estímulo cancelado.")

    # --- API usada por StimSession ----------------------------------------
    def stim_is_ready(self) -> bool:
        return (self.source is not None and self._configured and self.recorder is None)

    def stim_start(self, name: str) -> bool:
        if self.recorder is not None:
            return False
        self.name_edit.setText(name)
        self._start_recording()
        return self.recorder is not None

    def stim_samples(self) -> int:
        return self.recorder.n_samples if self.recorder is not None else 0

    def stim_marker(self, label: str) -> None:
        if self.recorder is not None:
            self.recorder.add_marker(str(label))

    def stim_finish(self, segments) -> None:
        """Termina la grabación colocando ``segments`` (lista de ``(inicio, fin,
        etiqueta)`` en MUESTRAS) — los segmentos EXACTOS del estímulo."""
        if self.recorder is None:
            return
        self._rec_segments = [(int(s), int(e), str(lbl)) for (s, e, lbl) in segments]
        self._seg_active = False
        self._stop_recording()

    def _insert_marker(self) -> None:
        if self.recorder is None:
            self.controller.info("Sin grabación", "Los marcadores se guardan en la grabación. "
                                 "Inicia una grabación primero.")
            return
        label = self.marker_edit.text().strip() or "marca"
        self.recorder.add_marker(label)
        self.status.setText(f"Marcador insertado: «{label}»")

    def _add_timed_marker(self) -> None:
        """Crea un segmento de DURACIÓN FIJA (N s) a partir del instante actual."""
        if self.recorder is None:
            self.controller.info("Sin grabación", "Las marcas se guardan durante la "
                                 "grabación. Inicia una grabación primero.")
            return
        secs = self.duration_spin.value()
        fs = self.source.sample_rate if self.source else 128.0
        start = self.recorder.n_samples
        stop = start + max(1, int(round(secs * fs)))     # se completará al grabar N s
        label = self.marker_edit.text().strip() or "marca"
        self._rec_segments.append((start, stop, label))
        self._flush_sidecar()                # guarda la marca en el lateral al instante
        self.status.setText(
            f"Marca «{label}» de {secs}s añadida (se completa al grabar {secs}s)  "
            f"· {len(self._rec_segments)} segmentos.")

    def _toggle_segment(self) -> None:
        """1er clic: marca el inicio del segmento; 2º clic: marca el fin y lo crea."""
        if self.recorder is None:
            self.controller.info("Sin grabación", "Los segmentos se marcan durante la "
                                 "grabación. Inicia una grabación primero.")
            return
        if not self._seg_active:                          # marcar INICIO
            self._seg_active = True
            self._seg_start = self.recorder.n_samples
            self._seg_label = self.marker_edit.text().strip() or "segmento"
            self.segment_btn.setText(f"⏹ Terminar segmento «{self._seg_label}»")
            self.status.setText(f"Segmento «{self._seg_label}» iniciado… (F4 para terminar)")
        else:                                             # marcar FIN
            start, stop = self._seg_start, self.recorder.n_samples
            self._seg_active = False
            self.segment_btn.setText("▶ Segmento · F4")
            if stop > start:
                self._rec_segments.append((start, stop, self._seg_label))
                self._flush_sidecar()         # guarda el segmento en el lateral
                fs = self.source.sample_rate if self.source else 128.0
                dur = (stop - start) / max(1.0, fs)
                self.status.setText(
                    f"Segmento «{self._seg_label}» [{start}–{stop}] · {dur:.1f}s  "
                    f"({len(self._rec_segments)} en esta grabación)")
            else:
                self.status.setText("Segmento vacío (ignorado).")

    # --- Estados de los botones ------------------------------------------
    def _update_states(self) -> None:
        connected = self.source is not None
        recording = self.recorder is not None
        self.connect_btn.setText("Desconectar" if connected else "Conectar")
        self.source_combo.setEnabled(not connected)
        self.params.setEnabled(not connected)
        self.record_btn.setEnabled(connected)
        self.marker_btn.setEnabled(connected)
        self.segment_btn.setEnabled(connected)
        self.timed_btn.setEnabled(connected)
        # Pausar/descartar solo tienen sentido mientras se graba.
        self.pause_btn.setEnabled(recording)
        self.discard_btn.setEnabled(recording)

    def shutdown(self) -> None:
        """Cierre limpio al salir de la aplicación."""
        self._timer.stop()
        if self.recorder is not None:
            if self.source is not None:
                self.source.set_tap(None)
            self._flush_sidecar()        # deja las marcas en disco (recuperables al reabrir)
            self.recorder.close()        # flush + fsync: todo lo grabado queda en disco
            self.recorder = None
        if self.source is not None:
            self.source.stop()
            self.source = None

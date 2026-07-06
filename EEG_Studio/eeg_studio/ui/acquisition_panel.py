"""Panel de adquisición en tiempo real (opcional).

Selecciona una fuente (Simulado / OpenViBE-LSL / CyKit-TCP), la conecta, muestra
la señal en vivo y permite grabar a un CSV local del proyecto e insertar
marcadores. La interfaz funciona sin este panel: la captura es opcional.
"""
from __future__ import annotations

import os
import re
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..acquisition import (
    EmotivDongleSource,
    LSLSource,
    SimulatedSource,
    TCPSource,
    emotiv_deps_available,
    pylsl_available,
    quality,
)
from ..acquisition.emotiv import quick_diagnose
from ..acquisition.recorder import CSVRecorder
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


class AcquisitionPanel(QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self.source = None
        self.recorder: CSVRecorder | None = None
        self._configured = False
        self._n_samples = 0
        self._t_start = 0.0
        # Segmentos marcados durante la grabación (start, stop, etiqueta).
        self._rec_segments: list[tuple[int, int, str]] = []
        self._seg_active = False
        self._seg_start = 0
        self._seg_label = ""
        self._roll: np.ndarray | None = None    # buffer circular para inferencia
        self._roll_filled = 0
        self._quality_tick = 0                   # para refrescar la calidad cada N ticks

        self._timer = QTimer(self)
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
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        layout.addWidget(self.source_combo)

        # Parámetros por fuente (apilados, en el mismo orden que el combo).
        self.params = QStackedWidget()
        self.params.addWidget(self._sim_params())
        self.params.addWidget(self._lsl_params())
        self.params.addWidget(self._emotiv_params())
        self.params.addWidget(self._tcp_params())
        layout.addWidget(self.params)

        conn = QHBoxLayout()
        self.connect_btn = QPushButton("Conectar")
        self.connect_btn.clicked.connect(self.toggle_connection)
        conn.addWidget(self.connect_btn)
        layout.addLayout(conn)

        # Grabación.
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

        self.record_btn = QPushButton("Iniciar grabación")
        self.record_btn.clicked.connect(self.toggle_recording)
        rec_layout.addWidget(self.record_btn)

        marker_row = QHBoxLayout()
        self.marker_edit = QLineEdit()
        self.marker_edit.setPlaceholderText("Etiqueta (p.ej. ojos_cerrados / mano_izq)")
        self.marker_btn = QPushButton("Marca (instante)")
        self.marker_btn.setToolTip("Marca un instante puntual en la grabación (F3).")
        self.marker_btn.clicked.connect(self._insert_marker)
        marker_row.addWidget(self.marker_edit, 1)
        marker_row.addWidget(self.marker_btn)
        rec_layout.addLayout(marker_row)

        # Segmento etiquetado (inicio/fin): 1º clic marca el inicio, 2º el fin.
        self.segment_btn = QPushButton("▶ Marcar inicio de segmento")
        self.segment_btn.setToolTip(
            "Marca un SEGMENTO (no un instante): el 1er clic marca el inicio y el 2º el "
            "fin, con la etiqueta de arriba. Se crea como segmento de la clase al añadir "
            "la grabación. Atajo: F4.")
        self.segment_btn.clicked.connect(self._toggle_segment)
        rec_layout.addWidget(self.segment_btn)

        # Marca de DURACIÓN FIJA: crea un segmento de N segundos desde ahora.
        timed_row = QHBoxLayout()
        timed_row.addWidget(QLabel("Duración:"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(1, 600)
        self.duration_spin.setValue(5)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setToolTip("Duración de la marca de duración fija.")
        timed_row.addWidget(self.duration_spin)
        self.timed_btn = QPushButton("Marca de duración fija")
        self.timed_btn.setToolTip(
            "Crea un segmento de la duración indicada a partir de AHORA (p. ej. 5 s "
            "de una clase), con la etiqueta de arriba. Atajo: F5.")
        self.timed_btn.clicked.connect(self._add_timed_marker)
        timed_row.addWidget(self.timed_btn, 1)
        rec_layout.addLayout(timed_row)

        mk_hint = QLabel("• Marca = un instante (para «Segmentos desde marcadores»).\n"
                         "• Segmento = un tramo con inicio y fin (se etiqueta como clase).\n"
                         "• Marca de duración fija = un segmento de N s desde ahora.\n"
                         "Atajos: F3 instante · F4 inicia/termina segmento · F5 duración fija.")
        mk_hint.setWordWrap(True)
        mk_hint.setStyleSheet("color: #8a929b; font-size: 11px;")
        rec_layout.addWidget(mk_hint)
        layout.addWidget(rec_box)

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

        self.status = QLabel("Desconectado.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        # Indicador de calidad/ruido de la señal recibida (verde/ámbar/rojo).
        self.quality_label = QLabel("")
        self.quality_label.setWordWrap(True)
        layout.addWidget(self.quality_label)
        layout.addStretch(1)

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
        self._n_samples = 0
        self._t_start = time.perf_counter()
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
        self.quality_label.setText("")
        self.status.setText("Desconectado.")
        self._update_states()

    # --- Bucle de adquisición (hilo principal vía QTimer) -----------------
    def _tick(self) -> None:
        if self.source is None:
            return
        if self.source.error:
            msg = self.source.error
            self._disconnect()
            self.controller.warn("Error de adquisición", msg)
            return
        chunk = self.source.read()
        if chunk is None:
            return
        if not self._configured:
            self.controller.live_view.configure(
                self.source.channel_names, self.source.sample_rate, LIVE_WINDOW_SECONDS
            )
            self._roll = np.zeros((self.source.n_channels, ONLINE_BUFFER_SAMPLES))
            self._roll_filled = 0
            self._configured = True
        self.controller.live_view.append(chunk)
        self._push_buffer(chunk)
        if self.recorder is not None:
            self.recorder.write(chunk)
        self._n_samples += chunk.shape[1]
        self._update_status()
        self._quality_tick = (self._quality_tick + 1) % 8
        if self._quality_tick == 0:              # ~4 Hz: evita parpadeo
            self._update_quality()

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
        # Reinicia los segmentos de esta grabación.
        self._rec_segments = []
        self._seg_active = False
        self.segment_btn.setText("▶ Marcar inicio de segmento")
        self.record_btn.setText("Detener grabación")
        self.status.setText(f"Grabando «{self._rec_alias}» en {os.path.basename(path)}…")

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
        # Cierra un segmento que quedara abierto (fin = última muestra grabada).
        if self._seg_active:
            stop = self.recorder.n_samples
            if stop > self._seg_start:
                self._rec_segments.append((self._seg_start, stop, self._seg_label))
            self._seg_active = False
            self.segment_btn.setText("▶ Marcar inicio de segmento")
        # Recorta segmentos cuyo fin caiga más allá de lo grabado (marcas de
        # duración fija que no llegaron a completarse); descarta los vacíos.
        final = self.recorder.n_samples
        segments = [(s, min(e, final), lbl) for (s, e, lbl) in self._rec_segments
                    if s < final and min(e, final) - s >= 1]
        self.recorder.close()
        path = self._rec_path
        alias = getattr(self, "_rec_alias", None)
        self.recorder = None
        self.record_btn.setText("Iniciar grabación")
        # Ofrecer añadir la grabación como fuente del proyecto (con nombre y segmentos).
        if self.controller.ask_add_recording(path):
            self.controller.add_recording_as_source(path, segments, alias)

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
            self.segment_btn.setText("▶ Marcar inicio de segmento")
            if stop > start:
                self._rec_segments.append((start, stop, self._seg_label))
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
        self.connect_btn.setText("Desconectar" if connected else "Conectar")
        self.source_combo.setEnabled(not connected)
        self.params.setEnabled(not connected)
        self.record_btn.setEnabled(connected)
        self.marker_btn.setEnabled(connected)
        self.segment_btn.setEnabled(connected)
        self.timed_btn.setEnabled(connected)

    def shutdown(self) -> None:
        """Cierre limpio al salir de la aplicación."""
        self._timer.stop()
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
        if self.source is not None:
            self.source.stop()
            self.source = None

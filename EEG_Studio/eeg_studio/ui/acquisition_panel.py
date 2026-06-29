"""Panel de adquisición en tiempo real (opcional).

Selecciona una fuente (Simulado / OpenViBE-LSL / CyKit-TCP), la conecta, muestra
la señal en vivo y permite grabar a un CSV local del proyecto e insertar
marcadores. La interfaz funciona sin este panel: la captura es opcional.
"""
from __future__ import annotations

import os
import time

import numpy as np
from PyQt6.QtCore import QTimer
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
)
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
        self._roll: np.ndarray | None = None    # buffer circular para inferencia
        self._roll_filled = 0

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
        self.record_btn = QPushButton("Iniciar grabación")
        self.record_btn.clicked.connect(self.toggle_recording)
        rec_layout.addWidget(self.record_btn)

        marker_row = QHBoxLayout()
        self.marker_edit = QLineEdit()
        self.marker_edit.setPlaceholderText("Etiqueta del marcador (p.ej. ojos_cerrados)")
        self.marker_btn = QPushButton("Marcar")
        self.marker_btn.clicked.connect(self._insert_marker)
        marker_row.addWidget(self.marker_edit, 1)
        marker_row.addWidget(self.marker_btn)
        rec_layout.addLayout(marker_row)
        mk_hint = QLabel("Cada marcador se guarda en la grabación y puede convertirse "
                         "en una clase: Dataset → «Segmentos desde marcadores».")
        mk_hint.setWordWrap(True)
        mk_hint.setStyleSheet("color: #8a929b; font-size: 11px;")
        rec_layout.addWidget(mk_hint)
        layout.addWidget(rec_box)

        self.status = QLabel("Desconectado.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
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
        return w

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
        fname = time.strftime("rec_%Y%m%d_%H%M%S.csv")
        path = os.path.join(rec_dir, fname)
        self.recorder = CSVRecorder(path, self.source.n_channels, self.source.sample_rate)
        self._rec_path = path
        self.record_btn.setText("Detener grabación")
        self.status.setText(f"Grabando en {fname}…")

    def _stop_recording(self) -> None:
        if self.recorder is None:
            return
        self.recorder.close()
        path = self._rec_path
        self.recorder = None
        self.record_btn.setText("Iniciar grabación")
        # Ofrecer añadir la grabación como fuente del proyecto.
        if self.controller.ask_add_recording(path):
            self.controller.add_recording_as_source(path)

    def _insert_marker(self) -> None:
        if self.recorder is None:
            self.controller.info("Sin grabación", "Los marcadores se guardan en la grabación. "
                                 "Inicia una grabación primero.")
            return
        label = self.marker_edit.text().strip() or "marca"
        self.recorder.add_marker(label)
        self.status.setText(f"Marcador insertado: «{label}»")

    # --- Estados de los botones ------------------------------------------
    def _update_states(self) -> None:
        connected = self.source is not None
        self.connect_btn.setText("Desconectar" if connected else "Conectar")
        self.source_combo.setEnabled(not connected)
        self.params.setEnabled(not connected)
        self.record_btn.setEnabled(connected)
        self.marker_btn.setEnabled(connected)

    def shutdown(self) -> None:
        """Cierre limpio al salir de la aplicación."""
        self._timer.stop()
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
        if self.source is not None:
            self.source.stop()
            self.source = None

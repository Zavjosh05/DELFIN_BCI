"""Modo de control en tiempo real.

Con un modelo entrenado y una fuente en vivo conectada (pestaña *Tiempo real*),
clasifica ventanas de la señal entrante —tratadas por el mismo preprocesamiento—
y envía la clase detectada a un controlador externo (brazo robótico, carrito…)
por UDP, puerto serie o registro.
"""
from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QTimer
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

from ..config import (
    ONLINE_INTERVAL_MS,
    ONLINE_SERIAL_BAUD,
    ONLINE_SMOOTH_K,
    ONLINE_UDP_HOST,
    ONLINE_UDP_PORT,
    ONLINE_WINDOW_SAMPLES,
)
from ..inference import PredictionSmoother, classify_window, make_sink, serial_available


class ControlPanel(QWidget):
    def __init__(self, controller) -> None:
        super().__init__()
        self.controller = controller
        self.sink = None
        self.smoother = PredictionSmoother(ONLINE_SMOOTH_K)
        self._cmd_edits: dict[str, QLineEdit] = {}
        self._classes: list[str] = []
        self._n_commands = 0
        self._run_model = None     # modelo fijado al iniciar el control

        self._timer = QTimer(self)
        self._timer.setInterval(ONLINE_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)

        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        # Modelo a usar (si el proyecto tiene varios entrenados).
        mform = QFormLayout()
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        mform.addRow("Modelo:", self.model_combo)
        layout.addLayout(mform)

        # Parámetros de inferencia.
        cfg = QGroupBox("Inferencia")
        form = QFormLayout(cfg)
        self.window = QSpinBox()
        self.window.setRange(16, 8192)
        self.window.setSingleStep(32)
        self.window.setValue(ONLINE_WINDOW_SAMPLES)
        self.window.setToolTip("Muestras por ventana a clasificar.")
        self.interval = QSpinBox()
        self.interval.setRange(20, 5000)
        self.interval.setValue(ONLINE_INTERVAL_MS)
        self.interval.setToolTip("Cada cuántos milisegundos se clasifica.")
        self.smooth_k = QSpinBox()
        self.smooth_k.setRange(1, 20)
        self.smooth_k.setValue(ONLINE_SMOOTH_K)
        self.smooth_k.setToolTip("Predicciones iguales seguidas para confirmar una clase "
                                 "(evita que el controlador oscile).")
        form.addRow("Ventana (muestras):", self.window)
        form.addRow("Intervalo (ms):", self.interval)
        form.addRow("Confirmación (K):", self.smooth_k)
        layout.addWidget(cfg)

        # Mapa clase -> comando.
        self.map_box = QGroupBox("Comando por clase")
        self.map_form = QFormLayout(self.map_box)
        layout.addWidget(self.map_box)

        # Salida hacia el controlador.
        out = QGroupBox("Salida al controlador")
        out_layout = QVBoxLayout(out)
        self.sink_combo = QComboBox()
        self.sink_combo.addItem("Registro (solo mostrar)", "log")
        self.sink_combo.addItem("UDP (red)", "udp")
        self.sink_combo.addItem("Puerto serie (Arduino)", "serial")
        self.sink_combo.currentIndexChanged.connect(self._on_sink_changed)
        out_layout.addWidget(self.sink_combo)

        self.sink_params = QStackedWidget()
        self.sink_params.addWidget(self._log_params())
        self.sink_params.addWidget(self._udp_params())
        self.sink_params.addWidget(self._serial_params())
        out_layout.addWidget(self.sink_params)
        layout.addWidget(out)

        self.start_btn = QPushButton("Iniciar control")
        self.start_btn.clicked.connect(self.toggle)
        layout.addWidget(self.start_btn)

        # Salida en vivo.
        self.pred_label = QLabel("—")
        self.pred_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pred_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #9be7c4;")
        layout.addWidget(self.pred_label)
        self.detail_label = QLabel("Detenido.")
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)
        layout.addStretch(1)

    def _log_params(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        lay.addRow(QLabel("Muestra el comando en pantalla (para probar sin hardware)."))
        return w

    def _udp_params(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        self.udp_host = QLineEdit(ONLINE_UDP_HOST)
        self.udp_port = QSpinBox()
        self.udp_port.setRange(1, 65535)
        self.udp_port.setValue(ONLINE_UDP_PORT)
        lay.addRow("Host:", self.udp_host)
        lay.addRow("Puerto:", self.udp_port)
        lay.addRow(QLabel("Envía «comando\\n» por UDP al controlador que escuche ahí."))
        return w

    def _serial_params(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        self.serial_port = QLineEdit("COM3")
        self.serial_baud = QSpinBox()
        self.serial_baud.setRange(300, 1000000)
        self.serial_baud.setValue(ONLINE_SERIAL_BAUD)
        lay.addRow("Puerto:", self.serial_port)
        lay.addRow("Baudios:", self.serial_baud)
        hint = "Envía «comando\\n» por el puerto serie (p. ej. un Arduino)."
        if not serial_available():
            hint = "⚠ pyserial no está instalado (pip install pyserial)."
        note = QLabel(hint)
        note.setWordWrap(True)
        lay.addRow(note)
        return w

    def _on_sink_changed(self, idx: int) -> None:
        self.sink_params.setCurrentIndex(idx)

    # ------------------------------------------------------------------ #
    def _selected_model(self):
        return self.controller.models.get(self.model_combo.currentData())

    def refresh(self) -> None:
        """Repuebla la lista de modelos del proyecto y el mapa de comandos."""
        prev = self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for name in self.controller.models:
            self.model_combo.addItem(name, name)
        target = prev if prev in self.controller.models else self.controller.active_model_name
        if target:
            idx = self.model_combo.findData(target)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)
        self._on_model_changed()

    def _on_model_changed(self, *_):
        model = self._selected_model()
        if model is None:
            self.status.setText("Sin modelos. Entrena uno en «Clasificación».")
            self.start_btn.setEnabled(False)
            return
        self.status.setText(
            f"Modelo «{self.model_combo.currentData()}» · clases: {', '.join(model.classes)}")
        if model.classes != self._classes:
            self._build_class_rows(model.classes)
        if model.input_kind == "raw" and model.nn_config:
            self.window.setValue(int(model.nn_config.get("window_samples", self.window.value())))
        self.start_btn.setEnabled(True)

    def _build_class_rows(self, classes: list[str]) -> None:
        while self.map_form.rowCount():
            self.map_form.removeRow(0)
        self._cmd_edits.clear()
        for cls in classes:
            edit = QLineEdit(cls)   # por defecto el comando es el nombre de la clase
            self._cmd_edits[cls] = edit
            self.map_form.addRow(cls, edit)
        self._classes = list(classes)

    def _command_for(self, cls: str) -> str:
        edit = self._cmd_edits.get(cls)
        return (edit.text().strip() if edit else "") or cls

    # ------------------------------------------------------------------ #
    def toggle(self) -> None:
        if self._timer.isActive():
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        self._run_model = self._selected_model()
        if self._run_model is None:
            self.controller.info("Sin modelo", "Entrena o importa un modelo primero.")
            return
        if not self.controller.acq_panel.is_streaming():
            self.controller.info(
                "Sin señal en vivo",
                "Conecta una fuente en la pestaña «Tiempo real» antes de iniciar el control.")
            return
        try:
            self.sink = make_sink(self.sink_combo.currentData(), **self._sink_kwargs())
        except Exception as exc:  # noqa: BLE001
            self.controller.warn("Salida no disponible", str(exc))
            return
        self.smoother = PredictionSmoother(self.smooth_k.value())
        self._n_commands = 0
        self._timer.setInterval(self.interval.value())
        self._timer.start()
        self.start_btn.setText("Detener control")
        self.detail_label.setText("Control en marcha…")
        self._set_inputs_enabled(False)

    def _stop(self) -> None:
        self._timer.stop()
        if self.sink is not None:
            self.sink.close()
            self.sink = None
        self.start_btn.setText("Iniciar control")
        self.detail_label.setText("Detenido.")
        self._set_inputs_enabled(True)

    def _sink_kwargs(self) -> dict:
        kind = self.sink_combo.currentData()
        if kind == "udp":
            return {"host": self.udp_host.text().strip() or ONLINE_UDP_HOST,
                    "port": self.udp_port.value()}
        if kind == "serial":
            return {"port": self.serial_port.text().strip(), "baud": self.serial_baud.value()}
        return {}

    def _set_inputs_enabled(self, enabled: bool) -> None:
        for w in (self.window, self.interval, self.smooth_k, self.sink_combo, self.map_box):
            w.setEnabled(enabled)

    # --- Bucle de inferencia (hilo principal vía QTimer) ------------------
    def _tick(self) -> None:
        acq = self.controller.acq_panel
        if not acq.is_streaming():
            self.detail_label.setText("Se perdió la señal en vivo. Control detenido.")
            self._stop()
            return
        window = acq.latest_window(self.window.value())
        if window is None:
            return   # aún no hay suficientes muestras
        try:
            pred, conf = classify_window(
                self._run_model, self.controller.project, window, acq.stream_fs())
        except Exception as exc:  # noqa: BLE001
            self.controller.warn("Error de clasificación", str(exc))
            self._stop()
            return

        conf_txt = f"  ({conf * 100:.0f}%)" if conf is not None else ""
        self.pred_label.setText(f"{pred}{conf_txt}")

        confirmed = self.smoother.update(pred)
        if confirmed is not None and self.sink is not None:
            command = self._command_for(confirmed)
            try:
                self.sink.send(command)
            except Exception as exc:  # noqa: BLE001
                self.controller.warn("Error de salida", str(exc))
                self._stop()
                return
            self._n_commands += 1
            ts = time.strftime("%H:%M:%S")
            self.detail_label.setText(
                f"Estable: «{confirmed}» → comando «{command}» enviado  ·  {ts}\n"
                f"Comandos enviados: {self._n_commands}")

    def shutdown(self) -> None:
        self._stop()

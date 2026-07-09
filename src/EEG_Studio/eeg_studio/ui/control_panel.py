"""Modo de control en tiempo real.

Con un modelo entrenado y una fuente en vivo conectada (pestaña *Tiempo real*),
clasifica ventanas de la señal entrante —tratadas por el mismo preprocesamiento—
y envía la clase detectada a un controlador externo (brazo robótico, carrito…)
por UDP, puerto serie o registro.
"""
from __future__ import annotations

import time

import threading

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
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

from ..inference.arm import (
    ARM_DISABLED,
    DEFAULT_HOST as ARM_HOST,
    DEFAULT_PORT as ARM_PORT,
    DEFAULT_PULSE_MS as ARM_PULSE,
    ArmClient,
)
from ..inference.sim_arm import SimArmSink, SimulatedArm
from .sim_arm_view import SimArmView

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
        self._sim_arm = SimulatedArm()             # brazo simulado (perfil sin hardware)
        self._cmd_buttons: dict[str, QPushButton] = {}

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

        # Actuador a controlar (perfiles): brazo real MaxArm o brazo simulado.
        layout.addWidget(self._actuator_section())

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
        self.sink_combo.addItem("Brazo MaxArm (HTTP)", "arm")
        self.sink_combo.addItem("Brazo simulado", "sim")
        self.sink_combo.addItem("UDP (red)", "udp")
        self.sink_combo.addItem("Puerto serie (Arduino)", "serial")
        self.sink_combo.currentIndexChanged.connect(self._on_sink_changed)
        out_layout.addWidget(self.sink_combo)

        self.sink_params = QStackedWidget()
        self.sink_params.addWidget(self._log_params())
        self.sink_params.addWidget(self._arm_sink_params())
        self.sink_params.addWidget(self._sim_sink_params())
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

    def _arm_sink_params(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        note = QLabel("Envía cada clase detectada al brazo MaxArm por HTTP.\n"
                      "Usa la IP/puerto/pulso del perfil «Brazo MaxArm» arriba.")
        note.setWordWrap(True)
        lay.addRow(note)
        return w

    def _sim_sink_params(self) -> QWidget:
        w = QWidget()
        lay = QFormLayout(w)
        note = QLabel("Mueve el brazo SIMULADO con cada clase detectada («controlar "
                      "con la mente»). Al iniciar se muestra el perfil «Brazo simulado».")
        note.setWordWrap(True)
        lay.addRow(note)
        return w

    # --- Actuador: perfiles de control ------------------------------------
    def _actuator_section(self) -> QGroupBox:
        box = QGroupBox("Actuador — perfil de control")
        lay = QVBoxLayout(box)

        pform = QFormLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("Brazo MaxArm (real, HTTP)", "maxarm")
        self.profile_combo.addItem("Brazo simulado", "sim")
        self.profile_combo.setToolTip(
            "Qué actuador controlan los botones y el clasificador. Podrás añadir "
            "más perfiles (otros brazos, carritos…) en el futuro.")
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)
        pform.addRow("Perfil:", self.profile_combo)
        lay.addLayout(pform)

        # Configuración específica de cada perfil (apilada).
        self.profile_stack = QStackedWidget()
        self.profile_stack.addWidget(self._maxarm_page())   # 0
        self.profile_stack.addWidget(self._sim_page())      # 1
        lay.addWidget(self.profile_stack)

        self.arm_status = QLabel("Elige un perfil y prueba los comandos.")
        self.arm_status.setWordWrap(True)
        self.arm_status.setStyleSheet("color: #8a929b; font-size: 11px;")
        lay.addWidget(self.arm_status)

        # D-pad compartido de los 6 comandos: controla el perfil ACTIVO.
        grid = QGridLayout()
        grid.setSpacing(6)
        for text, cmd, r, c in (
            ("▲ Arriba", "arriba", 0, 1),
            ("◀ Izquierda", "izquierda", 1, 0),
            ("▶ Derecha", "derecha", 1, 2),
            ("▼ Abajo", "abajo", 2, 1),
            ("✊ Agarre", "agarre", 3, 0),
            ("✋ Soltar", "soltar", 3, 2),
        ):
            b = QPushButton(text)
            b.setMinimumHeight(34)
            b.clicked.connect(lambda _=False, cc=cmd: self._profile_do(cc))
            self._cmd_buttons[cmd] = b
            grid.addWidget(b, r, c)
        lay.addLayout(grid)
        self._update_dpad_enabled()
        return box

    def _maxarm_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0)
        conn = QFormLayout()
        self.arm_host = QLineEdit(ARM_HOST)
        self.arm_port = QSpinBox(); self.arm_port.setRange(1, 65535); self.arm_port.setValue(ARM_PORT)
        self.arm_pulse = QSpinBox()
        self.arm_pulse.setRange(50, 5000); self.arm_pulse.setValue(ARM_PULSE); self.arm_pulse.setSuffix(" ms")
        self.arm_pulse.setToolTip("Cuánto dura cada pulsación de movimiento (comando discreto).")
        conn.addRow("IP del brazo:", self.arm_host)
        conn.addRow("Puerto:", self.arm_port)
        conn.addRow("Pulso de movimiento:", self.arm_pulse)
        lay.addLayout(conn)
        row = QHBoxLayout()
        test_btn = QPushButton("Probar conexión")
        test_btn.clicked.connect(self._test_arm)
        home_btn = QPushButton("HOME")
        home_btn.clicked.connect(lambda: self._profile_do("home"))
        row.addWidget(test_btn); row.addWidget(home_btn)
        lay.addLayout(row)
        hint = QLabel("Conecta el PC a la red WiFi «MaxArm_IPN» (clave: maxarm2024). "
                      "Izquierda/Derecha no están disponibles (base del MaxArm sin servicio).")
        hint.setWordWrap(True); hint.setStyleSheet("color: #8a929b; font-size: 11px;")
        lay.addWidget(hint)
        return w

    def _sim_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0)
        self.sim_view = SimArmView(self._sim_arm)
        lay.addWidget(self.sim_view)
        row = QHBoxLayout()
        home_btn = QPushButton("HOME (posición inicial)")
        home_btn.clicked.connect(lambda: self._profile_do("home"))
        row.addWidget(home_btn); row.addStretch(1)
        lay.addLayout(row)
        hint = QLabel("Brazo 4DOF simulado (sin hardware): arriba/abajo mueven el hombro, "
                      "izquierda/derecha giran la base, agarre/soltar cierran/abren la pinza.")
        hint.setWordWrap(True); hint.setStyleSheet("color: #8a929b; font-size: 11px;")
        lay.addWidget(hint)
        return w

    def _current_profile(self) -> str:
        return self.profile_combo.currentData()

    def _on_profile_changed(self, *_) -> None:
        self.profile_stack.setCurrentIndex(self.profile_combo.currentIndex())
        self._update_dpad_enabled()
        self.arm_status.setText(f"Perfil activo: {self.profile_combo.currentText()}.")

    def _update_dpad_enabled(self) -> None:
        """En el MaxArm real, Izquierda/Derecha (base) están sin servicio; en el
        simulado todos los comandos funcionan."""
        maxarm = self._current_profile() == "maxarm"
        for cmd, b in self._cmd_buttons.items():
            off = maxarm and cmd in ARM_DISABLED
            b.setEnabled(not off)
            b.setToolTip("Base giratoria del MaxArm sin servicio." if off else "")

    def _profile_do(self, command: str) -> None:
        """Despacha un comando al actuador del perfil activo."""
        if self._current_profile() == "sim":
            self._sim_do(command)
        else:
            self._arm_do(command)

    def _sim_do(self, command: str) -> None:
        if command == "home":
            self._sim_arm.reset()
            self.arm_status.setText("Brazo simulado en HOME.")
        else:
            self._sim_arm.execute(command)
            self.arm_status.setText(f"Comando «{command}» aplicado al brazo simulado.")
        self.sim_view.refresh()

    def _arm_client(self) -> ArmClient:
        return ArmClient(self.arm_host.text().strip() or ARM_HOST, self.arm_port.value())

    @staticmethod
    def _fire(fn) -> None:
        """Ejecuta ``fn`` en un hilo daemon, tragándose errores (no bloquea la GUI)."""
        def run():
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=run, daemon=True).start()

    def _test_arm(self) -> None:
        host, port = self.arm_host.text().strip() or ARM_HOST, self.arm_port.value()
        client = ArmClient(host, port)
        self.arm_status.setText("Probando conexión…")
        # En un hilo del controlador para reflejar el resultado en la GUI.
        self.controller._spawn(
            client.ping,
            lambda ok: self.arm_status.setText(
                f"✓ Brazo conectado en {host}:{port}." if ok else
                f"✗ Sin respuesta de {host}:{port}. ¿El PC está en la red «MaxArm_IPN»?"))

    def _arm_do(self, command: str) -> None:
        client = self._arm_client()
        if command == "home":
            self._fire(client.reset)
            self.arm_status.setText("HOME enviado (posición inicial).")
        else:
            client.execute_async(command, self.arm_pulse.value())
            self.arm_status.setText(f"Comando «{command}» enviado al brazo.")

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
        kind = self.sink_combo.currentData()
        try:
            if kind == "sim":
                # Enlaza el clasificador al brazo simulado visible y activa su perfil.
                self.profile_combo.setCurrentIndex(self.profile_combo.findData("sim"))
                self.sink = SimArmSink(self._sim_arm, on_change=self.sim_view.refresh)
            else:
                self.sink = make_sink(kind, **self._sink_kwargs())
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
        if kind == "arm":
            return {"host": self.arm_host.text().strip() or ARM_HOST,
                    "port": self.arm_port.value(), "pulse_ms": self.arm_pulse.value()}
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

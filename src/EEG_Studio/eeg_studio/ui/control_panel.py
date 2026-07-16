"""Modo de control en tiempo real.

Con un modelo entrenado y una fuente en vivo conectada (pestaña *Tiempo real*),
clasifica ventanas de la señal entrante —tratadas por el mismo preprocesamiento—
y envía la clase detectada a un controlador externo (brazo robótico, carrito…)
por UDP, puerto serie o registro.
"""
from __future__ import annotations

import os
import time

import threading

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
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
from ..inference.sim_arm import SIM_ARM_COMMAND_NAMES, SimArmSink, SimulatedArm
from .arm_builder import ArmBuilderWidget
from .sim_arm_controls import SimArmControls
from .sim_arm_view import SimArmView

from ..core import stim as stim_core
from ..core.csv_loader import load_recording
from ..config import (
    ONLINE_HOLD_MS,
    ONLINE_INTERVAL_MS,
    ONLINE_MIN_CONFIDENCE,
    ONLINE_SERIAL_BAUD,
    ONLINE_SMOOTH_K,
    ONLINE_UDP_HOST,
    ONLINE_UDP_PORT,
    ONLINE_WINDOW_SAMPLES,
)
from ..inference import (
    PredictionSmoother,
    classify_recording,
    classify_window,
    make_sink,
    serial_available,
)


class ControlPanel(QWidget):
    # La clasificación corre en un hilo aparte; su resultado vuelve al hilo de la
    # GUI por esta señal (Qt la encola sola al cruzar de hilo). El payload es
    # ``(run_id, pred, conf, error)``.
    _classified = pyqtSignal(int, object, object, object)

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
        self._file_path: str | None = None         # grabación para el modo «un archivo → un movimiento»

        # --- Estado del bucle de inferencia ---
        self._run_id = 0           # sube en cada arranque: descarta resultados rezagados
        self._inflight = False     # ya hay una ventana clasificándose: no encimar otra
        self._dropped = 0          # ticks saltados por ir la clasificación más lenta que el timer
        self._hold_until = 0.0     # monotonic hasta el que la acción en curso se mantiene
        self._hold_command: str | None = None      # comando que se está sosteniendo
        self._hold_class: str | None = None
        self._last_ms = 0.0        # cuánto tardó la última clasificación

        self._timer = QTimer(self)
        self._timer.setInterval(ONLINE_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._classified.connect(self._on_classified)

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

        # Estabilidad del comando: que una clase confirmada dé pie a una acción
        # que dure lo suficiente para ser útil, en vez de cambiar cada 250 ms.
        stab = QGroupBox("Estabilidad del comando")
        sform = QFormLayout(stab)
        self.min_conf = QSpinBox()
        self.min_conf.setRange(0, 99)
        self.min_conf.setSuffix(" %")
        self.min_conf.setValue(int(round(ONLINE_MIN_CONFIDENCE * 100)))
        self.min_conf.setToolTip(
            "Las predicciones por debajo de esta confianza se ignoran (no cuentan "
            "para la confirmación). 0 = aceptar todas.\n"
            "Si el modelo no da probabilidades, este filtro no se aplica.")
        self.hold_ms = QSpinBox()
        self.hold_ms.setRange(0, 10000)
        self.hold_ms.setSingleStep(250)
        self.hold_ms.setSuffix(" ms")
        self.hold_ms.setValue(ONLINE_HOLD_MS)
        self.hold_ms.setToolTip(
            "Una vez confirmada una clase, la acción se mantiene este tiempo y no "
            "se atiende ninguna otra predicción: da margen a que el movimiento se "
            "complete.\n0 = comportamiento anterior (reaccionar a cada confirmación).")
        self.hold_repeat = QCheckBox("Repetir el comando mientras dura la acción")
        self.hold_repeat.setChecked(True)
        self.hold_repeat.setToolTip(
            "Reenvía el comando en cada intervalo durante la retención. Como cada "
            "envío es un pulso/incremento del actuador, repetirlo convierte una "
            "orden suelta en un movimiento sostenido.")
        sform.addRow("Confianza mínima:", self.min_conf)
        sform.addRow("Duración de la acción:", self.hold_ms)
        sform.addRow(self.hold_repeat)
        layout.addWidget(stab)

        # Mapa clase -> comando.
        self.map_box = QGroupBox("Comando por clase")
        self.map_form = QFormLayout(self.map_box)
        layout.addWidget(self.map_box)

        # Control «un archivo → un movimiento» (sin diadema).
        layout.addWidget(self._recorded_file_section())

        # Salida hacia el controlador (solo perfiles con actuador EXTERNO, p. ej.
        # el MaxArm real). El brazo simulado no necesita salida: se mueve directo.
        self.output_group = QGroupBox("Salida al controlador")
        out_layout = QVBoxLayout(self.output_group)
        self.sink_combo = QComboBox()
        self.sink_combo.addItem("Registro (solo mostrar)", "log")
        self.sink_combo.addItem("Brazo MaxArm (HTTP)", "arm")
        self.sink_combo.addItem("UDP (red)", "udp")
        self.sink_combo.addItem("Puerto serie (Arduino)", "serial")
        self.sink_combo.currentIndexChanged.connect(self._on_sink_changed)
        out_layout.addWidget(self.sink_combo)

        self.sink_params = QStackedWidget()
        self.sink_params.addWidget(self._log_params())
        self.sink_params.addWidget(self._arm_sink_params())
        self.sink_params.addWidget(self._udp_params())
        self.sink_params.addWidget(self._serial_params())
        out_layout.addWidget(self.sink_params)
        self.sim_output_note = QLabel(
            "El brazo simulado se controla directamente (sin salida externa).")
        self.sim_output_note.setWordWrap(True)
        self.sim_output_note.setStyleSheet("color: #8a929b; font-size: 11px;")
        self.sim_output_note.setVisible(False)
        out_layout.addWidget(self.sim_output_note)
        layout.addWidget(self.output_group)

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

    # --- Control desde un archivo grabado (un archivo -> un movimiento) -----
    def _recorded_file_section(self) -> QGroupBox:
        """Sección para clasificar una grabación previa y mover el actuador del
        perfil activo, sin necesidad de la diadema. Complementa la reproducción en
        vivo (fuente «Reproducir grabación» de la pestaña Tiempo real)."""
        box = QGroupBox("Controlar desde archivo grabado (sin diadema)")
        lay = QVBoxLayout(box)

        row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("Ningún archivo elegido (p. ej. Sujeto001_Abajo.csv)")
        pick = QPushButton("Elegir archivo…")
        pick.clicked.connect(self._choose_control_file)
        row.addWidget(self.file_edit, 1)
        row.addWidget(pick, 0)
        lay.addLayout(row)

        self.file_btn = QPushButton("Clasificar y mover")
        self.file_btn.setToolTip("Analiza la grabación con el modelo elegido y mueve el "
                                 "actuador del perfil activo según la clase mayoritaria.")
        self.file_btn.clicked.connect(self._classify_file)
        lay.addWidget(self.file_btn)

        self.file_result = QLabel("Elige un archivo y un modelo para clasificarlo.")
        self.file_result.setWordWrap(True)
        self.file_result.setStyleSheet("color: #9aa4ae; font-size: 11px;")
        lay.addWidget(self.file_result)
        return box

    def _choose_control_file(self) -> None:
        start = ""
        proj = getattr(self.controller, "project", None)
        if proj is not None:
            start = getattr(proj, "path", "") or ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Elegir grabación para clasificar", start,
            "Grabaciones EEG (*.csv *.csv.gz);;Todos los archivos (*)")
        if path:
            self._file_path = path
            self.file_edit.setText(path)

    def _classify_file(self) -> None:
        model = self._selected_model()
        if model is None:
            self.controller.info("Sin modelo", "Entrena o importa un modelo primero.")
            return
        if not self._file_path:
            self.controller.info("Sin archivo", "Elige una grabación (CSV) primero.")
            return
        path = self._file_path
        project = self.controller.project
        win = self.window.value()
        # Clase esperada del nombre del archivo (p. ej. «Sujeto001_Abajo» -> «abajo»).
        expected = stim_core.class_from_filename(os.path.basename(path))

        def job():
            rec = load_recording(path)
            gt = None
            if expected is not None:                 # verdad-terreno constante = clase del nombre
                gt = np.full(rec.n_samples, expected, dtype=object)
            summary = classify_recording(model, project, rec.data, rec.sample_rate,
                                         window=win, ground_truth=gt)
            return {"summary": summary, "channels": rec.n_channels,
                    "expected": expected, "path": path}

        self.file_btn.setEnabled(False)
        self.file_result.setText("Clasificando la grabación…")
        self.controller._spawn(job, self._on_file_classified, self._on_file_error)

    def _on_file_classified(self, res: dict) -> None:
        self.file_btn.setEnabled(True)
        s = res["summary"]
        label = s["label"]
        command = self._command_for(label)
        # Mueve el actuador del perfil activo (mismo despacho que el D-pad).
        self._profile_do(command)

        parts = [f"Archivo: {os.path.basename(res['path'])}",
                 f"predicho: «{label}» → comando «{command}»"]
        if res["expected"]:
            hit = "✓" if res["expected"] == label else "✗"
            parts.append(f"esperado (nombre): «{res['expected']}» {hit}")
        if s["confidence"] is not None:
            parts.append(f"confianza {s['confidence'] * 100:.0f}%")
        if s["accuracy"] is not None:
            parts.append(f"exactitud {s['accuracy'] * 100:.0f}% de {s['n_windows']} ventanas")
        text = "  ·  ".join(parts)
        # Aviso de compatibilidad: si el comando no es un movimiento del brazo.
        if command not in SIM_ARM_COMMAND_NAMES:
            text += ("\n⚠ «" + command + "» no corresponde a un movimiento del brazo; "
                     "ajusta el mapeo «Comando por clase» si quieres que se mueva.")
        self.file_result.setText(text)
        self.file_result.setStyleSheet("color: #c8d0d8; font-size: 11px;")

    def _on_file_error(self, msg: str) -> None:
        self.file_btn.setEnabled(True)
        self.file_result.setText("")
        self.controller.warn(
            "No se pudo clasificar el archivo",
            "La grabación no se pudo clasificar. Posibles causas: no es un CSV válido, "
            "o su nº de canales no coincide con el que se usó para entrenar el modelo.\n\n"
            f"{msg}")

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
        """Todo el control del brazo simulado en una sola vista: la simulación
        arriba (3D + 2D) y los sliders por articulación justo debajo, para verlo
        moverse mientras se controla. El constructor va en un diálogo aparte."""
        w = QWidget()
        lay = QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        # `control=self`: la pantalla completa del brazo ofrece el control en vivo
        # delegando en este panel (no duplica el clasificador).
        self.sim_view = SimArmView(self._sim_arm, on_change=self._sim_refresh,
                                   control=self)
        lay.addWidget(self.sim_view)

        self.sim_controls = SimArmControls(self._sim_arm, on_change=self._sim_refresh)
        sc_box = QGroupBox("Control por articulación")
        scl = QVBoxLayout(sc_box); scl.setContentsMargins(6, 6, 6, 6)
        scl.addWidget(self.sim_controls)
        lay.addWidget(sc_box)

        build_btn = QPushButton("Construir / elegir brazo…")
        build_btn.setToolTip("Elige un preset o construye el brazo (joints, ejes, longitudes).")
        build_btn.clicked.connect(self._open_builder)
        lay.addWidget(build_btn)

        hint = QLabel("Muévelo con el D-pad de abajo o con los sliders; se ve arriba en 3D y 2D. "
                      "Arriba/abajo = hombro, izquierda/derecha = base, agarre/soltar = pinza.")
        hint.setWordWrap(True); hint.setStyleSheet("color: #8a929b; font-size: 11px;")
        lay.addWidget(hint)
        return w

    def _open_builder(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Construir / elegir brazo simulado")
        dlg.resize(720, 500)
        v = QVBoxLayout(dlg)
        builder = ArmBuilderWidget()

        def _apply(spec):
            self._on_arm_built(spec)
            dlg.accept()
        builder.applied.connect(_apply)
        v.addWidget(builder)
        dlg.exec()

    def _sim_refresh(self) -> None:
        """Redibuja la vista del brazo simulado y sincroniza los sliders."""
        self.sim_view.refresh()
        self.sim_controls.sync()

    def _on_arm_built(self, spec) -> None:
        """Reconstruye el brazo simulado con la spec del constructor."""
        self._sim_arm.apply_spec(spec)
        self.sim_view.set_arm(self._sim_arm)
        self.sim_controls.rebuild()
        self._sim_refresh()
        self.arm_status.setText(f"Brazo simulado reconstruido: «{spec.name}».")

    def _current_profile(self) -> str:
        return self.profile_combo.currentData()

    def _on_profile_changed(self, *_) -> None:
        self.profile_stack.setCurrentIndex(self.profile_combo.currentIndex())
        self._update_dpad_enabled()
        # El brazo simulado no usa salida externa: se oculta el selector de salida.
        sim = self._current_profile() == "sim"
        self.sink_combo.setVisible(not sim)
        self.sink_params.setVisible(not sim)
        self.sim_output_note.setVisible(sim)
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
        self._sim_refresh()

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
            if hasattr(self, "file_btn"):
                self.file_btn.setEnabled(False)
            return
        self.status.setText(
            f"Modelo «{self.model_combo.currentData()}» · clases: {', '.join(model.classes)}")
        if model.classes != self._classes:
            self._build_class_rows(model.classes)
        if model.input_kind == "raw" and model.nn_config:
            self.window.setValue(int(model.nn_config.get("window_samples", self.window.value())))
        self.start_btn.setEnabled(True)
        if hasattr(self, "file_btn"):
            self.file_btn.setEnabled(True)

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
            if self._current_profile() == "sim":
                # El perfil simulado NO usa salida externa: el clasificador mueve
                # directamente el brazo simulado visible.
                self.sink = SimArmSink(self._sim_arm, on_change=self._sim_refresh)
            else:
                self.sink = make_sink(self.sink_combo.currentData(), **self._sink_kwargs())
        except Exception as exc:  # noqa: BLE001
            self.controller.warn("Salida no disponible", str(exc))
            return
        self.smoother = PredictionSmoother(self.smooth_k.value())
        self._n_commands = 0
        self._run_id += 1          # invalida cualquier resultado del arranque anterior
        self._inflight = False
        self._dropped = 0
        self._last_ms = 0.0
        self._end_hold()
        self._timer.setInterval(self.interval.value())
        self._timer.start()
        self.start_btn.setText("Detener control")
        self.detail_label.setText("Control en marcha…")
        self._set_inputs_enabled(False)

    def _stop(self) -> None:
        self._timer.stop()
        # Sube el id ANTES de soltar la salida: si hay una ventana clasificándose,
        # su resultado llegará con un id viejo y se descartará en vez de intentar
        # enviar por un sink ya cerrado.
        self._run_id += 1
        self._inflight = False
        self._end_hold()
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
        # `min_conf`, `hold_ms` y `hold_repeat` NO se bloquean a propósito: son los
        # ajustes que uno quiere afinar viendo el control funcionar, y ninguno se
        # lee al arrancar (se consultan en cada tick), así que cambiarlos en marcha
        # es seguro y surte efecto al momento.
        for w in (self.window, self.interval, self.smooth_k, self.sink_combo,
                  self.map_box, self.profile_combo, self.file_btn):
            w.setEnabled(enabled)

    # --- Bucle de inferencia ----------------------------------------------
    # El QTimer solo TOMA la ventana (barato, hilo de la GUI) y despacha; el
    # trabajo pesado —el pipeline del proyecto, que con ICA ronda los 100 ms por
    # ventana— corre en un hilo aparte. Antes se hacía aquí mismo: cada 250 ms la
    # interfaz se quedaba muerta ~100-170 ms, y en ese hilo viven también la
    # adquisición y el visor, así que todo se trababa.
    def _tick(self) -> None:
        acq = self.controller.acq_panel
        if not acq.is_streaming():
            self.detail_label.setText("Se perdió la señal en vivo. Control detenido.")
            self._stop()
            return

        # Acción en curso: se mantiene (y opcionalmente se repite) sin atender
        # predicciones nuevas. Es lo que da tiempo a que el movimiento signifique algo.
        if self._hold_command is not None:
            if time.monotonic() < self._hold_until:
                self._hold_tick()
                return
            self._end_hold()        # la retención expiró: volver a escuchar

        if self._inflight:
            self._dropped += 1      # la clasificación va más lenta que el timer: se salta
            return
        window = acq.latest_window(self.window.value())
        if window is None:
            return   # aún no hay suficientes muestras

        # `latest_window` devuelve una copia, así que el hilo no comparte el buffer.
        self._inflight = True
        run_id, model, project, fs = self._run_id, self._run_model, self.controller.project, acq.stream_fs()

        def job():
            t0 = time.perf_counter()
            try:
                pred, conf = classify_window(model, project, window, fs)
                err = None
            except Exception as exc:  # noqa: BLE001
                pred, conf, err = None, None, str(exc)
            self._last_ms = (time.perf_counter() - t0) * 1000
            self._classified.emit(run_id, pred, conf, err)

        threading.Thread(target=job, daemon=True).start()

    def _on_classified(self, run_id: int, pred, conf, err) -> None:
        """Resultado de una ventana, ya de vuelta en el hilo de la GUI."""
        if run_id != self._run_id:
            return              # de una sesión de control anterior: ignorar
        self._inflight = False
        if err is not None:
            # DETENER primero y avisar después: `warn` es modal y un modal levanta un
            # bucle de eventos anidado, así que con el timer aún vivo seguirían
            # entrando ventanas -> más errores -> más diálogos encima del primero.
            self._stop()
            self.controller.warn("Error de clasificación", err)
            return
        if not self._timer.isActive():
            return

        conf_txt = f"  ({conf * 100:.0f}%)" if conf is not None else ""
        self.pred_label.setText(f"{pred}{conf_txt}")

        # Filtro de confianza: una predicción dudosa no confirma nada. Se corta la
        # racha para que el ruido no sume hacia la K.
        min_conf = self.min_conf.value() / 100.0
        if conf is not None and min_conf > 0 and conf < min_conf:
            self.smoother.reset()
            self._set_detail(f"Predicción «{pred}» descartada: confianza "
                             f"{conf * 100:.0f}% < {self.min_conf.value()}%.")
            return

        confirmed = self.smoother.update(pred)
        if confirmed is None or self.sink is None:
            return
        command = self._command_for(confirmed)
        if not self._send(command):
            return
        self._n_commands += 1
        hold = self.hold_ms.value()
        if hold > 0:
            self._hold_class, self._hold_command = confirmed, command
            self._hold_until = time.monotonic() + hold / 1000.0
        self._set_detail(f"Estable: «{confirmed}» → comando «{command}»  ·  "
                         f"{time.strftime('%H:%M:%S')}")

    # --- Retención de la acción -------------------------------------------
    def _holding(self) -> bool:
        return self._hold_command is not None and time.monotonic() < self._hold_until

    def _hold_tick(self) -> None:
        """Sostiene la acción confirmada: la repite y muestra lo que queda."""
        left = max(0.0, self._hold_until - time.monotonic())
        if self.hold_repeat.isChecked() and self.sink is not None:
            if not self._send(self._hold_command):
                return
            self._n_commands += 1
        self._set_detail(f"Acción «{self._hold_command}» en curso  ·  quedan {left:.1f} s"
                         + ("  (repitiendo)" if self.hold_repeat.isChecked() else ""))

    def _end_hold(self) -> None:
        """Cierra la retención y deja el suavizador listo para volver a confirmar
        la MISMA clase: así, mantener la imaginación motora encadena acciones."""
        self._hold_command = self._hold_class = None
        self._hold_until = 0.0
        self.smoother.reset()

    def _send(self, command: str) -> bool:
        """Envía un comando a la salida. ``False`` si falló (y ya se detuvo)."""
        try:
            self.sink.send(command)
            return True
        except Exception as exc:  # noqa: BLE001
            self._stop()            # antes de avisar: `warn` es modal (ver _on_classified)
            self.controller.warn("Error de salida", str(exc))
            return False

    def _set_detail(self, msg: str) -> None:
        extra = f"Comandos: {self._n_commands}"
        if self._last_ms:
            extra += f"  ·  {self._last_ms:.0f} ms/ventana"
        if self._dropped:
            extra += f"  ·  {self._dropped} ventanas saltadas"
        self.detail_label.setText(f"{msg}\n{extra}")

    def shutdown(self) -> None:
        self._stop()

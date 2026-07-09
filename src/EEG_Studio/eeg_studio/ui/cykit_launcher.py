"""Ventana para configurar y lanzar CyKit (CyKIT.py).

Permite activar/desactivar las banderas de CyKit y ajustar sus cantidades
(``ovdelay``, ``ovsamples``), construye el comando en vivo, lo puede copiar,
lanzar como proceso (``QProcess``) y aplicar el host/puerto a la fuente
*CyKit / TCP* de la app.

Los valores por defecto reproducen un comando habitual::

    python CyKIT.py 127.0.0.1 5151 6 \\
        openvibe+generic+nocounter+noheader+nobattery+ovdelay:100+float+ovsamples:004
"""
from __future__ import annotations

import os

from PyQt6.QtCore import QProcess
from PyQt6.QtWidgets import (
    QApplication,
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
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

# (clave, etiqueta, activada_por_defecto)
BOOL_FLAGS = [
    ("openvibe", "openvibe (modo OpenViBE)", True),
    ("generic", "generic (servidor TCP)", True),
    ("nocounter", "nocounter (sin COUNTER/INTERPOLATE)", True),
    ("noheader", "noheader (sin cabecera CyKIT)", True),
    ("nobattery", "nobattery (sin batería)", True),
    ("float", "float (formato decimal)", True),
    ("integer", "integer (formato entero)", False),
    ("info", "info (info en consola)", False),
    ("verbose", "verbose (detalle interno)", False),
    ("confirm", "confirm (confirmar dispositivo)", False),
    ("blankdata", "blankdata (señal de prueba)", False),
    ("pywinusb", "pywinusb (en vez de libusb)", False),
    ("bluetooth", "bluetooth (auto-detectar)", False),
]

# (clave, etiqueta, valor, min, max, relleno_ceros)
VALUE_FLAGS = [
    ("ovdelay", "ovdelay (retardo de envío)", 100, 1, 999, 3),
    ("ovsamples", "ovsamples (muestras por bloque)", 4, 1, 999, 3),
]

MODELS = [
    ("1 — Epoc (Premium)", 1),
    ("2 — Epoc (Consumer)", 2),
    ("3 — Insight (Premium)", 3),
    ("4 — Insight (Consumer)", 4),
    ("5 — Epoc+ (Premium)", 5),
    ("6 — Epoc+ (Consumer, 16-bit)", 6),
    ("7 — Epoc+ (Standard, 14-bit)", 7),
]


def autodetect_cykit_dir() -> str:
    """Busca CyKit-master/Py3/CyKIT.py junto al repositorio."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # EEG_Studio
    candidates = [
        os.path.join(os.path.dirname(here), "CyKit-master", "Py3"),
        os.path.join(here, "..", "CyKit-master", "Py3"),
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "CyKIT.py")):
            return os.path.abspath(c)
    return ""


def find_legacy_python() -> str:
    """Busca un intérprete Python 3.7–3.10 (CyKit no soporta versiones nuevas).

    Devuelve la ruta encontrada, o "python" si no halla ninguno compatible.
    """
    local = os.environ.get("LOCALAPPDATA", "")
    for ver in ("39", "38", "37", "310"):
        for path in (
            rf"C:\Python{ver}\python.exe",
            os.path.join(local, "Programs", "Python", f"Python{ver}", "python.exe"),
        ):
            if path and os.path.isfile(path):
                return path
    return "python"


class CyKitLauncherDialog(QDialog):
    def __init__(self, panel, parent=None) -> None:
        super().__init__(parent)
        self.panel = panel
        self.setWindowTitle("Configurar / lanzar CyKit")
        self.resize(560, 640)
        self._proc: QProcess | None = None
        self._bool_widgets: dict[str, QCheckBox] = {}
        self._value_widgets: dict[str, tuple[QCheckBox, QSpinBox, int]] = {}
        self._build_ui()
        self._update_preview()

    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        tip = QLabel(
            "Esto lanza el CyKit original (proceso aparte). CyKit requiere "
            "Python 3.7–3.9; si el intérprete es más nuevo puede fallar.\n"
            "Alternativa sin Python aparte: usa la fuente «Emotiv EPOC+ "
            "(lector integrado)», que hace lo mismo dentro de la app."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #9aa4ae;")
        layout.addWidget(tip)

        # Conexión / proceso.
        conn = QGroupBox("Ejecución")
        form = QFormLayout(conn)
        self.python_edit = QLineEdit(find_legacy_python())
        self.dir_edit = QLineEdit(autodetect_cykit_dir())
        dir_row = QHBoxLayout()
        dir_row.addWidget(self.dir_edit, 1)
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse)
        self.host_edit = QLineEdit("127.0.0.1")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1025, 65535)
        self.port_spin.setValue(5151)
        self.model_combo = QComboBox()
        for label, num in MODELS:
            self.model_combo.addItem(label, num)
        self.model_combo.setCurrentIndex(5)  # modelo 6
        form.addRow("Python:", self.python_edit)
        form.addRow("Carpeta CyKit:", self._wrap(dir_row))
        form.addRow("Host:", self.host_edit)
        form.addRow("Puerto:", self.port_spin)
        form.addRow("Modelo:", self.model_combo)
        layout.addWidget(conn)

        # Banderas booleanas.
        flags_box = QGroupBox("Banderas")
        grid = QGridLayout(flags_box)
        for i, (key, label, default) in enumerate(BOOL_FLAGS):
            chk = QCheckBox(label)
            chk.setChecked(default)
            chk.stateChanged.connect(self._update_preview)
            self._bool_widgets[key] = chk
            grid.addWidget(chk, i // 2, i % 2)
        layout.addWidget(flags_box)

        # Banderas con valor.
        val_box = QGroupBox("Cantidades")
        vform = QFormLayout(val_box)
        for key, label, default, lo, hi, pad in VALUE_FLAGS:
            row = QHBoxLayout()
            chk = QCheckBox("incluir")
            chk.setChecked(True)
            chk.stateChanged.connect(self._update_preview)
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setValue(default)
            spin.valueChanged.connect(self._update_preview)
            row.addWidget(chk)
            row.addWidget(spin, 1)
            self._value_widgets[key] = (chk, spin, pad)
            vform.addRow(label + ":", self._wrap(row))
        layout.addWidget(val_box)

        # Conectar señales de los campos de texto/combo al preview.
        for w in (self.host_edit,):
            w.textChanged.connect(self._update_preview)
        self.port_spin.valueChanged.connect(self._update_preview)
        self.model_combo.currentIndexChanged.connect(self._update_preview)

        # Comando + acciones.
        layout.addWidget(QLabel("Comando:"))
        self.cmd_view = QPlainTextEdit()
        self.cmd_view.setReadOnly(True)
        self.cmd_view.setMaximumHeight(70)
        layout.addWidget(self.cmd_view)

        actions = QHBoxLayout()
        copy_btn = QPushButton("Copiar comando")
        copy_btn.clicked.connect(self._copy)
        self.launch_btn = QPushButton("Lanzar CyKit")
        self.launch_btn.clicked.connect(self._launch)
        self.stop_btn = QPushButton("Detener")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        apply_btn = QPushButton("Aplicar a la fuente")
        apply_btn.clicked.connect(self._apply_to_source)
        for b in (copy_btn, self.launch_btn, self.stop_btn, apply_btn):
            actions.addWidget(b)
        layout.addLayout(actions)

        # Salida del proceso.
        layout.addWidget(QLabel("Salida de CyKit:"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

    @staticmethod
    def _wrap(inner_layout) -> QLabel:
        from PyQt6.QtWidgets import QWidget
        w = QWidget()
        w.setLayout(inner_layout)
        return w

    # ------------------------------------------------------------------ #
    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Carpeta de CyKIT.py", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    def _flag_string(self) -> str:
        flags = [k for k, _l, _d in BOOL_FLAGS if self._bool_widgets[k].isChecked()]
        for key, _label, _d, _lo, _hi, pad in VALUE_FLAGS:
            chk, spin, _pad = self._value_widgets[key]
            if chk.isChecked():
                flags.append(f"{key}:{str(spin.value()).zfill(pad)}")
        return "+".join(flags)

    def nocounter_enabled(self) -> bool:
        return self._bool_widgets["nocounter"].isChecked()

    def _command_args(self) -> list[str]:
        return [
            "CyKIT.py",
            self.host_edit.text().strip() or "127.0.0.1",
            str(self.port_spin.value()),
            str(self.model_combo.currentData()),
            self._flag_string(),
        ]

    def _command_string(self) -> str:
        return f"{self.python_edit.text().strip() or 'python'} " + " ".join(self._command_args())

    def _update_preview(self) -> None:
        self.cmd_view.setPlainText(self._command_string())

    def _copy(self) -> None:
        QApplication.clipboard().setText(self._command_string())
        self.log.appendPlainText("· Comando copiado al portapapeles.")

    # ------------------------------------------------------------------ #
    def _launch(self) -> None:
        cwd = self.dir_edit.text().strip()
        if not cwd or not os.path.isfile(os.path.join(cwd, "CyKIT.py")):
            self.log.appendPlainText("✗ No se encontró CyKIT.py en la carpeta indicada.")
            return
        if self._proc is not None:
            self.log.appendPlainText("· CyKit ya está en ejecución.")
            return
        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(cwd)
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._read_proc)
        self._proc.finished.connect(self._proc_finished)
        self._proc.errorOccurred.connect(
            lambda *_: self.log.appendPlainText("✗ Error al iniciar el proceso de CyKit.")
        )
        self._proc.start(self.python_edit.text().strip() or "python", self._command_args())
        self.log.appendPlainText("· Lanzando: " + self._command_string())
        self.launch_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def _read_proc(self) -> None:
        if self._proc is None:
            return
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if data.strip():
            self.log.appendPlainText(data.rstrip())

    def _proc_finished(self, *_args) -> None:
        self.log.appendPlainText("· CyKit finalizó.")
        self._proc = None
        self.launch_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _stop(self) -> None:
        if self._proc is not None:
            self._proc.kill()

    # ------------------------------------------------------------------ #
    def _apply_to_source(self) -> None:
        """Configura la fuente CyKit/TCP de la app con estos valores."""
        self.panel.apply_cykit_settings(
            host=self.host_edit.text().strip() or "127.0.0.1",
            port=self.port_spin.value(),
            # Con 'nocounter' los canales empiezan en la columna 0; si no, en la 2.
            channel_start=0 if self.nocounter_enabled() else 2,
        )
        self.log.appendPlainText("· Aplicado a la fuente CyKit / TCP de la app.")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop()
        super().closeEvent(event)

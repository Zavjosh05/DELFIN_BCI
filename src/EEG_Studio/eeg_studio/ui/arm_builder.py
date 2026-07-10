"""Constructor / selector del brazo simulado (elige un preset o edita los joints).

Adaptado de ``Arm3DBuilderDialog`` de ``Proyecto_RNN``, en forma de tabla: cada
fila es un joint de la cadena (nombre, eje de rotación, eslabón LinkX/Y/Z, masa y
límites articulares). Al aplicar produce una :class:`ArmSpec` que reconstruye el
brazo simulado.
"""
from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..inference.sim_arm import (
    ArmSpec,
    JointSpec,
    make_default_arm_spec,
    validate_arm_spec,
)

# Ejes de rotación disponibles (etiqueta -> vector).
_AXIS_OPTIONS = [
    ("+Z (yaw, plano XY)", (0.0, 0.0, 1.0)),
    ("+Y (pitch, plano XZ)", (0.0, 1.0, 0.0)),
    ("+X (roll, plano YZ)", (1.0, 0.0, 0.0)),
    ("-Z", (0.0, 0.0, -1.0)),
    ("-Y", (0.0, -1.0, 0.0)),
    ("-X", (-1.0, 0.0, 0.0)),
]
_COLS = ["Nombre", "Eje", "LinkX", "LinkY", "LinkZ", "Masa", "q min °", "q max °", "Home °"]

_PRESETS = {"Default 4DOF (histórico)": make_default_arm_spec}


class ArmBuilderWidget(QWidget):
    """Tabla para elegir un preset o construir el brazo joint por joint."""

    applied = pyqtSignal(object)     # emite la ArmSpec resultante

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(QLabel("Construye un brazo desde cero o parte de un preset."))

        top = QHBoxLayout()
        self.preset_combo = QComboBox()
        for name in _PRESETS:
            self.preset_combo.addItem(name)
        load = QPushButton("Cargar preset")
        load.clicked.connect(self._load_preset)
        top.addWidget(QLabel("Preset:")); top.addWidget(self.preset_combo, 1); top.addWidget(load)
        lay.addLayout(top)

        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.desc_edit = QLineEdit()
        form.addRow("Nombre:", self.name_edit)
        form.addRow("Descripción:", self.desc_edit)
        lay.addLayout(form)

        lay.addWidget(QLabel("Joints (filas = joints en orden de la cadena):"))
        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(30)
        lay.addWidget(self.table, 1)

        btns = QHBoxLayout()
        add = QPushButton("+ Agregar joint"); add.clicked.connect(self._add_joint)
        rm = QPushButton("− Eliminar último"); rm.clicked.connect(self._remove_last)
        apply = QPushButton("Aplicar brazo"); apply.clicked.connect(self._apply)
        btns.addWidget(add); btns.addWidget(rm); btns.addStretch(1); btns.addWidget(apply)
        lay.addLayout(btns)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #8a929b; font-size: 11px;")
        lay.addWidget(self.status)

        self.load_spec(make_default_arm_spec())

    # --- Cargar/volcar la tabla -------------------------------------------
    def _axis_combo(self, axis) -> QComboBox:
        cb = QComboBox()
        for label, vec in _AXIS_OPTIONS:
            cb.addItem(label, vec)
        # selecciona el eje más cercano
        best = min(range(len(_AXIS_OPTIONS)),
                   key=lambda k: sum((a - b) ** 2 for a, b in zip(_AXIS_OPTIONS[k][1], axis)))
        cb.setCurrentIndex(best)
        return cb

    def _add_row(self, j: JointSpec, home: float) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(j.name))
        self.table.setCellWidget(r, 1, self._axis_combo(j.axis))
        vals = [j.link_offset[0], j.link_offset[1], j.link_offset[2], j.mass,
                math.degrees(j.q_min), math.degrees(j.q_max), math.degrees(home)]
        for c, v in enumerate(vals, start=2):
            it = QTableWidgetItem(f"{v:.3g}")
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r, c, it)

    def load_spec(self, spec: ArmSpec) -> None:
        """Vuelca una ``ArmSpec`` a la tabla."""
        self.name_edit.setText(spec.name)
        self.desc_edit.setText(spec.description)
        self.table.setRowCount(0)
        home = list(spec.q_home) + [0.0] * (spec.n_joints - len(spec.q_home))
        for i, j in enumerate(spec.joints):
            self._add_row(j, home[i])
        self.status.setText(f"{spec.n_joints} joints cargados.")

    def _load_preset(self) -> None:
        factory = _PRESETS.get(self.preset_combo.currentText())
        if factory:
            self.load_spec(factory())

    def _add_joint(self) -> None:
        self._add_row(JointSpec(name=f"joint{self.table.rowCount() + 1}",
                                axis=(0.0, 1.0, 0.0), link_offset=(0.15, 0.0, 0.0),
                                q_min=-math.pi * 0.9, q_max=math.pi * 0.9, mass=0.5), 0.0)

    def _remove_last(self) -> None:
        if self.table.rowCount() > 1:
            self.table.removeRow(self.table.rowCount() - 1)

    # --- Construir la spec desde la tabla ---------------------------------
    def _cell_float(self, r: int, c: int, default: float = 0.0) -> float:
        it = self.table.item(r, c)
        try:
            return float(it.text()) if it and it.text().strip() else default
        except ValueError:
            return default

    def build_spec(self) -> ArmSpec:
        joints, home = [], []
        for r in range(self.table.rowCount()):
            name_it = self.table.item(r, 0)
            name = name_it.text().strip() if name_it else f"joint{r + 1}"
            axis = self.table.cellWidget(r, 1).currentData()
            link = (self._cell_float(r, 2), self._cell_float(r, 3), self._cell_float(r, 4))
            mass = self._cell_float(r, 5)
            q_min = math.radians(self._cell_float(r, 6, -162.0))
            q_max = math.radians(self._cell_float(r, 7, 162.0))
            joints.append(JointSpec(name, tuple(axis), link, q_min, q_max, mass))
            home.append(math.radians(self._cell_float(r, 8)))
        return ArmSpec(name=self.name_edit.text().strip() or "custom",
                       description=self.desc_edit.text().strip(),
                       joints=joints, q_home=tuple(home), floor_z=0.0)

    def _apply(self) -> None:
        spec = self.build_spec()
        ok, errors = validate_arm_spec(spec)
        if not ok:
            QMessageBox.warning(self, "Brazo no válido", "\n".join(errors))
            self.status.setText("⚠ Corrige los errores y vuelve a aplicar.")
            return
        self.status.setText(f"✓ Brazo «{spec.name}» aplicado ({spec.n_joints} joints).")
        self.applied.emit(spec)

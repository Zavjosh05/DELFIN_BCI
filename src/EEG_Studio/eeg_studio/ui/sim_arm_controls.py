"""Control manual por articulación del brazo simulado (un slider por joint).

Adaptado del panel de control manual de ``Proyecto_RNN``: cada joint tiene un
slider (mapeado a sus límites articulares) con la lectura del ángulo en grados y
un botón para volver a la posición inicial (HOME).
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..inference.sim_arm import SimulatedArm
from .theme import ACCENT, MUTED, TEXT

_SLIDER_MAX = 1000


class SimArmControls(QWidget):
    """Sliders por articulación + botón HOME para el brazo simulado."""

    def __init__(self, arm: SimulatedArm, on_change=None, parent=None) -> None:
        super().__init__(parent)
        self.arm = arm
        self._on_change = on_change
        self._sliders: list[QSlider] = []
        self._val_labels: list[QLabel] = []

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(4, 4, 4, 4)
        self._root.setSpacing(8)
        self._rows = QVBoxLayout()
        self._rows.setSpacing(8)
        self._root.addLayout(self._rows)

        home = QPushButton("🏠  Volver a posición inicial")
        home.clicked.connect(self._home)
        self._root.addWidget(home)
        hint = QLabel("Mueve cada articulación dentro de sus límites. Los mismos "
                      "comandos (arriba/abajo, izquierda/derecha, agarre/soltar) "
                      "siguen funcionando y actualizan estos sliders.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        self._root.addWidget(hint)
        self._root.addStretch(1)

        self.rebuild()

    def _pos_from_q(self, idx: int, q: float) -> int:
        lo, hi = self.arm.q_min[idx], self.arm.q_max[idx]
        if hi <= lo:
            return _SLIDER_MAX // 2
        frac = (q - lo) / (hi - lo)
        return int(round(min(1.0, max(0.0, frac)) * _SLIDER_MAX))

    def _q_from_pos(self, idx: int, pos: int) -> float:
        lo, hi = self.arm.q_min[idx], self.arm.q_max[idx]
        return float(lo + (pos / _SLIDER_MAX) * (hi - lo))

    def rebuild(self) -> None:
        """(Re)crea un slider por joint del brazo actual (tras cambiar la spec)."""
        while self._rows.count():
            item = self._rows.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._sliders = []
        self._val_labels = []
        names = self.arm.joint_names
        for i in range(self.arm.q.size):
            row = QWidget()
            rlay = QVBoxLayout(row)
            rlay.setContentsMargins(0, 0, 0, 0)
            rlay.setSpacing(2)
            top = QHBoxLayout()
            name = QLabel(names[i] if i < len(names) else f"q{i + 1}")
            name.setStyleSheet(f"color: {TEXT}; font-weight: 500;")
            val = QLabel(f"{np.degrees(self.arm.q[i]):+.1f}°")
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val.setStyleSheet(f"color: {ACCENT}; font-family: Consolas, monospace; font-weight: 600;")
            top.addWidget(name, 1); top.addWidget(val, 0)
            rlay.addLayout(top)
            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setMinimum(0); sld.setMaximum(_SLIDER_MAX)
            sld.setValue(self._pos_from_q(i, self.arm.q[i]))
            sld.valueChanged.connect(lambda pos, idx=i: self._on_slider(idx, pos))
            rlay.addWidget(sld)
            self._rows.addWidget(row)
            self._sliders.append(sld)
            self._val_labels.append(val)

    def _on_slider(self, idx: int, pos: int) -> None:
        self.arm.set_q(idx, self._q_from_pos(idx, pos))
        self._val_labels[idx].setText(f"{np.degrees(self.arm.q[idx]):+.1f}°")
        if self._on_change is not None:
            self._on_change()

    def _home(self) -> None:
        self.arm.reset()
        self.sync()
        if self._on_change is not None:
            self._on_change()

    def sync(self) -> None:
        """Actualiza los sliders/etiquetas desde el estado actual del brazo (tras
        moverlo por comandos/clasificador), sin re-disparar ``on_change``."""
        for i, sld in enumerate(self._sliders):
            if i >= self.arm.q.size:
                break
            sld.blockSignals(True)
            sld.setValue(self._pos_from_q(i, self.arm.q[i]))
            sld.blockSignals(False)
            self._val_labels[i].setText(f"{np.degrees(self.arm.q[i]):+.1f}°")

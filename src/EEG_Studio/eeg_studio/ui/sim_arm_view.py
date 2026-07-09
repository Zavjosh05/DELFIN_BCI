"""Vista 2D del brazo simulado (pyqtgraph, sin OpenGL).

Adaptada del módulo de simulación de ``Proyecto_RNN`` (``_Projection2D``): dibuja
la cadena de eslabones proyectada en dos planos —lateral (elevación, muestra
arriba/abajo) y superior (giro de la base, muestra izquierda/derecha)— y marca el
efector según el estado de la pinza (abierta/cerrada).
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..inference.sim_arm import SimulatedArm
from .theme import ACCENT, BORDER, MUTED, SURFACE, TEXT

_ARM_COL = "#5eead4"      # eslabones (turquesa)
_JOINT_COL = "#c8d0d8"    # articulaciones
_OPEN_COL = "#3d86cc"     # pinza abierta (azul)
_CLOSED_COL = "#ff6b6b"   # pinza cerrada / agarrando (rojo)


class _ArmProjection(pg.PlotWidget):
    """Proyección 2D del brazo en un plano ('side' = elevación, 'top' = giro)."""

    def __init__(self, arm: SimulatedArm, title: str, plane: str, parent=None) -> None:
        super().__init__(parent)
        self.arm = arm
        self.plane = plane
        r = arm.reach * 1.15
        self.setAspectLocked(True)
        self.setMenuEnabled(False)
        self.setMouseEnabled(False, False)
        self.hideButtons()
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setTitle(title, color=MUTED, size="9pt")
        self.getPlotItem().getViewBox().setBackgroundColor(SURFACE)
        if plane == "side":
            self.setRange(xRange=(-0.02, r), yRange=(-r * 0.15, r))
            self.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(
                BORDER, width=1, style=Qt.PenStyle.DashLine)))   # piso z=0
            theta = np.linspace(0, np.pi / 2, 40)
            self.plot(arm.reach * np.cos(theta), arm.reach * np.sin(theta),
                      pen=pg.mkPen(BORDER, width=1, style=Qt.PenStyle.DashLine))
        else:  # top
            self.setRange(xRange=(-r, r), yRange=(-r, r))
            theta = np.linspace(0, 2 * np.pi, 80)
            self.plot(arm.reach * np.cos(theta), arm.reach * np.sin(theta),
                      pen=pg.mkPen(BORDER, width=1, style=Qt.PenStyle.DashLine))

        self.arm_curve = self.plot([], [], pen=pg.mkPen(_ARM_COL, width=4))
        self.joints_scat = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(_JOINT_COL),
                                               pen=pg.mkPen("#000", width=1))
        self.addItem(self.joints_scat)
        self.ee_scat = pg.ScatterPlotItem(size=15, pen=pg.mkPen("#02201a", width=1))
        self.addItem(self.ee_scat)

    def _extract(self, pts: np.ndarray):
        if self.plane == "side":                       # (radio horizontal, altura)
            rxy = np.hypot(pts[:, 0], pts[:, 1])
            return rxy, pts[:, 2]
        return pts[:, 0], pts[:, 1]                     # vista superior (x, y)

    def refresh(self) -> None:
        pts = self.arm.fk()
        xs, ys = self._extract(pts)
        self.arm_curve.setData(xs, ys)
        self.joints_scat.setData(xs[:-1], ys[:-1])
        col = _CLOSED_COL if self.arm.gripper_closed else _OPEN_COL
        self.ee_scat.setData([xs[-1]], [ys[-1]], brush=pg.mkBrush(col))


class SimArmView(QWidget):
    """Vista del brazo simulado: proyección lateral + superior + estado de pinza."""

    def __init__(self, arm: SimulatedArm | None = None, parent=None) -> None:
        super().__init__(parent)
        self.arm = arm or SimulatedArm()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        plots = QHBoxLayout()
        self.side = _ArmProjection(self.arm, "Lateral (elevación)", "side")
        self.top = _ArmProjection(self.arm, "Superior (giro base)", "top")
        for p in (self.side, self.top):
            p.setMinimumHeight(150)
            plots.addWidget(p)
        lay.addLayout(plots)

        self.status = QLabel()
        self.status.setStyleSheet(f"color: {TEXT}; font-size: 11px;")
        lay.addWidget(self.status)
        self.refresh()

    def set_arm(self, arm: SimulatedArm) -> None:
        self.arm = arm
        self.side.arm = arm
        self.top.arm = arm
        self.refresh()

    def refresh(self) -> None:
        self.side.refresh()
        self.top.refresh()
        q = self.arm.q
        pinza = "cerrada ✊" if self.arm.gripper_closed else "abierta ✋"
        ee = self.arm.ee()
        self.status.setText(
            f"Pinza: {pinza}   ·   base {np.degrees(q[0]):+.0f}°   "
            f"hombro {np.degrees(q[1]):+.0f}°   ·   efector "
            f"({ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}) m")

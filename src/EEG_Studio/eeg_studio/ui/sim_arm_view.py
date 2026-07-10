"""Vistas del brazo simulado (3D con OpenGL + proyecciones 2D con pyqtgraph).

Adaptado del módulo de simulación de ``Proyecto_RNN`` (``ArmView3D`` y
``_Projection2D``). La vista 3D usa ``pyqtgraph.opengl`` (requiere PyOpenGL); si no
está disponible, la vista degrada con elegancia a solo las proyecciones 2D.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ..inference.sim_arm import SimulatedArm
from .theme import BORDER, MUTED, SURFACE, TEXT

try:                                    # 3D es opcional (PyOpenGL)
    import pyqtgraph.opengl as gl
    _GL_OK = True
except Exception:                       # noqa: BLE001
    gl = None
    _GL_OK = False

_ARM_COL = "#5eead4"       # eslabones (turquesa)
_JOINT_COL = "#c8d0d8"     # articulaciones
_OPEN_COL = "#3d86cc"      # pinza abierta (azul)
_CLOSED_COL = "#ff6b6b"    # pinza cerrada / agarrando (rojo)


class _ArmProjection(pg.PlotWidget):
    """Proyección 2D del brazo en un plano ('side' = elevación, 'top' = giro)."""

    def __init__(self, arm: SimulatedArm, title: str, plane: str, parent=None) -> None:
        super().__init__(parent)
        self.arm = arm
        self.plane = plane
        self.setMenuEnabled(False)
        self.setMouseEnabled(False, False)
        self.hideButtons()
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setTitle(title, color=MUTED, size="9pt")
        self.getPlotItem().getViewBox().setBackgroundColor(SURFACE)
        self._rebuild_static()
        self.arm_curve = self.plot([], [], pen=pg.mkPen(_ARM_COL, width=4))
        self.joints_scat = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(_JOINT_COL),
                                               pen=pg.mkPen("#000", width=1))
        self.addItem(self.joints_scat)
        self.ee_scat = pg.ScatterPlotItem(size=15, pen=pg.mkPen("#02201a", width=1))
        self.addItem(self.ee_scat)

    def _rebuild_static(self) -> None:
        r = self.arm.reach * 1.15
        pen = pg.mkPen(BORDER, width=1, style=Qt.PenStyle.DashLine)
        if self.plane == "side":
            self.setRange(xRange=(-0.02, r), yRange=(-r * 0.15, r))
            self.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pen))     # piso z=0
            theta = np.linspace(0, np.pi / 2, 40)
        else:
            self.setRange(xRange=(-r, r), yRange=(-r, r))
            theta = np.linspace(0, 2 * np.pi, 80)
        self.plot(self.arm.reach * np.cos(theta), self.arm.reach * np.sin(theta), pen=pen)

    def _extract(self, pts: np.ndarray):
        if self.plane == "side":                       # (radio horizontal, altura)
            return np.hypot(pts[:, 0], pts[:, 1]), pts[:, 2]
        return pts[:, 0], pts[:, 1]                     # vista superior (x, y)

    def refresh(self) -> None:
        pts = self.arm.fk()
        xs, ys = self._extract(pts)
        self.arm_curve.setData(xs, ys)
        self.joints_scat.setData(xs[:-1], ys[:-1])
        col = _CLOSED_COL if self.arm.gripper_closed else _OPEN_COL
        self.ee_scat.setData([xs[-1]], [ys[-1]], brush=pg.mkBrush(col))


if _GL_OK:
    _LINK_RAMP = [(0.37, 0.92, 0.83, 1.0), (0.24, 0.75, 0.95, 1.0),
                  (0.30, 0.80, 0.55, 1.0), (0.95, 0.65, 0.30, 1.0)]

    class _ArmView3D(gl.GLViewWidget):
        """Vista 3D del brazo con OpenGL (grid + ejes + eslabones + joints)."""

        def __init__(self, arm: SimulatedArm, parent=None) -> None:
            super().__init__(parent)
            self.arm = arm
            self.setCameraPosition(distance=arm.reach * 2.2, elevation=22, azimuth=-60)
            grid = gl.GLGridItem()
            grid.setSize(x=arm.reach * 2, y=arm.reach * 2)
            grid.setSpacing(x=0.1, y=0.1)
            grid.setColor((80, 100, 130, 90))
            self.addItem(grid)
            axis = gl.GLAxisItem()
            axis.setSize(x=0.18, y=0.18, z=0.18)
            self.addItem(axis)
            self._links: list = []
            self.joints = gl.GLScatterPlotItem(pos=np.zeros((1, 3)),
                                               color=(0.85, 0.88, 0.92, 1.0),
                                               size=12, pxMode=True)
            self.addItem(self.joints)
            self.ee = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=20, pxMode=True)
            self.addItem(self.ee)
            self.rebuild()

        def rebuild(self) -> None:
            for ln in self._links:
                try:
                    self.removeItem(ln)
                except Exception:      # noqa: BLE001
                    pass
            self._links = []
            n_seg = max(1, len(self.arm.fk()) - 1)
            for i in range(n_seg):
                ln = gl.GLLinePlotItem(pos=np.zeros((2, 3)), width=7, antialias=True,
                                       color=_LINK_RAMP[i % len(_LINK_RAMP)])
                self.addItem(ln)
                self._links.append(ln)
            self.refresh()

        def refresh(self) -> None:
            pts = self.arm.fk()
            if len(pts) - 1 != len(self._links):
                self.rebuild(); return
            for i in range(len(pts) - 1):
                self._links[i].setData(pos=np.array([pts[i], pts[i + 1]]))
            self.joints.setData(pos=pts)
            col = ((1.0, 0.42, 0.42, 1.0) if self.arm.gripper_closed
                   else (0.24, 0.53, 0.80, 1.0))
            self.ee.setData(pos=pts[-1].reshape(1, 3), color=col)


def _make_3d(arm: SimulatedArm):
    if not _GL_OK:
        return None
    try:
        return _ArmView3D(arm)
    except Exception:                   # noqa: BLE001 (sin contexto OpenGL, etc.)
        return None


class SimArmView(QWidget):
    """Vista del brazo simulado: 3D (si hay OpenGL) + proyecciones 2D + estado."""

    def __init__(self, arm: SimulatedArm | None = None, parent=None) -> None:
        super().__init__(parent)
        self.arm = arm or SimulatedArm()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self.view3d = _make_3d(self.arm)
        if self.view3d is not None:
            self.view3d.setMinimumHeight(220)
            lay.addWidget(self.view3d, 1)
        else:
            note = QLabel("Vista 3D no disponible (instala PyOpenGL). Se muestran las "
                          "proyecciones 2D.")
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
            lay.addWidget(note)

        plots = QHBoxLayout()
        self.side = _ArmProjection(self.arm, "Lateral (elevación)", "side")
        self.top = _ArmProjection(self.arm, "Superior (giro base)", "top")
        for p in (self.side, self.top):
            p.setMinimumHeight(130)
            plots.addWidget(p)
        lay.addLayout(plots)

        self.status = QLabel()
        self.status.setStyleSheet(f"color: {TEXT}; font-size: 11px;")
        lay.addWidget(self.status)
        self.refresh()

    def set_arm(self, arm: SimulatedArm) -> None:
        """Reasigna el brazo (p. ej. tras reconstruirlo) y redibuja todo."""
        self.arm = arm
        self.side.arm = arm; self.side._rebuild_static()
        self.top.arm = arm; self.top._rebuild_static()
        if self.view3d is not None:
            self.view3d.arm = arm
            self.view3d.rebuild()
        self.refresh()

    def refresh(self) -> None:
        if self.view3d is not None:
            self.view3d.refresh()
        self.side.refresh()
        self.top.refresh()
        q = self.arm.q
        pinza = "cerrada ✊" if self.arm.gripper_closed else "abierta ✋"
        ee = self.arm.ee()
        deg = "  ".join(f"{np.degrees(v):+.0f}°" for v in q)
        self.status.setText(
            f"Pinza: {pinza}   ·   q: {deg}   ·   "
            f"efector ({ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}) m")

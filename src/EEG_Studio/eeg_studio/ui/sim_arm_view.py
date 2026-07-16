"""Vistas del brazo simulado (3D con OpenGL + proyecciones 2D con pyqtgraph).

Adaptado del módulo de simulación de ``Proyecto_RNN`` (``ArmView3D`` y
``_Projection2D``). La vista 3D usa ``pyqtgraph.opengl`` (requiere PyOpenGL); si no
está disponible, la vista degrada con elegancia a solo las proyecciones 2D.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..inference.sim_arm import SimulatedArm
from .sim_arm_controls import SimArmActionPad, SimArmControls
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

    def __init__(self, arm: SimulatedArm, title: str, plane: str,
                 on_control=None, parent=None) -> None:
        super().__init__(parent)
        self.arm = arm
        self.plane = plane
        self._on_control = on_control          # callback tras mover el brazo por clic
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
        # Objetivo del último clic (marcador rojo tenue) para referencia visual.
        self.target_scat = pg.ScatterPlotItem(size=13, brush=pg.mkBrush("#ff6b6b88"),
                                               pen=pg.mkPen("#000", width=1))
        self.addItem(self.target_scat)
        if on_control is not None:
            self.setCursor(Qt.CursorShape.CrossCursor)
            self.setToolTip("Clic para mover el brazo hacia ese punto "
                            "(lateral: hombro/codo/muñeca · superior: giro de base).")
            self.scene().sigMouseClicked.connect(self._on_click)

    def _on_click(self, event) -> None:
        """Clic en la proyección: mueve el brazo hacia el punto pulsado.

        Vista lateral → IK planar (hombro/codo/muñeca); vista superior → giro de
        la base (yaw). Respeta límites y piso (lo garantiza el modelo del brazo)."""
        if self._on_control is None or event.button() != Qt.MouseButton.LeftButton:
            return
        vb = self.getPlotItem().getViewBox()
        pos = event.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
        mp = vb.mapSceneToView(pos)
        x, y = float(mp.x()), float(mp.y())
        if self.plane == "side":                       # (radio horizontal, altura)
            self.arm.aim_planar(max(0.0, x), y)
        else:                                          # vista superior (x, y)
            self.arm.aim_base_to(x, y)
        self.target_scat.setData([x], [y])
        self._on_control()

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

    class _ArmView3D(gl.GLViewWidget):
        """Vista 3D del brazo con OpenGL.

        El brazo se dibuja como **una sola polilínea** (line_strip) por todos los
        puntos de la cadena — más limpio y sin los artefactos de dibujar cada
        eslabón como un item GL suelto. Las articulaciones y el efector son
        *scatters* aparte."""

        def __init__(self, arm: SimulatedArm, parent=None) -> None:
            super().__init__(parent)
            self.arm = arm
            self.setBackgroundColor(pg.mkColor(SURFACE))
            self.grid = gl.GLGridItem()
            self.grid.setColor((90, 110, 140, 70))
            self.addItem(self.grid)
            self.arm_line = gl.GLLinePlotItem(
                pos=np.zeros((2, 3)), width=4.0, antialias=True,
                color=(0.37, 0.92, 0.83, 1.0), mode="line_strip")
            self.addItem(self.arm_line)
            self.joints = gl.GLScatterPlotItem(
                pos=np.zeros((1, 3)), color=(0.82, 0.86, 0.92, 1.0),
                size=8.0, pxMode=True)
            self.addItem(self.joints)
            self.ee = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=13.0, pxMode=True)
            self.addItem(self.ee)
            self._fit()
            self.refresh()

        def _fit(self) -> None:
            s = max(0.4, self.arm.reach * 1.6)
            self.grid.setSize(x=s, y=s)
            self.grid.setSpacing(x=0.1, y=0.1)
            self.setCameraPosition(distance=max(0.6, self.arm.reach * 2.6),
                                   elevation=22, azimuth=-55)

        def rebuild(self) -> None:
            self._fit()
            self.refresh()

        def refresh(self) -> None:
            pts = np.asarray(self.arm.fk(), dtype=float)
            self.arm_line.setData(pos=pts)
            self.joints.setData(pos=pts[:-1] if len(pts) > 1 else pts)
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


class _ArmFullscreen(QWidget):
    """Ventana a pantalla completa con el brazo **y sus controles**: el 3D grande
    a la izquierda y, a la derecha, el D-pad de acciones + los sliders por
    articulación (igual que en el panel). Incluye un botón visible para volver."""

    def __init__(self, arm: SimulatedArm, on_change=None, control=None) -> None:
        super().__init__()                 # top-level (sin padre) para pantalla completa
        self.arm = arm
        self._on_change = on_change        # sincroniza el panel principal al mover el brazo
        self._control = control            # panel de Control (se delega, no se duplica)
        self.setWindowTitle("Brazo simulado")
        self.setStyleSheet(f"background: {SURFACE};")
        # Para poder recibir el teclado aunque el foco no esté en ningún hijo.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Esc como atajo de VENTANA: funciona tenga el foco quien lo tenga (el 3D, un
        # botón, un slider…). Con solo keyPressEvent dependía de a quién llegara la tecla.
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self,
                  context=Qt.ShortcutContext.WindowShortcut, activated=self.close)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Barra superior con el botón de volver (visible, además de Esc).
        topbar = QHBoxLayout()
        topbar.setContentsMargins(12, 8, 12, 8)
        title = QLabel("Brazo simulado — pantalla completa")
        title.setStyleSheet(f"color: {TEXT}; font-size: 13px; font-weight: 600;")
        topbar.addWidget(title)
        topbar.addStretch(1)
        self.close_btn = QPushButton("✕  Cerrar (Esc)")
        self.close_btn.setToolTip("Cierra la pantalla completa y vuelve al panel.")
        self.close_btn.setMinimumHeight(30)
        self.close_btn.clicked.connect(self.close)
        topbar.addWidget(self.close_btn)
        outer.addLayout(topbar)

        body = QHBoxLayout()
        body.setContentsMargins(8, 0, 8, 8)
        body.setSpacing(8)
        # Brazo grande a la izquierda (3D si hay OpenGL; si no, la vista lateral 2D
        # interactiva para no perder el control por clic).
        self.view = _make_3d(arm) or _ArmProjection(
            arm, "Brazo (lateral)", "side", on_control=self._changed)
        body.addWidget(self.view, 3)

        # Columna de controles a la derecha (desplazable si no cabe).
        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(8)
        # Control en tiempo real (solo si hay panel de Control al que delegar).
        if control is not None:
            pl.addWidget(self._control_box())

        act_box = QGroupBox("Acciones")
        al = QVBoxLayout(act_box); al.setContentsMargins(6, 6, 6, 6)
        self.action_pad = SimArmActionPad(self._do_command)
        al.addWidget(self.action_pad)
        pl.addWidget(act_box)
        joint_box = QGroupBox("Control por articulación")
        jl = QVBoxLayout(joint_box); jl.setContentsMargins(6, 6, 6, 6)
        self.controls = SimArmControls(arm, on_change=self._changed)
        jl.addWidget(self.controls)
        pl.addWidget(joint_box)
        pl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(420)
        body.addWidget(scroll, 0)
        outer.addLayout(body, 1)

        # Espejo del estado del panel de Control. El bucle de inferencia (con su
        # estado asíncrono y la retención de comando) es SUYO: aquí solo se muestra
        # y se delega, para no tener dos clasificadores compitiendo.
        self._mirror = QTimer(self)
        self._mirror.setInterval(150)
        self._mirror.timeout.connect(self._sync_control)
        if control is not None:
            self._sync_control()
            self._mirror.start()

    # --- Control en tiempo real (delegado en el panel de Control) ----------
    def _control_box(self) -> QGroupBox:
        """Selector de modelo + iniciar/detener + comando predicho, en grande."""
        box = QGroupBox("Control en tiempo real")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 6, 6, 6)

        row = QHBoxLayout()
        row.addWidget(QLabel("Modelo:"))
        self.model_combo = QComboBox()
        self.model_combo.setToolTip("Modelo con el que se clasifica la señal en vivo. "
                                    "Es el mismo selector que el del panel de Control.")
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        row.addWidget(self.model_combo, 1)
        lay.addLayout(row)

        self.start_btn = QPushButton("Iniciar control")
        self.start_btn.setMinimumHeight(34)
        self.start_btn.setToolTip("Arranca o detiene la clasificación en vivo "
                                  "(necesita una fuente conectada en «Tiempo real»).")
        self.start_btn.clicked.connect(self._toggle_control)
        lay.addWidget(self.start_btn)

        self.pred_label = QLabel("—")
        self.pred_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pred_label.setStyleSheet(
            "font-size: 26px; font-weight: bold; color: #9be7c4;")
        self.pred_label.setToolTip("Comando predicho por el modelo (y su confianza).")
        lay.addWidget(self.pred_label)

        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        lay.addWidget(self.detail_label)
        return box

    def _on_model_changed(self) -> None:
        """Cambiar el modelo aquí lo cambia en el panel (una sola fuente de verdad)."""
        if self._control is None:
            return
        name = self.model_combo.currentData()
        i = self._control.model_combo.findData(name)
        if i >= 0 and i != self._control.model_combo.currentIndex():
            self._control.model_combo.setCurrentIndex(i)

    def _toggle_control(self) -> None:
        if self._control is not None:
            self._control.toggle()      # arranca/detiene el bucle del panel
            self._sync_control()

    def _sync_control(self) -> None:
        """Refleja el estado del panel de Control (modelos, botón, predicción)."""
        c = self._control
        if c is None:
            return
        names = [c.model_combo.itemData(i) for i in range(c.model_combo.count())]
        mine = [self.model_combo.itemData(i) for i in range(self.model_combo.count())]
        if names != mine:                       # la lista puede cambiar en caliente
            self.model_combo.blockSignals(True)
            self.model_combo.clear()
            for n in names:
                self.model_combo.addItem(str(n), n)
            self.model_combo.blockSignals(False)
        current = c.model_combo.currentData()
        if current != self.model_combo.currentData():
            i = self.model_combo.findData(current)
            if i >= 0:
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(i)
                self.model_combo.blockSignals(False)
        self.start_btn.setText(c.start_btn.text())
        self.start_btn.setEnabled(c.start_btn.isEnabled())
        self.pred_label.setText(c.pred_label.text())
        self.detail_label.setText(c.detail_label.text())

    def _do_command(self, command: str) -> None:
        """Ejecuta una acción del D-pad sobre el brazo y refresca todo."""
        if command == "home":
            self.arm.reset()
        else:
            self.arm.execute(command)
        self._changed()

    def _changed(self) -> None:
        """Tras cualquier control (D-pad, sliders o clic): redibuja el brazo,
        sincroniza los sliders y avisa al panel principal."""
        self.refresh()
        self.controls.sync()
        if self._on_change is not None:
            self._on_change()

    def refresh(self) -> None:
        try:
            self.view.refresh()
        except Exception:  # noqa: BLE001
            pass

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)


class SimArmView(QWidget):
    """Vista del brazo simulado: 3D (si hay OpenGL) + proyecciones 2D + estado."""

    def __init__(self, arm: SimulatedArm | None = None, on_change=None,
                 parent=None, control=None) -> None:
        super().__init__(parent)
        self.arm = arm or SimulatedArm()
        self._on_change = on_change            # avisa al panel al mover el brazo por clic
        self._control = control                # panel de Control (para la pantalla completa)
        self._fs: _ArmFullscreen | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        # Barra: colapsar las vistas 2D y abrir el brazo a pantalla completa.
        bar = QHBoxLayout()
        self.toggle2d_btn = QPushButton("▾ Vistas laterales")
        self.toggle2d_btn.setCheckable(True)
        self.toggle2d_btn.setChecked(True)
        self.toggle2d_btn.setToolTip("Mostrar/ocultar las proyecciones 2D (lateral y superior).")
        self.toggle2d_btn.toggled.connect(self._toggle_2d)
        bar.addWidget(self.toggle2d_btn)
        bar.addStretch(1)
        self.fs_btn = QPushButton("⛶ Pantalla completa")
        self.fs_btn.setToolTip("Abre SOLO el brazo a pantalla completa (Esc para volver).")
        self.fs_btn.clicked.connect(self._open_fullscreen)
        bar.addWidget(self.fs_btn)
        lay.addLayout(bar)

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

        # Contenedor colapsable de las proyecciones 2D.
        self.plots_container = QWidget()
        plots = QHBoxLayout(self.plots_container)
        plots.setContentsMargins(0, 0, 0, 0)
        self.side = _ArmProjection(self.arm, "Lateral (elevación)", "side",
                                   on_control=self._on_projection_control)
        self.top = _ArmProjection(self.arm, "Superior (giro base)", "top",
                                  on_control=self._on_projection_control)
        for p in (self.side, self.top):
            p.setMinimumHeight(130)
            plots.addWidget(p)
        lay.addWidget(self.plots_container)

        self.status = QLabel()
        self.status.setStyleSheet(f"color: {TEXT}; font-size: 11px;")
        lay.addWidget(self.status)
        self.refresh()

    def _toggle_2d(self, shown: bool) -> None:
        self.plots_container.setVisible(shown)
        self.toggle2d_btn.setText(("▾ " if shown else "▸ ") + "Vistas laterales")

    def _on_projection_control(self) -> None:
        """El usuario movió el brazo con un clic en una proyección 2D: redibuja
        todo y avisa al panel (para sincronizar sliders y estado)."""
        self.refresh()
        if self._on_change is not None:
            self._on_change()

    def _open_fullscreen(self) -> None:
        if self._fs is not None:
            self._fs.close()
        self._fs = _ArmFullscreen(self.arm, on_change=self._on_change,
                                  control=self._control)
        self._fs.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._fs.destroyed.connect(self._on_fs_closed)
        self._fs.showFullScreen()
        # Sin activar la ventana, el teclado sigue yendo a la ventana principal: se
        # veía la pantalla completa pero Esc no hacía nada (el ratón sí funcionaba,
        # porque no necesita el foco).
        self._fs.raise_()
        self._fs.activateWindow()
        self._fs.setFocus(Qt.FocusReason.OtherFocusReason)
        self._fs.refresh()

    def _on_fs_closed(self, *_args) -> None:
        self._fs = None

    def set_arm(self, arm: SimulatedArm) -> None:
        """Reasigna el brazo (p. ej. tras reconstruirlo) y redibuja todo."""
        self.arm = arm
        self.side.arm = arm; self.side._rebuild_static()
        self.top.arm = arm; self.top._rebuild_static()
        if self.view3d is not None:
            self.view3d.arm = arm
            self.view3d.rebuild()
        if self._fs is not None:               # la ventana FS quedaría con el brazo viejo
            self._fs.close()
        self.refresh()

    def refresh(self) -> None:
        if self.view3d is not None:
            self.view3d.refresh()
        self.side.refresh()
        self.top.refresh()
        if self._fs is not None:               # ventana a pantalla completa abierta
            self._fs.refresh()
        q = self.arm.q
        pinza = "cerrada ✊" if self.arm.gripper_closed else "abierta ✋"
        ee = self.arm.ee()
        deg = "  ".join(f"{np.degrees(v):+.0f}°" for v in q)
        self.status.setText(
            f"Pinza: {pinza}   ·   q: {deg}   ·   "
            f"efector ({ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}) m")

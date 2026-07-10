"""Ventana principal de EEG Studio: integra visor, paneles y control de cambios."""
from __future__ import annotations

import os
import threading
import time

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QSettings, QSize, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QDesktopServices, QKeySequence, QPainter
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QSpinBox,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    APP_NAME,
    DATASETS_DIR,
    EPOC_CHANNELS,
    FREQ_BANDS,
    IMPORTED_DIR,
    ORG_NAME,
    PROJECT_EXT,
)
from ..core import classification, dataset as dataset_mod, mat_loader, mne_loader
from ..core.csv_loader import compress_csv, load_recording
from ..core.processing import band_powers, extract_feature_vector, time_features
from ..core.project import Project
from ..workers import run_async
from .acquisition_panel import AcquisitionPanel
from .control_panel import ControlPanel
from .live_view import LiveSignalView
from .panels import ClassificationPanel, DatasetPanel, PreprocessingPanel
from .signal_view import SignalView
from .signal_window import SignalWindow
from .theme import ACCENT, BG, BORDER, ELEVATED, MUTED, SURFACE, TEXT

# Indicador de contenido de una fuente: un punto pequeño y discreto a la derecha.
_MARK_COLOR_ROLE = Qt.ItemDataRole.UserRole + 1
COLOR_HAS_SEGMENTS = QColor("#57c98a")   # verde: tiene segmentos etiquetados
COLOR_HAS_MARKERS = QColor("#d6a341")    # ámbar: solo tiene marcadores (Event Id)


class _SourceListWidget(QListWidget):
    """Lista de fuentes que avisa cuando se reordena arrastrando.

    ``QListWidget`` implementa el «internal move» como insertar + eliminar, así que
    ``model().rowsMoved`` no se dispara; emitimos ``reordered`` al terminar el drop.
    """

    reordered = pyqtSignal()

    def dropEvent(self, event) -> None:  # noqa: N802
        super().dropEvent(event)
        self.reordered.emit()


class _SourceItemDelegate(QStyledItemDelegate):
    """Pinta un punto pequeño a la derecha de la fila (indicador de contenido).

    Va aparte del texto para no ocupar la columna del icono (así la fila conserva
    su tamaño y el nombre no se indenta), y queda discreto."""

    def paint(self, painter, option, index) -> None:  # noqa: N802
        super().paint(painter, option, index)
        color = index.data(_MARK_COLOR_ROLE)
        if not isinstance(color, QColor):
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        rect = option.rect
        painter.drawEllipse(QPointF(rect.right() - 9, rect.center().y() + 1.0), 3.0, 3.0)
        painter.restore()


class MainWindow(QMainWindow):
    # Emitida desde el hilo (daemon) de escaneo de marcadores con {source_id: nº}.
    # Se entrega en el hilo GUI (conexión en cola) para pintar los indicadores.
    _markers_scanned = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self.project: Project | None = None
        self.dataset = None
        self.model: classification.TrainingResult | None = None       # modelo activo
        self.models: dict[str, classification.TrainingResult] = {}    # registro del proyecto
        self.active_model_name: str | None = None
        self.current_source_id: str | None = None
        self._threads: list = []  # mantiene vivos los hilos en ejecución
        self._signal_windows: set = set()  # ventanas de señal abiertas (varias a la vez)

        # Guardado continuo (autosave): guarda el proyecto poco después de cada
        # cambio, sin necesidad de Ctrl+S (que sigue funcionando).
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(800)
        self._autosave_timer.timeout.connect(self._autosave)
        self._dirty = False                       # cambios sin guardar (indicador ●)

        self.setWindowTitle(APP_NAME)
        # Tamaño y mínimo seguros para 1080p y 1440p (cabe en el área útil).
        self.setMinimumSize(1024, 640)
        self.resize(1360, 860)

        # Centro: pila con (0) bienvenida y (1) pestañas Análisis / Tiempo real.
        # La pestaña "Análisis (CSV)" es a su vez un cuaderno de pestañas: una por
        # fuente abierta (como pestañas de navegador). `signal_view` (propiedad)
        # apunta siempre a la pestaña activa.
        self._source_views: dict[str, SignalView] = {}
        self._stale_views: set[str] = set()     # fuentes cuya pestaña hay que redibujar
        self._empty_view = SignalView()         # respaldo cuando no hay ninguna abierta
        self._signal_tabs = QTabWidget()
        self._signal_tabs.setTabsClosable(True)
        self._signal_tabs.setMovable(True)
        self._signal_tabs.setDocumentMode(True)
        self._signal_tabs.setUsesScrollButtons(True)
        self._signal_tabs.setElideMode(Qt.TextElideMode.ElideRight)
        self._signal_tabs.tabCloseRequested.connect(self._close_source_tab)
        self._signal_tabs.currentChanged.connect(self._on_signal_tab_changed)
        self.live_view = LiveSignalView()
        self.center_tabs = QTabWidget()
        self.center_tabs.addTab(self._signal_tabs, "Análisis (CSV)")
        self.center_tabs.addTab(self.live_view, "Tiempo real")
        self.welcome = self._build_welcome()
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self.welcome)        # índice 0
        self.center_stack.addWidget(self.center_tabs)    # índice 1
        self.setCentralWidget(self.center_stack)

        self._build_docks()
        self._build_menu()
        self._build_toolbar()
        self._build_statusbar()
        self._update_actions()
        self._update_title()
        self._update_center()

    # ------------------------------------------------------------------ #
    # Construcción de interfaz
    # ------------------------------------------------------------------ #
    def _scrollable(self, widget: QWidget) -> QScrollArea:
        """Envuelve un panel en un área desplazable sin marco."""
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.Shape.NoFrame)
        sa.setWidget(widget)
        return sa

    # ------------------------------------------------------------------ #
    # Pantalla de bienvenida (cuando no hay proyecto abierto)
    # ------------------------------------------------------------------ #
    def _build_welcome(self) -> QWidget:
        from . import theme

        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(40, 40, 40, 40)
        outer.addStretch(1)

        card = QWidget()
        card.setMaximumWidth(560)
        cl = QVBoxLayout(card)
        cl.setSpacing(10)

        title = QLabel(APP_NAME)
        title.setStyleSheet(f"font-size: 30px; font-weight: 700; color: {theme.TEXT};")
        subtitle = QLabel("Análisis, preprocesamiento y clasificación de señales EEG")
        subtitle.setStyleSheet(f"font-size: 13px; color: {theme.MUTED};")
        cl.addWidget(title)
        cl.addWidget(subtitle)
        cl.addSpacing(10)

        btn_row = QHBoxLayout()
        btn_new = QPushButton("  Nuevo proyecto")
        btn_new.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        btn_new.setMinimumHeight(40)
        btn_new.setDefault(True)
        btn_new.setStyleSheet(
            f"QPushButton {{ background: {theme.ACCENT}; color: #ffffff; font-weight: 600;"
            f" border: none; border-radius: 6px; padding: 8px 16px; }}"
            f"QPushButton:hover {{ background: {theme.ACCENT_HI}; }}")
        btn_new.clicked.connect(self.new_project)
        btn_open = QPushButton("  Abrir proyecto…")
        btn_open.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        btn_open.setMinimumHeight(40)
        btn_open.clicked.connect(self.open_project)
        btn_row.addWidget(btn_new)
        btn_row.addWidget(btn_open)
        btn_row.addStretch(1)
        cl.addLayout(btn_row)
        cl.addSpacing(14)

        rec_title = QLabel("Proyectos recientes")
        rec_title.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {theme.MUTED};")
        cl.addWidget(rec_title)
        self.welcome_recent = QListWidget()
        self.welcome_recent.setMinimumHeight(150)
        self.welcome_recent.itemActivated.connect(self._on_welcome_recent)
        self.welcome_recent.itemClicked.connect(self._on_welcome_recent)
        cl.addWidget(self.welcome_recent)

        center = QHBoxLayout()
        center.addStretch(1)
        center.addWidget(card)
        center.addStretch(1)
        outer.addLayout(center)
        outer.addStretch(2)
        return page

    def _refresh_welcome_recents(self) -> None:
        self.welcome_recent.clear()
        recent = self._recent_projects()
        if not recent:
            it = QListWidgetItem("Aún no hay proyectos recientes. Crea o abre uno.")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.welcome_recent.addItem(it)
            return
        for path in recent:
            exists = os.path.isfile(os.path.join(path, "project.json"))
            name = os.path.basename(path.rstrip("/\\"))
            it = QListWidgetItem(f"{name}\n{path}")
            it.setData(Qt.ItemDataRole.UserRole, path)
            if not exists:
                it.setText(f"{name}  (no encontrado)\n{path}")
                it.setForeground(QColor("#727a83"))
                it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.welcome_recent.addItem(it)

    def _on_welcome_recent(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self._open_project_path(path)

    def _update_center(self) -> None:
        """Muestra la bienvenida si no hay proyecto, o el área de trabajo si lo hay.

        Sin proyecto se ocultan los paneles laterales para una bienvenida limpia.
        """
        docks = (self.src_dock, self.right_dock, self.log_dock)
        if self.project is None:
            for dock in docks:
                dock.setVisible(False)
            self._refresh_welcome_recents()
            self.center_stack.setCurrentWidget(self.welcome)
        else:
            # Solo al venir de la bienvenida se muestran los paneles; después el
            # usuario controla su visibilidad desde el menú «Ver» sin que se fuerce.
            if self.center_stack.currentWidget() is self.welcome:
                for dock in docks:
                    dock.setVisible(True)
            self.center_stack.setCurrentWidget(self.center_tabs)

    def _build_docks(self) -> None:
        # Permite arrastrar los paneles a cualquier borde y anidarlos/dividir áreas
        # (más libertad para acomodar la disposición). "Ver → Restaurar paneles"
        # deja todo en su sitio de nuevo.
        self.setDockNestingEnabled(True)

        # Izquierda: fuentes (CSV) del proyecto.
        #   Orden guardado entre sesiones (alfabético / fechas / propio) y un
        #   indicador de si cada archivo tiene segmentos (✂) o marcadores (⚑).
        self._source_sort = str(self._settings().value("source_sort", "custom"))
        if self._source_sort not in self._SORT_MODES:
            self._source_sort = "custom"
        self._src_event_counts: dict[str, int] = {}   # caché de nº de marcadores
        self._reordering = False
        self._scanning_markers = False
        self._scan_project = None
        self._markers_scanned.connect(self._on_markers_scanned)

        self.sources_list = _SourceListWidget()
        self.sources_list.setObjectName("sourcesList")
        self.sources_list.setItemDelegate(_SourceItemDelegate(self.sources_list))
        self.sources_list.setAlternatingRowColors(False)   # plano, estilo PyCharm
        self.sources_list.setFrameShape(QFrame.Shape.NoFrame)
        self.sources_list.setUniformItemSizes(True)
        self.sources_list.currentRowChanged.connect(self._on_source_selected)
        # Renombrar con clic izquierdo sobre la señal ya seleccionada (o F2):
        # edición en el sitio. «Abrir en ventana nueva» pasa al menú contextual.
        self.sources_list.setEditTriggers(
            QListWidget.EditTrigger.SelectedClicked
            | QListWidget.EditTrigger.DoubleClicked
            | QListWidget.EditTrigger.EditKeyPressed)
        self.sources_list.itemChanged.connect(self._on_source_renamed)
        self.sources_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.sources_list.customContextMenuRequested.connect(self._on_sources_menu)
        # Reordenar arrastrando (solo tiene efecto en modo «orden propio»).
        self.sources_list.reordered.connect(self._on_sources_reordered)

        # Cabecera: selector de orden (barra discreta sobre la lista).
        self.source_sort_combo = QComboBox()
        self.source_sort_combo.setObjectName("sourcesSortCombo")
        for key, label in self._SORT_MODES.items():
            self.source_sort_combo.addItem(label, key)
        self.source_sort_combo.setCurrentIndex(
            self.source_sort_combo.findData(self._source_sort))
        self.source_sort_combo.setToolTip("Orden de la lista de fuentes.")
        self.source_sort_combo.currentIndexChanged.connect(self._on_source_sort_changed)

        sort_label = QLabel("ORDEN")
        sort_label.setObjectName("sourcesSortLabel")
        sort_bar = QWidget()
        sort_bar.setObjectName("sourcesHeader")
        sort_lay = QHBoxLayout(sort_bar)
        sort_lay.setContentsMargins(8, 4, 6, 4)
        sort_lay.setSpacing(6)
        sort_lay.addWidget(sort_label)
        sort_lay.addWidget(self.source_sort_combo, 1)

        src_container = QWidget()
        src_container.setObjectName("sourcesPanel")
        src_layout = QVBoxLayout(src_container)
        src_layout.setContentsMargins(0, 0, 0, 0)
        src_layout.setSpacing(0)
        src_layout.addWidget(sort_bar)
        src_layout.addWidget(self.sources_list, 1)
        src_container.setStyleSheet(self._sources_panel_qss())

        self.src_dock = QDockWidget("Fuentes (CSV)", self)
        self.src_dock.setWidget(src_container)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.src_dock)
        self._apply_source_drag_mode()

        # Derecha: pestañas de procesamiento.
        self.preproc_panel = PreprocessingPanel(self)
        self.dataset_panel = DatasetPanel(self)
        self.clf_panel = ClassificationPanel(self)
        self.acq_panel = AcquisitionPanel(self)
        self.control_panel = ControlPanel(self)
        tabs = QTabWidget()
        # Cada panel va en un área desplazable: así su contenido no fuerza un
        # tamaño mínimo de ventana enorme (evita el aviso de geometría en 1080p).
        tabs.addTab(self._scrollable(self.acq_panel), "Tiempo real")
        tabs.addTab(self._scrollable(self.preproc_panel), "Preprocesamiento")
        tabs.addTab(self._scrollable(self.dataset_panel), "Dataset")
        tabs.addTab(self._scrollable(self.clf_panel), "Clasificación")
        tabs.addTab(self._scrollable(self.control_panel), "Control")
        self.right_dock = QDockWidget("Herramientas", self)
        self.right_dock.setWidget(tabs)
        self.right_dock.setMinimumWidth(340)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.right_dock)

        # Abajo: historial de cambios navegable, como ÁRBOL colapsable.
        hist_container = QWidget()
        hist_layout = QVBoxLayout(hist_container)
        hist_layout.setContentsMargins(6, 4, 6, 6)
        hrow = QHBoxLayout()
        hint = QLabel("Árbol de cambios · clic para navegar · ▶ actual · ⑂ bifurcación")
        hint.setStyleSheet("color: #8a929b; font-size: 11px;")
        hrow.addWidget(hint, 1)
        collapse_btn = QPushButton("Colapsar ramas")
        collapse_btn.setToolTip("Contrae las ramas para ver solo la estructura.")
        collapse_btn.clicked.connect(lambda: self.changelog_tree.collapseAll())
        expand_btn = QPushButton("Expandir")
        expand_btn.clicked.connect(lambda: self.changelog_tree.expandAll())
        for b in (collapse_btn, expand_btn):
            b.setStyleSheet("padding: 1px 8px; font-size: 11px;")
            hrow.addWidget(b)
        hist_layout.addLayout(hrow)
        self.changelog_tree = QTreeWidget()
        self.changelog_tree.setHeaderHidden(True)
        self.changelog_tree.setIndentation(16)
        self.changelog_tree.setUniformRowHeights(True)
        self.changelog_tree.setStyleSheet(
            "QTreeWidget { border: none; background: #14181d; }"
            "QTreeWidget::item { padding: 2px 2px; }"
            "QTreeWidget::item:hover { background: #243042; }"
            "QTreeWidget::item:selected { background: #24402f; color: #eaf6ef; }"
        )
        self.changelog_tree.itemClicked.connect(self._on_history_click)
        hist_layout.addWidget(self.changelog_tree)
        self.log_dock = QDockWidget("Historial", self)
        self.log_dock.setWidget(hist_container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

    def _restore_docks(self) -> None:
        """Vuelve a mostrar y recolocar todos los paneles (Ver → Restaurar paneles)."""
        for dock, area in ((self.src_dock, Qt.DockWidgetArea.LeftDockWidgetArea),
                           (self.right_dock, Qt.DockWidgetArea.RightDockWidgetArea),
                           (self.log_dock, Qt.DockWidgetArea.BottomDockWidgetArea)):
            dock.setFloating(False)
            self.addDockWidget(area, dock)
            dock.show()

    def _build_menu(self) -> None:
        bar = self.menuBar()

        m_proj = bar.addMenu("&Proyecto")
        self.act_new = QAction("Nuevo proyecto…", self, triggered=self.new_project)
        self.act_open = QAction("Abrir proyecto…", self, triggered=self.open_project)
        self.act_save = QAction("Guardar proyecto", self, triggered=self.save_project)
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_open_folder = QAction("Abrir carpeta del proyecto", self,
                                       triggered=self.open_project_folder)
        self.act_open_folder.setToolTip("Abre la carpeta .eegproj en el explorador de archivos.")
        self.act_export_cfg = QAction("Exportar configuración/bundle…", self,
                                      triggered=self.export_config)
        self.act_export_cfg.setToolTip("Exporta preprocesamiento, dataset (.npz) y modelos "
                                       "(.joblib) a un archivo .eegbundle autónomo.")
        self.act_import_cfg = QAction("Importar configuración/bundle…", self,
                                      triggered=self.import_config)
        self.act_import_cfg.setToolTip("Carga un .eegbundle: aplica el pipeline y trae dataset "
                                       "y modelos ya entrenados.")
        self.act_add = QAction("Añadir o importar señal…", self,
                               triggered=self.add_or_import_source)
        self.act_add.setToolTip("Añade CSV o importa datasets (.mat / .fif / .edf / .gdf…) "
                                "en un solo paso.")
        # Opción: al importar .mat (BCI 2a), excluir los EOG por defecto (las
        # etiquetas se conservan). Persistente entre sesiones.
        self.act_exclude_eog = QAction("Excluir EOG al importar .mat", self, checkable=True)
        self.act_exclude_eog.setChecked(
            self._settings().value("exclude_eog_on_mat", True, type=bool))
        self.act_exclude_eog.setToolTip("Recomendado: marca los canales EOG como "
                                        "excluidos del análisis. No borra nada del CSV "
                                        "y las etiquetas se mantienen.")
        self.act_exclude_eog.toggled.connect(
            lambda v: self._settings().setValue("exclude_eog_on_mat", bool(v)))
        self.act_compress = QAction("Comprimir fuentes a .csv.gz…", self,
                                    triggered=self.compress_sources)
        self.act_del_src = QAction("Quitar fuente del proyecto…", self,
                                   triggered=self.remove_current_source)
        for a in (self.act_new, self.act_open):
            m_proj.addAction(a)
        self.recent_menu = m_proj.addMenu("Abrir reciente")
        self.recent_menu.aboutToShow.connect(self._build_recent_menu)
        m_proj.addAction(self.act_save)
        m_proj.addAction(self.act_open_folder)
        m_proj.addSeparator()
        m_proj.addAction(self.act_add)
        m_proj.addAction(self.act_exclude_eog)
        m_proj.addAction(self.act_compress)
        m_proj.addAction(self.act_del_src)
        m_proj.addSeparator()
        m_proj.addAction(self.act_export_cfg)
        m_proj.addAction(self.act_import_cfg)
        m_proj.addSeparator()
        m_proj.addAction(QAction("Salir", self, triggered=self.close))

        m_edit = bar.addMenu("&Editar")
        self.act_undo = QAction("Deshacer", self, triggered=self.undo)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_redo = QAction("Rehacer", self, triggered=self.redo)
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        m_edit.addAction(self.act_undo)
        m_edit.addAction(self.act_redo)

        # Menú Ver: reabrir/ocultar los paneles (soluciona que no se pudieran reabrir).
        m_view = bar.addMenu("&Ver")
        for dock, label in ((self.src_dock, "Fuentes (CSV)"),
                            (self.right_dock, "Herramientas"),
                            (self.log_dock, "Historial")):
            act = dock.toggleViewAction()
            act.setText(label)
            m_view.addAction(act)
        m_view.addSeparator()
        m_view.addAction(QAction("Restaurar paneles", self,
                                 triggered=self._restore_docks))

        m_help = bar.addMenu("A&yuda")
        m_help.addAction(QAction("Acerca de", self, triggered=self._about))

        # Iconos estándar del estilo (sin archivos externos) para menú y barra.
        st = self.style()
        self.act_new.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        self.act_open.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.act_save.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.act_open_folder.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        self.act_add.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        self.act_undo.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        self.act_redo.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_ArrowForward))

    def _build_toolbar(self) -> None:
        tb = QToolBar("Principal", self)
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        # La misma QAction sirve en el menú y en la barra (Qt comparte su estado).
        for group in ((self.act_new, self.act_open, self.act_save),
                      (self.act_add,),
                      (self.act_undo, self.act_redo)):
            for a in group:
                tb.addAction(a)
            tb.addSeparator()
        self.addToolBar(tb)

    def _update_title(self) -> None:
        if self.project is None:
            self.setWindowTitle(APP_NAME)
        else:
            dot = "● " if self._dirty else ""        # ● = cambios sin guardar
            self.setWindowTitle(f"{dot}{self.project.name} — {APP_NAME}")

    def _build_statusbar(self) -> None:
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(320)
        self.progress.setFormat("%p%")
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Listo. Crea o abre un proyecto para empezar.")

    # ------------------------------------------------------------------ #
    # Proyecto
    # ------------------------------------------------------------------ #
    def new_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Carpeta donde crear el proyecto")
        if not folder:
            return
        name, ok = QInputDialog.getText(self, "Nuevo proyecto", "Nombre del proyecto:")
        if not ok or not name.strip():
            return
        self.project = Project.create(folder, name.strip())
        self.current_source_id = None
        self._reset_source_tabs()
        self.dataset = None
        self._dirty = False
        self._load_project_models()
        self._push_recent(self.project.path)
        self.refresh_all()
        self.statusBar().showMessage(f"Proyecto creado: {self.project.path}")
        # Ofrecer arrancar desde un bundle existente (pipeline + dataset + modelos).
        if QMessageBox.question(
                self, "Importar bundle",
                "Proyecto creado. ¿Quieres importar un bundle existente (.eegbundle) "
                "para traer pipeline, dataset y modelos ya entrenados?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) == QMessageBox.StandardButton.Yes:
            self.import_config()

    def open_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, f"Abrir proyecto ({PROJECT_EXT})")
        self._open_project_path(path)

    def _open_project_path(self, path: str) -> None:
        """Abre un proyecto por ruta (usado por «Abrir…» y por «Abrir reciente»)."""
        if not path:
            return
        if not os.path.isfile(os.path.join(path, "project.json")):
            QMessageBox.warning(self, "Proyecto inválido", "La carpeta no contiene project.json.")
            self._forget_recent(path)
            return
        try:
            self.project = Project.open(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error al abrir", str(exc))
            return
        self.current_source_id = None
        self._reset_source_tabs()
        self.dataset = None
        self._dirty = False
        self._load_project_models()
        self._push_recent(self.project.path)
        self.refresh_all()
        n = len(self.models)
        self.statusBar().showMessage(
            f"Proyecto abierto: {self.project.path}"
            + (f"  ·  {n} modelo(s) cargado(s)" if n else ""))
        self._offer_orphan_recordings()

    def _offer_orphan_recordings(self) -> None:
        """Si hay grabaciones en recordings/ sin añadir, ofrece incorporarlas."""
        if self.project is None:
            return
        orphans = self.project.orphan_recordings()
        if not orphans:
            return
        res = QMessageBox.question(
            self, "Grabaciones sin añadir",
            f"Hay {len(orphans)} grabación(es) en la carpeta «recordings/» que no están "
            f"en el proyecto (p. ej. de una sesión anterior).\n\n¿Añadirlas ahora?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if res == QMessageBox.StandardButton.Yes:
            self.add_orphan_recordings(orphans)

    def add_orphan_recordings(self, paths=None) -> int:
        """Añade como fuentes las grabaciones sueltas de recordings/. Devuelve cuántas."""
        if not self._require_project():
            return 0
        paths = paths if paths is not None else self.project.orphan_recordings()
        from ..core import marks_sidecar
        added, n_seg = 0, 0
        for p in paths:
            try:
                src = self.project.add_source(p)
                for start, stop, label in marks_sidecar.read_marks(p):   # restaura marcas
                    self.project.add_segment(src["id"], start, stop, label)
                    n_seg += 1
                added += 1
            except Exception:  # noqa: BLE001
                pass
        if added:
            self.refresh_all()
            self._persist_now()
            extra = f" · {n_seg} segmento(s) recuperados" if n_seg else ""
            self.statusBar().showMessage(f"{added} grabación(es) añadida(s){extra}.")
        return added

    # ------------------------------------------------------------------ #
    # Proyectos recientes (persisten entre sesiones vía QSettings)
    # ------------------------------------------------------------------ #
    def _settings(self) -> QSettings:
        return QSettings(ORG_NAME, APP_NAME)

    def _recent_projects(self) -> list[str]:
        val = self._settings().value("recent_projects", [])
        if isinstance(val, str):                 # QSettings puede devolver str si hay 1
            val = [val]
        return [p for p in (val or []) if p]

    def _push_recent(self, path: str) -> None:
        path = os.path.abspath(path)
        recent = [p for p in self._recent_projects() if os.path.abspath(p) != path]
        recent.insert(0, path)
        self._settings().setValue("recent_projects", recent[:8])

    def _forget_recent(self, path: str) -> None:
        path = os.path.abspath(path)
        recent = [p for p in self._recent_projects() if os.path.abspath(p) != path]
        self._settings().setValue("recent_projects", recent)

    def _build_recent_menu(self) -> None:
        self.recent_menu.clear()
        recent = self._recent_projects()
        if not recent:
            self.recent_menu.addAction(QAction("(sin proyectos recientes)", self,
                                               enabled=False))
            return
        for path in recent:
            exists = os.path.isfile(os.path.join(path, "project.json"))
            label = os.path.basename(path.rstrip("/\\")) + ("" if exists else "  (no encontrado)")
            act = QAction(label, self)
            act.setToolTip(path)
            act.setEnabled(exists)
            act.triggered.connect(lambda _=False, p=path: self._open_project_path(p))
            self.recent_menu.addAction(act)
        self.recent_menu.addSeparator()
        self.recent_menu.addAction(QAction("Vaciar lista", self,
                                           triggered=self._clear_recent))

    def _clear_recent(self) -> None:
        self._settings().setValue("recent_projects", [])

    def remove_current_source(self) -> None:
        """Quita la fuente seleccionada del proyecto; si su archivo está DENTRO del
        proyecto, ofrece borrarlo del disco. Nunca toca la carpeta de origen."""
        if not self._require_project() or self.current_source_id is None:
            self.info("Sin fuente", "Selecciona una fuente en el panel izquierdo.")
            return
        sid = self.current_source_id
        src = self.project.get_source(sid)
        if src is None:
            return
        internal = self.project.is_internal_path(src["path"])
        on_disk = os.path.isfile(src["path"])
        box = QMessageBox(self)
        box.setWindowTitle("Quitar fuente del proyecto")
        box.setIcon(QMessageBox.Icon.Question)
        text = f"¿Quitar «{src['alias']}» del proyecto?"
        if internal and on_disk:
            text += ("\n\nEl archivo está DENTRO del proyecto:\n"
                     f"{src['path']}\n\nMarca la casilla para borrarlo también del disco.")
        elif on_disk:
            text += "\n\nEl archivo de origen NO se modifica (solo se quita la referencia)."
        box.setText(text)
        del_chk = None
        if internal and on_disk:
            del_chk = QCheckBox("Borrar también el archivo del disco")
            box.setCheckBox(del_chk)
        box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() != QMessageBox.StandardButton.Yes:
            return
        if del_chk is not None and del_chk.isChecked():
            try:
                os.remove(src["path"])
            except OSError as exc:
                QMessageBox.warning(self, "No se pudo borrar", str(exc))
        self.project.remove_source(sid)
        self.current_source_id = None
        self._after_state_change()
        self.statusBar().showMessage(f"Fuente «{src['alias']}» quitada del proyecto.")

    def save_project(self) -> None:
        if self.project is None:
            return
        self._autosave_timer.stop()
        self.project.save()
        self._set_dirty(False)
        self.statusBar().showMessage("Proyecto guardado.")

    def open_project_folder(self) -> None:
        """Abre la carpeta del proyecto en el explorador de archivos del sistema."""
        if not self._require_project():
            return
        path = self.project.path
        if not os.path.isdir(path):
            self.warn("Carpeta no encontrada", f"No existe la carpeta:\n{path}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            self.warn("No se pudo abrir", f"No se pudo abrir la carpeta:\n{path}")

    def _ask_export_sections(self):
        """Diálogo con casillas para elegir qué exportar (incl. qué pipelines).

        Devuelve ``(sections, pipeline_indices)`` o ``None``. ``pipeline_indices``
        es la lista de pipelines elegidos (o ``None`` si no se exporta el
        preprocesamiento)."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Exportar configuración")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Selecciona qué exportar al bundle:"))
        chk_pre = QCheckBox("Preprocesamiento (pipelines, canales excluidos, alias)")
        chk_ds = QCheckBox("Dataset (características, segmentos etiquetados, recortes)")
        chk_mdl = QCheckBox(f"Modelos de clasificación ({len(self.models)})")
        n_src = len(self.project.sources) if self.project else 0
        chk_src = QCheckBox(f"Señales de origen — CSV ({n_src})")
        chk_pre.setChecked(True)
        chk_ds.setChecked(True)
        chk_mdl.setChecked(bool(self.models))
        chk_mdl.setEnabled(bool(self.models))
        chk_src.setEnabled(n_src > 0)                # opcional: aumenta el tamaño
        lay.addWidget(chk_pre)

        # Selección de qué PIPELINES incluir (con un selector global «Todas»).
        pls = self.project.pipelines() if self.project else []
        pl_box = QGroupBox("Pipelines a incluir")
        pl_lay = QVBoxLayout(pl_box)
        chk_all = QCheckBox("Todas las pipelines")
        chk_all.setTristate(True)
        chk_all.setCheckState(Qt.CheckState.Checked)
        pl_lay.addWidget(chk_all)
        pipe_checks: list[QCheckBox] = []
        active_i = self.project.active_pipeline_index() if self.project else 0
        for i, pl in enumerate(pls):
            star = "  ★ (activo)" if i == active_i else ""
            c = QCheckBox(f"{pl['name']}{star}")
            c.setChecked(True)
            pl_lay.addWidget(c)
            pipe_checks.append(c)

        def _apply_all():                            # «Todas» → marca/desmarca todas
            checked = chk_all.checkState() != Qt.CheckState.Unchecked
            for c in pipe_checks:
                c.blockSignals(True)
                c.setChecked(checked)
                c.blockSignals(False)

        def _sync_all():                             # refleja el estado global (tri-estado)
            n = sum(c.isChecked() for c in pipe_checks)
            chk_all.blockSignals(True)
            chk_all.setCheckState(Qt.CheckState.Checked if n == len(pipe_checks)
                                  else Qt.CheckState.Unchecked if n == 0
                                  else Qt.CheckState.PartiallyChecked)
            chk_all.blockSignals(False)

        chk_all.clicked.connect(lambda *_: (_apply_all(), _sync_all()))
        for c in pipe_checks:
            c.stateChanged.connect(lambda *_: _sync_all())
        chk_pre.toggled.connect(pl_box.setEnabled)
        lay.addWidget(pl_box)

        for c in (chk_ds, chk_mdl, chk_src):
            lay.addWidget(c)
        hint = QLabel("El bundle NO incluye la caché (regenerable), por lo que suele "
                      "pesar menos que la carpeta del proyecto aunque incluyas las señales.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8a929b; font-size: 11px;")
        lay.addWidget(hint)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        sections = set()
        pipeline_indices = None
        if chk_pre.isChecked():
            sections.add("preprocessing")
            pipeline_indices = [i for i, c in enumerate(pipe_checks) if c.isChecked()]
        if chk_ds.isChecked():
            sections.add("dataset")
        if chk_mdl.isChecked():
            sections.add("models")
        if chk_src.isChecked():
            sections.add("sources")
        if not sections:
            return None
        return sections, pipeline_indices

    def export_config(self) -> None:
        """Exporta preprocesamiento/dataset(.npz)/modelos(.joblib) a un .eegbundle."""
        if not self._require_project():
            return
        choice = self._ask_export_sections()
        if not choice:
            return
        sections, pipeline_indices = choice
        from ..core import config_export
        # Si se pide el dataset y hay uno en memoria sin guardar, se guarda primero.
        if "dataset" in sections and self.dataset is not None:
            ds_dir = os.path.join(self.project.path, DATASETS_DIR)
            has_npz = os.path.isdir(ds_dir) and any(f.endswith(".npz") for f in os.listdir(ds_dir))
            if not has_npz:
                dataset_mod.save_dataset(self.project, self.dataset, "dataset")
        default = os.path.join(self.project.path,
                               f"{self.project.name}{config_export.BUNDLE_EXT}")
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar bundle", default,
            f"Bundle EEG (*{config_export.BUNDLE_EXT})")
        if not path:
            return
        self._busy("Exportando bundle…")
        self.progress.setRange(0, 0)
        self.progress.show()

        proj_size = self._dir_size(self.project.path)

        def task():
            return config_export.export_bundle(self.project, self.models, sections, path,
                                               pipeline_indices)

        def done(info):
            self._idle()
            mb = info["size"] / 1e6
            cmp = (f"  ·  {mb:.1f} MB (proyecto: {proj_size / 1e6:.1f} MB)"
                   if proj_size else f"  ·  {mb:.1f} MB")
            self.statusBar().showMessage(
                f"Bundle exportado: {info['models']} modelo(s), {info['datasets']} "
                f"dataset(s), {info.get('sources', 0)} fuente(s){cmp}")
            # Aviso si, pese a omitir la caché, el bundle pesa más que el proyecto.
            if proj_size and info["size"] > proj_size:
                extra = ("\n\nSugerencia: vuelve a exportar SIN marcar «Señales de "
                         "origen» para un archivo más ligero."
                         if "sources" in sections else "")
                QMessageBox.information(
                    self, "El bundle resultó grande",
                    f"El bundle ({mb:.1f} MB) es MÁS grande que la carpeta del "
                    f"proyecto ({proj_size / 1e6:.1f} MB).{extra}")

        self._spawn(task, done)

    @staticmethod
    def _dir_size(path: str) -> int:
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    def import_config(self) -> None:
        """Importa un .eegbundle: aplica pipeline y carga dataset(s) y modelos."""
        if not self._require_project():
            return
        from ..core import config_export
        path, _ = QFileDialog.getOpenFileName(
            self, "Importar bundle/configuración", self.project.path,
            f"Bundle EEG (*{config_export.BUNDLE_EXT});;Configuración EEG (*.eegcfg *.json)")
        if not path:
            return
        try:
            if path.lower().endswith(config_export.BUNDLE_EXT):
                cfg, model_blobs, ds_blobs, src_blobs = config_export.read_bundle(path)
            else:
                cfg, model_blobs, ds_blobs, src_blobs = config_export.load_config(path), {}, {}, {}
        except Exception as exc:  # noqa: BLE001
            self.warn("No se pudo leer", f"{path}\n\n{exc}")
            return
        summary = self._apply_config(cfg, model_blobs, ds_blobs, src_blobs)
        self._after_state_change()
        self.statusBar().showMessage(f"Importado: {summary}")

    def _apply_config(self, cfg: dict, model_blobs: dict, ds_blobs: dict,
                      src_blobs: dict | None = None) -> str:
        """Aplica una configuración/bundle al proyecto actual. Devuelve un resumen."""
        parts: list[str] = []
        # Fuentes primero (conservando su id) para que los segmentos sigan válidos.
        srcs = cfg.get("sources")
        if srcs and src_blobs:
            imp = os.path.join(self.project.path, IMPORTED_DIR)
            os.makedirs(imp, exist_ok=True)
            n_src = 0
            for meta in srcs:
                arc = meta.get("file")
                if not arc or arc not in src_blobs:
                    continue
                dest = os.path.join(imp, os.path.basename(arc))
                with open(dest, "wb") as f:
                    f.write(src_blobs[arc])
                try:
                    rec = load_recording(dest)
                except Exception:  # noqa: BLE001
                    rec = None
                try:
                    self.project.add_source(dest, alias=meta.get("alias"),
                                            recording=rec, source_id=meta.get("id"))
                    n_src += 1
                except Exception:  # noqa: BLE001
                    pass
            if n_src:
                parts.append(f"{n_src} fuente(s)")
        pre = cfg.get("preprocessing")
        if pre:
            if pre.get("pipelines"):     # bundles nuevos: todos los pipelines
                self.project.set_pipelines(pre["pipelines"], pre.get("active_pipeline", 0),
                                           "Importar pipelines")
            else:                        # bundles antiguos: un solo pipeline
                self.project.set_active_pipeline_steps(pre.get("pipeline", []), "Importar pipeline")
            self.project.edit("excluded_channels", pre.get("excluded_channels", []),
                              "Importar canales excluidos")
            if pre.get("channel_aliases"):
                self.project.edit("channel_aliases", pre["channel_aliases"],
                                  "Importar alias de canal")
            parts.append("preprocesamiento")
        ds = cfg.get("dataset")
        if ds:
            if ds.get("config"):
                self.project.edit("dataset", ds["config"], "Importar config de dataset")
            if ds.get("segments"):
                self.project.edit("segments", ds["segments"], "Importar segmentos")
            if ds.get("cuts"):
                self.project.edit("cuts", ds["cuts"], "Importar recortes")
            if ds_blobs:
                out_dir = os.path.join(self.project.path, DATASETS_DIR)
                os.makedirs(out_dir, exist_ok=True)
                for fname, data in ds_blobs.items():
                    with open(os.path.join(out_dir, fname), "wb") as f:
                        f.write(data)
                try:                                    # carga el primero en memoria
                    self.dataset = dataset_mod.load_dataset(
                        os.path.join(out_dir, sorted(ds_blobs)[0]))
                except Exception:  # noqa: BLE001
                    pass
            parts.append(f"dataset ({len(ds_blobs)} archivo(s))")
        if model_blobs:
            for name, data in model_blobs.items():
                try:
                    res = classification.result_from_bytes(data)
                    self._register_model(res, name=name, persist=True)
                except Exception:  # noqa: BLE001
                    pass
            parts.append(f"{len(model_blobs)} modelo(s)")
        self.clf_panel.refresh()
        return ", ".join(parts) or "nada"

    def request_autosave(self) -> None:
        """Programa un guardado automático poco después del último cambio."""
        if self.project is not None:
            self._set_dirty(True)
            self._autosave_timer.start()

    def _set_dirty(self, dirty: bool) -> None:
        if dirty != self._dirty:
            self._dirty = dirty
            self._update_title()

    def _autosave(self) -> None:
        if self.project is None:
            return
        try:
            self.project.save()
            self._set_dirty(False)
            self.statusBar().showMessage("Guardado automáticamente.", 1500)
        except Exception as exc:  # noqa: BLE001
            # No se rinde: reintenta (deja el proyecto marcado como pendiente).
            self.statusBar().showMessage(f"No se pudo autoguardar (reintentando)…: {exc}", 3000)
            self._autosave_timer.start()

    def _auto_exclude_eog(self) -> int:
        """Marca como excluidos los canales EOG de todas las fuentes (no destructivo).

        No borra nada del CSV ni toca los marcadores/etiquetas: solo añade los
        nombres EOG a ``excluded_channels`` para que no entren en CAR, características
        ni modelos. Devuelve cuántos canales nuevos se excluyeron.
        """
        if self.project is None:
            return 0
        excluded = set(self.project.excluded_channels())
        found: set[str] = set()
        for src in self.project.sources:
            try:
                rec = self.project.get_recording(src["id"])
            except Exception:  # noqa: BLE001 — fuente no disponible
                continue
            for n in rec.channel_names:
                if str(n).strip().upper().startswith("EOG"):
                    found.add(n)
        new_excl = found - excluded
        if new_excl:
            self.project.edit("excluded_channels", sorted(excluded | found),
                              "Excluir canales EOG")
        return len(new_excl)

    def add_or_import_source(self) -> None:
        """Añade fuentes CSV **o** importa datasets (.mat/.fif/.edf…) en un solo paso.

        Los ``.csv``/``.csv.gz`` se referencian tal cual; el resto se convierte a
        ``.csv.gz`` dentro del proyecto (carpeta ``imported/``) sin tocar el origen.
        """
        if not self._require_project():
            return
        mne_exts = " ".join(f"*{e}" for e in mne_loader.supported_extensions())
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Añadir o importar señal EEG", "",
            f"Señal EEG (*.csv *.csv.gz *.mat {mne_exts});;"
            f"CSV de OpenViBE (*.csv *.csv.gz);;MATLAB BNCI 2a (*.mat);;"
            f"MNE / FIF / EDF / GDF ({mne_exts})")
        if not paths:
            return
        has_mat = any(p.lower().endswith(".mat") for p in paths)
        self._busy("Añadiendo / convirtiendo señal…")
        self.progress.setRange(0, 0)
        self.progress.show()

        # Las conversiones se guardan DENTRO del proyecto (no se toca el origen).
        imported_dir = os.path.join(self.project.path, IMPORTED_DIR)
        os.makedirs(imported_dir, exist_ok=True)

        def task():
            out = []
            for p in paths:
                low = p.lower()
                if low.endswith(".csv") or low.endswith(".csv.gz"):
                    csv = p                             # CSV: se referencia directo
                else:
                    base = os.path.splitext(os.path.basename(p))[0]
                    gz = os.path.join(imported_dir, base + ".csv.gz")
                    if os.path.isfile(gz):              # reutiliza si ya se importó
                        csv = gz
                    elif os.path.splitext(p)[1].lower() == ".mat":
                        csv = mat_loader.convert_bnci_mat(p, gz)
                    else:
                        csv = mne_loader.convert_with_mne(p, gz)
                # Carga la grabación AQUÍ (en el hilo) para no bloquear la GUI al añadir.
                try:
                    rec = load_recording(csv)
                except Exception:  # noqa: BLE001 — se validará/reintentará en add_source
                    rec = None
                out.append((csv, rec))
            return out

        def done(results):
            added = 0
            for c, rec in results:
                try:
                    self.project.add_source(c, recording=rec)
                    added += 1
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.warning(self, "No se pudo añadir", f"{c}\n\n{exc}")
            msg = f"{added} fuente(s) añadida(s) al proyecto."
            # Al importar .mat (BCI 2a): excluir EOG por defecto, conservando etiquetas.
            if has_mat and self.act_exclude_eog.isChecked():
                n_eog = self._auto_exclude_eog()
                if n_eog:
                    msg += f" {n_eog} canal(es) EOG excluido(s) (las etiquetas se conservan)."
            self._idle()
            self.refresh_all()
            self._update_signal_view()
            self.request_autosave()
            self.statusBar().showMessage(msg)

        self._spawn(task, done)

    def compress_sources(self) -> None:
        """Recomprime las fuentes en .csv plano a .csv.gz para ahorrar espacio."""
        if not self._require_project():
            return
        plain = [s for s in self.project.sources
                 if s["path"].lower().endswith(".csv") and os.path.isfile(s["path"])]
        if not plain:
            self.info("Nada que comprimir",
                      "No hay fuentes en .csv plano (ya están comprimidas o no existen).")
            return
        total = sum(os.path.getsize(s["path"]) for s in plain)
        res = QMessageBox.question(
            self, "Comprimir fuentes",
            f"Se comprimirán {len(plain)} fuente(s) (~{total / 1e6:.1f} MB) a .csv.gz.\n"
            "¿Eliminar los .csv originales tras comprimir?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel)
        if res == QMessageBox.StandardButton.Cancel:
            return
        delete = res == QMessageBox.StandardButton.Yes

        self._busy("Comprimiendo fuentes…")
        self.progress.setRange(0, 0)
        self.progress.show()
        ids = [s["id"] for s in plain]
        paths = {s["id"]: s["path"] for s in plain}

        def task():
            results = []
            for sid in ids:
                old = paths[sid]
                new = compress_csv(old)        # old + ".gz"
                results.append((sid, old, new, os.path.getsize(old), os.path.getsize(new)))
            return results

        def done(results):
            saved = 0
            for sid, old, new, osz, nsz in results:
                self.project.set_source_path(sid, new, "Comprimir fuente a .csv.gz")
                saved += osz - nsz
                if delete:
                    try:
                        os.remove(old)
                    except OSError:
                        pass
            self._idle()
            self.refresh_all()
            self._update_signal_view()
            self.request_autosave()
            self.info("Comprimido",
                      f"{len(results)} fuente(s) comprimida(s).\n"
                      f"Ahorro: {saved / 1e6:.1f} MB"
                      + ("\nOriginales eliminados." if delete else "\nOriginales conservados."))

        self._spawn(task, done)

    # ------------------------------------------------------------------ #
    # Fuentes / visor
    # ------------------------------------------------------------------ #
    @property
    def signal_view(self) -> SignalView:
        """Visor de la fuente en la pestaña activa (o un respaldo vacío)."""
        w = self._signal_tabs.currentWidget()
        return w if isinstance(w, SignalView) else self._empty_view

    def _sid_for_view(self, view) -> str | None:
        for sid, v in self._source_views.items():
            if v is view:
                return sid
        return None

    def _on_source_selected(self, row: int) -> None:
        if self.project is None or row < 0:
            return
        item = self.sources_list.item(row)
        if item is None:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        if sid is None or self.project.get_source(sid) is None:
            return
        self.current_source_id = sid
        src = self.project.get_source(sid)
        if src and not os.path.isfile(src["path"]):
            self._handle_missing_source(sid, src)
            return
        self._open_source_tab(sid)

    # --- Pestañas de fuentes (centro multi-fuente, estilo navegador) -----
    def _ensure_source_tab(self, sid: str) -> SignalView:
        """Devuelve la pestaña (SignalView) de ``sid``, creándola si no existe."""
        view = self._source_views.get(sid)
        if view is None:
            view = SignalView()
            view.segment_requested.connect(self._on_segment_requested)
            view.cut_requested.connect(self._on_cut_requested)
            view.delete_segments_requested.connect(self._on_delete_segments_in_selection)
            view.relabel_segment_requested.connect(self.relabel_segment)
            view.delete_segment_requested.connect(self.remove_segment)
            view.generate_periodic_requested.connect(self.generate_periodic_segments)
            view.mode_changed.connect(self._update_signal_view)
            self._source_views[sid] = view
            src = self.project.get_source(sid) if self.project else None
            alias = src["alias"] if src else sid
            idx = self._signal_tabs.addTab(view, alias)
            self._signal_tabs.setTabToolTip(idx, alias)
        return view

    def _open_source_tab(self, sid: str) -> SignalView:
        """Abre (o enfoca) la pestaña de una fuente y la dibuja."""
        view = self._ensure_source_tab(sid)
        self._stale_views.discard(sid)
        self._signal_tabs.setCurrentWidget(view)      # dispara _on_signal_tab_changed
        self._render_view(sid, view)
        return view

    def _on_signal_tab_changed(self, index: int) -> None:
        if index < 0:
            return
        view = self._signal_tabs.widget(index)
        sid = self._sid_for_view(view)
        if sid is None:
            return
        self.current_source_id = sid
        self._sync_sources_selection(sid)
        if sid in self._stale_views:                  # redibujado perezoso
            self._stale_views.discard(sid)
            self._render_view(sid, view)

    def _close_source_tab(self, index: int) -> None:
        sid = self._sid_for_view(self._signal_tabs.widget(index))
        if sid is not None:
            self._remove_source_tab(sid)

    def _remove_source_tab(self, sid: str) -> None:
        view = self._source_views.pop(sid, None)
        self._stale_views.discard(sid)
        if view is not None:
            idx = self._signal_tabs.indexOf(view)
            if idx >= 0:
                self._signal_tabs.removeTab(idx)
            view.deleteLater()

    def _reset_source_tabs(self) -> None:
        """Cierra todas las pestañas de fuentes (al cambiar de proyecto)."""
        self._signal_tabs.blockSignals(True)
        for sid in list(self._source_views):
            view = self._source_views.pop(sid)
            idx = self._signal_tabs.indexOf(view)
            if idx >= 0:
                self._signal_tabs.removeTab(idx)
            view.deleteLater()
        self._stale_views.clear()
        self._signal_tabs.blockSignals(False)

    def _sync_source_tabs(self) -> None:
        """Quita pestañas de fuentes borradas y refresca los títulos (alias)."""
        valid = {s["id"] for s in self.project.sources} if self.project else set()
        for sid in list(self._source_views):
            if sid not in valid:
                self._remove_source_tab(sid)
        if self.project:
            for sid, view in self._source_views.items():
                idx = self._signal_tabs.indexOf(view)
                src = self.project.get_source(sid)
                if idx >= 0 and src:
                    self._signal_tabs.setTabText(idx, src["alias"])

    def _sync_sources_selection(self, sid: str) -> None:
        """Marca en la lista de fuentes la que corresponde a la pestaña activa."""
        if not self.project:
            return
        it = self._item_for_sid(sid)
        if it is not None:
            self.sources_list.blockSignals(True)
            self.sources_list.setCurrentItem(it)
            self.sources_list.blockSignals(False)

    def _refresh_source_tabs(self) -> None:
        """Tras un cambio de estado: marca todas las pestañas para redibujar y
        redibuja la activa (las demás se redibujan al activarlas)."""
        self._sync_source_tabs()
        self._stale_views = set(self._source_views)
        self._update_signal_view()

    def _handle_missing_source(self, source_id: str, src: dict) -> None:
        """Una fuente cuyo archivo ya no existe: ofrecer reubicarla o quitarla."""
        self._remove_source_tab(source_id)
        res = QMessageBox.question(
            self, "Archivo de la fuente no encontrado",
            f"No se encuentra el archivo de «{src['alias']}»:\n{src['path']}\n\n"
            "¿Quieres localizarlo (reubicar la fuente)? Elige «No» para quitarla del proyecto.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel)
        if res == QMessageBox.StandardButton.Yes:
            path, _ = QFileDialog.getOpenFileName(
                self, "Localizar archivo de la fuente", "", "CSV (*.csv *.csv.gz)")
            if path:
                self.project.set_source_path(source_id, path)
                self._after_state_change()
        elif res == QMessageBox.StandardButton.No:
            self.project.remove_source(source_id)
            self.current_source_id = None
            self._after_state_change()

    def _render_view(self, sid: str, view: SignalView) -> None:
        """Rellena una pestaña concreta con la señal (cruda o procesada) de ``sid``."""
        if self.project is None or sid is None:
            view.clear()
            return
        try:
            rec = self.project.get_recording(sid)
        except FileNotFoundError:
            view.clear()
            self.statusBar().showMessage("La fuente seleccionada no se encuentra en disco.")
            return
        names = self.project.kept_display_names(rec)   # solo canales activos
        cuts = self.project.cut_intervals(sid)
        # Tramos eliminados (sombreados en gris, excluidos del dataset).
        view.set_cuts(cuts)
        # Marcadores (Event Id); se ocultan los que caen en tramos recortados.
        view.set_markers([
            (e["sample"], e["id"]) for e in rec.events
            if not any(ca <= e["sample"] < cb for ca, cb in cuts)
        ])
        # Segmentos ya etiquetados de esta fuente, sombreados por clase (con su id
        # para poder reetiquetarlos/eliminarlos con clic derecho).
        view.set_segments([
            (s["start"], s["stop"], s["label"], s["id"])
            for s in self.project.state["segments"]
            if s["source_id"] == sid
        ])

        if view.mode == "raw" or not self.project.state["pipeline"]:
            raw = rec.data[self.project.kept_indices(rec)]
            view.set_data(raw, rec.sample_rate, names)
            return

        # Si el resultado del pipeline ya está en caché, dibujar al instante
        # (sin hilo ni «ocupado»): evita el parpadeo al editar segmentos, etc.
        cached = self.project.processed_if_cached(sid)
        if cached is not None:
            view.set_data(cached, rec.sample_rate, names)
            return

        # Primera vez / tras invalidar: procesamiento en segundo plano.
        self._busy("Aplicando preprocesamiento…")
        self.progress.setRange(0, 0)        # indeterminado hasta el 1er paso
        self.progress.show()

        def done(data):
            self._idle()
            # La pestaña pudo cerrarse (o reemplazarse) mientras se procesaba:
            # no escribir sobre un visor ya destruido.
            if self._source_views.get(sid) is view:
                view.set_data(data, rec.sample_rate, names)

        self._spawn(lambda progress=None: self.project.get_processed(sid, progress=progress),
                    done, on_progress=self._on_filter_progress)

    def _update_signal_view(self) -> None:
        """Redibuja la pestaña de la fuente activa (creándola si hace falta)."""
        sid = self.current_source_id
        if sid is None:
            return
        view = self._source_views.get(sid)
        if view is None:
            if self.project and self.project.get_source(sid):
                view = self._ensure_source_tab(sid)
                self._signal_tabs.setCurrentWidget(view)
            else:
                return
        self._stale_views.discard(sid)
        self._render_view(sid, view)

    def _on_filter_progress(self, done_n: int, total: int) -> None:
        """Barra determinada con el paso de filtrado en curso."""
        if total <= 0:
            return
        if self.progress.maximum() != total:
            self.progress.setRange(0, total)
        self.progress.setValue(done_n)
        self.statusBar().showMessage(f"Aplicando preprocesamiento… paso {done_n}/{total}")

    # ------------------------------------------------------------------ #
    # Preprocesamiento (delegado desde el panel)
    # ------------------------------------------------------------------ #
    def add_pipeline_step(self, step_type: str) -> None:
        if not self._require_project():
            return
        self.project.add_pipeline_step(step_type)
        self._after_state_change()

    def remove_pipeline_step(self, index: int) -> None:
        self.project.remove_pipeline_step(index)
        self._after_state_change()

    def move_pipeline_step(self, index: int, delta: int) -> None:
        self.project.move_pipeline_step(index, delta)
        self._after_state_change()

    def update_pipeline_step(self, index: int, params: dict) -> None:
        self.project.update_pipeline_step(index, params)
        self._after_state_change()

    def set_pipeline_step_enabled(self, index: int, enabled: bool) -> None:
        """Activa/desactiva un paso del pipeline sin eliminarlo."""
        if not self._require_project():
            return
        self.project.set_step_enabled(index, enabled)
        self._after_state_change()

    # --- Varios pipelines por proyecto ------------------------------------
    def add_pipeline(self) -> None:
        if not self._require_project():
            return
        self.project.add_pipeline()
        self._after_state_change()

    def remove_pipeline(self, index: int) -> None:
        if not self._require_project():
            return
        if not self.project.remove_pipeline(index):
            QMessageBox.information(self, "Pipelines",
                                    "Debe quedar al menos un pipeline en el proyecto.")
            return
        self._after_state_change()

    def rename_pipeline(self, index: int, name: str) -> None:
        if not self._require_project():
            return
        self.project.rename_pipeline(index, name)
        self._after_state_change()

    def set_active_pipeline(self, index: int) -> None:
        if not self._require_project():
            return
        self.project.set_active_pipeline(index)
        self._after_state_change()

    def cache_processed(self) -> None:
        if not self._require_project() or self.current_source_id is None:
            QMessageBox.information(self, "Sin fuente", "Selecciona una fuente CSV primero.")
            return
        path = self.project.cache_processed_to_disk(self.current_source_id)
        QMessageBox.information(self, "Guardado", f"Señal procesada escrita en:\n{path}")

    def select_channels(self) -> None:
        """Diálogo para activar/desactivar canales (p. ej. excluir los EOG)."""
        if not self._require_project() or not self.project.sources:
            QMessageBox.information(self, "Sin fuente", "Añade una fuente CSV primero.")
            return
        sid = self.current_source_id or self.project.sources[0]["id"]
        rec = self.project.get_recording(sid)
        names = rec.channel_names
        display = self.project.display_channel_names(rec)
        excluded = set(self.project.excluded_channels())

        dlg = QDialog(self)
        dlg.setWindowTitle("Canales activos")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Marca los canales a incluir (desmarca para excluir, p. ej. EOG):"))

        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        checks: list[tuple[str, QCheckBox]] = []
        for name, disp in zip(names, display):
            chk = QCheckBox(disp if disp == name else f"{disp}  ({name})")
            chk.setChecked(name not in excluded)
            inner_lay.addWidget(chk)
            checks.append((name, chk))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setMinimumHeight(280)
        lay.addWidget(scroll)

        quick = QHBoxLayout()
        btn_all = QPushButton("Todos")
        btn_all.clicked.connect(lambda: [c.setChecked(True) for _, c in checks])
        btn_none = QPushButton("Ninguno")
        btn_none.clicked.connect(lambda: [c.setChecked(False) for _, c in checks])
        btn_eog = QPushButton("Excluir EOG")
        btn_eog.clicked.connect(
            lambda: [c.setChecked("eog" not in n.lower()) for n, c in checks])
        quick.addWidget(btn_all)
        quick.addWidget(btn_none)
        quick.addWidget(btn_eog)
        lay.addLayout(quick)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        new_excluded = [n for n, c in checks if not c.isChecked()]
        if len(new_excluded) >= len(names):
            QMessageBox.warning(self, "Sin canales", "Debe quedar al menos un canal activo.")
            return
        self.project.edit("excluded_channels", new_excluded, "Cambiar canales activos")
        self._after_state_change()
        kept = len(names) - len(new_excluded)
        self.statusBar().showMessage(f"Canales activos: {kept} de {len(names)}.")

    # ------------------------------------------------------------------ #
    # Segmentos
    # ------------------------------------------------------------------ #
    def _on_segment_requested(self, start: int, stop: int) -> None:
        if not self._require_project() or self.current_source_id is None:
            return
        labels = self.project.labels()
        label, ok = QInputDialog.getItem(
            self, "Nuevo segmento", "Etiqueta (clase):", labels or ["clase_1"], 0, True
        )
        if not ok or not label.strip():
            return
        self.project.add_segment(self.current_source_id, start, stop, label.strip())
        self._after_state_change()

    def _on_cut_requested(self, start: int, stop: int) -> None:
        """Recorta (marca como eliminado) el tramo seleccionado de la señal."""
        if not self._require_project() or self.current_source_id is None:
            return
        if stop - start < 1:
            return
        n_seg = sum(1 for s in self.project.state["segments"]
                    if s["source_id"] == self.current_source_id
                    and s["start"] < stop and s["stop"] > start)
        if n_seg and QMessageBox.question(
                self, "Recortar señal",
                f"El tramo seleccionado solapa {n_seg} segmento(s) etiquetado(s), "
                "que se eliminarán.\n¿Continuar? (Ctrl+Z para deshacer)"
                ) != QMessageBox.StandardButton.Yes:
            return
        self.project.add_cut(self.current_source_id, start, stop)
        self._after_state_change()
        self.statusBar().showMessage(
            "Tramo recortado (excluido del dataset). Ctrl+Z para deshacer.")

    def _on_delete_segments_in_selection(self, start: int, stop: int) -> None:
        """Borra los segmentos etiquetados que caen en la selección."""
        if not self._require_project() or self.current_source_id is None:
            return
        n = self.project.remove_segments_in_range(self.current_source_id, start, stop)
        if n == 0:
            self.info("Sin segmentos", "No hay segmentos etiquetados en la selección.")
            return
        self._after_state_change()
        self.statusBar().showMessage(f"{n} segmento(s) eliminado(s) de la selección.")

    def show_features(self) -> None:
        """Muestra las características extraídas de la región seleccionada."""
        if not self._require_project() or self.current_source_id is None:
            self.info("Sin fuente", "Selecciona una fuente y una región en la señal.")
            return
        s0, s1 = self.signal_view.selection_samples()
        if s1 - s0 < 8:
            self.info("Selección muy corta", "Selecciona una región más amplia en la señal.")
            return
        data, fs = self.project.segment_data(
            {"source_id": self.current_source_id, "start": s0, "stop": s1, "channels": None})
        rec = self.project.get_recording(self.current_source_id)
        from .feature_view import show_feature_dialog
        show_feature_dialog(self, self.project.kept_display_names(rec),
                            band_powers(data, fs), time_features(data))

    def relabel_segment(self, segment_id: str) -> None:
        labels = self.project.labels()
        label, ok = QInputDialog.getItem(self, "Reetiquetar", "Nueva etiqueta:", labels, 0, True)
        if ok and label.strip():
            self.project.relabel_segment(segment_id, label.strip())
            self._after_state_change()

    def remove_segment(self, segment_id: str) -> None:
        self.project.remove_segment(segment_id)
        self._after_state_change()

    def generate_periodic_segments(self, segment_id: str) -> None:
        """Repite un segmento hacia adelante a intervalos regulares (protocolos
        periódicos: p. ej. 5 s de tarea cada 15 s). Marcas el 1º y genera el resto."""
        if not self._require_project():
            return
        seg = next((s for s in self.project.state["segments"] if s["id"] == segment_id), None)
        if seg is None:
            return
        sid = seg["source_id"]
        try:
            rec = self.project.get_recording(sid)
            n_samples, fs = rec.n_samples, rec.sample_rate
        except Exception:  # noqa: BLE001
            self.warn("No disponible", "No se pudo leer la grabación de este segmento.")
            return
        dur = seg["stop"] - seg["start"]

        dlg = QDialog(self)
        dlg.setWindowTitle("Generar segmentos periódicos")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            f"Repite «{seg['label']}» ({dur / fs:.1f} s) hacia adelante, a intervalos "
            f"regulares.\nSe crean con la misma duración y etiqueta."))
        form = QFormLayout()
        period_spin = QDoubleSpinBox()
        period_spin.setRange(0.5, 600.0); period_spin.setValue(15.0)
        period_spin.setDecimals(1); period_spin.setSuffix(" s")
        period_spin.setToolTip("Tiempo entre INICIOS de segmentos (5 s tarea + 10 s reposo = 15 s).")
        total_spin = QSpinBox()
        total_spin.setRange(2, 200); total_spin.setValue(4)
        total_spin.setToolTip("Nº TOTAL de segmentos (incluido el que ya marcaste).")
        fill_chk = QCheckBox("Hasta el final de la señal")
        form.addRow("Periodo entre inicios:", period_spin)
        form.addRow("Nº total de segmentos:", total_spin)
        lay.addLayout(form)
        lay.addWidget(fill_chk)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        period = int(round(period_spin.value() * fs))
        count = None if fill_chk.isChecked() else total_spin.value()
        created = self.project.repeat_segment(segment_id, period, count, n_samples)
        if created:
            self._after_state_change()
        self.statusBar().showMessage(
            f"{created} segmento(s) «{seg['label']}» generado(s) cada {period_spin.value():g} s.")

    def clear_all_segments(self) -> None:
        """Elimina todos los segmentos de una vez (en vez de uno por uno)."""
        if not self._require_project():
            return
        n = len(self.project.state["segments"])
        if n == 0:
            self.info("Sin segmentos", "No hay segmentos que eliminar.")
            return
        if QMessageBox.question(
                self, "Vaciar segmentos",
                f"¿Eliminar los {n} segmentos etiquetados del proyecto?\n"
                "Se puede deshacer con Ctrl+Z.") != QMessageBox.StandardButton.Yes:
            return
        self.project.clear_segments()
        self.dataset = None                       # el dataset construido queda obsoleto
        self.dataset_panel.set_info("Dataset vacío. Crea segmentos y vuelve a construir.")
        self._after_state_change()
        self.statusBar().showMessage(f"{n} segmentos eliminados.")

    def create_segments_from_markers(self, window: int, offset: int = 0,
                                     all_sources: bool = False) -> None:
        if not self._require_project():
            return
        if all_sources:
            if not self.project.sources:
                self.info("Sin fuentes", "Añade al menos una fuente primero.")
                return
            n = self.project.segments_from_markers_all(window, offset)
            scope = f"{len(self.project.sources)} fuentes"
        else:
            if self.current_source_id is None:
                self.info("Sin fuente", "Selecciona una fuente en el panel izquierdo primero.")
                return
            n = self.project.segments_from_markers(self.current_source_id, window, offset)
            scope = "la fuente actual"
        if n == 0:
            self.info("Sin marcadores",
                      f"No se encontraron marcadores en {scope} (columna «Event Id» vacía).\n"
                      "Graba en vivo insertando marcadores, o usa un CSV con estimulaciones.")
            return
        self._after_state_change()
        self.statusBar().showMessage(f"{n} segmentos creados desde marcadores ({scope}).")

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    def build_dataset(self) -> None:
        if not self._require_project():
            return
        if not self.project.state["segments"]:
            QMessageBox.information(self, "Sin segmentos", "Crea segmentos etiquetados primero.")
            return
        self._busy("Construyendo dataset (multiproceso)…")
        self.progress.setRange(0, 0)
        self.progress.show()

        def done(ds):
            self._idle()
            self.dataset = ds
            skipped = (f"\n⚠ {ds.skipped} segmento(s) omitido(s): su fuente no está "
                       "disponible (reubícala o quítala)." if ds.skipped else "")
            import numpy as _np
            vals, cnts = _np.unique(ds.y, return_counts=True)
            por_clase = " · ".join(f"{v}: {c}" for v, c in zip(vals, cnts))
            self.dataset_panel.set_info(
                f"Dataset: {ds.n_samples} muestras × {ds.n_features} características.\n"
                f"Por clase ⟶ {por_clase}{skipped}"
            )
            self.clf_panel.refresh()   # actualiza la capa de entrada (nº exacto)
            self.statusBar().showMessage(
                "Dataset construido." + (f" {ds.skipped} omitido(s)." if ds.skipped else ""))

        self._spawn(lambda: dataset_mod.build_dataset(self.project), done)

    def save_dataset(self) -> None:
        if self.dataset is None:
            QMessageBox.information(self, "Sin dataset", "Construye el dataset primero.")
            return
        name, ok = QInputDialog.getText(self, "Guardar dataset", "Nombre:", text="dataset")
        if not ok:
            return
        path = dataset_mod.save_dataset(self.project, self.dataset, name.strip() or "dataset")
        QMessageBox.information(self, "Guardado", f"Dataset escrito en:\n{path}")

    # ------------------------------------------------------------------ #
    # Clasificación
    # ------------------------------------------------------------------ #
    def train_model(self) -> None:
        key = self.clf_panel.classifier_key
        nn_config = self.clf_panel.nn_config() if classification.is_nn(key) else None

        # CNN/LSTM trabajan sobre señal cruda: se construye su dataset desde los
        # segmentos. El resto usa el dataset de características ya construido.
        if classification.requires_raw(key):
            if not self._require_project() or not self.project.state["segments"]:
                QMessageBox.information(self, "Sin segmentos",
                                        "Crea segmentos etiquetados primero.")
                return
            if classification.is_nn(key):
                window = nn_config.get("window_samples", 512)
                task = lambda progress=None: classification.train_raw(
                    dataset_mod.build_raw_dataset(self.project, window), key, nn_config,
                    progress=progress
                )
            else:  # Riemann / CSP
                window = self.clf_panel.raw_window_value()
                task = lambda progress=None: classification.train_riemann(
                    dataset_mod.build_raw_dataset(self.project, window), key,
                    raw_window=window
                )
        else:
            if self.dataset is None:
                QMessageBox.information(self, "Sin dataset", "Construye el dataset primero.")
                return
            clf_params = self.clf_panel.classic_params()
            task = lambda progress=None: classification.train(
                self.dataset, key, nn_config=nn_config, clf_params=clf_params,
                progress=progress
            )

        self._busy("Entrenando modelo…")
        self.progress.setRange(0, 0)        # indeterminado hasta la 1ª época
        self.progress.show()

        def done(result):
            self._idle()
            name = self._register_model(result)
            cv = result.cv_scores
            acc = f"  ·  exactitud≈{float(cv.mean()):.0%}" if getattr(cv, "size", 0) else ""
            self.statusBar().showMessage(f"Modelo «{name}» entrenado y añadido.{acc}")

        self._spawn(task, done, on_progress=self._on_train_progress)

    def _on_train_progress(self, done_n: int, total: int) -> None:
        """Convierte la barra en determinada y muestra la época en curso."""
        if total <= 0:
            return
        if self.progress.maximum() != total:
            self.progress.setRange(0, total)
        self.progress.setValue(done_n)
        self.statusBar().showMessage(f"Entrenando red… época {done_n}/{total}")

    # --- Registro de varios modelos por proyecto --------------------------
    def _auto_model_name(self, classifier_name: str) -> str:
        i = 1
        while f"{classifier_name}_{i}" in self.models:
            i += 1
        return f"{classifier_name}_{i}"

    def _register_model(self, result, name: str | None = None, persist: bool = True) -> str:
        name = name or self._auto_model_name(result.classifier_name)
        self.models[name] = result
        if persist and self.project is not None:
            try:
                classification.save_model(self.project, result, name)
            except Exception:  # noqa: BLE001
                pass
        self.activate_model(name)
        return name

    def activate_model(self, name: str) -> None:
        if name in self.models:
            self.active_model_name = name
            self.model = self.models[name]
        self.clf_panel.refresh_models()
        self.control_panel.refresh()

    def remove_model(self, name: str) -> None:
        if name not in self.models:
            return
        self.models.pop(name)
        if self.project is not None:
            path = os.path.join(self.project.path, "models", f"{name}.joblib")
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass
        if self.active_model_name == name:
            self.active_model_name = next(iter(self.models), None)
            self.model = self.models.get(self.active_model_name) if self.active_model_name else None
        self.clf_panel.refresh_models()
        self.control_panel.refresh()

    def export_model(self, name: str) -> None:
        if name not in self.models:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar modelo", f"{name}.joblib", "Modelo (*.joblib)")
        if not path:
            return
        classification.save_model_to(self.models[name], path)
        QMessageBox.information(self, "Exportado", f"Modelo escrito en:\n{path}")

    def import_model(self) -> None:
        if not self._require_project():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Importar modelo", "", "Modelo (*.joblib)")
        if not path:
            return
        try:
            result = classification.load_model(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "No se pudo importar", str(exc))
            return
        base = os.path.splitext(os.path.basename(path))[0]
        name = base if base not in self.models else self._auto_model_name(result.classifier_name)
        self._register_model(result, name)
        self.statusBar().showMessage(f"Modelo «{name}» importado.")

    def show_model_metrics(self, name: str) -> None:
        if name not in self.models:
            return
        result = self.models[name]
        label = classification.CLASSIFIER_LABELS.get(result.classifier_name, result.classifier_name)
        header = (f"{name}  ·  {label}\n{result.score_label}: "
                  f"{result.cv_mean * 100:.1f}%  (±{result.cv_std * 100:.1f})\n"
                  f"Clases: {', '.join(result.classes)}")
        data_note = result.split_report()          # con cuántos datos se entrenó/evaluó
        text = (header + "\n" + data_note + "\n" + "-" * 40 + "\n"
                + classification.metrics_report(result))
        from . import metrics_view
        # Vista con gráficos si hay métricas y matplotlib; si no, el texto de siempre.
        if result.metrics and metrics_view.matplotlib_available():
            metrics_view.show_metrics_dialog(self, f"Métricas — {name}", header,
                                             result.metrics, text, data_note=data_note)
        else:
            self._show_text_dialog(f"Métricas — {name}", text)

    def configure_model(self, name: str) -> None:
        """Muestra la configuración de un modelo y permite editarla y reentrenar."""
        if name not in self.models:
            return
        result = self.models[name]
        key = result.classifier_name

        # ¿Hay datos para reentrenar con la nueva configuración?
        if classification.requires_raw(key):
            retrainable = bool(self.project and self.project.state.get("segments"))
            reason = "crea segmentos etiquetados primero."
        else:
            retrainable = self.dataset is not None
            reason = "construye el dataset primero."

        from . import model_config
        choice = model_config.edit_model_config(self, name, result, retrainable, reason)
        if choice is None:
            return
        kind, payload = choice

        if kind == "classic":
            task = lambda progress=None: classification.train(
                self.dataset, key, clf_params=payload)
        elif kind == "nn":
            if classification.requires_raw(key):
                window = int(payload.get("window_samples", 512))
                task = lambda progress=None: classification.train_raw(
                    dataset_mod.build_raw_dataset(self.project, window), key, payload,
                    progress=progress)
            else:
                task = lambda progress=None: classification.train(
                    self.dataset, key, nn_config=payload, progress=progress)
        elif kind == "riemann":
            window = int(payload)
            task = lambda progress=None: classification.train_riemann(
                dataset_mod.build_raw_dataset(self.project, window), key, raw_window=window)
        else:
            return

        self._busy(f"Reentrenando «{name}»…")
        self.progress.setRange(0, 0)
        self.progress.show()

        def done(new_result):
            self._idle()
            self._register_model(new_result, name=name)   # sustituye con el mismo nombre
            cv = new_result.cv_scores
            acc = f"  ·  exactitud≈{float(cv.mean()):.0%}" if getattr(cv, "size", 0) else ""
            self.statusBar().showMessage(f"Modelo «{name}» reentrenado.{acc}")

        self._spawn(task, done, on_progress=self._on_train_progress)

    def _load_project_models(self) -> None:
        """Carga los modelos guardados (.joblib) de la carpeta models/."""
        self.models = {}
        self.active_model_name = None
        self.model = None
        if self.project is None:
            return
        mdir = os.path.join(self.project.path, "models")
        if not os.path.isdir(mdir):
            return
        for fn in sorted(os.listdir(mdir)):
            if fn.endswith(".joblib"):
                try:
                    self.models[fn[:-7]] = classification.load_model(os.path.join(mdir, fn))
                except Exception:  # noqa: BLE001 - p. ej. falta torch/pyriemann
                    pass
        self.active_model_name = next(iter(self.models), None)
        self.model = self.models.get(self.active_model_name) if self.active_model_name else None

    def _show_text_dialog(self, title: str, text: str) -> None:
        from PyQt6.QtWidgets import QPlainTextEdit
        from PyQt6.QtGui import QFont
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(560, 460)
        lay = QVBoxLayout(dlg)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setFont(QFont("Consolas", 10))
        view.setPlainText(text)
        lay.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        lay.addWidget(buttons)
        dlg.exec()

    def predict_selection(self) -> None:
        if self.model is None:
            QMessageBox.information(self, "Sin modelo", "Entrena o carga un modelo primero.")
            return
        if self.current_source_id is None:
            return
        s0, s1 = self.signal_view.selection_samples()
        if s1 - s0 < 2:
            QMessageBox.information(self, "Selección vacía", "Selecciona una región en la señal.")
            return
        data, fs = self.project.segment_data(
            {"source_id": self.current_source_id, "start": s0, "stop": s1, "channels": None}
        )
        if self.model.input_kind == "raw":
            # Redes CNN/LSTM: ventana cruda con el mismo tamaño que en el entrenamiento.
            window = (self.model.nn_config or {}).get("window_samples", 512)
            X = dataset_mod.fit_window(data, window)[np.newaxis, ...]
        else:
            cfg = self.project.state.get("dataset", {})
            vec, _ = extract_feature_vector(
                data, fs, cfg.get("use_bands", True), cfg.get("use_time", True)
            )
            X = vec.reshape(1, -1)
        try:
            pred = classification.predict(self.model, X)[0]
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Error de predicción",
                                f"{exc}\n\n¿El dataset usa las mismas características y canales?")
            return
        proba = classification.predict_proba(self.model, X)
        extra = ""
        if proba is not None:
            pairs = ", ".join(f"{c}:{p:.2f}" for c, p in zip(self.model.classes, proba[0]))
            extra = f"\nProbabilidades: {pairs}"
        self.clf_panel.set_result(f"Predicción de la selección: {pred}{extra}")
        self.statusBar().showMessage(f"Predicción: {pred}")

    # ------------------------------------------------------------------ #
    # Control de cambios
    # ------------------------------------------------------------------ #
    def undo(self) -> None:
        if self.project and self.project.undo():
            self._after_state_change()

    def redo(self) -> None:
        if self.project and self.project.redo():
            self._after_state_change()

    # ------------------------------------------------------------------ #
    # Refrescos / utilidades
    # ------------------------------------------------------------------ #
    def _after_state_change(self) -> None:
        self.refresh_all()
        self._refresh_source_tabs()
        self._refresh_signal_windows()
        self.request_autosave()

    def refresh_all(self) -> None:
        self._refresh_sources()
        self._refresh_history()
        self.preproc_panel.refresh()
        self.dataset_panel.refresh()
        self.clf_panel.refresh()
        self.acq_panel.refresh_stim()
        self._update_actions()
        self._update_title()
        self._update_center()

    # --- Dimensiones para mostrar las capas de la red ---------------------
    def class_count(self) -> int:
        """Número de clases distintas entre los segmentos etiquetados."""
        return len(self.project.labels()) if self.project else 0

    def channel_count(self) -> int:
        """Número de canales **activos** (de la primera fuente, o el del EPOC+)."""
        if self.project and self.project.sources:
            try:
                rec = self.project.get_recording(self.project.sources[0]["id"])
                return len(self.project.kept_indices(rec))
            except Exception:  # noqa: BLE001
                pass
        return len(EPOC_CHANNELS)

    def feature_count(self) -> int:
        """Nº de características por segmento (exacto si el dataset ya existe)."""
        if self.dataset is not None:
            return self.dataset.n_features
        if self.project is None:
            return 0
        cfg = self.project.state.get("dataset", {})
        n_ch = self.channel_count()
        n_bands = len(FREQ_BANDS) if cfg.get("use_bands", True) else 0
        n_time = 8 if cfg.get("use_time", True) else 0   # ver processing.time_features
        return (n_bands + n_time) * n_ch

    def _on_sources_menu(self, pos) -> None:
        if self.project is None:
            return
        item = self.sources_list.itemAt(pos)
        if item is None:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        if sid is None or self.project.get_source(sid) is None:
            return
        self.sources_list.setCurrentItem(item)
        menu = QMenu(self)
        menu.addAction("Abrir en ventana nueva", lambda: self.open_source_window(sid))
        menu.addAction("Renombrar…", lambda: self._rename_source_dialog(sid))
        menu.addAction("Ver datos (tabla numérica)…", lambda: self.view_source_data(sid))
        menu.addAction("Exportar CSV (descomprimido)…", lambda: self.export_source_csv(sid))
        menu.addSeparator()
        menu.addAction("Buscar grabaciones sueltas…", self._offer_orphan_recordings)
        menu.addAction("Quitar del proyecto…", self.remove_current_source)
        menu.exec(self.sources_list.mapToGlobal(pos))

    def _on_source_renamed(self, item) -> None:
        """Edición en el sitio de la lista de fuentes → renombra la señal."""
        if self.project is None:
            return
        sid = item.data(Qt.ItemDataRole.UserRole)
        if sid is None or self.project.get_source(sid) is None:
            return
        self.rename_source(sid, item.text())

    def _rename_source_dialog(self, source_id: str) -> None:
        src = self.project.get_source(source_id) if self.project else None
        if src is None:
            return
        name, ok = QInputDialog.getText(self, "Renombrar señal", "Nuevo nombre:",
                                        text=src["alias"])
        if ok:
            self.rename_source(source_id, name)

    def rename_source(self, source_id: str, new_name: str) -> None:
        if not self._require_project():
            return
        try:
            changed = self.project.rename_source(source_id, new_name)
        except OSError as exc:
            QMessageBox.warning(self, "No se pudo renombrar el archivo", str(exc))
            self._refresh_sources()             # revierte el texto mostrado
            return
        if changed:
            self._after_state_change()
        else:
            self._refresh_sources()             # nombre vacío o igual: revierte

    def export_source_csv(self, source_id: str) -> None:
        """Exporta el CSV de una fuente **descomprimido** a la ubicación elegida.

        Útil para abrirlo en VS Code u otro editor, que no leen ``.csv.gz``."""
        if not self._require_project():
            return
        src = self.project.get_source(source_id)
        if src is None:
            return
        path = self.project._resolve_path(src["path"])
        if not os.path.isfile(path):
            QMessageBox.warning(self, "No encontrado",
                                f"No se encuentra el archivo de la fuente:\n{path}")
            return
        default = os.path.join(os.path.expanduser("~"), f"{src['alias']}.csv")
        out, _ = QFileDialog.getSaveFileName(
            self, "Exportar CSV (descomprimido)", default, "CSV (*.csv)")
        if not out:
            return
        try:
            self._write_plain_csv(path, out)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "No se pudo exportar", f"{out}\n\n{exc}")
            return
        self.statusBar().showMessage(f"CSV exportado: {out}")
        QMessageBox.information(self, "CSV exportado",
                               f"Guardado (descomprimido) en:\n{out}")

    @staticmethod
    def _write_plain_csv(src_path: str, out_path: str) -> None:
        """Escribe ``src_path`` en ``out_path`` como CSV plano (descomprime si es .gz)."""
        import shutil
        if src_path.lower().endswith(".gz"):
            import gzip
            with gzip.open(src_path, "rb") as fin, open(out_path, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        else:
            shutil.copyfile(src_path, out_path)

    def view_source_data(self, source_id: str) -> None:
        """Abre un visor con los datos de la fuente en forma numérica (tabla)."""
        if not self._require_project():
            return
        try:
            rec = self.project.get_recording(source_id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "No se pudo cargar", str(exc))
            return
        src = self.project.get_source(source_id)
        alias = src["alias"] if src else source_id
        from .csv_view import build_data_dialog
        dlg = build_data_dialog(self, rec, rec.channel_names, f"Datos — {alias}",
                                on_export=lambda: self.export_source_csv(source_id))
        dlg.exec()

    def open_source_window(self, source_id: str | None = None) -> None:
        """Abre una fuente en una ventana aparte (se pueden tener varias a la vez)."""
        if not self._require_project():
            return
        sid = source_id or self.current_source_id
        if sid is None or self.project.get_source(sid) is None:
            self.info("Sin fuente", "Selecciona una fuente en el panel izquierdo.")
            return
        win = SignalWindow(self, sid)
        self._signal_windows.add(win)
        win.show()
        win.raise_()

    def _refresh_signal_windows(self) -> None:
        """Resincroniza las ventanas de señal abiertas tras un cambio de estado."""
        for win in list(self._signal_windows):
            win.reload()

    # ---- Lista de fuentes: orden e indicadores de contenido -------------- #
    _SORT_MODES = {
        "custom":   "Orden propio",
        "alpha":    "Alfabético (A→Z)",
        "created":  "Fecha de creación",
        "modified": "Última modificación",
    }

    def _segment_counts(self) -> dict[str, int]:
        """Nº de segmentos etiquetados por fuente (barato, desde el estado)."""
        counts: dict[str, int] = {}
        if self.project:
            for s in self.project.state["segments"]:
                counts[s["source_id"]] = counts.get(s["source_id"], 0) + 1
        return counts

    def _sorted_sources(self) -> list[dict]:
        """Fuentes en el orden de vista actual (no muta el orden del proyecto)."""
        if not self.project:
            return []
        srcs = list(self.project.sources)
        mode = self._source_sort
        if mode == "alpha":
            srcs.sort(key=lambda s: s.get("alias", "").lower())
        elif mode in ("created", "modified"):
            getter = os.path.getctime if mode == "created" else os.path.getmtime

            def _stamp(s):
                try:
                    return getter(s.get("path", ""))
                except OSError:
                    return 0.0
            srcs.sort(key=_stamp)            # más antiguo primero
        return srcs                          # "custom" => tal cual el proyecto

    def _sources_panel_qss(self) -> str:
        """Estilo del panel de Fuentes, inspirado en el árbol de proyecto de PyCharm
        (cabecera discreta, filas planas con selección redondeada y hover)."""
        return f"""
        #sourcesPanel {{ background: {SURFACE}; }}
        #sourcesHeader {{ background: {BG}; border-bottom: 1px solid {BORDER}; }}
        #sourcesSortLabel {{
            color: {MUTED}; font-size: 10px; font-weight: 600; letter-spacing: 1px;
        }}
        #sourcesSortCombo {{
            background: {SURFACE}; border: 1px solid {BORDER};
            border-radius: 4px; padding: 2px 6px; min-height: 18px;
        }}
        #sourcesSortCombo:hover {{ border-color: {ACCENT}; }}
        QListWidget#sourcesList {{
            background: {SURFACE}; border: none; outline: 0; padding: 4px;
        }}
        QListWidget#sourcesList::item {{
            color: {TEXT}; padding: 4px 8px; margin: 1px 4px; border-radius: 5px;
        }}
        QListWidget#sourcesList::item:hover {{ background: {ELEVATED}; }}
        QListWidget#sourcesList::item:selected {{ background: {ACCENT}; color: #ffffff; }}
        """

    def _decorate_source_item(self, item, n_seg: int, n_mark: int) -> None:
        """Fija el indicador de contenido de una fuente sin tocar su nombre.

        Un punto pequeño y discreto a la derecha (lo pinta ``_SourceItemDelegate``):
        verde si tiene segmentos, ámbar si solo tiene marcadores. El texto (alias)
        queda intacto para no romper el renombrado en el sitio.
        """
        parts = []
        if n_seg:
            parts.append(f"{n_seg} segmento" + ("s" if n_seg != 1 else ""))
        if n_mark:
            parts.append(f"{n_mark} marcador" + ("es" if n_mark != 1 else ""))
        if parts:
            item.setData(_MARK_COLOR_ROLE,
                         COLOR_HAS_SEGMENTS if n_seg else COLOR_HAS_MARKERS)
            item.setToolTip("Contiene " + " y ".join(parts) +
                            ".  ·  Clic para renombrar (F2).")
        else:
            item.setData(_MARK_COLOR_ROLE, None)
            item.setToolTip("Sin segmentos ni marcadores.  ·  Clic para renombrar (F2).")

    def _refresh_sources(self) -> None:
        self.sources_list.blockSignals(True)
        self.sources_list.clear()
        if self.project:
            seg_counts = self._segment_counts()
            for src in self._sorted_sources():
                sid = src["id"]
                item = QListWidgetItem(src["alias"])
                item.setData(Qt.ItemDataRole.UserRole, sid)
                # Editable en el sitio para renombrar (clic izquierdo / F2).
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                self._decorate_source_item(item, seg_counts.get(sid, 0),
                                           self._src_event_counts.get(sid, 0))
                self.sources_list.addItem(item)
            if self.current_source_id:
                it = self._item_for_sid(self.current_source_id)
                if it is not None:
                    self.sources_list.setCurrentRow(self.sources_list.row(it))
        self.sources_list.blockSignals(False)
        self._scan_markers_async()

    def _item_for_sid(self, sid: str):
        for i in range(self.sources_list.count()):
            it = self.sources_list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == sid:
                return it
        return None

    def _update_source_indicators(self) -> None:
        """Repinta los indicadores de las fuentes ya listadas (sin reconstruir)."""
        if not self.project:
            return
        seg_counts = self._segment_counts()
        self.sources_list.blockSignals(True)
        for i in range(self.sources_list.count()):
            item = self.sources_list.item(i)
            sid = item.data(Qt.ItemDataRole.UserRole)
            self._decorate_source_item(item, seg_counts.get(sid, 0),
                                       self._src_event_counts.get(sid, 0))
        self.sources_list.blockSignals(False)

    def _scan_markers_async(self) -> None:
        """Cuenta los marcadores (Event Id) de cada fuente en segundo plano.

        Carga las grabaciones que falten (una sola pasada, sin bloquear la GUI) y
        cachea el resultado; al terminar, actualiza los indicadores. Las que fallen
        se cuentan como 0 (no rompe). Usa un hilo *daemon* y una señal en cola: no
        bloquea la GUI y no deja hilos vivos que estorben al cerrar."""
        if not self.project or self._scanning_markers:
            return
        todo = [s["id"] for s in self.project.sources
                if s["id"] not in self._src_event_counts]
        if not todo:
            return
        proj = self.project
        self._scanning_markers = True
        self._scan_project = proj

        def work():
            out: dict[str, int] = {}
            for sid in todo:
                try:
                    out[sid] = len(proj.get_recording(sid).events or [])
                except Exception:                     # noqa: BLE001
                    out[sid] = 0
            self._markers_scanned.emit(out)           # se entrega en el hilo GUI

        threading.Thread(target=work, daemon=True).start()

    def _on_markers_scanned(self, result: dict) -> None:
        self._scanning_markers = False
        if self.project is not self._scan_project:    # cambió de proyecto entretanto
            return
        self._src_event_counts.update(result)
        self._update_source_indicators()

    def _apply_source_drag_mode(self) -> None:
        """Habilita arrastrar para reordenar solo en modo «orden propio»."""
        custom = self._source_sort == "custom"
        if custom:
            self.sources_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self.sources_list.setDefaultDropAction(Qt.DropAction.MoveAction)
            self.sources_list.setToolTip("Arrastra para reordenar (orden propio).")
        else:
            self.sources_list.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
            self.sources_list.setToolTip("Cambia a «Orden propio» para reordenar arrastrando.")

    def _on_source_sort_changed(self, index: int) -> None:
        mode = self.source_sort_combo.itemData(index)
        if not mode or mode == self._source_sort:
            return
        self._source_sort = mode
        self._settings().setValue("source_sort", mode)
        self._apply_source_drag_mode()
        self._refresh_sources()

    def _on_sources_reordered(self, *args) -> None:
        """Tras arrastrar en modo «orden propio»: persiste el nuevo orden."""
        if self._reordering or self.project is None or self._source_sort != "custom":
            return
        ids = [self.sources_list.item(i).data(Qt.ItemDataRole.UserRole)
               for i in range(self.sources_list.count())]
        ids = [s for s in ids if s]
        self._reordering = True
        try:
            changed = self.project.reorder_sources(ids)
        finally:
            self._reordering = False
        if changed:
            self._after_state_change()

    # Iconos por sección, para identificar de un vistazo el tipo de cambio.
    _HISTORY_ICONS = {
        "pipeline": "🎛", "pipelines": "🎛", "active_pipeline": "🔀",
        "segments": "✂", "sources": "📁",
        "dataset": "📊", "channel_aliases": "🏷", "excluded_channels": "🚫",
    }

    def _refresh_history(self) -> None:
        """Pinta el historial como ÁRBOL colapsable (QTreeWidget).

        Cada nodo cuelga de su padre; la rama actual va resaltada (▶) y en negrita,
        las ramas/estados no aplicados se atenúan. Un clic navega a ese nodo.
        """
        tree = self.changelog_tree
        tree.clear()
        if not self.project:
            return
        stack: list[QTreeWidgetItem] = []      # stack[d] = último ítem visto a profundidad d
        current_item = None
        for node in self.project.changelog.nodes():
            depth = node["depth"]
            if node["is_root"]:
                text = "⏮  Estado inicial"
            else:
                icon = self._HISTORY_ICONS.get(node["section"], "•")
                ts = time.strftime("%H:%M:%S", time.localtime(node["timestamp"]))
                branch = "   ⑂" if node["n_children"] > 1 else ""
                text = f"{icon}  {node['description']}{branch}    ·  {ts}"
            item = QTreeWidgetItem([("▶  " if node["is_current"] else "") + text])
            item.setData(0, Qt.ItemDataRole.UserRole, node["id"])
            item.setToolTip(0, text)
            if node["is_current"]:
                f = item.font(0); f.setBold(True); item.setFont(0, f)
                item.setForeground(0, QColor("#9be7c4"))
                current_item = item
            elif not node["on_path"]:            # rama no aplicada: atenuada/cursiva
                f = item.font(0); f.setItalic(True); item.setFont(0, f)
                item.setForeground(0, QColor("#7c848d"))
            else:
                item.setForeground(0, QColor("#c8d0d8"))

            parent = stack[depth - 1] if depth > 0 and len(stack) >= depth else None
            if parent is None:
                tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            del stack[depth:]
            stack.append(item)

        tree.expandAll()
        if current_item is not None:
            tree.setCurrentItem(current_item)
            tree.scrollToItem(current_item)

    def _on_history_click(self, item, *_args) -> None:
        target = item.data(0, Qt.ItemDataRole.UserRole)
        if self.project is None or target is None:
            return
        if int(target) != self.project.changelog.current_id:
            self.project.goto_node(int(target))
            self._after_state_change()

    def _update_actions(self) -> None:
        has_proj = self.project is not None
        for a in (self.act_save, self.act_open_folder, self.act_export_cfg,
                  self.act_import_cfg, self.act_add, self.act_compress, self.act_del_src):
            a.setEnabled(has_proj)
        self.act_undo.setEnabled(has_proj and self.project.changelog.can_undo())
        self.act_redo.setEnabled(has_proj and self.project.changelog.can_redo())

    def _require_project(self) -> bool:
        if self.project is None:
            QMessageBox.information(self, "Sin proyecto", "Crea o abre un proyecto primero.")
            return False
        return True

    # --- Apoyo para el panel de adquisición -------------------------------
    def warn(self, title: str, msg: str) -> None:
        QMessageBox.warning(self, title, msg)

    def info(self, title: str, msg: str) -> None:
        QMessageBox.information(self, title, msg)

    def show_live_view(self) -> None:
        # La adquisición en vivo puede usarse sin proyecto: salir de la bienvenida
        # y asegurar que el dock de herramientas (donde está su panel) esté visible.
        self.right_dock.setVisible(True)
        self.center_stack.setCurrentWidget(self.center_tabs)
        self.center_tabs.setCurrentWidget(self.live_view)

    def ask_add_recording(self, path: str) -> bool:
        res = QMessageBox.question(
            self, "Grabación finalizada",
            f"Se guardó la grabación en:\n{path}\n\n¿Añadirla como fuente del proyecto?",
        )
        return res == QMessageBox.StandardButton.Yes

    def add_recording_as_source(self, path: str, segments=None, alias: str | None = None) -> None:
        if not self._require_project():
            return
        try:
            src = self.project.add_source(path, alias=alias or None)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "No se pudo añadir", f"{path}\n\n{exc}")
            return
        # Si no llegan segmentos, recupéralos del archivo lateral (.marks.json).
        if not segments:
            from ..core import marks_sidecar
            segments = marks_sidecar.read_marks(path)
        n_seg = 0
        if segments and isinstance(src, dict) and src.get("id"):
            sid = src["id"]
            for start, stop, label in segments:
                self.project.add_segment(sid, int(start), int(stop), str(label))
                n_seg += 1
        self.refresh_all()
        self._persist_now()              # guarda YA (blindaje: no se pierde al cerrar)
        extra = f" con {n_seg} segmento(s)" if n_seg else ""
        self.statusBar().showMessage(f"Grabación añadida como fuente{extra}.")

    def _persist_now(self) -> None:
        """Guarda el proyecto inmediatamente (no depende del temporizador de autosave)."""
        if self.project is None:
            return
        try:
            self._autosave_timer.stop()
            self.project.save()
            self._set_dirty(False)
        except Exception:  # noqa: BLE001
            self.request_autosave()      # si falla, al menos deja el autosave programado

    def closeEvent(self, event) -> None:  # noqa: N802 (API de Qt)
        try:
            self.control_panel.shutdown()
            self.acq_panel.shutdown()     # cierra grabación en curso (deja CSV + lateral)
            # Guardado de PRECAUCIÓN al cerrar: siempre que haya proyecto, no solo si
            # había un autosave pendiente (blindaje contra pérdida de datos).
            if self.project is not None:
                try:
                    self._autosave_timer.stop()
                    self.project.save()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            super().closeEvent(event)

    # --- Concurrencia -----------------------------------------------------
    def _spawn(self, fn, on_done, on_progress=None) -> None:
        def _err(msg):
            self._idle()
            QMessageBox.critical(self, "Error en tarea en segundo plano", msg)

        handle = run_async(self, fn, on_done=on_done, on_error=_err,
                           on_progress=on_progress)
        thread = handle[0]
        # Conservar la tupla (hilo, worker, proxy) viva hasta que el hilo acabe.
        self._threads.append(handle)
        thread.finished.connect(
            lambda: self._threads.remove(handle) if handle in self._threads else None
        )

    def _busy(self, msg: str) -> None:
        self.statusBar().showMessage(msg)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

    def _idle(self) -> None:
        QApplication.restoreOverrideCursor()
        self.progress.hide()
        self.progress.setRange(0, 100)
        self._update_actions()

    def _about(self) -> None:
        QMessageBox.about(
            self, "Acerca de",
            f"{APP_NAME}\n\nVisualización, preprocesamiento, construcción de datasets y "
            "clasificación de señales EEG (Emotiv EPOC+) a partir de CSV de OpenViBE.\n\n"
            "Edición no destructiva: el CSV original nunca se modifica.",
        )

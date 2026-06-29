"""Ventana principal de EEG Studio: integra visor, paneles y control de cambios."""
from __future__ import annotations

import os
import time

import numpy as np
from PyQt6.QtCore import Qt, QSettings, QSize, QTimer
from PyQt6.QtGui import QAction, QColor, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
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
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..config import APP_NAME, EPOC_CHANNELS, FREQ_BANDS, IMPORTED_DIR, ORG_NAME, PROJECT_EXT
from ..core import classification, dataset as dataset_mod, mat_loader, mne_loader
from ..core.csv_loader import compress_csv
from ..core.processing import band_powers, extract_feature_vector, time_features
from ..core.project import Project
from ..workers import run_async
from .acquisition_panel import AcquisitionPanel
from .control_panel import ControlPanel
from .live_view import LiveSignalView
from .panels import ClassificationPanel, DatasetPanel, PreprocessingPanel
from .signal_view import SignalView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.project: Project | None = None
        self.dataset = None
        self.model: classification.TrainingResult | None = None       # modelo activo
        self.models: dict[str, classification.TrainingResult] = {}    # registro del proyecto
        self.active_model_name: str | None = None
        self.current_source_id: str | None = None
        self._threads: list = []  # mantiene vivos los hilos en ejecución

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
        self.signal_view = SignalView()
        self.signal_view.segment_requested.connect(self._on_segment_requested)
        self.signal_view.mode_changed.connect(self._update_signal_view)
        self.live_view = LiveSignalView()
        self.center_tabs = QTabWidget()
        self.center_tabs.addTab(self.signal_view, "Análisis (CSV)")
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
        has_proj = self.project is not None
        for dock in (self.src_dock, self.right_dock, self.log_dock):
            dock.setVisible(has_proj)
        if not has_proj:
            self._refresh_welcome_recents()
            self.center_stack.setCurrentWidget(self.welcome)
        else:
            self.center_stack.setCurrentWidget(self.center_tabs)

    def _build_docks(self) -> None:
        # Izquierda: fuentes (CSV) del proyecto.
        self.sources_list = QListWidget()
        self.sources_list.currentRowChanged.connect(self._on_source_selected)
        self.sources_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.sources_list.customContextMenuRequested.connect(self._on_sources_menu)
        self.src_dock = QDockWidget("Fuentes (CSV)", self)
        self.src_dock.setWidget(self.sources_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.src_dock)

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

        # Abajo: historial de cambios navegable (línea de tiempo).
        hist_container = QWidget()
        hist_layout = QVBoxLayout(hist_container)
        hist_layout.setContentsMargins(6, 4, 6, 6)
        hint = QLabel("Línea de tiempo · haz clic en un punto para navegar")
        hint.setStyleSheet("color: #8a929b; font-size: 11px;")
        hist_layout.addWidget(hint)
        self.changelog_list = QListWidget()
        self.changelog_list.setAlternatingRowColors(True)
        self.changelog_list.setStyleSheet(
            "QListWidget { border: none; background: #14181d; }"
            "QListWidget::item { padding: 3px 4px; }"
            "QListWidget::item:alternate { background: #181d23; }"
            "QListWidget::item:hover { background: #243042; }"
        )
        self.changelog_list.itemClicked.connect(self._on_history_click)
        hist_layout.addWidget(self.changelog_list)
        self.log_dock = QDockWidget("Historial", self)
        self.log_dock.setWidget(hist_container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)

    def _build_menu(self) -> None:
        bar = self.menuBar()

        m_proj = bar.addMenu("&Proyecto")
        self.act_new = QAction("Nuevo proyecto…", self, triggered=self.new_project)
        self.act_open = QAction("Abrir proyecto…", self, triggered=self.open_project)
        self.act_save = QAction("Guardar proyecto", self, triggered=self.save_project)
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_add_src = QAction("Añadir CSV…", self, triggered=self.add_source)
        self.act_import_mat = QAction("Importar dataset (.mat / .fif / .edf…)…",
                                      self, triggered=self.import_dataset)
        self.act_compress = QAction("Comprimir fuentes a .csv.gz…", self,
                                    triggered=self.compress_sources)
        self.act_del_src = QAction("Quitar fuente del proyecto…", self,
                                   triggered=self.remove_current_source)
        for a in (self.act_new, self.act_open):
            m_proj.addAction(a)
        self.recent_menu = m_proj.addMenu("Abrir reciente")
        self.recent_menu.aboutToShow.connect(self._build_recent_menu)
        m_proj.addAction(self.act_save)
        m_proj.addSeparator()
        m_proj.addAction(self.act_add_src)
        m_proj.addAction(self.act_import_mat)
        m_proj.addAction(self.act_compress)
        m_proj.addAction(self.act_del_src)
        m_proj.addSeparator()
        m_proj.addAction(QAction("Salir", self, triggered=self.close))

        m_edit = bar.addMenu("&Editar")
        self.act_undo = QAction("Deshacer", self, triggered=self.undo)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_redo = QAction("Rehacer", self, triggered=self.redo)
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        m_edit.addAction(self.act_undo)
        m_edit.addAction(self.act_redo)

        m_help = bar.addMenu("A&yuda")
        m_help.addAction(QAction("Acerca de", self, triggered=self._about))

        # Iconos estándar del estilo (sin archivos externos) para menú y barra.
        st = self.style()
        self.act_new.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_FileIcon))
        self.act_open.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.act_save.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.act_add_src.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_FileDialogNewFolder))
        self.act_import_mat.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        self.act_undo.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_ArrowBack))
        self.act_redo.setIcon(st.standardIcon(QStyle.StandardPixmap.SP_ArrowForward))

    def _build_toolbar(self) -> None:
        tb = QToolBar("Principal", self)
        tb.setMovable(False)
        tb.setIconSize(QSize(18, 18))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        # La misma QAction sirve en el menú y en la barra (Qt comparte su estado).
        for group in ((self.act_new, self.act_open, self.act_save),
                      (self.act_add_src, self.act_import_mat),
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
        self.dataset = None
        self._dirty = False
        self._load_project_models()
        self._push_recent(self.project.path)
        self.refresh_all()
        self.statusBar().showMessage(f"Proyecto creado: {self.project.path}")

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
        self.dataset = None
        self._dirty = False
        self._load_project_models()
        self._push_recent(self.project.path)
        self.refresh_all()
        n = len(self.models)
        self.statusBar().showMessage(
            f"Proyecto abierto: {self.project.path}"
            + (f"  ·  {n} modelo(s) cargado(s)" if n else ""))

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
            self.statusBar().showMessage(f"No se pudo autoguardar: {exc}", 3000)

    def add_source(self) -> None:
        if not self._require_project():
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Añadir CSV de EEG", "", "CSV (*.csv *.csv.gz)")
        for p in paths:
            try:
                self.project.add_source(p)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "No se pudo cargar", f"{p}\n\n{exc}")
        self.refresh_all()
        self.request_autosave()

    def import_dataset(self) -> None:
        if not self._require_project():
            return
        mne_exts = " ".join(f"*{e}" for e in mne_loader.supported_extensions())
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Importar dataset", "",
            f"Datasets (*.mat {mne_exts});;MATLAB (*.mat);;MNE/FIF/EDF ({mne_exts})")
        if not paths:
            return
        self._busy("Convirtiendo a CSV (puede tardar)…")
        self.progress.setRange(0, 0)
        self.progress.show()

        # El CSV convertido se guarda DENTRO del proyecto (no se toca el origen).
        imported_dir = os.path.join(self.project.path, IMPORTED_DIR)
        os.makedirs(imported_dir, exist_ok=True)

        def task():
            out = []
            for p in paths:
                base = os.path.splitext(os.path.basename(p))[0]
                gz = os.path.join(imported_dir, base + ".csv.gz")   # comprimido, en el proyecto
                if os.path.isfile(gz):                              # reutiliza si ya se importó
                    csv = gz
                elif os.path.splitext(p)[1].lower() == ".mat":
                    csv = mat_loader.convert_bnci_mat(p, gz)
                else:
                    csv = mne_loader.convert_with_mne(p, gz)
                out.append(csv)
            return out

        def done(csvs):
            for c in csvs:
                try:
                    self.project.add_source(c)
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.warning(self, "No se pudo añadir", f"{c}\n\n{exc}")
            self._idle()
            self.refresh_all()
            self.request_autosave()
            self.statusBar().showMessage(f"{len(csvs)} archivo(s) importado(s) como CSV.")

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
    def _on_source_selected(self, row: int) -> None:
        if self.project is None or row < 0 or row >= len(self.project.sources):
            return
        self.current_source_id = self.project.sources[row]["id"]
        src = self.project.get_source(self.current_source_id)
        if src and not os.path.isfile(src["path"]):
            self._handle_missing_source(self.current_source_id, src)
            return
        self._update_signal_view()

    def _handle_missing_source(self, source_id: str, src: dict) -> None:
        """Una fuente cuyo archivo ya no existe: ofrecer reubicarla o quitarla."""
        self.signal_view.clear()
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

    def _update_signal_view(self) -> None:
        if self.project is None or self.current_source_id is None:
            self.signal_view.clear()
            return
        try:
            rec = self.project.get_recording(self.current_source_id)
        except FileNotFoundError:
            self.signal_view.clear()
            self.statusBar().showMessage("La fuente seleccionada no se encuentra en disco.")
            return
        names = self.project.kept_display_names(rec)   # solo canales activos
        # Marcadores (Event Id) como ayuda visual para etiquetar manualmente.
        self.signal_view.set_markers([(e["sample"], e["id"]) for e in rec.events])
        # Segmentos ya etiquetados de esta fuente, sombreados por clase.
        self.signal_view.set_segments([
            (s["start"], s["stop"], s["label"])
            for s in self.project.state["segments"]
            if s["source_id"] == self.current_source_id
        ])

        if self.signal_view.mode == "raw" or not self.project.state["pipeline"]:
            raw = rec.data[self.project.kept_indices(rec)]
            self.signal_view.set_data(raw, rec.sample_rate, names)
            return

        # Procesamiento en segundo plano para no bloquear la interfaz.
        sid = self.current_source_id
        self._busy("Aplicando preprocesamiento…")

        def done(data):
            self._idle()
            self.signal_view.set_data(data, rec.sample_rate, names)

        self._spawn(lambda: self.project.get_processed(sid), done)

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
            self.dataset_panel.set_info(
                f"Dataset: {ds.n_samples} segmentos × {ds.n_features} características.\n"
                f"Clases: {', '.join(ds.classes)}{skipped}"
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
                    dataset_mod.build_raw_dataset(self.project, window), key
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
                  f"Clases: {', '.join(result.classes)}\n" + "-" * 40 + "\n")
        self._show_text_dialog(f"Métricas — {name}", header + classification.metrics_report(result))

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
        self._update_signal_view()
        self.request_autosave()

    def refresh_all(self) -> None:
        self._refresh_sources()
        self._refresh_history()
        self.preproc_panel.refresh()
        self.dataset_panel.refresh()
        self.clf_panel.refresh()
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
        row = self.sources_list.indexAt(pos).row()
        if row < 0 or row >= len(self.project.sources):
            return
        self.sources_list.setCurrentRow(row)
        menu = QMenu(self)
        menu.addAction("Quitar del proyecto…", self.remove_current_source)
        menu.exec(self.sources_list.mapToGlobal(pos))

    def _refresh_sources(self) -> None:
        self.sources_list.blockSignals(True)
        self.sources_list.clear()
        if self.project:
            for src in self.project.sources:
                self.sources_list.addItem(QListWidgetItem(src["alias"]))
            if self.current_source_id:
                ids = [s["id"] for s in self.project.sources]
                if self.current_source_id in ids:
                    self.sources_list.setCurrentRow(ids.index(self.current_source_id))
        self.sources_list.blockSignals(False)

    # Iconos por sección, para identificar de un vistazo el tipo de cambio.
    _HISTORY_ICONS = {
        "pipeline": "🎛", "segments": "✂", "sources": "📁",
        "dataset": "📊", "channel_aliases": "🏷", "excluded_channels": "🚫",
    }

    def _add_history_item(self, text: str, target: int, *, current: bool,
                          applied: bool) -> None:
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, target)
        if current:
            font = item.font(); font.setBold(True); item.setFont(font)
            item.setForeground(QColor("#9be7c4"))
            item.setBackground(QColor("#1f3a2e"))
        elif not applied:
            font = item.font(); font.setItalic(True); item.setFont(font)
            item.setForeground(QColor("#727a83"))   # rehacible: atenuado
        else:
            item.setForeground(QColor("#c8d0d8"))
        self.changelog_list.addItem(item)

    def _refresh_history(self) -> None:
        self.changelog_list.clear()
        if not self.project:
            return
        cl = self.project.changelog
        applied = cl.applied_count()

        # Punto de partida (antes de cualquier cambio).
        self._add_history_item(
            ("▶  " if applied == 0 else "      ") + "⏮  Estado inicial",
            target=0, current=(applied == 0), applied=True,
        )
        current_item = None
        for idx, entry in enumerate(cl.timeline()):
            target = idx + 1
            icon = self._HISTORY_ICONS.get(entry["section"], "•")
            ts = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
            is_current = entry["applied"] and target == applied
            prefix = "▶  " if is_current else ("↪  " if not entry["applied"] else "      ")
            text = f"{prefix}{icon}  {entry['description']}    ·  {ts}"
            self._add_history_item(text, target, current=is_current, applied=entry["applied"])
            if is_current:
                current_item = self.changelog_list.item(self.changelog_list.count() - 1)
        if current_item is not None:
            self.changelog_list.scrollToItem(current_item)

    def _on_history_click(self, item) -> None:
        target = item.data(Qt.ItemDataRole.UserRole)
        if self.project is None or target is None:
            return
        if target != self.project.changelog.applied_count():
            self.project.goto_history(int(target))
            self._after_state_change()

    def _update_actions(self) -> None:
        has_proj = self.project is not None
        for a in (self.act_save, self.act_add_src, self.act_import_mat, self.act_compress,
                  self.act_del_src):
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

    def add_recording_as_source(self, path: str) -> None:
        if not self._require_project():
            return
        try:
            self.project.add_source(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "No se pudo añadir", f"{path}\n\n{exc}")
            return
        self.refresh_all()
        self.statusBar().showMessage("Grabación añadida como fuente.")

    def closeEvent(self, event) -> None:  # noqa: N802 (API de Qt)
        try:
            if self._autosave_timer.isActive() and self.project is not None:
                self._autosave_timer.stop()
                self.project.save()       # vaciar guardado pendiente al salir
            self.control_panel.shutdown()
            self.acq_panel.shutdown()
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

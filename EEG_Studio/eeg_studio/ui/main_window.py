"""Ventana principal de EEG Studio: integra visor, paneles y control de cambios."""
from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QTabWidget,
)

from ..config import APP_NAME, EPOC_CHANNELS, FREQ_BANDS, PROJECT_EXT
from ..core import classification, dataset as dataset_mod
from ..core.processing import extract_feature_vector
from ..core.project import Project
from ..workers import run_async
from .acquisition_panel import AcquisitionPanel
from .live_view import LiveSignalView
from .panels import ClassificationPanel, DatasetPanel, PreprocessingPanel
from .signal_view import SignalView


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.project: Project | None = None
        self.dataset = None
        self.model: classification.TrainingResult | None = None
        self.current_source_id: str | None = None
        self._threads: list = []  # mantiene vivos los hilos en ejecución

        self.setWindowTitle(APP_NAME)
        self.resize(1280, 800)

        # Centro: pestañas Análisis (estático) / Tiempo real (en vivo).
        self.signal_view = SignalView()
        self.signal_view.segment_requested.connect(self._on_segment_requested)
        self.signal_view.mode_changed.connect(self._update_signal_view)
        self.live_view = LiveSignalView()
        self.center_tabs = QTabWidget()
        self.center_tabs.addTab(self.signal_view, "Análisis (CSV)")
        self.center_tabs.addTab(self.live_view, "Tiempo real")
        self.setCentralWidget(self.center_tabs)

        self._build_docks()
        self._build_menu()
        self._build_statusbar()
        self._update_actions()

    # ------------------------------------------------------------------ #
    # Construcción de interfaz
    # ------------------------------------------------------------------ #
    def _build_docks(self) -> None:
        # Izquierda: fuentes (CSV) del proyecto.
        self.sources_list = QListWidget()
        self.sources_list.currentRowChanged.connect(self._on_source_selected)
        src_dock = QDockWidget("Fuentes (CSV)", self)
        src_dock.setWidget(self.sources_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, src_dock)

        # Derecha: pestañas de procesamiento.
        self.preproc_panel = PreprocessingPanel(self)
        self.dataset_panel = DatasetPanel(self)
        self.clf_panel = ClassificationPanel(self)
        self.acq_panel = AcquisitionPanel(self)
        tabs = QTabWidget()
        tabs.addTab(self.acq_panel, "Tiempo real")
        tabs.addTab(self.preproc_panel, "Preprocesamiento")
        tabs.addTab(self.dataset_panel, "Dataset")
        tabs.addTab(self.clf_panel, "Clasificación")
        right_dock = QDockWidget("Herramientas", self)
        right_dock.setWidget(tabs)
        right_dock.setMinimumWidth(340)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right_dock)

        # Abajo: historial de cambios (control de cambios).
        self.changelog_list = QListWidget()
        log_dock = QDockWidget("Historial de cambios", self)
        log_dock.setWidget(self.changelog_list)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)

    def _build_menu(self) -> None:
        bar = self.menuBar()

        m_proj = bar.addMenu("&Proyecto")
        self.act_new = QAction("Nuevo proyecto…", self, triggered=self.new_project)
        self.act_open = QAction("Abrir proyecto…", self, triggered=self.open_project)
        self.act_save = QAction("Guardar proyecto", self, triggered=self.save_project)
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_add_src = QAction("Añadir CSV…", self, triggered=self.add_source)
        for a in (self.act_new, self.act_open, self.act_save):
            m_proj.addAction(a)
        m_proj.addSeparator()
        m_proj.addAction(self.act_add_src)
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

    def _build_statusbar(self) -> None:
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
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
        self.model = None
        self.refresh_all()
        self.statusBar().showMessage(f"Proyecto creado: {self.project.path}")

    def open_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, f"Abrir proyecto ({PROJECT_EXT})")
        if not path:
            return
        if not os.path.isfile(os.path.join(path, "project.json")):
            QMessageBox.warning(self, "Proyecto inválido", "La carpeta no contiene project.json.")
            return
        try:
            self.project = Project.open(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error al abrir", str(exc))
            return
        self.current_source_id = None
        self.dataset = None
        self.model = None
        self.refresh_all()
        self.statusBar().showMessage(f"Proyecto abierto: {self.project.path}")

    def save_project(self) -> None:
        if self.project is None:
            return
        self.project.save()
        self.statusBar().showMessage("Proyecto guardado.")

    def add_source(self) -> None:
        if not self._require_project():
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Añadir CSV de EEG", "", "CSV (*.csv)")
        for p in paths:
            try:
                self.project.add_source(p)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "No se pudo cargar", f"{p}\n\n{exc}")
        self.refresh_all()

    # ------------------------------------------------------------------ #
    # Fuentes / visor
    # ------------------------------------------------------------------ #
    def _on_source_selected(self, row: int) -> None:
        if self.project is None or row < 0 or row >= len(self.project.sources):
            return
        self.current_source_id = self.project.sources[row]["id"]
        self._update_signal_view()

    def _update_signal_view(self) -> None:
        if self.project is None or self.current_source_id is None:
            self.signal_view.clear()
            return
        rec = self.project.get_recording(self.current_source_id)
        names = self.project.display_channel_names(rec)

        if self.signal_view.mode == "raw" or not self.project.state["pipeline"]:
            self.signal_view.set_data(rec.data, rec.sample_rate, names)
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

    def relabel_segment(self, segment_id: str) -> None:
        labels = self.project.labels()
        label, ok = QInputDialog.getItem(self, "Reetiquetar", "Nueva etiqueta:", labels, 0, True)
        if ok and label.strip():
            self.project.relabel_segment(segment_id, label.strip())
            self._after_state_change()

    def remove_segment(self, segment_id: str) -> None:
        self.project.remove_segment(segment_id)
        self._after_state_change()

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
            self.dataset_panel.set_info(
                f"Dataset: {ds.n_samples} segmentos × {ds.n_features} características.\n"
                f"Clases: {', '.join(ds.classes)}"
            )
            self.clf_panel.refresh()   # actualiza la capa de entrada (nº exacto)
            self.statusBar().showMessage("Dataset construido.")

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
                task = lambda: classification.train_raw(
                    dataset_mod.build_raw_dataset(self.project, window), key, nn_config
                )
            else:  # Riemann / CSP
                window = self.clf_panel.raw_window_value()
                task = lambda: classification.train_riemann(
                    dataset_mod.build_raw_dataset(self.project, window), key
                )
        else:
            if self.dataset is None:
                QMessageBox.information(self, "Sin dataset", "Construye el dataset primero.")
                return
            clf_params = self.clf_panel.svm_params() if key == "svm" else None
            task = lambda: classification.train(
                self.dataset, key, nn_config=nn_config, clf_params=clf_params
            )

        self._busy("Entrenando modelo…")
        self.progress.setRange(0, 0)
        self.progress.show()

        def done(result):
            self._idle()
            self.model = result
            label = classification.CLASSIFIER_LABELS[result.classifier_name]
            if result.cv_scores.size:
                msg = (f"Modelo: {label}\n"
                       f"{result.score_label}: {result.cv_mean:.3f} ± {result.cv_std:.3f}\n"
                       f"Clases: {', '.join(result.classes)}")
            else:
                msg = (f"Modelo entrenado ({label}).\n"
                       f"Clases: {', '.join(result.classes)}\n"
                       f"(Pocas muestras para validar.)")
            self.clf_panel.set_result(msg)
            self.statusBar().showMessage("Modelo entrenado.")

        self._spawn(task, done)

    def save_model(self) -> None:
        if self.model is None:
            QMessageBox.information(self, "Sin modelo", "Entrena un modelo primero.")
            return
        name, ok = QInputDialog.getText(self, "Guardar modelo", "Nombre:", text="model")
        if not ok:
            return
        path = classification.save_model(self.project, self.model, name.strip() or "model")
        QMessageBox.information(self, "Guardado", f"Modelo escrito en:\n{path}")

    def load_model(self) -> None:
        if not self._require_project():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Cargar modelo", os.path.join(self.project.path, "models"), "Modelo (*.joblib)"
        )
        if not path:
            return
        self.model = classification.load_model(path)
        self.clf_panel.set_result(
            f"Modelo cargado: {classification.CLASSIFIER_LABELS.get(self.model.classifier_name, '?')}\n"
            f"Clases: {', '.join(self.model.classes)}"
        )

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

    def refresh_all(self) -> None:
        self._refresh_sources()
        self._refresh_changelog()
        self.preproc_panel.refresh()
        self.dataset_panel.refresh()
        self.clf_panel.refresh()
        self._update_actions()

    # --- Dimensiones para mostrar las capas de la red ---------------------
    def class_count(self) -> int:
        """Número de clases distintas entre los segmentos etiquetados."""
        return len(self.project.labels()) if self.project else 0

    def channel_count(self) -> int:
        """Número de canales EEG (de la primera fuente, o el del EPOC+)."""
        if self.project and self.project.sources:
            try:
                return self.project.get_recording(self.project.sources[0]["id"]).n_channels
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

    def _refresh_changelog(self) -> None:
        self.changelog_list.clear()
        if not self.project:
            return
        for entry in self.project.changelog.history[-200:]:
            ev = {"do": "✓", "undo": "↶", "redo": "↷"}.get(entry.get("event"), "•")
            self.changelog_list.addItem(f"{ev} {entry.get('description', '')}")
        self.changelog_list.scrollToBottom()

    def _update_actions(self) -> None:
        has_proj = self.project is not None
        for a in (self.act_save, self.act_add_src):
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
            self.acq_panel.shutdown()
        finally:
            super().closeEvent(event)

    # --- Concurrencia -----------------------------------------------------
    def _spawn(self, fn, on_done) -> None:
        def _err(msg):
            self._idle()
            QMessageBox.critical(self, "Error en tarea en segundo plano", msg)

        handle = run_async(self, fn, on_done=on_done, on_error=_err)
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

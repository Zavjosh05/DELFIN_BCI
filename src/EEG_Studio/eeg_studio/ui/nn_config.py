"""Configuración detallada de redes neuronales en la interfaz.

Permite definir, capa por capa, el número de unidades/filtros, la función de
activación, el dropout (y kernel/bidireccional según el tipo de red), además de
los hiperparámetros de entrenamiento. Construye el ``config`` que consume
:mod:`eeg_studio.core.neuralnet`.
"""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.neuralnet import (
    ACTIVATION_LABELS,
    ACTIVATIONS,
    OPTIMIZERS,
    default_config,
)


class LayerEditor(QFrame):
    """Fila editable para una capa de la red."""

    def __init__(self, net_type: str, layer: dict, on_remove) -> None:
        super().__init__()
        self.net_type = net_type
        self.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)

        units_label = {"mlp": "Unid.", "cnn": "Filtros", "lstm": "Oculto"}[net_type]
        lay.addWidget(QLabel(units_label))
        self.units = QSpinBox()
        self.units.setRange(1, 4096)
        self.units.setValue(int(layer.get("units", 32)))
        lay.addWidget(self.units)

        if net_type == "cnn":
            lay.addWidget(QLabel("Kernel"))
            self.kernel = QSpinBox()
            self.kernel.setRange(1, 51)
            self.kernel.setSingleStep(2)
            self.kernel.setValue(int(layer.get("kernel_size", 3)))
            lay.addWidget(self.kernel)

        lay.addWidget(QLabel("Activ."))
        self.activation = QComboBox()
        for key in ACTIVATIONS:
            self.activation.addItem(ACTIVATION_LABELS[key], key)
        self._set_combo(self.activation, layer.get("activation", "relu"))
        lay.addWidget(self.activation)

        lay.addWidget(QLabel("Drop"))
        self.dropout = QDoubleSpinBox()
        self.dropout.setRange(0.0, 0.9)
        self.dropout.setSingleStep(0.05)
        self.dropout.setDecimals(2)
        self.dropout.setValue(float(layer.get("dropout", 0.0)))
        lay.addWidget(self.dropout)

        if net_type == "lstm":
            self.bidir = QCheckBox("Bi")
            self.bidir.setChecked(bool(layer.get("bidirectional", False)))
            lay.addWidget(self.bidir)

        lay.addStretch(1)
        remove = QPushButton("✕")
        remove.setFixedWidth(28)
        remove.clicked.connect(lambda: on_remove(self))
        lay.addWidget(remove)

    @staticmethod
    def _set_combo(combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def to_dict(self) -> dict:
        d = {
            "units": self.units.value(),
            "activation": self.activation.currentData(),
            "dropout": float(self.dropout.value()),
        }
        if self.net_type == "cnn":
            d["kernel_size"] = self.kernel.value()
        if self.net_type == "lstm":
            d["bidirectional"] = self.bidir.isChecked()
        return d


class NNConfigWidget(QGroupBox):
    """Editor completo de la arquitectura y el entrenamiento de la red."""

    def __init__(self) -> None:
        super().__init__("Configuración de la red neuronal")
        self.net_type = "mlp"
        self._editors: list[LayerEditor] = []
        self._build_ui()
        self.set_net_type("mlp")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Capa de entrada (su nº de neuronas depende de los datos).
        self.io_input = QLabel("Capa de entrada: —")
        self.io_input.setWordWrap(True)
        self.io_input.setStyleSheet("color: #7fd1b9; font-weight: bold;")
        layout.addWidget(self.io_input)

        # Lista de capas (con scroll).
        self._layers_box = QWidget()
        self._layers_layout = QVBoxLayout(self._layers_box)
        self._layers_layout.setContentsMargins(0, 0, 0, 0)
        self._layers_scroll = QScrollArea()
        self._layers_scroll.setWidgetResizable(True)
        self._layers_scroll.setWidget(self._layers_box)
        self._layers_scroll.setMinimumHeight(140)
        self._layers_label = QLabel("Capas (en orden):")
        layout.addWidget(self._layers_label)
        layout.addWidget(self._layers_scroll)

        self._add_btn = QPushButton("➕ Añadir capa")
        self._add_btn.clicked.connect(lambda: self.add_layer())
        layout.addWidget(self._add_btn)

        # Hiperparámetros propios de EEGNet (arquitectura fija; doi:10.1088/1741-2552/aace8c).
        self._eegnet_box = QGroupBox("Parámetros de EEGNet")
        eform = QFormLayout(self._eegnet_box)
        self.eeg_F1 = QSpinBox(); self.eeg_F1.setRange(1, 64); self.eeg_F1.setValue(8)
        self.eeg_D = QSpinBox(); self.eeg_D.setRange(1, 16); self.eeg_D.setValue(2)
        self.eeg_F2 = QSpinBox(); self.eeg_F2.setRange(1, 256); self.eeg_F2.setValue(16)
        self.eeg_kern = QSpinBox(); self.eeg_kern.setRange(8, 512); self.eeg_kern.setValue(64)
        self.eeg_drop = QDoubleSpinBox(); self.eeg_drop.setRange(0.0, 0.9)
        self.eeg_drop.setSingleStep(0.05); self.eeg_drop.setValue(0.25)
        self.eeg_F1.setToolTip("Nº de filtros temporales (frecuenciales).")
        self.eeg_D.setToolTip("Multiplicador de profundidad (filtros espaciales por filtro temporal).")
        self.eeg_F2.setToolTip("Nº de filtros separables del segundo bloque.")
        self.eeg_kern.setToolTip("Longitud del kernel temporal (≈ fs/2 capta hasta ~2 Hz).")
        eform.addRow("F1 (temporales):", self.eeg_F1)
        eform.addRow("D (profundidad):", self.eeg_D)
        eform.addRow("F2 (separables):", self.eeg_F2)
        eform.addRow("Kernel temporal:", self.eeg_kern)
        eform.addRow("Dropout:", self.eeg_drop)
        self._eegnet_box.setVisible(False)
        layout.addWidget(self._eegnet_box)

        # Hiperparámetros de entrenamiento.
        form = QFormLayout()
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5000)
        self.batch = QSpinBox()
        self.batch.setRange(1, 1024)
        self.lr = QDoubleSpinBox()
        self.lr.setRange(0.00001, 1.0)
        self.lr.setDecimals(5)
        self.lr.setSingleStep(0.0005)
        self.optimizer = QComboBox()
        self.optimizer.addItems(OPTIMIZERS)
        form.addRow("Épocas:", self.epochs)
        form.addRow("Batch size:", self.batch)
        form.addRow("Learning rate:", self.lr)
        form.addRow("Optimizador:", self.optimizer)

        # Ventana (solo CNN/LSTM, señal cruda).
        self.window = QSpinBox()
        self.window.setRange(16, 8192)
        self.window.setSingleStep(32)
        self.window_label = QLabel("Ventana (muestras):")
        form.addRow(self.window_label, self.window)
        layout.addLayout(form)

        # Capa de salida (nº de neuronas = nº de clases).
        self.io_output = QLabel("Capa de salida: —")
        self.io_output.setWordWrap(True)
        self.io_output.setStyleSheet("color: #e0a96d; font-weight: bold;")
        layout.addWidget(self.io_output)

    def set_io_info(self, input_text: str, output_text: str) -> None:
        self.io_input.setText(input_text)
        self.io_output.setText(output_text)

    # --- Tipo de red ------------------------------------------------------
    def set_net_type(self, net_type: str) -> None:
        """Reinicia el editor con los valores por defecto del tipo dado."""
        self.net_type = net_type
        cfg = default_config(net_type)
        is_eegnet = net_type == "eegnet"

        # EEGNet tiene arquitectura fija: se oculta la lista de capas.
        for w in (self._layers_label, self._layers_scroll, self._add_btn):
            w.setVisible(not is_eegnet)
        self._eegnet_box.setVisible(is_eegnet)

        self._clear_layers()
        if not is_eegnet:
            for layer in cfg.get("layers", []):
                self.add_layer(layer)
        else:
            self.eeg_F1.setValue(cfg["F1"]); self.eeg_D.setValue(cfg["D"])
            self.eeg_F2.setValue(cfg["F2"]); self.eeg_kern.setValue(cfg["kernel_length"])
            self.eeg_drop.setValue(cfg["dropout"])

        self.epochs.setValue(cfg["epochs"])
        self.batch.setValue(cfg["batch_size"])
        self.lr.setValue(cfg["learning_rate"])
        self._set_combo_text(self.optimizer, cfg["optimizer"])
        is_raw = net_type in ("cnn", "lstm", "eegnet")
        self.window.setVisible(is_raw)
        self.window_label.setVisible(is_raw)
        if is_raw:
            self.window.setValue(cfg.get("window_samples", 512))

    def _clear_layers(self) -> None:
        for ed in self._editors:
            ed.setParent(None)
        self._editors.clear()

    def add_layer(self, layer: dict | None = None) -> None:
        layer = layer or {"units": 32, "activation": "relu", "dropout": 0.0}
        editor = LayerEditor(self.net_type, layer, self._remove_layer)
        self._editors.append(editor)
        self._layers_layout.addWidget(editor)

    def _remove_layer(self, editor: LayerEditor) -> None:
        if len(self._editors) <= 1:
            return  # mantener al menos una capa
        editor.setParent(None)
        self._editors.remove(editor)

    @staticmethod
    def _set_combo_text(combo: QComboBox, text: str) -> None:
        i = combo.findText(text)
        if i >= 0:
            combo.setCurrentIndex(i)

    # --- Resultado --------------------------------------------------------
    def config(self) -> dict:
        cfg = {
            "type": self.net_type,
            "epochs": self.epochs.value(),
            "batch_size": self.batch.value(),
            "learning_rate": float(self.lr.value()),
            "optimizer": self.optimizer.currentText(),
        }
        if self.net_type == "eegnet":
            cfg.update({
                "F1": self.eeg_F1.value(), "D": self.eeg_D.value(),
                "F2": self.eeg_F2.value(), "kernel_length": self.eeg_kern.value(),
                "dropout": float(self.eeg_drop.value()),
            })
        else:
            cfg["layers"] = [ed.to_dict() for ed in self._editors]
        if self.net_type in ("cnn", "lstm", "eegnet"):
            cfg["window_samples"] = self.window.value()
        return cfg

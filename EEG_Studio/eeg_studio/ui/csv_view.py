"""Visor numérico de una grabación (tabla eficiente para archivos grandes).

Usa un :class:`QAbstractTableModel` sobre la matriz de la grabación, de modo que
la tabla es *virtual* (solo pinta las celdas visibles): abre grabaciones de
cientos de miles de muestras sin copiarlas a widgets ni consumir memoria.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableView,
    QVBoxLayout,
)


class RecordingTableModel(QAbstractTableModel):
    """Modelo de solo lectura: columnas ``#, t(s), <canales…>, Event Id``."""

    def __init__(self, data: np.ndarray, channel_names: list[str], fs: float,
                 events=None, parent=None) -> None:
        super().__init__(parent)
        self._data = np.asarray(data)
        self._names = list(channel_names)
        self._fs = float(fs) or 1.0
        self._nch, self._n = (self._data.shape if self._data.ndim == 2 else (0, 0))
        self._events = {int(e.get("sample", -1)): str(e.get("id", ""))
                        for e in (events or [])}
        self._cols = ["#", "t (s)"] + self._names + ["Event Id"]

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else self._n

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._cols)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        r, c = index.row(), index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if c == 0:
                return str(r)
            if c == 1:
                return f"{r / self._fs:.4f}"
            ci = c - 2
            if ci < self._nch:
                return f"{float(self._data[ci, r]):.3f}"
            return self._events.get(r, "")
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(self._cols):
            return self._cols[section]
        return None


def build_data_dialog(parent, recording, channel_names: list[str], title: str,
                      on_export=None) -> QDialog:
    """Diálogo con la tabla numérica de la grabación (no la ejecuta)."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(940, 620)
    lay = QVBoxLayout(dlg)

    n = recording.data.shape[1] if recording.data.ndim == 2 else 0
    info = QLabel(f"{len(channel_names)} canales · {n} muestras · "
                  f"{recording.sample_rate:g} Hz · {n / (recording.sample_rate or 1):.1f} s")
    info.setStyleSheet("color: #8a929b;")
    lay.addWidget(info)

    top = QHBoxLayout()
    top.addWidget(QLabel("Ir a muestra:"))
    goto = QLineEdit()
    goto.setPlaceholderText("nº de muestra")
    goto.setMaximumWidth(140)
    top.addWidget(goto)
    top.addStretch(1)
    if on_export is not None:
        exp = QPushButton("Exportar CSV…")
        exp.setToolTip("Guarda este CSV descomprimido en la ubicación que elijas.")
        exp.clicked.connect(lambda: on_export())
        top.addWidget(exp)
    lay.addLayout(top)

    model = RecordingTableModel(recording.data, channel_names, recording.sample_rate,
                                recording.events, dlg)
    view = QTableView()
    view.setModel(model)
    view.setFont(QFont("Consolas", 10))
    view.verticalHeader().setVisible(False)
    view.setAlternatingRowColors(True)
    view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
    lay.addWidget(view, 1)

    def _goto():
        try:
            r = int(goto.text())
        except ValueError:
            return
        r = max(0, min(model.rowCount() - 1, r))
        idx = model.index(r, 0)
        view.scrollTo(idx)
        view.selectRow(r)
    goto.returnPressed.connect(_goto)

    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    bb.rejected.connect(dlg.reject)
    lay.addWidget(bb)
    dlg._model = model            # mantener vivo el modelo
    return dlg

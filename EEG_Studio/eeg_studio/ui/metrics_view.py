"""Vista embellecida de métricas de un modelo.

Muestra la **matriz de confusión** como mapa de calor (matplotlib) junto a un
gráfico de **F1 por clase**, y una **tabla de scores** con color. La figura se
puede **guardar como imagen** (PNG). Se conserva el informe de **texto** (botón
«Ver texto…»). Si matplotlib no está disponible, el llamador usa el texto.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    _MPL_OK = True
except Exception:  # noqa: BLE001
    _MPL_OK = False

# Colores a juego con el tema oscuro de la app.
_BG = "#15191e"
_SURF = "#1e242b"
_TEXT = "#c8d0d8"
_MUTED = "#8a929b"
_BORDER = "#2c343d"
_TITLE = "#e8edf2"
_ACCENT = "#4f9fe0"


def matplotlib_available() -> bool:
    return _MPL_OK


def build_figure(metrics: dict):
    """Figura con la matriz de confusión (mapa de calor) y F1 por clase."""
    labels = [str(x) for x in metrics["labels"]]
    cm = np.asarray(metrics["confusion"], dtype=float)
    f1 = np.asarray(metrics.get("f1", []), dtype=float)

    fig = Figure(figsize=(9.0, 4.3), facecolor=_SURF)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    im = ax1.imshow(cm, cmap="Blues", aspect="auto")
    ax1.set_title("Matriz de confusión", color=_TITLE)
    ax1.set_xlabel("Predicho")
    ax1.set_ylabel("Real")
    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_yticklabels(labels)
    thr = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thr else "#12233a", fontsize=9)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=_MUTED)
    cbar.outline.set_edgecolor(_BORDER)

    y = np.arange(len(labels))
    ax2.barh(y, f1, color=_ACCENT)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()
    ax2.set_title("F1 por clase", color=_TITLE)
    ax2.set_xlabel("F1")
    for i, v in enumerate(f1):
        ax2.text(min(v + 0.02, 0.98), i, f"{v:.2f}", va="center", color=_TEXT, fontsize=9)

    for ax in (ax1, ax2):
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_MUTED)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        for spine in ax.spines.values():
            spine.set_color(_BORDER)
    fig.tight_layout()
    return fig


def _heat(value: float) -> QColor:
    """Mezcla del fondo con verde según ``value`` en [0, 1] (mayor = más verde)."""
    v = 0.0 if value != value else max(0.0, min(1.0, float(value)))   # nan -> 0
    base = (30, 36, 43)
    good = (46, 125, 91)
    rgb = tuple(int(base[k] + (good[k] - base[k]) * v) for k in range(3))
    return QColor(*rgb)


def build_scores_table(metrics: dict) -> QTableWidget:
    """Tabla de precisión/recall/F1/soporte por clase, con color por valor."""
    labels = [str(x) for x in metrics["labels"]]
    cols = ["Precisión", "Recall", "F1", "Soporte"]
    keys = ["precision", "recall", "f1", "support"]
    table = QTableWidget(len(labels), len(cols))
    table.setHorizontalHeaderLabels(cols)
    table.setVerticalHeaderLabels(labels)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.setMaximumHeight(38 + 30 * len(labels))
    for r in range(len(labels)):
        for c, key in enumerate(keys):
            arr = metrics.get(key, [])
            val = arr[r] if r < len(arr) else 0
            if key == "support":
                item = QTableWidgetItem(str(int(val)))
            else:
                item = QTableWidgetItem(f"{float(val):.2f}")
                item.setBackground(QBrush(_heat(val)))
                lum = _heat(val).lightness()
                item.setForeground(QBrush(QColor("#ffffff" if lum < 140 else "#12233a")))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(r, c, item)
    return table


def _save_figure(parent, fig) -> None:
    path, _ = QFileDialog.getSaveFileName(parent, "Guardar imagen", "matriz_confusion.png",
                                          "Imagen PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
    if not path:
        return
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")


def _show_text(parent, title: str, text: str) -> None:
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(560, 460)
    lay = QVBoxLayout(dlg)
    view = QPlainTextEdit()
    view.setReadOnly(True)
    view.setFont(QFont("Consolas", 10))
    view.setPlainText(text)
    lay.addWidget(view)
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    bb.rejected.connect(dlg.reject)
    lay.addWidget(bb)
    dlg.exec()


def build_metrics_dialog(parent, title: str, header: str, metrics: dict,
                         text_report: str) -> QDialog:
    """Construye el diálogo de métricas (sin ejecutarlo, para poder probarlo)."""
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(900, 660)
    lay = QVBoxLayout(dlg)

    hdr = QLabel(header)
    hdr.setWordWrap(True)
    hdr.setStyleSheet(f"color: {_TITLE}; font-weight: 600;")
    lay.addWidget(hdr)

    acc = metrics.get("accuracy")
    if acc is not None:
        a = QLabel(f"Exactitud global: {float(acc) * 100:.1f}%")
        a.setStyleSheet(f"color: {_ACCENT}; font-weight: 700; font-size: 14px;")
        lay.addWidget(a)

    fig = build_figure(metrics)
    canvas = FigureCanvas(fig)
    canvas.setMinimumHeight(320)
    lay.addWidget(canvas, 1)

    lay.addWidget(QLabel("Scores por clase:"))
    lay.addWidget(build_scores_table(metrics))

    row = QHBoxLayout()
    save_btn = QPushButton("Guardar imagen…")
    save_btn.clicked.connect(lambda: _save_figure(dlg, fig))
    text_btn = QPushButton("Ver texto…")
    text_btn.clicked.connect(lambda: _show_text(dlg, title, text_report))
    row.addWidget(save_btn)
    row.addWidget(text_btn)
    row.addStretch(1)
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    bb.rejected.connect(dlg.reject)
    row.addWidget(bb)
    lay.addLayout(row)
    dlg._figure = fig            # mantener viva la figura
    return dlg


def show_metrics_dialog(parent, title: str, header: str, metrics: dict,
                        text_report: str) -> None:
    build_metrics_dialog(parent, title, header, metrics, text_report).exec()

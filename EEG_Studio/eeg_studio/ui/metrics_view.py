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
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
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


def _draw_cm(ax, metrics: dict, normalize: bool):
    """Dibuja la matriz de confusión (conteos o normalizada por fila) y devuelve ``im``."""
    labels = [str(x) for x in metrics["labels"]]
    cm = np.asarray(metrics["confusion"], dtype=float)
    if normalize:
        rows = cm.sum(axis=1, keepdims=True)
        disp = np.divide(cm, rows, out=np.zeros_like(cm), where=rows != 0)
        title, vmax = "Matriz de confusión (normalizada)", 1.0
    else:
        disp, title, vmax = cm, "Matriz de confusión", (cm.max() if cm.size else 1.0)
    im = ax.imshow(disp, cmap="Blues", aspect="auto", vmin=0, vmax=vmax or 1.0)
    ax.set_title(title, color=_TITLE)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    thr = (vmax or 1.0) / 2.0
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            txt = f"{disp[i, j] * 100:.0f}%" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if disp[i, j] > thr else "#12233a", fontsize=9)
    return im


def _draw_f1(ax, metrics: dict) -> None:
    labels = [str(x) for x in metrics["labels"]]
    f1 = np.asarray(metrics.get("f1", []), dtype=float)
    y = np.arange(len(labels))
    ax.barh(y, f1, color=_ACCENT)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("F1 por clase", color=_TITLE)
    ax.set_xlabel("F1")
    for i, v in enumerate(f1):
        ax.text(min(v + 0.02, 0.98), i, f"{v:.2f}", va="center", color=_TEXT, fontsize=9)


def _style_axes(*axes) -> None:
    for ax in axes:
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_MUTED)
        ax.xaxis.label.set_color(_TEXT)
        ax.yaxis.label.set_color(_TEXT)
        for spine in ax.spines.values():
            spine.set_color(_BORDER)


def _populate_figure(fig, metrics: dict, normalize: bool) -> None:
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    im = _draw_cm(ax1, metrics, normalize)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=_MUTED)
    cbar.outline.set_edgecolor(_BORDER)
    _draw_f1(ax2, metrics)
    _style_axes(ax1, ax2)
    fig.tight_layout()


def build_figure(metrics: dict, normalize: bool = False):
    """Figura con la matriz de confusión (conteos o normalizada) y F1 por clase."""
    fig = Figure(figsize=(9.0, 4.3), facecolor=_SURF)
    _populate_figure(fig, metrics, normalize)
    return fig


def _heat_hex(value: float) -> str:
    v = 0.0 if value != value else max(0.0, min(1.0, float(value)))
    base, good = (30, 36, 43), (46, 125, 91)
    return "#%02x%02x%02x" % tuple(int(base[k] + (good[k] - base[k]) * v) for k in range(3))


def _mpl_table(ax, col_labels, cell_text, cell_colors=None, title=None):
    """Dibuja una tabla en ``ax`` con el estilo oscuro de la app."""
    ax.axis("off")
    if title:
        ax.text(0.0, 1.0, title, transform=ax.transAxes, ha="left", va="bottom",
                color=_TITLE, fontsize=11, fontweight="bold")
    # bbox deja un margen arriba para el título (evita solapes).
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center",
                   bbox=[0, 0, 1, 0.88])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(_BORDER)
        if r == 0:                                   # cabecera
            cell.set_facecolor("#232a32")
            cell.get_text().set_color(_MUTED)
            cell.get_text().set_fontweight("bold")
        else:
            color = "#1a1f25"
            if cell_colors is not None and (r - 1) < len(cell_colors):
                color = cell_colors[r - 1][c] or color
            cell.set_facecolor(color)
            cell.get_text().set_color(_TEXT)
    return tbl


def build_report_figure(metrics: dict, header: str, normalize: bool = False):
    """Informe COMPLETO para guardar: matriz + F1 + tabla por clase + tabla global.

    Se compone con matplotlib para que **todas** las filas de las tablas se vean
    (sin barras de desplazamiento) en la imagen.
    """
    labels = [str(x) for x in metrics["labels"]]
    fig = Figure(figsize=(9.5, 9.6), facecolor=_SURF)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.8, 1.3 + 0.28 * len(labels), 1.7],
                          hspace=1.15, wspace=0.28, top=0.93, bottom=0.05,
                          left=0.11, right=0.95)
    ax_cm = fig.add_subplot(gs[0, 0])
    ax_f1 = fig.add_subplot(gs[0, 1])
    im = _draw_cm(ax_cm, metrics, normalize)
    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=_MUTED)
    cbar.outline.set_edgecolor(_BORDER)
    _draw_f1(ax_f1, metrics)
    _style_axes(ax_cm, ax_f1)

    # Tabla por clase (con color en precisión/recall/F1).
    prec = metrics.get("precision", [])
    rec = metrics.get("recall", [])
    f1 = metrics.get("f1", [])
    sup = metrics.get("support", [])
    pc_text, pc_colors = [], []
    for i, lab in enumerate(labels):
        vals = [prec[i] if i < len(prec) else 0, rec[i] if i < len(rec) else 0,
                f1[i] if i < len(f1) else 0]
        pc_text.append([lab] + [f"{v:.2f}" for v in vals]
                       + [str(int(sup[i]) if i < len(sup) else 0)])
        pc_colors.append([None] + [_heat_hex(v) for v in vals] + [None])
    ax_pc = fig.add_subplot(gs[1, :])
    _mpl_table(ax_pc, ["Clase", "Precisión", "Recall", "F1", "Soporte"], pc_text,
               pc_colors, title="Scores por clase")

    # Tabla global (todas las métricas del modelo).
    g = global_metrics(metrics)
    g_text = [
        ["Exactitud (accuracy)", f"{g['accuracy'] * 100:.1f}%"],
        ["Precisión (macro)", f"{g['precision_macro']:.2f}"],
        ["Recall (macro)", f"{g['recall_macro']:.2f}"],
        ["F1 (macro)", f"{g['f1_macro']:.2f}"],
        ["F1 (ponderado)", f"{g['f1_weighted']:.2f}"],
        ["Muestras evaluadas (soporte total)", str(g["support_total"])],
    ]
    ax_g = fig.add_subplot(gs[2, :])
    _mpl_table(ax_g, ["Métrica global", "Valor"], g_text, title="Métricas globales")

    fig.suptitle(header.replace("\n", "  ·  "), color=_TITLE, fontsize=11, y=0.995)
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


def global_metrics(metrics: dict) -> dict:
    """Métricas GLOBALES del modelo (no por clase): exactitud + promedios.

    * ``accuracy``  — exactitud global (aciertos / total).
    * ``*_macro``   — media simple de la métrica entre clases (todas cuentan igual).
    * ``*_weighted``— media ponderada por el soporte (nº de muestras de cada clase).
    """
    p = np.asarray(metrics.get("precision", []), dtype=float)
    r = np.asarray(metrics.get("recall", []), dtype=float)
    f = np.asarray(metrics.get("f1", []), dtype=float)
    s = np.asarray(metrics.get("support", []), dtype=float)
    total = float(s.sum()) if s.size else 0.0

    def wavg(x):
        return float((x * s).sum() / total) if total and x.size else 0.0

    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "precision_macro": float(p.mean()) if p.size else 0.0,
        "recall_macro": float(r.mean()) if r.size else 0.0,
        "f1_macro": float(f.mean()) if f.size else 0.0,
        "precision_weighted": wavg(p),
        "recall_weighted": wavg(r),
        "f1_weighted": wavg(f),
        "support_total": int(total),
    }


def build_global_table(metrics: dict) -> QTableWidget:
    """Tabla resumen del modelo en general (métricas globales, no por clase)."""
    g = global_metrics(metrics)
    rows = [
        ("Exactitud (accuracy)", f"{g['accuracy'] * 100:.1f}%"),
        ("Precisión (macro)", f"{g['precision_macro']:.2f}"),
        ("Recall (macro)", f"{g['recall_macro']:.2f}"),
        ("F1 (macro)", f"{g['f1_macro']:.2f}"),
        ("F1 (ponderado)", f"{g['f1_weighted']:.2f}"),
        ("Muestras evaluadas (soporte total)", str(g["support_total"])),
    ]
    table = QTableWidget(len(rows), 2)
    table.setHorizontalHeaderLabels(["Métrica global", "Valor"])
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
    table.setMaximumHeight(38 + 30 * len(rows))
    for i, (metric, value) in enumerate(rows):
        table.setItem(i, 0, QTableWidgetItem(metric))
        vi = QTableWidgetItem(value)
        vi.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        if i == 0:                                   # exactitud global, destacada
            vi.setForeground(QBrush(QColor(_ACCENT)))
            f = vi.font(); f.setBold(True); vi.setFont(f)
        table.setItem(i, 1, vi)
    return table


def _save_report(parent, metrics: dict, header: str, normalize: bool) -> None:
    """Guarda el informe COMPLETO (matriz + F1 + tablas) como imagen."""
    path, _ = QFileDialog.getSaveFileName(
        parent, "Guardar imagen de métricas", "metricas.png",
        "Imagen PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
    if not path:
        return
    fig = build_report_figure(metrics, header, normalize)
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


def _section(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {_MUTED}; font-weight: 600; padding-top: 4px;")
    return lbl


def build_metrics_dialog(parent, title: str, header: str, metrics: dict,
                         text_report: str) -> QDialog:
    """Diálogo de métricas: matriz de confusión + F1 + tabla por clase + tabla global.

    El botón «Guardar imagen» captura TODO el informe (figura + tablas) en un PNG.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(920, 780)
    outer = QVBoxLayout(dlg)

    # Todo el informe va en 'content' para poder guardarlo como una sola imagen.
    content = QWidget()
    content.setStyleSheet(f"background: {_BG};")
    lay = QVBoxLayout(content)

    hdr = QLabel(header)
    hdr.setWordWrap(True)
    hdr.setStyleSheet(f"color: {_TITLE}; font-weight: 600;")
    lay.addWidget(hdr)

    g = global_metrics(metrics)
    acc = QLabel(f"Exactitud global del modelo: {g['accuracy'] * 100:.1f}%")
    acc.setStyleSheet(f"color: {_ACCENT}; font-weight: 700; font-size: 15px;")
    lay.addWidget(acc)

    # Botón para normalizar la matriz de confusión (por defecto: conteos).
    norm_chk = QCheckBox("Matriz de confusión normalizada (%)")
    norm_chk.setToolTip("Muestra porcentajes por fila (útil si el soporte por clase "
                        "es desigual). Se refleja también en la imagen guardada.")
    lay.addWidget(norm_chk)

    fig = build_figure(metrics, normalize=False)
    canvas = FigureCanvas(fig)
    canvas.setMinimumHeight(300)
    canvas.draw()
    lay.addWidget(canvas)

    def _redraw_cm(_state=None):
        fig.clear()
        _populate_figure(fig, metrics, norm_chk.isChecked())
        canvas.draw()

    norm_chk.stateChanged.connect(_redraw_cm)

    lay.addWidget(_section("Scores por clase:"))
    lay.addWidget(build_scores_table(metrics))
    lay.addWidget(_section("Métricas globales (todo el modelo):"))
    lay.addWidget(build_global_table(metrics))

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    outer.addWidget(scroll, 1)

    row = QHBoxLayout()
    save_btn = QPushButton("Guardar imagen…")
    save_btn.setToolTip("Guarda un PNG con la matriz de confusión, el F1 y las tablas "
                        "completas (respeta la normalización).")
    save_btn.clicked.connect(lambda: _save_report(dlg, metrics, header, norm_chk.isChecked()))
    text_btn = QPushButton("Ver texto…")
    text_btn.clicked.connect(lambda: _show_text(dlg, title, text_report))
    row.addWidget(save_btn)
    row.addWidget(text_btn)
    row.addStretch(1)
    bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    bb.rejected.connect(dlg.reject)
    row.addWidget(bb)
    outer.addLayout(row)
    dlg._figure = fig            # mantener viva la figura
    dlg._content = content
    return dlg


def show_metrics_dialog(parent, title: str, header: str, metrics: dict,
                        text_report: str) -> None:
    build_metrics_dialog(parent, title, header, metrics, text_report).exec()

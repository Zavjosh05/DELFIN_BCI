"""Mapas topográficos de los componentes independientes (ICA).

Dibuja, para cada componente ICA, la **distribución espacial** de su peso por
canal sobre un esquema de la cabeza (vista superior) — como en la literatura de
EEG: zonas **rojas** = peso positivo, **azules** = negativo. Los componentes con
kurtosis alta (candidatos a artefacto: parpadeos, músculo…) se **resaltan**, para
identificar de un vistazo dónde surgen los artefactos que ``ica_artifact`` elimina.

Requiere ``matplotlib`` (opcional). La interpolación usa ``scipy`` (ya es
dependencia). Si matplotlib no está, ``topomaps_available()`` devuelve ``False`` y
el llamador avisa.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..core.montage import positions_2d

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure
    # Mapa de color divergente PROPIO: rojo y azul base más CLAROS y vivos que el
    # "RdBu_r" de matplotlib, cuyos extremos son vino y azul marino muy oscuros
    # (cuesta diferenciar entre tonos). Azul (−) → casi blanco (0) → rojo (+).
    _ICA_CMAP = LinearSegmentedColormap.from_list("ica_div", [
        "#3d7ff0",   # azul base (más claro que el navy de RdBu)
        "#a9c9f7",   # azul claro
        "#f4f6f8",   # centro (peso ~0)
        "#f9b0a9",   # rojo claro
        "#f5564a",   # rojo base (más claro que el vino de RdBu)
    ])
    _MPL_OK = True
except Exception:  # noqa: BLE001
    _ICA_CMAP = None
    _MPL_OK = False

_BG = "#15191e"
_TEXT = "#c8d0d8"
_MUTED = "#8a929b"
_TITLE = "#e8edf2"
_ARTIFACT = "#ff6b6b"


def topomaps_available() -> bool:
    """True si se pueden dibujar los mapas (matplotlib disponible)."""
    return _MPL_OK


def _draw_one(ax, weights: np.ndarray, xy: list, title: str,
              is_artifact: bool) -> None:
    """Dibuja un topomapa en ``ax``: interpola los pesos por canal sobre el disco
    de la cabeza y añade el contorno (cabeza, nariz, orejas) y los electrodos."""
    from scipy.interpolate import griddata

    pts = np.array([p for p in xy if p is not None], dtype=float)
    w = np.array([wi for wi, p in zip(weights, xy) if p is not None], dtype=float)

    # Contorno de la cabeza: círculo unitario + nariz (triángulo) + orejas.
    head = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(head), np.sin(head), color="#4a5560", lw=1.5)
    ax.plot([-0.10, 0.0, 0.10], [0.99, 1.12, 0.99], color="#4a5560", lw=1.5)  # nariz
    for sx in (-1.0, 1.0):                                                     # orejas
        ear = np.linspace(-0.35, 0.35, 20)
        ax.plot(sx * (1.0 + 0.06 * np.cos(ear * 2)), 0.5 * np.sin(ear * 2.4),
                color="#4a5560", lw=1.5)

    if pts.shape[0] >= 3 and np.ptp(w) > 0:
        # Rejilla sobre el disco; interpola y enmascara fuera de la cabeza.
        g = np.linspace(-1.1, 1.1, 120)
        gx, gy = np.meshgrid(g, g)
        zi = griddata(pts, w, (gx, gy), method="cubic")
        zi_lin = griddata(pts, w, (gx, gy), method="linear")
        zi = np.where(np.isnan(zi), zi_lin, zi)              # rellena huecos del cúbico
        zi[gx ** 2 + gy ** 2 > 1.0] = np.nan                 # fuera de la cabeza
        vmax = float(np.nanmax(np.abs(w))) or 1.0
        ax.contourf(gx, gy, zi, levels=14, cmap=_ICA_CMAP, vmin=-vmax, vmax=vmax)
        ax.contour(gx, gy, zi, levels=6, colors="#00000022", linewidths=0.4)

    ax.scatter(pts[:, 0], pts[:, 1], s=6, c="#1a1a1a", zorder=5)  # electrodos
    ax.set_xlim(-1.25, 1.28)
    ax.set_ylim(-1.25, 1.28)
    ax.set_aspect("equal")
    ax.axis("off")
    color = _ARTIFACT if is_artifact else _TITLE
    suffix = "  ⚠" if is_artifact else ""
    ax.set_title(title + suffix, color=color, fontsize=9,
                 fontweight="bold" if is_artifact else "normal")


def build_topomap_figure(mixing: np.ndarray, ch_names: list, kurtosis: np.ndarray,
                         artifact: np.ndarray, ncols: int = 5):
    """Figura matplotlib con la rejilla de topomapas (uno por componente ICA).

    ``mixing`` es ``(n_canales, n_comp)``: cada COLUMNA es el mapa espacial de un
    componente. ``artifact`` marca los componentes de kurtosis alta.
    """
    xy = positions_2d(ch_names)
    n_comp = int(mixing.shape[1])
    ncols = max(1, min(ncols, n_comp))
    nrows = int(np.ceil(n_comp / ncols))
    fig = Figure(figsize=(2.15 * ncols, 2.35 * nrows), facecolor=_BG)
    for j in range(n_comp):
        ax = fig.add_subplot(nrows, ncols, j + 1)
        ax.set_facecolor(_BG)
        _draw_one(ax, mixing[:, j], xy, f"ICA{j:03d}", bool(artifact[j]))
    fig.suptitle("Componentes ICA (distribución espacial)", color=_TITLE,
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def show_ica_topomaps_dialog(parent, mixing: np.ndarray, ch_names: list,
                             kurtosis: np.ndarray, artifact: np.ndarray,
                             title: str = "Mapas espaciales ICA",
                             kurt_threshold: float = 5.0) -> None:
    """Diálogo con la rejilla de topomapas de los componentes ICA.

    Rojo = peso positivo, azul = negativo. Los componentes con kurtosis alta se
    marcan (⚠) como candidatos a artefacto (mismo criterio que ``ica_artifact``).
    """
    if not _MPL_OK:
        return
    mixing = np.asarray(mixing, dtype=float)
    kurtosis = np.asarray(kurtosis, dtype=float)
    artifact = np.asarray(artifact, dtype=bool)

    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(900, 640)
    root = QVBoxLayout(dlg)

    n_art = int(artifact.sum())
    caption = QLabel(
        f"Cada mapa es un componente independiente proyectado sobre el cuero "
        f"cabelludo (vista superior, nariz arriba). <b>Rojo</b> = actividad "
        f"positiva, <b>azul</b> = negativa. Los <b>{n_art}</b> componente(s) con "
        f"kurtosis &gt; {kurt_threshold:g} se marcan con <span style='color:{_ARTIFACT}'>"
        f"⚠</span> (candidatos a artefacto que el filtro ICA elimina). Los "
        f"artefactos suelen mostrar actividad intensa y localizada (p. ej. frontal "
        f"= parpadeos); la actividad cerebral tiende a ser más distribuida."
    )
    caption.setWordWrap(True)
    caption.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
    root.addWidget(caption)

    fig = build_topomap_figure(mixing, ch_names, kurtosis, artifact)
    canvas = FigureCanvas(fig)
    canvas.setMinimumSize(int(fig.get_figwidth() * fig.dpi),
                          int(fig.get_figheight() * fig.dpi))
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(canvas)
    root.addWidget(scroll, 1)

    bar = QHBoxLayout()
    save_btn = QPushButton("Guardar imagen…")

    def _save() -> None:
        path, _ = QFileDialog.getSaveFileName(
            dlg, "Guardar mapas ICA", "ica_topomaps.png",
            "Imagen PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            fig.savefig(path, facecolor=_BG, bbox_inches="tight", dpi=150)

    save_btn.clicked.connect(_save)
    bar.addWidget(save_btn)
    bar.addStretch(1)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dlg.reject)
    buttons.accepted.connect(dlg.accept)
    bar.addWidget(buttons)
    root.addLayout(bar)

    # Conservar refs para que el GC no destruya la figura mientras el diálogo vive.
    dlg._figure = fig            # type: ignore[attr-defined]
    dlg._canvas = canvas         # type: ignore[attr-defined]
    dlg.exec()

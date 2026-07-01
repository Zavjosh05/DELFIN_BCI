"""Vista embellecida de métricas: figura de matriz de confusión + tabla + guardar."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from PyQt6.QtWidgets import QApplication

from eeg_studio.ui import metrics_view


def main() -> int:
    app = QApplication(sys.argv)
    if not metrics_view.matplotlib_available():
        print("matplotlib no disponible; se omite (la app usaría el texto).")
        return 0

    metrics = {
        "labels": ["izq", "der", "pies"],
        "accuracy": 0.78,
        "confusion": [[20, 3, 1], [4, 18, 2], [1, 2, 21]],
        "precision": [0.80, 0.78, 0.88],
        "recall": [0.83, 0.75, 0.88],
        "f1": [0.81, 0.76, 0.88],
        "support": [24, 24, 24],
    }

    print("[1] La figura (matriz de confusión + F1) se construye y guarda")
    fig = metrics_view.build_figure(metrics)
    png = os.path.join(tempfile.mkdtemp(), "cm.png")
    fig.savefig(png, dpi=100)
    assert os.path.isfile(png) and os.path.getsize(png) > 1000, os.path.getsize(png)
    print(f"    imagen guardada: {os.path.getsize(png)} bytes")

    print("[2] La tabla de scores tiene filas por clase y valores")
    table = metrics_view.build_scores_table(metrics)
    assert table.rowCount() == 3 and table.columnCount() == 4
    assert table.item(0, 2).text() == "0.81"          # F1 de la clase 0
    assert table.item(0, 3).text() == "24"            # soporte
    assert table.item(0, 0).background().color().isValid()
    print(f"    {table.rowCount()}×{table.columnCount()} con color en las celdas")

    print("[3] El diálogo completo se construye (sin ejecutarlo)")
    dlg = metrics_view.build_metrics_dialog(None, "Métricas — rf_1",
                                            "rf_1 · Random Forest", metrics, "informe de texto")
    assert dlg is not None and dlg.findChildren(type(table))
    print("    diálogo con canvas + tabla + botones ✓")

    print("\nVISOR DE MÉTRICAS OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

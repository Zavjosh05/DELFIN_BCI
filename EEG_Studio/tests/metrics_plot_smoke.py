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

    print("[1] La figura (conteos y normalizada) se construye y guarda")
    tmp = tempfile.mkdtemp()
    for norm in (False, True):
        fig = metrics_view.build_figure(metrics, normalize=norm)
        png = os.path.join(tmp, f"cm_{norm}.png")
        fig.savefig(png, dpi=100)
        assert os.path.isfile(png) and os.path.getsize(png) > 1000, os.path.getsize(png)
    print("    conteos y normalizada (%) ✓")

    print("[2] La tabla de scores por clase tiene filas por clase y valores")
    table = metrics_view.build_scores_table(metrics)
    assert table.rowCount() == 3 and table.columnCount() == 4
    assert table.item(0, 2).text() == "0.81"          # F1 de la clase 0
    assert table.item(0, 3).text() == "24"            # soporte
    print(f"    {table.rowCount()}×{table.columnCount()} con color en las celdas")

    print("[3] Métricas GLOBALES (no por clase)")
    g = metrics_view.global_metrics(metrics)
    assert g["accuracy"] == 0.78 and g["support_total"] == 72, g
    assert abs(g["f1_macro"] - (0.81 + 0.76 + 0.88) / 3) < 1e-6, g["f1_macro"]
    gt = metrics_view.build_global_table(metrics)
    assert gt.rowCount() == 6 and gt.item(0, 1).text() == "78.0%"
    print(f"    exactitud={g['accuracy']:.2f} · F1 macro={g['f1_macro']:.3f} · soporte={g['support_total']}")

    print("[4] La imagen-informe incluye matriz + F1 + AMBAS tablas (todas las filas)")
    for norm in (False, True):
        rep = metrics_view.build_report_figure(metrics, "rf_1 · Random Forest", normalize=norm)
        # ejes: matriz, F1, colorbar, tabla por clase, tabla global (todo visible, sin scroll)
        assert len(rep.axes) >= 4, len(rep.axes)
        path = os.path.join(tmp, f"report_{norm}.png")
        rep.savefig(path, dpi=110)
        assert os.path.isfile(path) and os.path.getsize(path) > 8000, os.path.getsize(path)
    print("    informe con matriz(+norm) + F1 + tabla por clase + tabla global ✓")

    print("[5] El diálogo se construye con el toggle de normalización")
    dlg = metrics_view.build_metrics_dialog(None, "Métricas — rf_1",
                                            "rf_1 · Random Forest", metrics, "informe de texto")
    assert dlg.findChildren(type(gt)), "sin tablas"
    from PyQt6.QtWidgets import QCheckBox
    assert dlg.findChildren(QCheckBox), "falta el toggle de normalización"
    print("    diálogo + checkbox de normalización ✓")

    print("\nVISOR DE MÉTRICAS OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

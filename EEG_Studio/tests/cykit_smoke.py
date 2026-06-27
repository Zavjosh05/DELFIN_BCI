"""Verifica el configurador/lanzador de CyKit (offscreen).

* El comando por defecto reproduce un comando habitual (banderas y cantidades).
* Activar/desactivar banderas y cambiar cantidades altera el comando.
* «Aplicar a la fuente» configura la fuente CyKit/TCP del panel.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.ui.cykit_launcher import CyKitLauncherDialog  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402

EXPECTED_FLAGS = "openvibe+generic+nocounter+noheader+nobattery+float+ovdelay:100+ovsamples:004"


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    panel = win.acq_panel
    dlg = CyKitLauncherDialog(panel, panel)

    print("[1] Comando por defecto = un comando habitual")
    cmd = dlg._command_string()
    print(f"    {cmd}")
    assert dlg._flag_string() == EXPECTED_FLAGS, f"banderas: {dlg._flag_string()}"
    assert "127.0.0.1 5151 6" in cmd, "host/puerto/modelo por defecto incorrectos"

    print("[2] Desactivar 'nocounter' y cambiar 'ovsamples'")
    dlg._bool_widgets["nocounter"].setChecked(False)
    chk, spin, _ = dlg._value_widgets["ovsamples"]
    spin.setValue(8)
    assert "nocounter" not in dlg._flag_string(), "no se quitó nocounter"
    assert "ovsamples:008" in dlg._flag_string(), "no cambió ovsamples"

    print("[3] Aplicar a la fuente (con nocounter OFF -> columna inicial 2)")
    dlg._apply_to_source()
    assert panel.tcp_host.text() == "127.0.0.1"
    assert panel.tcp_port.value() == 5151
    assert panel.tcp_start.value() == 2, "columna inicial debería ser 2 sin nocounter"
    assert panel.source_combo.currentData() == "tcp", "no seleccionó la fuente CyKit/TCP"
    print(f"    fuente=tcp host={panel.tcp_host.text()} puerto={panel.tcp_port.value()} "
          f"col_inicial={panel.tcp_start.value()}")

    print("[4] Con nocounter ON -> columna inicial 0")
    dlg._bool_widgets["nocounter"].setChecked(True)
    dlg._apply_to_source()
    assert panel.tcp_start.value() == 0, "con nocounter, la columna inicial debería ser 0"
    print("    columna inicial=0 ✓")

    print("\nCYKIT LAUNCHER OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

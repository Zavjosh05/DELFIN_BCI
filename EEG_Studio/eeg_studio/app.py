"""Arranque de la aplicación Qt."""
from __future__ import annotations

import multiprocessing as mp
import sys

from PyQt6.QtWidgets import QApplication

from .config import APP_NAME, ORG_NAME
from .ui.main_window import MainWindow


def main() -> int:
    # Necesario para que ProcessPoolExecutor funcione al congelar/empaquetar.
    mp.freeze_support()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

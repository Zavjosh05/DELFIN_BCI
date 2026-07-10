"""Arranque de la aplicación Qt."""
from __future__ import annotations

import multiprocessing as mp
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from .config import APP_NAME, ORG_NAME
from .ui.main_window import MainWindow
from .ui.theme import apply_dark_theme


def main() -> int:
    # Necesario para que ProcessPoolExecutor funcione al congelar/empaquetar.
    mp.freeze_support()

    # Contextos OpenGL COMPARTIDOS: al reubicar/flotar el panel que contiene la
    # vista 3D del brazo (GLViewWidget), el widget se reparenta y su contexto
    # OpenGL se recrea; sin contextos compartidos eso hacía crashear la app.
    # Debe fijarse ANTES de crear la QApplication.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    apply_dark_theme(app)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

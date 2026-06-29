"""Tema oscuro coherente para toda la aplicación.

El visor de señal y el historial ya eran oscuros; sin un tema global, el resto
de la interfaz (menús, paneles, botones) salía en gris claro con Fusion y
desentonaba. Aquí se define una paleta + hoja de estilo única a juego con el
fondo del visor (``#101317``) y los acentos ya usados (azul de selección,
verde de predicción, ámbar de marcadores).
"""
from __future__ import annotations

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

# --- Paleta base (azul-grisáceos) -----------------------------------------
BG = "#15191e"          # fondo de ventana
SURFACE = "#1e242b"     # paneles, campos, listas
ELEVATED = "#232a32"    # filas alternas, emergentes
BORDER = "#2c343d"      # bordes
TEXT = "#c8d0d8"        # texto principal (igual que el foreground del visor)
MUTED = "#8a929b"       # texto secundario / pistas
DISABLED = "#5b646e"    # deshabilitado
ACCENT = "#2f6fb0"      # azul de acento (selección, foco)
ACCENT_HI = "#3d86cc"   # azul más claro (hover/enlaces)
SUCCESS = "#7fd1b9"     # verde (estados positivos)

_QSS = f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}

QToolTip {{
    background: {ELEVATED}; color: {TEXT};
    border: 1px solid {BORDER}; padding: 4px 6px;
}}

/* Menús */
QMenuBar {{ background: {BG}; }}
QMenuBar::item {{ padding: 4px 10px; background: transparent; }}
QMenuBar::item:selected {{ background: {SURFACE}; }}
QMenu {{ background: {SURFACE}; border: 1px solid {BORDER}; }}
QMenu::item {{ padding: 5px 22px; }}
QMenu::item:selected {{ background: {ACCENT}; color: #ffffff; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}

/* Barra de herramientas */
QToolBar {{ background: {BG}; border: none; spacing: 2px; padding: 3px; }}
QToolButton {{ padding: 4px 8px; border-radius: 4px; color: {TEXT}; }}
QToolButton:hover {{ background: {SURFACE}; }}
QToolButton:pressed {{ background: {ACCENT}; color: #ffffff; }}
QToolButton:disabled {{ color: {DISABLED}; }}

/* Docks */
QDockWidget {{ titlebar-close-icon: none; color: {TEXT}; }}
QDockWidget::title {{
    background: {SURFACE}; padding: 5px 8px;
    border-bottom: 1px solid {BORDER};
}}

/* Pestañas */
QTabWidget::pane {{ border: 1px solid {BORDER}; background: {BG}; }}
QTabBar::tab {{
    background: {BG}; color: {MUTED};
    padding: 6px 12px; border: 1px solid transparent; border-bottom: none;
}}
QTabBar::tab:selected {{ background: {SURFACE}; color: {TEXT}; border-color: {BORDER}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}

/* Botones */
QPushButton {{
    background: {SURFACE}; border: 1px solid {BORDER};
    padding: 5px 12px; border-radius: 4px;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT}; color: #ffffff; }}
QPushButton:disabled {{ color: {DISABLED}; border-color: {SURFACE}; }}
QPushButton:default {{ border-color: {ACCENT}; }}

/* Campos de entrada */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QAbstractSpinBox, QPlainTextEdit, QTextEdit {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 4px; padding: 3px 6px; selection-background-color: {ACCENT};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QAbstractSpinBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{
    background: {SURFACE}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT}; selection-color: #ffffff;
}}

/* Listas y tablas */
QListWidget, QTreeWidget, QTableWidget {{
    background: {SURFACE}; border: 1px solid {BORDER};
    alternate-background-color: {ELEVATED};
}}
QListWidget::item:selected, QTreeWidget::item:selected {{ background: {ACCENT}; color: #ffffff; }}
QHeaderView::section {{
    background: {ELEVATED}; color: {MUTED};
    padding: 4px 6px; border: none; border-right: 1px solid {BORDER};
}}

/* Grupos */
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 6px;
    margin-top: 10px; padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {MUTED};
}}

/* Casillas y radios */
QCheckBox, QRadioButton {{ spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; }}

/* Barra de progreso */
QProgressBar {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 4px; text-align: center; color: {TEXT};
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}

/* Barra de estado */
QStatusBar {{ background: {BG}; color: {MUTED}; border-top: 1px solid {BORDER}; }}

/* Barras de desplazamiento */
QScrollBar:vertical {{ background: {BG}; width: 11px; margin: 0; }}
QScrollBar:horizontal {{ background: {BG}; height: 11px; margin: 0; }}
QScrollBar::handle {{ background: {BORDER}; border-radius: 5px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {ACCENT}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""


def apply_dark_theme(app: QApplication) -> None:
    """Aplica estilo Fusion + paleta oscura + hoja de estilo global."""
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(BG))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(ELEVATED))
    pal.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Button, QColor(SURFACE))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(ELEVATED))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, QColor(ACCENT_HI))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(MUTED))
    dis = QPalette.ColorGroup.Disabled
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText,
                 QPalette.ColorRole.WindowText):
        pal.setColor(dis, role, QColor(DISABLED))
    app.setPalette(pal)
    app.setStyleSheet(_QSS)

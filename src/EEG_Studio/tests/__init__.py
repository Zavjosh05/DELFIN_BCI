"""Utilidades compartidas por las pruebas de humo."""
from __future__ import annotations

import os


def data_dir() -> str:
    """Carpeta de datos de ejemplo (con ``Prueba_001.csv``, etc.).

    Tras la reestructuración vive en ``data/raw/EEG/`` en la raíz del repositorio.
    Se localiza subiendo por los directorios padre (con respaldo a la ubicación
    histórica ``EEG/`` en la raíz), así sigue funcionando aunque el árbol cambie de
    profundidad (p. ej. al pasar de ``EEG_Studio/`` a ``src/EEG_Studio/``).
    """
    d = here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        for cand in (os.path.join(d, "data", "raw", "EEG"),
                     os.path.join(d, "EEG")):        # respaldo a la ubicación vieja
            if os.path.isdir(cand):
                return cand
        parent = os.path.dirname(d)
        if parent == d:                      # llegamos a la raíz del disco
            break
        d = parent
    return os.path.normpath(os.path.join(here, "..", "..", "..", "data", "raw", "EEG"))

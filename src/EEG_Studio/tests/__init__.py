"""Utilidades compartidas por las pruebas de humo."""
from __future__ import annotations

import os


def data_dir() -> str:
    """Carpeta de datos de ejemplo ``EEG/`` (con ``Prueba_001.csv``, etc.).

    Se localiza subiendo por los directorios padre desde este paquete, así sigue
    funcionando aunque el árbol del proyecto cambie de profundidad (p. ej. al pasar
    de ``EEG_Studio/`` a ``src/EEG_Studio/``). Si no se encuentra, cae en la
    ubicación histórica (dos niveles por encima de ``tests/``).
    """
    d = here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        cand = os.path.join(d, "EEG")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:                      # llegamos a la raíz del disco
            break
        d = parent
    return os.path.normpath(os.path.join(here, "..", "..", "EEG"))

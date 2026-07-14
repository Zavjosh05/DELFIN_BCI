"""Posiciones 2D de los electrodos para los mapas topográficos.

Coordenadas normalizadas dentro del **disco unitario** de la cabeza vista desde
arriba: el origen es el vértex (Cz), la **nariz** apunta a ``+y`` y la **oreja
derecha** a ``+x`` (convención habitual de los topomapas EEG). Son posiciones
aproximadas del sistema 10-20 para el montaje del casco **Emotiv EPOC+** (14
canales); bastan para visualizar la distribución espacial (no son medidas
clínicas).

Se usan en los mapas topográficos de los componentes ICA (``ui/ica_topomap_view``).
"""
from __future__ import annotations

# Nombre de electrodo (mayúsculas) -> (x, y) en el disco unitario.
# x: oreja izquierda (-) ↔ derecha (+);  y: occipucio (-) ↔ nariz (+).
EPOC_POSITIONS_2D: dict[str, tuple[float, float]] = {
    "AF3": (-0.25, 0.62), "AF4": (0.25, 0.62),
    "F7":  (-0.68, 0.40), "F8":  (0.68, 0.40),
    "F3":  (-0.34, 0.44), "F4":  (0.34, 0.44),
    "FC5": (-0.57, 0.16), "FC6": (0.57, 0.16),
    "T7":  (-0.85, 0.00), "T8":  (0.85, 0.00),
    "P7":  (-0.68, -0.40), "P8": (0.68, -0.40),
    "O1":  (-0.25, -0.72), "O2": (0.25, -0.72),
}


def positions_2d(names) -> list[tuple[float, float] | None]:
    """Devuelve la posición 2D de cada nombre de canal (en el mismo orden), o
    ``None`` para los que no tienen una posición conocida.

    El emparejamiento es por nombre normalizado (sin espacios, en mayúsculas),
    así que tolera alias como ``" af3 "``.
    """
    out: list[tuple[float, float] | None] = []
    for n in names:
        out.append(EPOC_POSITIONS_2D.get(str(n).strip().upper()))
    return out


def known_count(names) -> int:
    """Cuántos de ``names`` tienen posición conocida (para decidir si se puede
    dibujar un topomapa)."""
    return sum(1 for p in positions_2d(names) if p is not None)

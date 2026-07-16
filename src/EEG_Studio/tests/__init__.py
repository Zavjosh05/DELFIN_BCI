"""Utilidades compartidas por las pruebas de humo.

Incluye un generador de grabaciones EEG sintéticas (`sample_csv`) para que las
pruebas NO dependan de archivos grandes fuera de git (`data/raw/EEG/*.csv` está en
`.gitignore`): así cualquier máquina, con un clon limpio, puede correr la batería.
El CSV sintético es pequeño (rápido de parsear) pero estructuralmente idéntico a
uno de OpenViBE real: columnas ``Time:<fs>Hz``, ``Epoch``, ``Channel 1..N`` y las de
evento, con señal reproducible (senoides por banda + ruido) para que los filtros,
la ICA y las bandas de potencia tengan algo real que procesar.
"""
from __future__ import annotations

import os
import tempfile


def data_dir() -> str:
    """Carpeta de datos de ejemplo (con ``Prueba_001.csv``, etc.).

    OJO: esa carpeta está en `.gitignore`, así que puede no existir en un clon
    limpio. Las pruebas ya no deberían depender de ella; usa `sample_csv` en su
    lugar. Se conserva por compatibilidad y para el modo interactivo (pasar un CSV
    real por línea de comandos).

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


def sample_csv(path: str, *, n_epochs: int = 10, epoch_samples: int = 128,
               n_channels: int = 14, fs: float = 128.0, seed: int = 0,
               markers: list[tuple[int, str]] | None = None) -> str:
    """Escribe un CSV OpenViBE pequeño pero válido en ``path`` y lo devuelve.

    Por defecto: 14 canales × (10 épocas · 128 muestras) = 1280 muestras (~10 s a
    128 Hz), frente a los ~6000 de los CSV reales. Suficiente para segmentos, épocas
    y entrenamiento, pero varias veces más rápido de parsear.

    ``markers``: lista de ``(muestra, código)`` para marcar eventos (opcional).
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    n = n_epochs * epoch_samples
    t = np.arange(n) / fs
    cols: dict[str, object] = {f"Time:{int(fs)}Hz": t,
                               "Epoch": np.repeat(np.arange(n_epochs), epoch_samples)}
    for c in range(n_channels):
        f1 = 8.0 + (c % 5) * 2.0                       # ritmos 8..16 Hz por canal
        sig = (50.0 * np.sin(2 * np.pi * f1 * t + c)
               + 20.0 * np.sin(2 * np.pi * (f1 / 2) * t)
               + rng.normal(0, 5.0, n)
               + 4000.0)                               # offset crudo tipo EPOC+ (µV)
        cols[f"Channel {c + 1}"] = sig
    df = pd.DataFrame(cols)
    df["Event Id"] = ""
    df["Event Date"] = ""
    df["Event Duration"] = ""
    for sample_idx, code in (markers or []):
        if 0 <= sample_idx < n:
            df.loc[sample_idx, "Event Id"] = code
            df.loc[sample_idx, "Event Date"] = float(t[sample_idx])
            df.loc[sample_idx, "Event Duration"] = 0
    df.to_csv(path, index=False)
    return path


_SAMPLE_CACHE: dict[tuple, str] = {}


def sample_csv_path(name: str = "muestra.csv", **opts) -> str:
    """Ruta a un CSV sintético generado UNA vez por combinación de opciones.

    Para pruebas que solo necesitan «un CSV válido» y no les importa dónde vive.
    Se cachea para no reescribirlo en cada llamada dentro de una misma corrida.
    """
    key = (name, tuple(sorted(opts.items())))
    cached = _SAMPLE_CACHE.get(key)
    if cached and os.path.isfile(cached):
        return cached
    d = tempfile.mkdtemp(prefix="eeg_sample_")
    path = sample_csv(os.path.join(d, name), **opts)
    _SAMPLE_CACHE[key] = path
    return path

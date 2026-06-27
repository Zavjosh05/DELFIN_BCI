"""Carga de CSV de OpenViBE hacia el modelo :class:`Recording`.

El formato esperado tiene cabecera::

    Time:128Hz,Epoch,Channel 1,...,Channel 14,Event Id,Event Date,Event Duration

Este módulo es de *solo lectura*: nunca escribe sobre el CSV de origen.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from ..config import (
    EPOCH_COLUMN,
    EVENT_COLUMNS,
    TIME_COLUMN_PREFIX,
)
from .recording import Recording


def _detect_sample_rate(time_col_name: str, default: float) -> float:
    """Extrae la frecuencia de muestreo de un encabezado tipo ``Time:128Hz``."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*hz", time_col_name, re.IGNORECASE)
    return float(m.group(1)) if m else default


def load_recording(path: str, default_sample_rate: float = 128.0) -> Recording:
    """Lee un CSV de OpenViBE y devuelve una :class:`Recording` inmutable.

    Detecta automáticamente columnas de tiempo, época, eventos y canales EEG.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    cols = list(df.columns)

    # Columna de tiempo (primera que empieza por "Time:").
    time_cols = [c for c in cols if c.startswith(TIME_COLUMN_PREFIX)]
    if not time_cols:
        # Fallback: primera columna numérica monótona.
        time_col = cols[0]
        sample_rate = default_sample_rate
    else:
        time_col = time_cols[0]
        sample_rate = _detect_sample_rate(time_col, default_sample_rate)

    epoch_col = EPOCH_COLUMN if EPOCH_COLUMN in cols else None
    present_event_cols = [c for c in EVENT_COLUMNS if c in cols]

    # Los canales son el resto de columnas (numéricas) que no son estructurales.
    structural = {time_col, *([epoch_col] if epoch_col else []), *present_event_cols}
    channel_cols = [c for c in cols if c not in structural]

    # Asegurar que las columnas de canal son numéricas.
    data = df[channel_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    # data shape -> (n_samples, n_channels); guardamos como (n_channels, n_samples).
    data = data.T.copy()

    time = df[time_col].to_numpy(dtype=np.float64)

    epochs = None
    if epoch_col is not None:
        epochs = df[epoch_col].to_numpy()
        try:
            epochs = epochs.astype(np.int64)
        except (ValueError, TypeError):
            epochs = None

    events = _parse_events(df, present_event_cols)

    return Recording(
        source_path=os.path.abspath(path),
        channel_names=channel_cols,
        data=data,
        time=time,
        sample_rate=sample_rate,
        epochs=epochs,
        events=events,
    )


def _parse_events(df: pd.DataFrame, event_cols: list[str]) -> list[dict]:
    """Extrae marcadores no vacíos como lista de diccionarios."""
    if not event_cols or "Event Id" not in df.columns:
        return []
    events: list[dict] = []
    ev = df["Event Id"]
    mask = ev.notna() & (ev.astype(str).str.strip() != "")
    for idx in np.flatnonzero(mask.to_numpy()):
        row = df.iloc[int(idx)]
        events.append(
            {
                "sample": int(idx),
                "id": str(row.get("Event Id", "")).strip(),
                "date": str(row.get("Event Date", "")).strip(),
                "duration": str(row.get("Event Duration", "")).strip(),
            }
        )
    return events

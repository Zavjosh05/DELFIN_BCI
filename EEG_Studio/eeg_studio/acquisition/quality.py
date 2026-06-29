"""Indicadores de calidad de la señal en vivo: detecta canales ruidosos o sin señal.

Heurística simple sobre una ventana reciente (en µV), pensada como aviso visual,
no como medida clínica. Clasifica cada canal en:

* ``ok``       — varianza razonable y dentro de rango (buen contacto).
* ``plano``    — varianza casi nula: electrodo desconectado / sin señal.
* ``saturado`` — valores en el extremo del rango: electrodo a riel.
* ``ruido``    — oscilación excesiva: mal contacto, movimiento o interferencia.
"""
from __future__ import annotations

import numpy as np

# Umbrales heurísticos (en µV) sobre la ventana analizada.
FLAT_STD_UV = 0.5         # por debajo: canal plano (desconectado)
NOISE_PTP_UV = 1500.0     # pico-a-pico por encima: oscilación excesiva (ruido)
RAIL_ABS_UV = 40000.0     # |valor| extremo: saturado / a riel


def channel_status(x: np.ndarray) -> str:
    """Clasifica un canal (vector de muestras en µV)."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return "ruido"
    if float(np.max(np.abs(x))) > RAIL_ABS_UV:
        return "saturado"
    if float(np.std(x)) < FLAT_STD_UV:
        return "plano"
    if float(np.ptp(x)) > NOISE_PTP_UV:
        return "ruido"
    return "ok"


def assess(data: np.ndarray) -> dict:
    """Evalúa la calidad de ``data`` ``(n_canales, n_muestras)`` en µV.

    Devuelve estado por canal, recuentos y si la señal es mayoritariamente ruido.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    statuses = [channel_status(data[i]) for i in range(data.shape[0])]
    n = len(statuses)
    n_ok = sum(1 for s in statuses if s == "ok")
    n_bad = n - n_ok
    return {
        "statuses": statuses,
        "n": n,
        "n_ok": n_ok,
        "n_bad": n_bad,
        "is_noise": n > 0 and n_ok < n / 2,     # mayoría de canales malos
    }

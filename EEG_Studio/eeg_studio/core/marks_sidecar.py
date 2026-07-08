"""Archivo lateral ``<csv>.marks.json`` con los segmentos/marcas de cada grabación.

Se escribe **durante** la grabación (escritura atómica) para que los segmentos no
dependan de que se guarde el proyecto: si la app se cierra o falla, quedan en
disco junto al CSV y se recuperan al añadir la grabación como fuente. Es un
blindaje contra la pérdida de marcas.
"""
from __future__ import annotations

import json
import os
import time


def sidecar_path(csv_path: str) -> str:
    """Ruta del archivo lateral para un CSV (``rec.csv`` → ``rec.csv.marks.json``)."""
    return csv_path + ".marks.json"


def write_marks(csv_path: str, segments, fs: float = 128.0) -> None:
    """Escribe (atómicamente) los segmentos ``(inicio, fin, etiqueta)`` del CSV."""
    data = {
        "version": 1,
        "fs": float(fs),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "segments": [{"start": int(a), "stop": int(b), "label": str(lbl)}
                     for a, b, lbl in segments],
    }
    dst = sidecar_path(csv_path)
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, dst)                     # atómico: nunca deja el archivo a medias


def read_marks(csv_path: str) -> list[tuple[int, int, str]]:
    """Lee los segmentos del archivo lateral (``[]`` si no existe o está corrupto)."""
    path = sidecar_path(csv_path)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return [(int(s["start"]), int(s["stop"]), str(s["label"]))
                for s in data.get("segments", [])
                if int(s["stop"]) > int(s["start"])]
    except Exception:  # noqa: BLE001 — un lateral corrupto no debe romper nada
        return []


def remove_marks(csv_path: str) -> None:
    """Borra el archivo lateral (al descartar una grabación)."""
    try:
        path = sidecar_path(csv_path)
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass

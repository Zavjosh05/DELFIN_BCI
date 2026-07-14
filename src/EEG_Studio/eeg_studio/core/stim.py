"""Estimulación sincronizada: descubrir los videos de estímulo, mapearlos a las
clases del proyecto Delfin y calcular los **segmentos exactos** a partir de la
línea de tiempo configurada (elimina el error humano al etiquetar).
"""
from __future__ import annotations

import os

# Clases del proyecto Delfin (mismas 6 del brazo).
DELFIN_CLASSES = ["arriba", "abajo", "izquierda", "derecha", "agarre", "soltar"]

# Palabra clave en el nombre del archivo -> clase.
_NAME_TO_CLASS = {
    "arriba": "arriba", "abajo": "abajo", "izquierda": "izquierda",
    "derecha": "derecha", "agarre": "agarre", "agarrar": "agarre",
    "soltar": "soltar",
}

_VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def class_from_filename(name: str) -> str | None:
    """Detecta la clase Delfin por el nombre del archivo (o ``None``)."""
    low = os.path.basename(name).lower()
    for key, cls in _NAME_TO_CLASS.items():
        if key in low:
            return cls
    return None


def find_videos_dir(start: str | None = None) -> str | None:
    """Localiza ``data/videos`` subiendo por los directorios padre."""
    d = os.path.abspath(start or __file__)
    if os.path.isfile(d):
        d = os.path.dirname(d)
    for _ in range(8):
        cand = os.path.join(d, "data", "videos")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def discover_videos(videos_dir: str | None = None) -> list[dict]:
    """Lista los videos de estímulo disponibles con su clase autodetectada."""
    vd = videos_dir or find_videos_dir()
    if not vd or not os.path.isdir(vd):
        return []
    out = []
    for fn in sorted(os.listdir(vd)):
        if fn.lower().endswith(_VIDEO_EXT):
            out.append({"path": os.path.abspath(os.path.join(vd, fn)),
                        "name": fn, "label": class_from_filename(fn)})
    return out


def compute_segments(events, fs: float, base_sample: int = 0,
                     n_samples: int | None = None) -> list[tuple]:
    """Convierte los eventos de tipo ``segment`` (tiempos en ms, relativos al inicio
    del video) en tuplas de **muestras** ``(inicio, fin, etiqueta)`` de la grabación.

    ``base_sample`` es la muestra de la grabación que coincide con el inicio del
    video (para descontar el pequeño desfase entre iniciar la grabación y el video).
    """
    segs = []
    for e in events:
        if e.get("kind") != "segment":
            continue
        s = base_sample + int(round(float(e["start"]) / 1000.0 * fs))
        t = base_sample + int(round(float(e["stop"]) / 1000.0 * fs))
        s, t = sorted((s, t))
        if n_samples is not None:
            t = min(t, n_samples)
            s = min(s, n_samples)
        if t - s >= 1:
            segs.append((s, t, str(e.get("label", ""))))
    return segs


def markers_in_order(events) -> list[dict]:
    """Eventos de tipo ``marker`` ordenados por tiempo (ms)."""
    ms = [e for e in events if e.get("kind") == "marker"]
    return sorted(ms, key=lambda e: float(e.get("t", 0)))


def project_classes(project) -> list[str]:
    """Clases disponibles en el proyecto: de los segmentos ya etiquetados y de los
    estímulos configurados. General — NO hardcodea las clases de Delfin (esas solo
    se usan para autodetectar la clase por el nombre del archivo)."""
    classes: set[str] = set()
    if project is not None:
        try:
            classes.update(project.labels())
        except Exception:  # noqa: BLE001
            pass
        for c in project.stim_videos():
            for e in c.get("events", []):
                if e.get("label"):
                    classes.add(str(e["label"]))
    return sorted(classes)


def relocate_video(path: str, search_dir: str | None = None) -> str | None:
    """Ruta válida del video, buscando por orden:

    1. la ruta original, si existe (mismo equipo);
    2. el MISMO nombre dentro de ``search_dir`` (la carpeta que indique el usuario);
    3. el MISMO nombre en la carpeta **``data/videos``** del proyecto — así, al
       importar una configuración de otro equipo, los videos de siempre se
       encuentran solos y no hace falta preguntar nada.

    Devuelve ``None`` si no aparece en ninguna."""
    if path and os.path.isfile(path):
        return path
    if not path:
        return None
    base = os.path.basename(path)
    for folder in (search_dir, find_videos_dir()):
        if not folder:
            continue
        cand = os.path.join(folder, base)
        if os.path.isfile(cand):
            return cand
    return None


def default_events(label: str, duration_ms: int) -> list[dict]:
    """Configuración inicial razonable: una marca al inicio del movimiento y un
    segmento cubriendo el grueso del video (el usuario lo ajusta en la línea de
    tiempo)."""
    if duration_ms <= 0:
        return []
    start = int(duration_ms * 0.15)
    stop = int(duration_ms * 0.85)
    return [
        {"kind": "marker", "t": start, "label": label},
        {"kind": "segment", "start": start, "stop": stop, "label": label},
    ]

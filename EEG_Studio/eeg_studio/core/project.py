"""Modelo de proyecto de EEG Studio.

Un proyecto es una carpeta ``<nombre>.eegproj`` que **referencia** los CSV de
origen por ruta (sin copiarlos ni modificarlos) y guarda en archivos locales:

* ``project.json``  -> manifiesto: fuentes, pipeline, segmentos, alias, dataset.
* ``changelog.json``-> bitácora de control de cambios (undo/redo + auditoría).
* ``cache/``        -> arreglos procesados en caché (``.npz``).
* ``datasets/``     -> datasets exportados para el modelo.
* ``models/``       -> modelos entrenados.

El estado editable se organiza en "secciones" (pipeline, segments, ...). Toda
modificación pasa por :meth:`Project.edit`, que la registra en el ``ChangeLog``
para poder deshacerse. La fuente original (CSV) jamás se escribe.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..config import (
    APP_VERSION,
    CACHE_DIR,
    CHANGELOG_FILE,
    DATASETS_DIR,
    DISK_CACHE_PROCESSED,
    EPOC_CHANNELS,
    MODELS_DIR,
    PROJECT_EXT,
    PROJECT_MANIFEST,
    RECORDINGS_DIR,
)
from . import preprocessing
from .changelog import ChangeLog, EditCommand, snapshot
from .csv_loader import load_recording
from .recording import Recording


def _default_state() -> dict:
    return {
        "pipeline": [],          # lista de pasos de preprocesamiento
        "segments": [],          # lista de segmentos etiquetados
        "channel_aliases": {},   # nombre_original -> alias mostrado
        "excluded_channels": [], # nombres de canal excluidos
        "dataset": {"use_bands": True, "use_time": True},
    }


class Project:
    def __init__(self, path: str, name: str) -> None:
        self.path = os.path.abspath(path)        # carpeta .eegproj
        self.name = name
        self.state: dict = _default_state()
        self.sources: list[dict] = []            # {id, path, alias}
        self.changelog = ChangeLog()
        self._recordings: dict[str, Recording] = {}      # id -> Recording (lazy)
        self._processed: dict[str, tuple[str, np.ndarray]] = {}  # id -> (firma, datos)
        self._lock = threading.Lock()                    # protege las cachés entre hilos

    # ------------------------------------------------------------------ #
    # Creación / apertura / guardado
    # ------------------------------------------------------------------ #
    @classmethod
    def create(cls, folder: str, name: str) -> "Project":
        path = folder if folder.endswith(PROJECT_EXT) else os.path.join(folder, name + PROJECT_EXT)
        os.makedirs(path, exist_ok=True)
        for sub in (CACHE_DIR, DATASETS_DIR, MODELS_DIR, RECORDINGS_DIR):
            os.makedirs(os.path.join(path, sub), exist_ok=True)
        proj = cls(path, name)
        proj.save()
        return proj

    @classmethod
    def open(cls, path: str) -> "Project":
        manifest_path = os.path.join(path, PROJECT_MANIFEST)
        with open(manifest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        proj = cls(path, data.get("name", os.path.basename(path)))
        proj.state = {**_default_state(), **data.get("state", {})}
        proj.sources = data.get("sources", [])
        log_path = os.path.join(path, CHANGELOG_FILE)
        if os.path.isfile(log_path):
            with open(log_path, "r", encoding="utf-8") as fh:
                proj.changelog = ChangeLog.from_dict(json.load(fh))
        return proj

    def save(self) -> None:
        for sub in (CACHE_DIR, DATASETS_DIR, MODELS_DIR, RECORDINGS_DIR):
            os.makedirs(os.path.join(self.path, sub), exist_ok=True)
        manifest = {
            "name": self.name,
            "version": APP_VERSION,
            "sources": self.sources,
            "state": self.state,
        }
        with open(os.path.join(self.path, PROJECT_MANIFEST), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        with open(os.path.join(self.path, CHANGELOG_FILE), "w", encoding="utf-8") as fh:
            json.dump(self.changelog.to_dict(), fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # Fuentes (CSV)
    # ------------------------------------------------------------------ #
    def add_source(self, csv_path: str, alias: str | None = None) -> dict:
        csv_path = os.path.abspath(csv_path)
        alias = alias or os.path.splitext(os.path.basename(csv_path))[0]
        source = {"id": uuid.uuid4().hex[:8], "path": csv_path, "alias": alias}
        new_sources = self.sources + [source]
        self._commit("sources", new_sources, f"Añadir fuente «{alias}»",
                     setter=lambda v: setattr(self, "sources", v))
        # Carga inmediata para validar y para fijar alias de canales por defecto.
        rec = self.get_recording(source["id"])
        if not self.state["channel_aliases"]:
            self._set_default_channel_aliases(rec)
        return source

    def remove_source(self, source_id: str) -> None:
        new_sources = [s for s in self.sources if s["id"] != source_id]
        self._commit("sources", new_sources, "Eliminar fuente",
                     setter=lambda v: setattr(self, "sources", v))
        self._recordings.pop(source_id, None)
        self._processed.pop(source_id, None)

    def get_source(self, source_id: str) -> dict | None:
        return next((s for s in self.sources if s["id"] == source_id), None)

    def get_recording(self, source_id: str) -> Recording:
        with self._lock:
            rec = self._recordings.get(source_id)
        if rec is not None:
            return rec
        src = self.get_source(source_id)
        if src is None:
            raise KeyError(f"Fuente inexistente: {source_id}")
        rec = load_recording(src["path"])           # E/S pesada fuera del lock
        with self._lock:
            self._recordings[source_id] = rec
        return rec

    def _set_default_channel_aliases(self, rec: Recording) -> None:
        aliases = {}
        for i, name in enumerate(rec.channel_names):
            aliases[name] = EPOC_CHANNELS[i] if i < len(EPOC_CHANNELS) else name
        self.state["channel_aliases"] = aliases

    def display_channel_names(self, rec: Recording) -> list[str]:
        aliases = self.state.get("channel_aliases", {})
        return [aliases.get(n, n) for n in rec.channel_names]

    # ------------------------------------------------------------------ #
    # Procesamiento (no destructivo): pipeline aplicado sobre copias
    # ------------------------------------------------------------------ #
    def _pipeline_signature(self) -> str:
        raw = json.dumps(self.state["pipeline"], sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get_processed(self, source_id: str) -> np.ndarray:
        """Devuelve la señal completa con el pipeline aplicado (en caché en RAM).

        Seguro para hilos: el cómputo pesado (filtros, ICA) corre fuera del lock,
        de modo que varias fuentes pueden procesarse en paralelo. Nunca modifica
        la grabación original; opera sobre una copia.
        """
        rec = self.get_recording(source_id)
        sig = self._pipeline_signature()
        with self._lock:
            cached = self._processed.get(source_id)
            if cached and cached[0] == sig:
                return cached[1]

        pipeline = self.state["pipeline"]
        if not pipeline:
            out = rec.data.copy()                 # sin procesar: no se cachea en disco
        else:
            out = self._load_disk_cache(source_id, sig)
            if out is None:
                out = preprocessing.apply_pipeline(rec.data, rec.sample_rate, pipeline)
                self._write_disk_cache(source_id, sig, out)
        with self._lock:
            self._processed[source_id] = (sig, out)
        return out

    # --- Caché en disco de la señal procesada -----------------------------
    def _disk_cache_path(self, source_id: str, sig: str) -> str:
        return os.path.join(self.path, CACHE_DIR, f"{source_id}_{sig[:16]}.cache.npz")

    def _load_disk_cache(self, source_id: str, sig: str) -> np.ndarray | None:
        if not DISK_CACHE_PROCESSED:
            return None
        path = self._disk_cache_path(source_id, sig)
        if not os.path.isfile(path):
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["sig"]) == sig:          # verifica la firma completa
                    return np.ascontiguousarray(z["data"], dtype=np.float64)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _write_disk_cache(self, source_id: str, sig: str, data: np.ndarray) -> None:
        if not DISK_CACHE_PROCESSED:
            return
        cache_dir = os.path.join(self.path, CACHE_DIR)
        try:
            os.makedirs(cache_dir, exist_ok=True)
            path = self._disk_cache_path(source_id, sig)
            # Conservar solo la versión vigente: borrar cachés antiguas de la fuente.
            for f in os.listdir(cache_dir):
                if f.startswith(f"{source_id}_") and f.endswith(".cache.npz") \
                        and f != os.path.basename(path):
                    try:
                        os.remove(os.path.join(cache_dir, f))
                    except OSError:
                        pass
            np.savez_compressed(path, data=data, sig=np.array(sig))
        except Exception:  # noqa: BLE001 - la caché es best-effort
            pass

    def prewarm(self, source_ids: list[str], max_workers: int | None = None) -> None:
        """Pre-calcula en paralelo (hilos) la señal procesada de varias fuentes.

        scipy/numpy liberan el GIL, así que los filtros de distintas fuentes se
        solapan en varios núcleos. Acelera la construcción de datasets multi-CSV.
        """
        ids = list(dict.fromkeys(source_ids))
        if len(ids) <= 1:
            for sid in ids:
                self.get_processed(sid)
            return
        workers = max_workers or min(len(ids), (os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(self.get_processed, ids))

    def invalidate_processed(self) -> None:
        with self._lock:
            self._processed.clear()

    def cache_processed_to_disk(self, source_id: str) -> str:
        """Escribe la señal procesada como ``.npz`` dentro del proyecto.

        Refuerza la regla de no destructividad: el resultado se guarda en los
        archivos locales del proyecto, no en el CSV de origen.
        """
        data = self.get_processed(source_id)
        rec = self.get_recording(source_id)
        out_path = os.path.join(self.path, CACHE_DIR, f"{source_id}_processed.npz")
        np.savez_compressed(
            out_path,
            data=data,
            sample_rate=rec.sample_rate,
            channels=np.array(self.display_channel_names(rec), dtype=object),
        )
        return out_path

    # ------------------------------------------------------------------ #
    # Segmentos (agrupación / aislamiento de señales para el dataset)
    # ------------------------------------------------------------------ #
    def add_segment(self, source_id: str, start: int, stop: int, label: str,
                    channels: list[int] | None = None, note: str = "") -> dict:
        seg = {
            "id": uuid.uuid4().hex[:8],
            "source_id": source_id,
            "start": int(start),
            "stop": int(stop),
            "label": label,
            "channels": channels,   # None => todos los canales
            "note": note,
        }
        new_segments = self.state["segments"] + [seg]
        self._commit("segments", new_segments, f"Añadir segmento «{label}»")
        return seg

    def remove_segment(self, segment_id: str) -> None:
        new_segments = [s for s in self.state["segments"] if s["id"] != segment_id]
        self._commit("segments", new_segments, "Eliminar segmento")

    def relabel_segment(self, segment_id: str, label: str) -> None:
        new_segments = snapshot(self.state["segments"])
        for s in new_segments:
            if s["id"] == segment_id:
                s["label"] = label
        self._commit("segments", new_segments, f"Reetiquetar a «{label}»")

    def segment_data(self, seg: dict) -> tuple[np.ndarray, float]:
        """Datos procesados de un segmento: ``(array(n_canales, n_muestras), fs)``."""
        rec = self.get_recording(seg["source_id"])
        full = self.get_processed(seg["source_id"])
        start, stop = seg["start"], seg["stop"]
        if seg.get("channels"):
            data = full[np.asarray(seg["channels"]), start:stop]
        else:
            data = full[:, start:stop]
        return np.ascontiguousarray(data), rec.sample_rate

    def labels(self) -> list[str]:
        return sorted({s["label"] for s in self.state["segments"]})

    def segments_from_markers(self, source_id: str, window: int = 0) -> int:
        """Crea segmentos a partir de los marcadores (columna Event Id).

        Cada marcador genera un segmento etiquetado con el texto del marcador, de
        modo que los marcadores **sirven como clases**. ``window`` = nº de muestras
        tras el marcador (0 = hasta el siguiente marcador o el final). Devuelve el
        número de segmentos creados.
        """
        rec = self.get_recording(source_id)
        events = rec.events
        if not events:
            return 0
        starts = [int(e["sample"]) for e in events]
        new_segments = list(self.state["segments"])
        created = 0
        for i, ev in enumerate(events):
            start = starts[i]
            if window and window > 0:
                stop = min(start + int(window), rec.n_samples)
            else:
                stop = starts[i + 1] if i + 1 < len(starts) else rec.n_samples
            if stop - start < 2:
                continue
            label = str(ev.get("id", "")).strip() or "marcador"
            new_segments.append({
                "id": uuid.uuid4().hex[:8],
                "source_id": source_id,
                "start": start,
                "stop": int(stop),
                "label": label,
                "channels": None,
                "note": "desde marcador",
            })
            created += 1
        if created:
            self._commit("segments", new_segments,
                         f"Segmentos desde marcadores ({created})")
        return created

    # ------------------------------------------------------------------ #
    # Preprocesamiento: edición del pipeline
    # ------------------------------------------------------------------ #
    def add_pipeline_step(self, step_type: str, params: dict | None = None) -> None:
        params = params if params is not None else dict(preprocessing.STEP_DEFAULTS.get(step_type, {}))
        new_pipeline = self.state["pipeline"] + [{"type": step_type, "params": params}]
        label = preprocessing.STEP_LABELS.get(step_type, step_type)
        self._commit("pipeline", new_pipeline, f"Añadir paso: {label}")
        self.invalidate_processed()

    def remove_pipeline_step(self, index: int) -> None:
        new_pipeline = [s for i, s in enumerate(self.state["pipeline"]) if i != index]
        self._commit("pipeline", new_pipeline, "Eliminar paso de preprocesamiento")
        self.invalidate_processed()

    def update_pipeline_step(self, index: int, params: dict) -> None:
        new_pipeline = snapshot(self.state["pipeline"])
        new_pipeline[index]["params"] = params
        self._commit("pipeline", new_pipeline, "Modificar parámetros de paso")
        self.invalidate_processed()

    def move_pipeline_step(self, index: int, delta: int) -> None:
        new_pipeline = snapshot(self.state["pipeline"])
        j = index + delta
        if 0 <= j < len(new_pipeline):
            new_pipeline[index], new_pipeline[j] = new_pipeline[j], new_pipeline[index]
            self._commit("pipeline", new_pipeline, "Reordenar pipeline")
            self.invalidate_processed()

    # ------------------------------------------------------------------ #
    # Control de cambios genérico
    # ------------------------------------------------------------------ #
    def _commit(self, section: str, new_value, description: str, setter=None) -> None:
        """Aplica un cambio a una sección y lo registra para undo/redo."""
        before = snapshot(self.sources if section == "sources" else self.state.get(section))
        after = snapshot(new_value)
        cmd = EditCommand(section=section, before=before, after=after, description=description)
        self._apply_section(section, after, setter)
        self.changelog.push(cmd)

    def edit(self, section: str, new_value, description: str) -> None:
        """API pública para editar una sección arbitraria del estado."""
        self._commit(section, new_value, description)
        if section == "pipeline":
            self.invalidate_processed()

    def _apply_section(self, section: str, value, setter=None) -> None:
        if setter is not None:
            setter(value)
        elif section == "sources":
            self.sources = value
        else:
            self.state[section] = value

    def undo(self) -> bool:
        cmd = self.changelog.undo()
        if cmd is None:
            return False
        self._apply_section(cmd.section, snapshot(cmd.before))
        if cmd.section == "pipeline":
            self.invalidate_processed()
        return True

    def redo(self) -> bool:
        cmd = self.changelog.redo()
        if cmd is None:
            return False
        self._apply_section(cmd.section, snapshot(cmd.after))
        if cmd.section == "pipeline":
            self.invalidate_processed()
        return True

    def goto_history(self, target_applied: int) -> None:
        """Navega en el historial hasta dejar ``target_applied`` comandos aplicados."""
        guard = 0
        while self.changelog.applied_count() > target_applied and guard < 10000:
            if not self.undo():
                break
            guard += 1
        while self.changelog.applied_count() < target_applied and guard < 10000:
            if not self.redo():
                break
            guard += 1

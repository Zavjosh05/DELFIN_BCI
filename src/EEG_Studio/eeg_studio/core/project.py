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
import re
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
    IMPORTED_DIR,
    MODELS_DIR,
    PROJECT_EXT,
    PROJECT_MANIFEST,
    RECORDINGS_DIR,
)
from . import preprocessing
from .changelog import ChangeLog, EditCommand, snapshot
from .csv_loader import load_recording
from .recording import Recording


def _safe_filename(name: str) -> str:
    """Nombre de archivo seguro (sin caracteres problemáticos; espacios → '_')."""
    name = re.sub(r"[^\w\- ]", "", name or "", flags=re.UNICODE).strip()
    return re.sub(r"\s+", "_", name)


def _split_csv_ext(basename: str) -> str:
    """Devuelve la extensión conservando ``.csv.gz`` como una sola."""
    low = basename.lower()
    if low.endswith(".csv.gz"):
        return ".csv.gz"
    return os.path.splitext(basename)[1]


def _default_state() -> dict:
    return {
        # Varios pipelines por proyecto (pestañas). `pipeline` es un ESPEJO de los
        # pasos del pipeline activo, para que todo el código que lo lee siga igual.
        "pipelines": [{"name": "Pipeline 1", "steps": []}],
        "active_pipeline": 0,
        "pipeline": [],          # espejo del pipeline activo (no editar directamente)
        "segments": [],          # lista de segmentos etiquetados
        "channel_aliases": {},   # nombre_original -> alias mostrado
        "excluded_channels": [], # nombres de canal excluidos
        "cuts": {},              # source_id -> [[inicio, fin], ...] tramos eliminados
        "dataset": {"use_bands": True, "use_time": True},
        "stim_videos": [],       # estímulos configurados (video -> marcas/segmentos en tiempo)
        # Configuraciones de modelo guardadas (hiperparámetros SIN entrenar):
        # {"name", "classifier_name", "clf_params"|"nn_config"|"raw_window"}
        "model_configs": [],
    }


class Project:
    def __init__(self, path: str, name: str) -> None:
        self.path = os.path.abspath(path)        # carpeta .eegproj
        self.name = name
        self.state: dict = _default_state()
        self._sync_pipeline_mirror()
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
        for sub in (CACHE_DIR, DATASETS_DIR, MODELS_DIR, RECORDINGS_DIR, IMPORTED_DIR):
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
        loaded = data.get("state", {})
        proj.state = {**_default_state(), **loaded}
        proj._migrate_pipelines(has_pipelines="pipelines" in loaded)
        # Resuelve las rutas (relativas) de las fuentes contra la ubicación ACTUAL
        # del proyecto: así abrir la carpeta movida/renombrada sigue encontrándolas.
        proj.sources = data.get("sources", [])
        for s in proj.sources:
            if s.get("path"):
                s["path"] = proj._resolve_path(s["path"])
        log_path = os.path.join(path, CHANGELOG_FILE)
        if os.path.isfile(log_path):
            with open(log_path, "r", encoding="utf-8") as fh:
                proj.changelog = ChangeLog.from_dict(json.load(fh))
            proj._resolve_changelog_paths()
        return proj

    def save(self) -> None:
        for sub in (CACHE_DIR, DATASETS_DIR, MODELS_DIR, RECORDINGS_DIR, IMPORTED_DIR):
            os.makedirs(os.path.join(self.path, sub), exist_ok=True)
        # Las rutas de las fuentes internas se guardan RELATIVAS al proyecto para
        # que la carpeta sea portátil (mover/copiar/renombrar sin romper enlaces).
        sources_persist = [{**s, "path": self._persist_path(s.get("path", ""))}
                           for s in self.sources]
        manifest = {
            "name": self.name,
            "version": APP_VERSION,
            "sources": sources_persist,
            "state": self.state,
        }
        # Escritura ATÓMICA (a .tmp y os.replace): un fallo/corte a mitad de guardado
        # nunca deja el project.json/changelog.json corrupto.
        self._atomic_write_json(os.path.join(self.path, PROJECT_MANIFEST), manifest)
        self._atomic_write_json(os.path.join(self.path, CHANGELOG_FILE),
                                self._persisted_changelog())

    @staticmethod
    def _atomic_write_json(path: str, data) -> None:
        """Escribe ``data`` como JSON de forma atómica (temporal + reemplazo)."""
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    # ------------------------------------------------------------------ #
    # Fuentes (CSV)
    # ------------------------------------------------------------------ #
    def add_source(self, csv_path: str, alias: str | None = None,
                   recording: Recording | None = None,
                   source_id: str | None = None) -> dict:
        """Añade una fuente CSV al proyecto.

        ``recording`` permite pasar la grabación **ya cargada** (p. ej. en un hilo
        de trabajo) para no bloquear con la lectura del CSV; si es ``None``, se
        carga aquí (valida el archivo y fija los alias de canal por defecto).
        ``source_id`` permite conservar un id concreto (p. ej. al importar un
        bundle, para que los segmentos sigan apuntando a la fuente correcta).
        """
        csv_path = os.path.abspath(csv_path)
        alias = alias or os.path.splitext(os.path.basename(csv_path))[0]
        source = {"id": source_id or uuid.uuid4().hex[:8], "path": csv_path, "alias": alias}
        new_sources = self.sources + [source]
        self._commit("sources", new_sources, f"Añadir fuente «{alias}»",
                     setter=lambda v: setattr(self, "sources", v))
        if recording is not None:                    # ya cargada fuera del hilo GUI
            with self._lock:
                self._recordings[source["id"]] = recording
        rec = self.get_recording(source["id"])       # acierto de caché si venía precargada
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

    def orphan_recordings(self) -> list[str]:
        """CSV en ``recordings/`` que NO están registrados como fuente del proyecto.

        Sirve para recuperar grabaciones que quedaron en disco sin añadirse."""
        rec_dir = os.path.join(self.path, RECORDINGS_DIR)
        if not os.path.isdir(rec_dir):
            return []
        known = {os.path.normcase(os.path.abspath(self._resolve_path(s.get("path", ""))))
                 for s in self.sources if s.get("path")}
        out = []
        for fn in sorted(os.listdir(rec_dir)):
            low = fn.lower()
            if not (low.endswith(".csv") or low.endswith(".csv.gz")):
                continue
            full = os.path.abspath(os.path.join(rec_dir, fn))
            if os.path.normcase(full) not in known:
                out.append(full)
        return out

    def is_internal_path(self, path: str) -> bool:
        """True si ``path`` vive dentro de la carpeta del proyecto.

        Sirve para decidir si un archivo se puede borrar del disco (solo los del
        proyecto) sin tocar nunca la carpeta de datos de origen.
        """
        if not path:
            return False
        root = os.path.abspath(self.path)
        try:
            return os.path.commonpath([os.path.abspath(path), root]) == root
        except ValueError:                       # distinto volumen (Windows)
            return False

    def _persist_path(self, path: str) -> str:
        """Ruta a GUARDAR en el manifiesto: relativa a la carpeta del proyecto si
        el archivo está dentro (portátil al mover/renombrar el proyecto); absoluta
        si es externo."""
        if not path:
            return path
        ap = os.path.abspath(path)
        if self.is_internal_path(ap):
            return os.path.relpath(ap, self.path).replace(os.sep, "/")
        return ap

    def _resolve_path(self, path: str) -> str:
        """Ruta ABSOLUTA a partir de la guardada (resuelve las relativas contra la
        ubicación actual del proyecto)."""
        if not path:
            return path
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.path, path))

    def _persisted_changelog(self) -> dict:
        """``to_dict()`` del historial con las rutas de fuentes en forma relativa.

        No hace un *deep copy* de todo el historial (costoso en autosave): solo
        reescribe las listas de fuentes de los comandos «sources», sin mutar el
        historial en memoria (los ``cmd`` de ``to_dict`` ya son dicts nuevos)."""
        d = self.changelog.to_dict()
        for node in d.get("nodes", {}).values():
            cmd = node.get("cmd")
            if not (cmd and cmd.get("section") == "sources"):
                continue
            for field in ("before", "after"):
                val = cmd.get(field)
                if isinstance(val, list):
                    cmd[field] = [
                        {**s, "path": self._persist_path(s["path"])}
                        if isinstance(s, dict) and s.get("path") else s
                        for s in val
                    ]
        return d

    def _resolve_changelog_paths(self) -> None:
        """Resuelve a absoluto las rutas de fuentes del historial recién cargado."""
        for node in self.changelog._nodes.values():
            cmd = node.get("cmd")
            if cmd is None or cmd.section != "sources":
                continue
            for field in ("before", "after"):
                val = getattr(cmd, field)
                if isinstance(val, list):
                    for s in val:
                        if isinstance(s, dict) and s.get("path"):
                            s["path"] = self._resolve_path(s["path"])

    def set_source_path(self, source_id: str, new_path: str,
                        description: str = "Cambiar ruta de fuente") -> None:
        """Actualiza la ruta de una fuente (p. ej. tras comprimirla a .csv.gz)."""
        new_sources = [dict(s) for s in self.sources]
        for s in new_sources:
            if s["id"] == source_id:
                s["path"] = os.path.abspath(new_path)
        self._commit("sources", new_sources, description,
                     setter=lambda v: setattr(self, "sources", v))
        with self._lock:                       # forzar recarga desde la nueva ruta
            self._recordings.pop(source_id, None)

    def rename_source(self, source_id: str, new_alias: str) -> bool:
        """Renombra una fuente: cambia su nombre mostrado (alias) y, si el archivo
        es **interno** al proyecto, también lo renombra en disco (conservando la
        extensión). No toca los archivos de origen externos. Devuelve si cambió."""
        src = self.get_source(source_id)
        if src is None:
            return False
        new_alias = (new_alias or "").strip()
        if not new_alias or new_alias == src["alias"]:
            return False
        new_sources = [dict(s) for s in self.sources]
        target = next(s for s in new_sources if s["id"] == source_id)
        target["alias"] = new_alias

        # Renombra el archivo si vive dentro del proyecto (recordings/, imported/…).
        old_path = self._resolve_path(src["path"])
        if self.is_internal_path(old_path) and os.path.isfile(old_path):
            folder = os.path.dirname(old_path)
            ext = _split_csv_ext(os.path.basename(old_path))
            base = _safe_filename(new_alias) or "senal"
            cand = os.path.join(folder, base + ext)
            i = 2
            while os.path.exists(cand) and os.path.abspath(cand) != os.path.abspath(old_path):
                cand = os.path.join(folder, f"{base}_{i}{ext}")
                i += 1
            if os.path.abspath(cand) != os.path.abspath(old_path):
                os.rename(old_path, cand)       # el id no cambia → caché sigue válida
            target["path"] = cand

        self._commit("sources", new_sources, f"Renombrar fuente a «{new_alias}»",
                     setter=lambda v: setattr(self, "sources", v))
        return True

    def reorder_sources(self, ordered_ids: list[str]) -> bool:
        """Reordena las fuentes según ``ordered_ids`` (el «orden propio» del usuario).

        Solo cambia el orden, no los datos. Ignora ids desconocidos y conserva al
        final (en su orden actual) las fuentes que no aparezcan en la lista, para no
        perder ninguna. Devuelve ``True`` si el orden efectivamente cambió.
        """
        by_id = {s["id"]: s for s in self.sources}
        seen: set[str] = set()
        new_sources = []
        for sid in ordered_ids:
            if sid in by_id and sid not in seen:
                new_sources.append(by_id[sid])
                seen.add(sid)
        for s in self.sources:                       # las que falten, al final
            if s["id"] not in seen:
                new_sources.append(s)
        if [s["id"] for s in new_sources] == [s["id"] for s in self.sources]:
            return False
        self._commit("sources", new_sources, "Reordenar fuentes",
                     setter=lambda v: setattr(self, "sources", v))
        return True

    def get_recording(self, source_id: str) -> Recording:
        with self._lock:
            rec = self._recordings.get(source_id)
        if rec is not None:
            return rec
        src = self.get_source(source_id)
        if src is None:
            raise KeyError(f"Fuente inexistente: {source_id}")
        rec = load_recording(self._resolve_path(src["path"]))   # E/S pesada fuera del lock
        with self._lock:
            self._recordings[source_id] = rec
        return rec

    def _set_default_channel_aliases(self, rec: Recording) -> None:
        import re

        names = rec.channel_names
        # Solo se aplican los nombres del EPOC+ cuando los canales son genéricos
        # ("Channel 1".."Channel 14"), como en los CSV de OpenViBE. Si el CSV ya
        # trae nombres reales (p. ej. un dataset ajeno: Fz, C3, …), se respetan.
        generic = all(re.fullmatch(r"Channel \d+", str(n)) for n in names)
        if generic and len(names) == len(EPOC_CHANNELS):
            aliases = {n: EPOC_CHANNELS[i] for i, n in enumerate(names)}
        else:
            aliases = {n: n for n in names}
        self.state["channel_aliases"] = aliases

    def display_channel_names(self, rec: Recording) -> list[str]:
        aliases = self.state.get("channel_aliases", {})
        return [aliases.get(n, n) for n in rec.channel_names]

    # --- Canales activos (exclusión, p. ej. de los EOG) -------------------
    def excluded_channels(self) -> list[str]:
        return list(self.state.get("excluded_channels", []))

    def kept_indices(self, rec: Recording) -> list[int]:
        """Índices de los canales NO excluidos, en orden original."""
        excluded = set(self.state.get("excluded_channels", []))
        return [i for i, n in enumerate(rec.channel_names) if n not in excluded]

    def kept_channel_names(self, rec: Recording) -> list[str]:
        excluded = set(self.state.get("excluded_channels", []))
        return [n for n in rec.channel_names if n not in excluded]

    def kept_display_names(self, rec: Recording) -> list[str]:
        """Nombres mostrados de los canales activos (coinciden con get_processed)."""
        aliases = self.state.get("channel_aliases", {})
        return [aliases.get(n, n) for n in self.kept_channel_names(rec)]

    # ------------------------------------------------------------------ #
    # Procesamiento (no destructivo): pipeline aplicado sobre copias
    # ------------------------------------------------------------------ #
    def _pipeline_signature(self) -> str:
        # La firma incluye los canales excluidos: cambiarlos invalida la caché.
        raw = json.dumps([self.state["pipeline"], sorted(self.state.get("excluded_channels", []))],
                         sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def processed_if_cached(self, source_id: str) -> np.ndarray | None:
        """Señal procesada SOLO si ya está en la caché en memoria (no recalcula).

        Permite a la interfaz redibujar al instante (sin lanzar un hilo ni mostrar
        «ocupado») cuando el resultado del pipeline ya está calculado."""
        sig = self._pipeline_signature()
        with self._lock:
            cached = self._processed.get(source_id)
            if cached and cached[0] == sig:
                return cached[1]
        return None

    def get_processed(self, source_id: str, progress=None) -> np.ndarray:
        """Señal de los **canales activos** con el pipeline aplicado (en caché).

        Selecciona primero los canales no excluidos y luego aplica el pipeline, de
        modo que CAR y los filtros operan solo sobre esos canales. Seguro para
        hilos; nunca modifica la grabación original. ``progress(hechos, total)``
        informa del avance cuando hay que recalcular (no si sale de caché).
        """
        rec = self.get_recording(source_id)
        sig = self._pipeline_signature()
        with self._lock:
            cached = self._processed.get(source_id)
            if cached and cached[0] == sig:
                return cached[1]

        base = np.ascontiguousarray(rec.data[self.kept_indices(rec)], dtype=np.float64)
        pipeline = self.state["pipeline"]
        if not pipeline:
            out = base                            # sin procesar: no se cachea en disco
        else:
            out = self._load_disk_cache(source_id, sig)
            if out is None:
                out = preprocessing.apply_pipeline(base, rec.sample_rate, pipeline,
                                                   progress=progress)
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
            channels=np.array(self.kept_display_names(rec), dtype=object),
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

    def clear_segments(self) -> int:
        """Elimina TODOS los segmentos de una vez. Devuelve cuántos se quitaron."""
        n = len(self.state["segments"])
        if n:
            self._commit("segments", [], f"Eliminar todos los segmentos ({n})")
        return n

    # ------------------------------------------------------------------ #
    # Estímulos sincronizados (video -> marcas/segmentos en el tiempo)
    # ------------------------------------------------------------------ #
    def stim_videos(self) -> list[dict]:
        """Configuraciones de estímulo guardadas en el proyecto."""
        return list(self.state.get("stim_videos", []))

    def save_stim_video(self, config: dict) -> dict:
        """Añade o actualiza (por ``id``) la configuración de un video de estímulo.

        La configuración lleva la ruta del video, su clase y los eventos definidos
        en tiempo (marcas y segmentos). Devuelve la config guardada (con ``id``)."""
        cfg = dict(config)
        if not cfg.get("id"):
            cfg["id"] = uuid.uuid4().hex[:8]
        videos = [dict(v) for v in self.state.get("stim_videos", [])]
        for i, v in enumerate(videos):
            if v.get("id") == cfg["id"]:
                videos[i] = cfg
                break
        else:
            videos.append(cfg)
        self._commit("stim_videos", videos,
                     f"Configurar estímulo «{cfg.get('label', '')}»")
        return cfg

    def remove_stim_video(self, video_id: str) -> None:
        keep = [v for v in self.state.get("stim_videos", []) if v.get("id") != video_id]
        self._commit("stim_videos", keep, "Quitar estímulo")

    # --- Configuraciones de modelo (hiperparámetros, SIN entrenar) ---------
    def model_configs(self) -> list[dict]:
        """Configuraciones de modelo guardadas en el proyecto."""
        return [dict(c) for c in self.state.get("model_configs", [])]

    def save_model_config(self, config: dict) -> dict:
        """Guarda (o actualiza, por ``name``) una configuración de modelo.

        Es solo la **receta** de hiperparámetros: no entrena ni necesita datos."""
        cfg = dict(config)
        name = str(cfg.get("name", "")).strip()
        if not name:
            raise ValueError("La configuración necesita un nombre.")
        cfg["name"] = name
        configs = [dict(c) for c in self.state.get("model_configs", [])]
        for i, c in enumerate(configs):
            if c.get("name") == name:
                configs[i] = cfg
                break
        else:
            configs.append(cfg)
        self._commit("model_configs", configs, f"Guardar configuración «{name}»")
        return cfg

    def remove_model_config(self, name: str) -> None:
        keep = [c for c in self.state.get("model_configs", []) if c.get("name") != name]
        if len(keep) != len(self.state.get("model_configs", [])):
            self._commit("model_configs", keep, f"Quitar configuración «{name}»")

    def repeat_segment(self, segment_id: str, period: int, count: int | None = None,
                       n_samples: int | None = None) -> int:
        """Repite un segmento hacia adelante cada ``period`` muestras (protocolos
        periódicos). ``count`` = nº TOTAL (incluido el original); ``None`` = hasta
        ``n_samples``. No duplica los que ya existan. Devuelve cuántos creó."""
        seg = next((s for s in self.state["segments"] if s["id"] == segment_id), None)
        if seg is None or period <= 0:
            return 0
        sid = seg["source_id"]
        dur = seg["stop"] - seg["start"]
        existing = {(s["start"], s["stop"]) for s in self.state["segments"]
                    if s["source_id"] == sid}
        new_segments = list(self.state["segments"])
        created, i = 0, 1
        while (count is None or i < count) and i <= 1000:
            start = seg["start"] + i * period
            stop = start + dur
            if n_samples is not None and stop > n_samples:
                break
            if (start, stop) not in existing:
                new_segments.append({
                    "id": uuid.uuid4().hex[:8], "source_id": sid,
                    "start": int(start), "stop": int(stop), "label": seg["label"],
                    "channels": seg.get("channels"), "note": seg.get("note", ""),
                })
                existing.add((start, stop))
                created += 1
            i += 1
        if created:
            self._commit("segments", new_segments,
                         f"Generar {created} segmento(s) «{seg['label']}»")
        return created

    def remove_segments_in_range(self, source_id: str, start: int, stop: int) -> int:
        """Elimina los segmentos de ``source_id`` que se solapen con ``[start, stop)``."""
        a, b = sorted((int(start), int(stop)))
        segs = self.state["segments"]
        keep = [s for s in segs if not (s["source_id"] == source_id
                                        and s["start"] < b and s["stop"] > a)]
        n = len(segs) - len(keep)
        if n:
            self._commit("segments", keep, f"Eliminar {n} segmento(s) de la selección")
        return n

    # --- Recorte de tramos de señal (no destructivo, reversible) ----------
    @staticmethod
    def _merge_intervals(intervals: list) -> list:
        ivs = sorted([sorted((int(a), int(b))) for a, b in intervals])
        out: list = []
        for a, b in ivs:
            if b - a < 1:
                continue
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return out

    def cut_intervals(self, source_id: str) -> list[tuple[int, int]]:
        """Tramos eliminados de una fuente, como lista de ``(inicio, fin)``."""
        return [(int(a), int(b)) for a, b in self.state.get("cuts", {}).get(source_id, [])]

    def add_cut(self, source_id: str, start: int, stop: int) -> None:
        """Marca ``[start, stop)`` como eliminado (excluido del dataset).

        No borra nada del CSV: guarda el tramo en el estado (reversible con undo).
        Elimina además los segmentos etiquetados que se solapen con el recorte.
        """
        a, b = sorted((int(start), int(stop)))
        if b - a < 1:
            return
        cuts = {k: [list(iv) for iv in v] for k, v in self.state.get("cuts", {}).items()}
        cuts[source_id] = self._merge_intervals(cuts.get(source_id, []) + [[a, b]])
        self.remove_segments_in_range(source_id, a, b)     # primero, mientras hay índices válidos
        self._commit("cuts", cuts, f"Recortar señal [{a}, {b}]")

    def clear_cuts(self, source_id: str) -> int:
        """Restaura (des-recorta) todos los tramos eliminados de una fuente."""
        cuts = {k: list(v) for k, v in self.state.get("cuts", {}).items()}
        n = len(cuts.get(source_id, []))
        if n:
            cuts.pop(source_id, None)
            self._commit("cuts", cuts, "Restaurar tramos recortados")
        return n

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

    def _build_marker_segments(self, source_id: str, window: int, offset: int) -> list[dict]:
        """Segmentos (sin registrar) a partir de los marcadores de una fuente."""
        rec = self.get_recording(source_id)
        events = rec.events
        if not events:
            return []
        starts = [int(e["sample"]) for e in events]
        cuts = self.cut_intervals(source_id)
        out: list[dict] = []
        for i, ev in enumerate(events):
            start = starts[i] + int(offset)
            if window and window > 0:
                stop = min(start + int(window), rec.n_samples)
            else:
                stop = starts[i + 1] if i + 1 < len(starts) else rec.n_samples
            if stop - start < 2:
                continue
            if any(start < cb and stop > ca for ca, cb in cuts):   # no crear en tramos recortados
                continue
            out.append({
                "id": uuid.uuid4().hex[:8],
                "source_id": source_id,
                "start": start,
                "stop": int(stop),
                "label": str(ev.get("id", "")).strip() or "marcador",
                "channels": None,
                "note": "desde marcador",
            })
        return out

    def segments_from_markers(self, source_id: str, window: int = 0, offset: int = 0) -> int:
        """Crea segmentos a partir de los marcadores (Event Id) de **una** fuente.

        Cada marcador genera un segmento etiquetado con el texto del marcador, de
        modo que los marcadores **sirven como clases**. ``offset`` = muestras a
        saltar tras el marcador; ``window`` = nº de muestras (0 = hasta el siguiente
        marcador o el final). Devuelve el número de segmentos creados.
        """
        new = self._build_marker_segments(source_id, window, offset)
        if new:
            self._commit("segments", self.state["segments"] + new,
                         f"Segmentos desde marcadores ({len(new)})")
        return len(new)

    def segments_from_markers_all(self, window: int = 0, offset: int = 0,
                                  skip_existing: bool = True) -> int:
        """Crea segmentos desde los marcadores de **todas** las fuentes (un solo paso).

        Por defecto omite las fuentes que ya tienen segmentos (para no duplicar) y
        las que no se puedan cargar del disco (se saltan sin romper la operación).
        """
        have = ({s["source_id"] for s in self.state["segments"]}
                if skip_existing else set())
        new: list[dict] = []
        used = 0
        for src in self.sources:
            if src["id"] in have:
                continue
            try:
                seg = self._build_marker_segments(src["id"], window, offset)
            except Exception:  # noqa: BLE001 — fuente no disponible: se omite
                continue
            if seg:
                new += seg
                used += 1
        if new:
            self._commit("segments", self.state["segments"] + new,
                         f"Segmentos desde marcadores · {used} fuentes ({len(new)})")
        return len(new)

    # ------------------------------------------------------------------ #
    # Varios pipelines por proyecto (pestañas)
    # ------------------------------------------------------------------ #
    def _sync_pipeline_mirror(self) -> None:
        """Deja ``state['pipeline']`` apuntando a los pasos del pipeline activo."""
        pls = self.state.get("pipelines") or [{"name": "Pipeline 1", "steps": []}]
        self.state["pipelines"] = pls
        i = self.state.get("active_pipeline", 0)
        if not (0 <= i < len(pls)):
            i = 0
            self.state["active_pipeline"] = 0
        self.state["pipeline"] = pls[i]["steps"]

    def _migrate_pipelines(self, has_pipelines: bool) -> None:
        """Proyectos antiguos (un solo ``pipeline``) → estructura de varios pipelines."""
        if not has_pipelines:
            steps = self.state.get("pipeline", [])
            self.state["pipelines"] = [{"name": "Pipeline 1", "steps": snapshot(steps)}]
            self.state["active_pipeline"] = 0
        self._sync_pipeline_mirror()

    def pipelines(self) -> list[dict]:
        return self.state.get("pipelines", [])

    def pipelines_snapshot(self) -> list[dict]:
        """Copia de todos los pipelines, con el activo reconciliado desde el espejo.

        Robustez: si alguien fijó ``state['pipeline']`` directamente, el activo
        refleja ese valor (así el export/persistencia no lo pierde)."""
        pls = snapshot(self.state.get("pipelines", []))
        i = self.active_pipeline_index()
        if 0 <= i < len(pls):
            pls[i]["steps"] = snapshot(self.state.get("pipeline", []))
        return pls

    def active_pipeline_index(self) -> int:
        return int(self.state.get("active_pipeline", 0))

    def active_pipeline_name(self) -> str:
        pls = self.pipelines()
        i = self.active_pipeline_index()
        return pls[i]["name"] if 0 <= i < len(pls) else "Pipeline 1"

    def _commit_active_steps(self, new_steps: list, description: str) -> None:
        """Reemplaza los pasos del pipeline activo y lo registra (undo/redo)."""
        pls = snapshot(self.state["pipelines"])
        i = self.active_pipeline_index()
        if not (0 <= i < len(pls)):
            return
        pls[i]["steps"] = new_steps
        self._commit("pipelines", pls, description)
        self.invalidate_processed()

    def add_pipeline(self, name: str | None = None) -> int:
        pls = snapshot(self.state["pipelines"])
        name = (name or "").strip() or f"Pipeline {len(pls) + 1}"
        pls.append({"name": name, "steps": []})
        self._commit("pipelines", pls, f"Añadir pipeline «{name}»")
        self.set_active_pipeline(len(pls) - 1)
        return len(pls) - 1

    def remove_pipeline(self, index: int) -> bool:
        pls = snapshot(self.state["pipelines"])
        if len(pls) <= 1 or not (0 <= index < len(pls)):
            return False                     # siempre queda al menos un pipeline
        name = pls[index]["name"]
        del pls[index]
        active = self.active_pipeline_index()
        self._commit("pipelines", pls, f"Eliminar pipeline «{name}»")
        if active >= len(pls):
            active = len(pls) - 1
        self._commit("active_pipeline", active, f"Pipeline activo: «{pls[active]['name']}»")
        self.invalidate_processed()
        return True

    def rename_pipeline(self, index: int, name: str) -> None:
        pls = snapshot(self.state["pipelines"])
        if not (0 <= index < len(pls)):
            return
        name = (name or "").strip()
        if not name or name == pls[index]["name"]:
            return
        old = pls[index]["name"]
        pls[index]["name"] = name
        self._commit("pipelines", pls, f"Renombrar pipeline «{old}» → «{name}»")

    def set_active_pipeline(self, index: int) -> None:
        pls = self.state["pipelines"]
        if not (0 <= index < len(pls)) or index == self.active_pipeline_index():
            return
        self._commit("active_pipeline", index, f"Cambiar a pipeline «{pls[index]['name']}»")
        self.invalidate_processed()

    def set_active_pipeline_steps(self, steps: list, description: str = "Actualizar pipeline") -> None:
        """Sustituye por completo los pasos del pipeline activo (p. ej. al importar)."""
        self._commit_active_steps(snapshot(steps), description)

    def set_pipelines(self, pipelines: list, active: int = 0,
                      description: str = "Importar pipelines") -> None:
        """Reemplaza TODOS los pipelines y el activo (p. ej. al importar un bundle)."""
        pls = snapshot(pipelines) or [{"name": "Pipeline 1", "steps": []}]
        self._commit("pipelines", pls, description)
        a = int(active) if 0 <= int(active) < len(pls) else 0
        self._commit("active_pipeline", a, f"Pipeline activo: «{pls[a]['name']}»")
        self.invalidate_processed()

    # ------------------------------------------------------------------ #
    # Preprocesamiento: edición del pipeline (activo)
    # ------------------------------------------------------------------ #
    def add_pipeline_step(self, step_type: str, params: dict | None = None) -> None:
        params = params if params is not None else dict(preprocessing.STEP_DEFAULTS.get(step_type, {}))
        new_pipeline = self.state["pipeline"] + [{"type": step_type, "params": params}]
        label = preprocessing.STEP_LABELS.get(step_type, step_type)
        self._commit_active_steps(new_pipeline, f"Añadir paso: {label}")

    def remove_pipeline_step(self, index: int) -> None:
        new_pipeline = [s for i, s in enumerate(self.state["pipeline"]) if i != index]
        self._commit_active_steps(new_pipeline, "Eliminar paso de preprocesamiento")

    def set_step_enabled(self, index: int, enabled: bool) -> None:
        """Activa/desactiva un paso sin borrarlo (queda en el pipeline)."""
        new_pipeline = snapshot(self.state["pipeline"])
        if not (0 <= index < len(new_pipeline)):
            return
        if bool(new_pipeline[index].get("enabled", True)) == bool(enabled):
            return
        new_pipeline[index]["enabled"] = bool(enabled)
        label = preprocessing.STEP_LABELS.get(new_pipeline[index]["type"], "paso")
        verb = "Activar" if enabled else "Desactivar"
        self._commit_active_steps(new_pipeline, f"{verb}: {label}")

    def update_pipeline_step(self, index: int, params: dict) -> None:
        new_pipeline = snapshot(self.state["pipeline"])
        new_pipeline[index]["params"] = params
        self._commit_active_steps(new_pipeline, "Modificar parámetros de paso")

    def move_pipeline_step(self, index: int, delta: int) -> None:
        new_pipeline = snapshot(self.state["pipeline"])
        j = index + delta
        if 0 <= j < len(new_pipeline):
            new_pipeline[index], new_pipeline[j] = new_pipeline[j], new_pipeline[index]
            self._commit_active_steps(new_pipeline, "Reordenar pipeline")

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

    # Cambios en estas secciones invalidan la señal procesada en caché.
    _REPROCESS_SECTIONS = ("pipelines", "active_pipeline", "excluded_channels")

    def edit(self, section: str, new_value, description: str) -> None:
        """API pública para editar una sección arbitraria del estado."""
        self._commit(section, new_value, description)
        if section in self._REPROCESS_SECTIONS:
            self.invalidate_processed()

    def _apply_section(self, section: str, value, setter=None) -> None:
        if setter is not None:
            setter(value)
        elif section == "sources":
            self.sources = value
        else:
            self.state[section] = value
        # El pipeline activo (o su lista) cambió: reactualiza el espejo `pipeline`.
        if section in ("pipelines", "active_pipeline"):
            self._sync_pipeline_mirror()

    def undo(self) -> bool:
        cmd = self.changelog.undo()
        if cmd is None:
            return False
        self._apply_section(cmd.section, snapshot(cmd.before))
        if cmd.section in self._REPROCESS_SECTIONS:
            self.invalidate_processed()
        return True

    def redo(self) -> bool:
        cmd = self.changelog.redo()
        if cmd is None:
            return False
        self._apply_section(cmd.section, snapshot(cmd.after))
        if cmd.section in self._REPROCESS_SECTIONS:
            self.invalidate_processed()
        return True

    def goto_history(self, target_applied: int) -> None:
        """Navega en la RAMA actual hasta dejar ``target_applied`` comandos aplicados."""
        guard = 0
        while self.changelog.applied_count() > target_applied and guard < 10000:
            if not self.undo():
                break
            guard += 1
        while self.changelog.applied_count() < target_applied and guard < 10000:
            if not self.redo():
                break
            guard += 1

    def goto_node(self, node_id: int) -> None:
        """Navega a un nodo cualquiera del árbol (puede saltar entre ramas)."""
        steps = self.changelog.steps_to(int(node_id))
        reprocess = False
        for direction, cmd in steps:
            value = cmd.before if direction == "undo" else cmd.after
            self._apply_section(cmd.section, snapshot(value))
            if cmd.section in self._REPROCESS_SECTIONS:
                reprocess = True
        self.changelog.set_current(int(node_id))
        if reprocess:
            self.invalidate_processed()

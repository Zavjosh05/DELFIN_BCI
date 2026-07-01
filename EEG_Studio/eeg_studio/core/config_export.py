"""Exportar/inspeccionar la configuración de un proyecto en un único archivo.

Genera un archivo ``.eegcfg`` (JSON legible) con las secciones que se elijan:

* ``preprocessing`` — pipeline (pasos + parámetros + activado), canales excluidos
  y alias de canal.
* ``dataset`` — configuración de características, segmentos etiquetados y recortes.
* ``models`` — por cada modelo entrenado: tipo, hiperparámetros (clásico o red),
  clases, exactitud de validación y métricas. (No incluye los pesos entrenados;
  para eso está «Exportar modelo…», que guarda el ``.joblib``.)

Es una *receta* portable para reproducir una configuración en otro proyecto.
"""
from __future__ import annotations

import copy
import json
import os
import time
import zipfile

from ..config import APP_NAME, APP_VERSION, DATASETS_DIR

SECTIONS = ("preprocessing", "dataset", "models", "sources")
CONFIG_EXT = ".eegcfg"        # JSON legible (solo configuración, sin binarios)
BUNDLE_EXT = ".eegbundle"     # ZIP autónomo: configuración + modelos + datasets


def _compress_type(name: str) -> int:
    """No recomprime lo que ya está comprimido (.gz/.npz/.joblib); comprime el resto."""
    already = name.lower().endswith((".gz", ".npz", ".joblib", ".zip"))
    return zipfile.ZIP_STORED if already else zipfile.ZIP_DEFLATED


def _model_entry(name: str, res) -> dict:
    cv = getattr(res, "cv_scores", None)
    return {
        "name": name,
        "classifier_name": res.classifier_name,
        "input_kind": getattr(res, "input_kind", "features"),
        "classes": list(res.classes),
        "clf_params": getattr(res, "clf_params", None),   # hiperparámetros clásicos
        "nn_config": getattr(res, "nn_config", None),      # config de la red (si es NN)
        "cv_mean": (float(cv.mean()) if cv is not None and cv.size else None),
        "metrics": getattr(res, "metrics", None),
    }


def build_config(project, models: dict, sections) -> dict:
    """Ensambla el dict de configuración con las secciones pedidas."""
    sections = set(sections)
    cfg: dict = {
        "app": APP_NAME,
        "version": APP_VERSION,
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "project": project.name,
        "sections": sorted(s for s in SECTIONS if s in sections),
    }
    if "preprocessing" in sections:
        cfg["preprocessing"] = {
            "pipeline": copy.deepcopy(project.state.get("pipeline", [])),
            "excluded_channels": list(project.state.get("excluded_channels", [])),
            "channel_aliases": dict(project.state.get("channel_aliases", {})),
        }
    if "dataset" in sections:
        cfg["dataset"] = {
            "config": dict(project.state.get("dataset", {})),
            "segments": copy.deepcopy(project.state.get("segments", [])),
            "cuts": copy.deepcopy(project.state.get("cuts", {})),
        }
    if "models" in sections:
        cfg["models"] = [_model_entry(n, r) for n, r in (models or {}).items()]
    if "sources" in sections:
        cfg["sources"] = [{"id": s["id"], "alias": s["alias"], "file": None}
                          for s in project.sources]
    return cfg


def save_config(config: dict, path: str) -> str:
    """Escribe la configuración como JSON con codificación UTF-8."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return path


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --- Bundle autónomo (.eegbundle = ZIP) ------------------------------------
def export_bundle(project, models: dict, sections, out_path: str) -> dict:
    """Escribe un ``.eegbundle`` (ZIP) con la configuración + **binarios**.

    Incluye, según las secciones elegidas: ``bundle.json`` (config), los modelos
    entrenados (``models/<nombre>.joblib``) y los datasets guardados del proyecto
    (``datasets/*.npz``). Devuelve un resumen ``{"models": n, "datasets": n}``.
    """
    from . import classification

    sections = set(sections)
    cfg = build_config(project, models, sections)
    n_models = 0
    ds_files: list[str] = []
    n_sources = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        if "models" in sections:
            for name, res in (models or {}).items():
                arc = f"models/{name}.joblib"
                z.writestr(arc, classification.result_to_bytes(res),
                           compress_type=_compress_type(arc))
                n_models += 1
            for m in cfg.get("models", []):                # referencia al binario
                m["file"] = f"models/{m['name']}.joblib"
        if "dataset" in sections:
            # Solo los datos (.npz). NO se incluyen imágenes ni gráficos: la matriz de
            # confusión y demás se regeneran al vuelo desde las métricas del modelo.
            ds_dir = os.path.join(project.path, DATASETS_DIR)
            if os.path.isdir(ds_dir):
                for f in sorted(os.listdir(ds_dir)):
                    if f.endswith(".npz"):
                        z.write(os.path.join(ds_dir, f), f"datasets/{f}",
                                compress_type=_compress_type(f))
                        ds_files.append(f)
            cfg["dataset"]["files"] = ds_files
        if "sources" in sections:
            # Las señales de origen se guardan comprimidas; la caché (regenerable)
            # NO se incluye, por eso el bundle suele pesar menos que el proyecto.
            for src, meta in zip(project.sources, cfg.get("sources", [])):
                p = src.get("path", "")
                if not os.path.isfile(p):
                    continue
                arc = f"sources/{src['id']}__{os.path.basename(p)}"
                z.write(p, arc, compress_type=_compress_type(p))
                meta["file"] = arc
                n_sources += 1
        z.writestr("bundle.json", json.dumps(cfg, ensure_ascii=False, indent=2),
                   compress_type=zipfile.ZIP_DEFLATED)
    return {"models": n_models, "datasets": len(ds_files), "sources": n_sources,
            "path": out_path, "size": os.path.getsize(out_path)}


def read_bundle(path: str) -> tuple[dict, dict, dict, dict]:
    """Lee un ``.eegbundle``.

    Devuelve ``(config, {modelo: bytes}, {dataset.npz: bytes}, {arc_fuente: bytes})``.
    """
    with zipfile.ZipFile(path) as z:
        cfg = json.loads(z.read("bundle.json").decode("utf-8"))
        models = {n[len("models/"):-len(".joblib")]: z.read(n)
                  for n in z.namelist()
                  if n.startswith("models/") and n.endswith(".joblib")}
        datasets = {os.path.basename(n): z.read(n)
                    for n in z.namelist()
                    if n.startswith("datasets/") and n.endswith(".npz")}
        sources = {n: z.read(n) for n in z.namelist() if n.startswith("sources/")}
    return cfg, models, datasets, sources

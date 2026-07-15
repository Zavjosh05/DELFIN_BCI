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

SECTIONS = ("preprocessing", "dataset", "models", "model_configs", "sources")
CONFIG_EXT = ".eegcfg"        # JSON legible (solo configuración, sin binarios)
BUNDLE_EXT = ".eegbundle"     # ZIP autónomo: configuración + modelos + datasets


def _source_arcname(source_id: str, path: str) -> str:
    """Nombre del CSV dentro del ZIP: ``<id>__<archivo>``.

    El prefijo con el id evita choques entre fuentes que se llamen igual. Si el
    archivo **ya** viene prefijado (porque a su vez se importó de otro bundle) el
    prefijo se **normaliza a uno solo**: si no, cada ciclo importar→exportar iba
    encadenando prefijos (``id__id__id__señal.csv``).
    """
    base = os.path.basename(path)
    prefix = f"{source_id}__"
    while base.startswith(prefix):
        base = base[len(prefix):]
    return prefix + base


def source_filename(arcname: str, source_id: str | None = None) -> str:
    """Nombre con el que guardar en disco un CSV que viene de un bundle.

    Dentro del ZIP viaja prefijado (``<id>__señal.csv``) para que dos fuentes que se
    llamen igual no choquen **dentro del paquete**. Al extraerlo se **quita** el
    prefijo, para que el archivo se llame como la señal (y coincida con lo que se ve
    en la interfaz); si ese nombre ya estuviera ocupado, quien lo guarda le añade un
    sufijo.
    """
    base = os.path.basename(arcname)
    if source_id:
        prefix = f"{source_id}__"
        while base.startswith(prefix):
            base = base[len(prefix):]
    return base


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
        # Aumento de datos con el que se entrenó: parte de la receta reproducible.
        "augment_config": getattr(res, "augment_config", None),
        # Ventana de señal cruda (Riemann/CSP/redes): hace falta para poder
        # RE-ENTRENAR el modelo con los datos de otro proyecto.
        "raw_window": int(getattr(res, "raw_window", 0) or 0),
        "cv_mean": (float(cv.mean()) if cv is not None and cv.size else None),
        "metrics": getattr(res, "metrics", None),
    }


def reusable_model_configs(cfg: dict) -> list[dict]:
    """Configuraciones de modelo **reutilizables** que trae una config/bundle.

    Reúne dos orígenes, ambos con los mismos campos de hiperparámetros:

    * ``cfg["models"]`` — los modelos **entrenados** que viajan en el bundle (de
      cada uno se puede reaprovechar su receta).
    * ``cfg["model_configs"]`` — las configuraciones **sin entrenar** guardadas en
      el proyecto de origen.

    Devuelve las que sirven para **volver a entrenar** en otro proyecto (clásicos
    con ``clf_params``, redes con ``nn_config``, o Riemann/CSP con ``raw_window``),
    sin repetir nombres.
    """
    out: list[dict] = []
    seen: set = set()
    for m in list(cfg.get("models") or []) + list(cfg.get("model_configs") or []):
        if not isinstance(m, dict) or not m.get("classifier_name"):
            continue
        if not (m.get("clf_params") or m.get("nn_config") or m.get("raw_window")):
            continue
        key = (m.get("name"), m.get("classifier_name"))
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def build_config(project, models: dict, sections, pipeline_indices=None) -> dict:
    """Ensambla el dict de configuración con las secciones pedidas.

    ``pipeline_indices`` (opcional) limita qué **pipelines** se exportan; ``None``
    exporta todos. Si el pipeline activo no está en la selección, se toma el
    primero elegido como activo del bundle.
    """
    sections = set(sections)
    cfg: dict = {
        "app": APP_NAME,
        "version": APP_VERSION,
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "project": project.name,
        "sections": sorted(s for s in SECTIONS if s in sections),
    }
    if "preprocessing" in sections:
        all_pls = project.pipelines_snapshot()
        if pipeline_indices is None:
            sel = list(range(len(all_pls)))
        else:
            sel = [i for i in pipeline_indices if 0 <= i < len(all_pls)]
        if not sel:                                  # siempre al menos uno
            sel = [project.active_pipeline_index() if 0 <= project.active_pipeline_index()
                   < len(all_pls) else 0]
        chosen = [copy.deepcopy(all_pls[i]) for i in sel]
        active = project.active_pipeline_index()
        new_active = sel.index(active) if active in sel else 0
        cfg["preprocessing"] = {
            # `pipeline` (el activo) se mantiene por compatibilidad con bundles
            # antiguos; `pipelines` lleva los pipelines elegidos.
            "pipeline": copy.deepcopy(chosen[new_active]["steps"]) if chosen else [],
            "pipelines": chosen,
            "active_pipeline": new_active,
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
    if "model_configs" in sections:
        # Recetas de hiperparámetros guardadas en el proyecto (sin entrenar): son
        # solo JSON, no llevan binarios asociados.
        cfg["model_configs"] = copy.deepcopy(project.model_configs())
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
def export_bundle(project, models: dict, sections, out_path: str,
                  pipeline_indices=None) -> dict:
    """Escribe un ``.eegbundle`` (ZIP) con la configuración + **binarios**.

    Incluye, según las secciones elegidas: ``bundle.json`` (config), los modelos
    entrenados (``models/<nombre>.joblib``) y los datasets guardados del proyecto
    (``datasets/*.npz``). ``pipeline_indices`` limita qué pipelines exportar
    (``None`` = todos). Devuelve un resumen ``{"models": n, "datasets": n, ...,
    "skipped": [...]}``.

    **Blindado**: se escribe primero en un archivo temporal (``.part``) junto al
    destino y solo al final se **reemplaza atómicamente** el ``out_path`` — un
    fallo a mitad nunca deja un bundle corrupto en su sitio. Cada binario se
    empaqueta de forma **tolerante**: si un modelo/dataset/fuente falla (p. ej. un
    archivo bloqueado o ilegible) se **omite** y se anota en ``skipped`` en vez de
    abortar todo el export. Al terminar se **verifica la integridad** del ZIP.
    """
    from . import classification

    sections = set(sections)
    cfg = build_config(project, models, sections, pipeline_indices)
    n_models = 0
    ds_files: list[str] = []
    n_sources = 0
    skipped: list[str] = []

    out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = out_path + ".part"                      # temporal junto al destino
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as z:
            if "models" in sections:
                for name, res in (models or {}).items():
                    arc = f"models/{name}.joblib"
                    try:
                        z.writestr(arc, classification.result_to_bytes(res),
                                   compress_type=_compress_type(arc))
                        n_models += 1
                    except Exception as exc:  # noqa: BLE001
                        skipped.append(f"modelo «{name}»: {exc}")
                for m in cfg.get("models", []):            # referencia al binario
                    m["file"] = f"models/{m['name']}.joblib"
            if "dataset" in sections:
                # Solo los datos (.npz). NO se incluyen imágenes ni gráficos: la matriz
                # de confusión y demás se regeneran al vuelo desde las métricas.
                ds_dir = os.path.join(project.path, DATASETS_DIR)
                if os.path.isdir(ds_dir):
                    for f in sorted(os.listdir(ds_dir)):
                        if not f.endswith(".npz"):
                            continue
                        try:
                            z.write(os.path.join(ds_dir, f), f"datasets/{f}",
                                    compress_type=_compress_type(f))
                            ds_files.append(f)
                        except Exception as exc:  # noqa: BLE001
                            skipped.append(f"dataset «{f}»: {exc}")
                cfg["dataset"]["files"] = ds_files
            if "sources" in sections:
                # Las señales de origen se guardan comprimidas; la caché (regenerable)
                # NO se incluye, por eso el bundle suele pesar menos que el proyecto.
                for src, meta in zip(project.sources, cfg.get("sources", [])):
                    p = src.get("path", "")
                    if not os.path.isfile(p):
                        skipped.append(f"fuente «{os.path.basename(p) or src.get('id')}»: "
                                       "archivo no encontrado")
                        continue
                    arc = f"sources/{_source_arcname(src['id'], p)}"
                    try:
                        z.write(p, arc, compress_type=_compress_type(p))
                        meta["file"] = arc
                        n_sources += 1
                    except Exception as exc:  # noqa: BLE001
                        skipped.append(f"fuente «{os.path.basename(p)}»: {exc}")
            z.writestr("bundle.json", json.dumps(cfg, ensure_ascii=False, indent=2),
                       compress_type=zipfile.ZIP_DEFLATED)

        # Verificación de integridad antes de publicar el archivo definitivo.
        with zipfile.ZipFile(tmp_path) as z:
            bad = z.testzip()
            if bad is not None:
                raise OSError(f"el ZIP quedó corrupto (entrada dañada: {bad})")
            if "bundle.json" not in z.namelist():
                raise OSError("falta bundle.json en el paquete")
        os.replace(tmp_path, out_path)                 # publicación atómica
    except BaseException:
        # No dejes el temporal a medias si algo falla (incluye Ctrl-C/errores).
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise
    return {"models": n_models, "datasets": len(ds_files), "sources": n_sources,
            "skipped": skipped, "path": out_path, "size": os.path.getsize(out_path)}


def read_bundle(path: str) -> tuple[dict, dict, dict, dict]:
    """Lee un ``.eegbundle``.

    Devuelve ``(config, {modelo: bytes}, {dataset.npz: bytes}, {arc_fuente: bytes})``.
    Blindado: rechaza con un error claro si el archivo no es un ZIP válido o si le
    falta ``bundle.json`` (no es un bundle de EEG Studio).
    """
    if not zipfile.is_zipfile(path):
        raise ValueError("El archivo no es un .eegbundle válido (no es un ZIP).")
    with zipfile.ZipFile(path) as z:
        if "bundle.json" not in z.namelist():
            raise ValueError("El .eegbundle no contiene bundle.json "
                             "(¿archivo incompleto o de otro programa?).")
        cfg = json.loads(z.read("bundle.json").decode("utf-8"))
        models = {n[len("models/"):-len(".joblib")]: z.read(n)
                  for n in z.namelist()
                  if n.startswith("models/") and n.endswith(".joblib")}
        datasets = {os.path.basename(n): z.read(n)
                    for n in z.namelist()
                    if n.startswith("datasets/") and n.endswith(".npz")}
        sources = {n: z.read(n) for n in z.namelist() if n.startswith("sources/")}
    return cfg, models, datasets, sources

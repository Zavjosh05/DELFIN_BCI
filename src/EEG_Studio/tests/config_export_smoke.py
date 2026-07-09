"""Exportar configuración (preprocesamiento/dataset/modelos) a un .eegcfg."""
from __future__ import annotations

import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.core import classification, config_export
from eeg_studio.core.dataset import Dataset
from eeg_studio.core.project import Project


def main() -> int:
    tmp = tempfile.mkdtemp()
    proj = Project.create(tmp, "cfg")
    proj.state["pipeline"] = [
        {"type": "bandpass", "params": {"low": 8.0, "high": 30.0}, "enabled": True}]
    proj.state["excluded_channels"] = ["EOG-left", "EOG-central", "EOG-right"]
    proj.state["segments"] = [
        {"id": "s1", "source_id": "x", "start": 0, "stop": 500, "label": "izq", "channels": None}]
    proj.state["cuts"] = {"x": [[100, 200]]}
    proj.state["dataset"] = {"use_bands": True, "use_time": False}

    print("[1] Los parámetros del clasificador clásico quedan en el resultado")
    rng = np.random.default_rng(0)
    ds = Dataset(X=rng.normal(0, 1, (20, 6)).astype(float),
                 y=np.array(["a", "b"] * 10, dtype=object),
                 feature_names=[f"f{i}" for i in range(6)],
                 segment_ids=[str(i) for i in range(20)])
    res = classification.train(ds, "random_forest", clf_params={"n_estimators": 50})
    assert res.clf_params == {"n_estimators": 50}, res.clf_params
    models = {"rf_1": res}
    print(f"    clf_params guardados: {res.clf_params}")

    print("[2] build_config con TODAS las secciones")
    cfg = config_export.build_config(proj, models, {"preprocessing", "dataset", "models"})
    assert cfg["preprocessing"]["pipeline"][0]["params"]["low"] == 8.0
    assert cfg["preprocessing"]["excluded_channels"] == ["EOG-left", "EOG-central", "EOG-right"]
    assert cfg["dataset"]["config"] == {"use_bands": True, "use_time": False}
    assert cfg["dataset"]["cuts"] == {"x": [[100, 200]]}
    assert len(cfg["dataset"]["segments"]) == 1
    assert cfg["models"][0]["classifier_name"] == "random_forest"
    assert cfg["models"][0]["clf_params"] == {"n_estimators": 50}
    print(f"    secciones={cfg['sections']}  modelos={len(cfg['models'])}")

    print("[3] Selección parcial: solo preprocesamiento")
    only_pre = config_export.build_config(proj, models, {"preprocessing"})
    assert "preprocessing" in only_pre and "dataset" not in only_pre and "models" not in only_pre
    print(f"    exporta solo: {only_pre['sections']}")

    print("[4] Guardar y volver a leer (round-trip)")
    path = os.path.join(tmp, "MiProyecto_config" + config_export.CONFIG_EXT)
    config_export.save_config(cfg, path)
    assert os.path.isfile(path)
    back = config_export.load_config(path)
    assert back["dataset"]["config"]["use_bands"] is True
    assert back["models"][0]["clf_params"]["n_estimators"] == 50
    print(f"    escrito y releído: {os.path.basename(path)} ({os.path.getsize(path)} bytes)")

    print("[5] Bundle .eegbundle (ZIP) con binarios de modelo y dataset")
    from eeg_studio.core import dataset as dataset_mod
    dataset_mod.save_dataset(proj, ds, "dataset")           # deja un .npz en datasets/
    bundle = os.path.join(tmp, "MiProyecto" + config_export.BUNDLE_EXT)
    info = config_export.export_bundle(proj, models,
                                       {"preprocessing", "dataset", "models"}, bundle)
    assert os.path.isfile(bundle) and info["models"] == 1 and info["datasets"] == 1, info
    print(f"    bundle: {info['models']} modelo(s), {info['datasets']} dataset(s), "
          f"{os.path.getsize(bundle)} bytes")

    print("[6] Leer el bundle: el modelo y el dataset se reconstruyen")
    cfg2, model_blobs, ds_blobs, _src_blobs = config_export.read_bundle(bundle)
    assert set(model_blobs) == {"rf_1"} and set(ds_blobs) == {"dataset.npz"}, (model_blobs, ds_blobs)
    res2 = classification.result_from_bytes(model_blobs["rf_1"])
    assert res2.classifier_name == "random_forest" and res2.clf_params == {"n_estimators": 50}
    pred = res2.model.predict(ds.X[:3])                      # el modelo funciona
    assert len(pred) == 3
    npz_path = os.path.join(tmp, "dataset.npz")
    with open(npz_path, "wb") as f:
        f.write(ds_blobs["dataset.npz"])
    ds2 = dataset_mod.load_dataset(npz_path)
    assert ds2.X.shape == ds.X.shape, (ds2.X.shape, ds.X.shape)
    print(f"    modelo predice {list(pred)} · dataset {ds2.X.shape} recuperado")

    print("\nEXPORTAR CONFIGURACIÓN / BUNDLE OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

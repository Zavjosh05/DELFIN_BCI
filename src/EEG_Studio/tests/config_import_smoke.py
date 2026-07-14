"""Importar un .eegbundle a un proyecto nuevo: pipeline + dataset + modelos (offscreen)."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
from PyQt6.QtWidgets import QApplication

from eeg_studio.config import IMPORTED_DIR
from eeg_studio.core import classification, config_export
from eeg_studio.core import dataset as dataset_mod
from eeg_studio.core.dataset import Dataset
from eeg_studio.core.mat_loader import write_openvibe_csv
from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)

    print("[1] Proyecto ORIGEN con fuente, pipeline, dataset y modelo -> bundle")
    tmp = tempfile.mkdtemp()
    src = Project.create(tmp, "origen")
    imp = os.path.join(src.path, IMPORTED_DIR)
    os.makedirs(imp, exist_ok=True)
    csv = os.path.join(imp, "sig.csv.gz")
    write_openvibe_csv(csv, np.random.default_rng(1).normal(0, 1, (600, 3)).astype(np.float32),
                       250.0, ["C3", "Cz", "C4"], [(50, "a")])
    sid = src.add_source(csv)["id"]
    src.add_segment(sid, 0, 200, "a")                    # segmento que referencia la fuente
    src.state["pipeline"] = [
        {"type": "bandpass", "params": {"low": 8.0, "high": 30.0}, "enabled": True}]
    src.state["excluded_channels"] = ["EOG-left"]
    rng = np.random.default_rng(0)
    ds = Dataset(X=rng.normal(0, 1, (24, 6)).astype(float),
                 y=np.array(["a", "b"] * 12, dtype=object),
                 feature_names=[f"f{i}" for i in range(6)],
                 segment_ids=[str(i) for i in range(24)])
    dataset_mod.save_dataset(src, ds, "dataset")
    res = classification.train(ds, "random_forest", clf_params={
        "n_estimators": 40, "min_samples_leaf": 2, "class_weight": "balanced"})
    # Recetas de hiperparámetros guardadas SIN entrenar (también viajan al bundle).
    src.save_model_config({"name": "lda_receta", "classifier_name": "lda",
                           "clf_params": {"solver": "lsqr", "shrinkage": "auto"}})
    src.save_model_config({"name": "ts_receta", "classifier_name": "riemann_ts",
                           "raw_window": 256})
    bundle = os.path.join(tmp, "origen" + config_export.BUNDLE_EXT)
    config_export.export_bundle(
        src, {"rf_1": res},
        {"preprocessing", "dataset", "models", "model_configs", "sources"}, bundle)
    print(f"    bundle creado: {os.path.basename(bundle)}")

    print("[2] Importar el bundle en un proyecto NUEVO (otra 'máquina')")
    win = MainWindow()
    tmp2 = tempfile.mkdtemp()
    win.project = Project.create(tmp2, "destino")
    cfg, model_blobs, ds_blobs, src_blobs = config_export.read_bundle(bundle)
    summary = win._apply_config(cfg, model_blobs, ds_blobs, src_blobs)
    print(f"    aplicado: {summary}")

    print("[3] El destino tiene fuente (id conservado), pipeline, dataset y modelo")
    assert win.project.state["pipeline"][0]["params"]["low"] == 8.0
    assert "EOG-left" in win.project.excluded_channels()
    assert win.dataset is not None and win.dataset.X.shape == ds.X.shape, win.dataset
    assert "rf_1" in win.models, list(win.models)
    dst_ids = {s["id"] for s in win.project.sources}
    assert sid in dst_ids, (sid, dst_ids)                # fuente reconstruida con su id
    assert win.project.state["segments"][0]["source_id"] == sid  # el segmento sigue válido
    assert os.path.isfile(win.project.get_source(sid)["path"])
    # los binarios quedaron escritos en el proyecto destino
    assert os.path.isfile(os.path.join(win.project.path, "models", "rf_1.joblib"))
    assert os.path.isfile(os.path.join(win.project.path, "datasets", "dataset.npz"))
    # el modelo importado predice
    pred = win.models["rf_1"].model.predict(ds.X[:2])
    assert len(pred) == 2
    # conserva las métricas -> los gráficos (matriz de confusión) se regeneran, sin
    # haber guardado ninguna imagen en el bundle
    assert win.models["rf_1"].metrics is not None, "el modelo importado perdió sus métricas"
    print(f"    pipeline={win.project.state['pipeline'][0]['type']} · "
          f"dataset={win.dataset.X.shape} · modelos={list(win.models)} · predice {list(pred)}")

    print("[3b] Reimportar el MISMO bundle: fuentes y etiquetas ya presentes se OMITEN")
    n_src, n_seg = len(win.project.sources), len(win.project.state["segments"])
    win._apply_config(*config_export.read_bundle(bundle))         # importa otra vez
    assert len(win.project.sources) == n_src, "duplicó fuentes al reimportar"
    assert len(win.project.state["segments"]) == n_seg, "duplicó segmentos al reimportar"
    print(f"    sin duplicados: {n_src} fuente(s), {n_seg} segmento(s)")

    print("[4] El bundle NO contiene imágenes (solo datos)")
    import zipfile
    with zipfile.ZipFile(bundle) as z:
        names = z.namelist()
    assert not any(n.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf"))
                   for n in names), names
    print(f"    entradas del bundle: {names}")

    print("[5] El bundle trae los HIPERPARÁMETROS del modelo y se detectan")
    entries = config_export.reusable_model_configs(cfg)
    assert entries and entries[0]["classifier_name"] == "random_forest", entries
    e = entries[0]
    assert e["clf_params"]["n_estimators"] == 40, e["clf_params"]
    assert e["clf_params"]["min_samples_leaf"] == 2, e["clf_params"]
    assert e["clf_params"]["class_weight"] == "balanced", e["clf_params"]
    assert "raw_window" in e, "falta raw_window (necesario para reentrenar Riemann/CSP/redes)"
    from eeg_studio.ui import model_config as mc
    assert "n_estimators=40" in mc.describe_model_config(e), mc.describe_model_config(e)
    print(f"    detectado: {e['name']} · {mc.describe_model_config(e)}")

    print("[6] Usar esos parámetros con los datos del proyecto DESTINO")
    task = win._task_for_config(e)
    assert task is not None, "debería poder entrenar (hay dataset)"
    out = task()
    clf = out.model.named_steps["clf"]
    assert (clf.n_estimators, clf.min_samples_leaf, clf.class_weight) == (40, 2, "balanced")
    print(f"    entrenado local: n_estimators={clf.n_estimators}, "
          f"min_samples_leaf={clf.min_samples_leaf}, class_weight={clf.class_weight}")

    print("[6b] La oferta al importar entrena y AÑADE el modelo (sin pisar el importado)")
    import time
    orig = mc.choose_imported_configs
    mc.choose_imported_configs = staticmethod(lambda *a, **k: entries[:1])  # simula «sí»
    try:
        win.offer_imported_model_configs(cfg)
        end = time.time() + 30
        while time.time() < end and "rf_1_local" not in win.models:
            app.processEvents()
            time.sleep(0.02)
    finally:
        mc.choose_imported_configs = orig
    assert "rf_1_local" in win.models, list(win.models)
    assert "rf_1" in win.models, "no debe pisar el modelo importado"
    print(f"    modelos ahora: {sorted(win.models)}")

    print("[6c] Sin los datos necesarios, la configuración no se ofrece")
    saved, win.dataset = win.dataset, None
    assert win._task_for_config(e) is None, "sin dataset no debería poder entrenar"
    win.dataset = saved
    print("    sin dataset -> no disponible")

    print("[7] Las configuraciones SIN entrenar viajan en el bundle y se importan")
    assert {c["name"] for c in cfg["model_configs"]} == {"lda_receta", "ts_receta"}, \
        cfg["model_configs"]
    got = {c["name"]: c for c in win.project.model_configs()}
    assert set(got) == {"lda_receta", "ts_receta"}, list(got)     # llegaron al destino
    assert got["lda_receta"]["clf_params"] == {"solver": "lsqr", "shrinkage": "auto"}
    assert got["ts_receta"]["raw_window"] == 256, got["ts_receta"]
    # y se ofrecen para reentrenar junto a los modelos entrenados
    names = {x.get("name") for x in config_export.reusable_model_configs(cfg)}
    assert {"rf_1", "lda_receta", "ts_receta"} <= names, names
    print(f"    importadas: {sorted(got)} · reutilizables: {sorted(names)}")

    print("[7b] Reimportar NO duplica las configuraciones ya presentes")
    n_cfg = len(win.project.model_configs())
    win._apply_config(*config_export.read_bundle(bundle))
    assert len(win.project.model_configs()) == n_cfg, win.project.model_configs()
    print(f"    siguen siendo {n_cfg}")

    print("[10] Importar NO borra los pipelines/canales propios (los añade)")
    # Regresión: set_pipelines REEMPLAZABA todos los pipelines, así que importar un
    # bundle borraba los que tenía el usuario (y recuperarlos exigía varios undo).
    o2 = Project.create(tempfile.mkdtemp(), "otro_origen")
    o2.set_pipelines([{"name": "Pipeline del bundle", "steps": [
        {"type": "bandpass", "params": {"low": 8.0, "high": 30.0}, "enabled": True}]}], 0)
    o2.state["excluded_channels"] = ["EOG-left"]
    b3 = os.path.join(o2.path, "pre" + config_export.BUNDLE_EXT)
    config_export.export_bundle(o2, {}, {"preprocessing"}, b3)

    win4 = MainWindow()
    win4.project = Project.create(tempfile.mkdtemp(), "con_pipelines")
    win4.project.set_pipelines([
        {"name": "Mi pipeline A", "steps": [{"type": "car", "params": {}, "enabled": True}]},
        {"name": "Mi pipeline B", "steps": [
            {"type": "notch", "params": {"freq": 60.0}, "enabled": True}]}], 1)
    win4.project.edit("excluded_channels", ["P7"], "propio")
    win4._apply_config(*config_export.read_bundle(b3))
    got = [p["name"] for p in win4.project.pipelines_snapshot()]
    assert got == ["Mi pipeline A", "Mi pipeline B", "Pipeline del bundle"], got
    assert win4.project.active_pipeline_index() == 1, "no debe cambiar el activo"
    # canales excluidos: se fusionan, no se pisan
    assert set(win4.project.excluded_channels()) == {"P7", "EOG-left"}, \
        win4.project.excluded_channels()
    win4._apply_config(*config_export.read_bundle(b3))          # reimportar
    assert len(win4.project.pipelines_snapshot()) == 3, "duplicó pipelines al reimportar"
    # Un proyecto en blanco (un pipeline vacío) sí adopta el del bundle.
    win5 = MainWindow()
    win5.project = Project.create(tempfile.mkdtemp(), "en_blanco")
    win5._apply_config(*config_export.read_bundle(b3))
    assert [p["name"] for p in win5.project.pipelines_snapshot()] == ["Pipeline del bundle"]
    win4.acq_panel.shutdown(); win5.acq_panel.shutdown()
    print(f"    {got} · activo conservado · excluidos fusionados · sin duplicar")

    print("[9] Importar VARIOS bundles: el segundo NO pisa lo del primero")
    # Regresión: todos los bundles traen su dataset como «dataset.npz» y suelen
    # repetir nombres de modelo (rf_1), así que el segundo import borraba en
    # silencio el dataset y el modelo del primero.
    from eeg_studio.config import DATASETS_DIR, MODELS_DIR

    def _mk_bundle(name, n_feats, label):
        p = Project.create(tempfile.mkdtemp(), name)
        r = np.random.default_rng(0)
        d = Dataset(X=r.normal(0, 1, (20, n_feats)),
                    y=np.array([label, "z"] * 10, dtype=object),
                    feature_names=[f"f{i}" for i in range(n_feats)],
                    segment_ids=[str(i) for i in range(20)])
        dataset_mod.save_dataset(p, d, "dataset")            # siempre «dataset.npz»
        r2 = classification.train(d, "random_forest", clf_params={"n_estimators": 10})
        out = os.path.join(p.path, name + config_export.BUNDLE_EXT)
        config_export.export_bundle(p, {"rf_1": r2}, {"dataset", "models"}, out)
        return out

    bA, bB = _mk_bundle("A", 6, "claseA"), _mk_bundle("B", 9, "claseB")
    win3 = MainWindow()
    win3.project = Project.create(tempfile.mkdtemp(), "varios")
    win3._apply_config(*config_export.read_bundle(bA))
    win3._apply_config(*config_export.read_bundle(bB))
    dsdir = os.path.join(win3.project.path, DATASETS_DIR)
    assert sorted(os.listdir(dsdir)) == ["dataset.npz", "dataset_2.npz"], os.listdir(dsdir)
    d1 = dataset_mod.load_dataset(os.path.join(dsdir, "dataset.npz"))
    d2 = dataset_mod.load_dataset(os.path.join(dsdir, "dataset_2.npz"))
    assert d1.y[0] == "claseA", "¡el dataset del primer bundle se perdió!"
    assert d2.y[0] == "claseB" and d2.X.shape[1] == 9, (d2.y[0], d2.X.shape)
    assert set(win3.models) == {"rf_1", "rf_1_2"}, list(win3.models)   # ninguno pisado
    mdir = os.path.join(win3.project.path, MODELS_DIR)
    assert sorted(os.listdir(mdir)) == ["rf_1.joblib", "rf_1_2.joblib"], os.listdir(mdir)
    win3.acq_panel.shutdown()
    print("    datasets y modelos de ambos bundles conviven (renombrando el 2º)")

    print("[8] Ventana de importación: elegir QUÉ importar (todo marcado por defecto)")
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QCheckBox
    win2 = MainWindow()
    win2.project = Project.create(tempfile.mkdtemp(), "seleccion")
    seen = {}

    def _shoot():
        w = app.activeModalWidget()
        if w:
            cbs = w.findChildren(QCheckBox)
            seen["labels"] = [c.text() for c in cbs]
            seen["all_checked"] = all(c.isChecked() for c in cbs)
            w.reject()                                   # cancelar
    QTimer.singleShot(300, _shoot)
    assert win2._ask_import_sections(cfg, model_blobs, ds_blobs, src_blobs) is None
    assert seen.get("all_checked") is True, seen
    assert len(seen["labels"]) == 5, seen["labels"]      # las 5 partes del bundle
    print(f"    {len(seen['labels'])} casillas, todas marcadas; cancelar no importa nada")

    print("[8b] Importar solo una parte respeta la selección")
    win2._apply_config(cfg, model_blobs, ds_blobs, src_blobs,
                       sections={"preprocessing", "model_configs"})
    assert win2.project.state["pipeline"][0]["params"]["low"] == 8.0     # sí
    assert len(win2.project.sources) == 0, "no debía importar fuentes"
    assert win2.dataset is None, "no debía importar el dataset"
    assert win2.models == {}, "no debía importar modelos"
    assert {c["name"] for c in win2.project.model_configs()} == {"lda_receta", "ts_receta"}
    win2.acq_panel.shutdown()
    print("    solo preprocesamiento + configuraciones (nada más)")

    print("[7c] Sin marcar la casilla, el bundle NO las lleva")
    b2 = os.path.join(tmp, "sin_cfg" + config_export.BUNDLE_EXT)
    config_export.export_bundle(src, {"rf_1": res}, {"preprocessing", "models"}, b2)
    cfg2, _m, _d, _s = config_export.read_bundle(b2)
    assert "model_configs" not in cfg2, cfg2.get("model_configs")
    assert "model_configs" in config_export.SECTIONS
    print("    sección opcional respetada")

    win.acq_panel.shutdown()
    print("\nIMPORTAR BUNDLE OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

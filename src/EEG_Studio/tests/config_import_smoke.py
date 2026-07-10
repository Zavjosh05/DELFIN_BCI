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
    res = classification.train(ds, "random_forest", clf_params={"n_estimators": 40})
    bundle = os.path.join(tmp, "origen" + config_export.BUNDLE_EXT)
    config_export.export_bundle(src, {"rf_1": res},
                                {"preprocessing", "dataset", "models", "sources"}, bundle)
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

    win.acq_panel.shutdown()
    print("\nIMPORTAR BUNDLE OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prueba integral del flujo completo (núcleo, sin GUI).

Encadena: proyecto → fuente CSV → pipeline con ICA → segmentos → dataset de
características y de señal cruda → entrenamiento de modelos clásicos, EEGNet y
Riemann/CSP → predicción. Verifica que el CSV de origen no se modifica.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.core import classification as clf
from eeg_studio.core import dataset as ds
from eeg_studio.core import neuralnet
from eeg_studio.core.dataset import fit_window
from eeg_studio.core.processing import extract_feature_vector
from eeg_studio.core.project import Project

from tests import data_dir
EEG_DIR = data_dir()


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(8192), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    csv = os.path.join(EEG_DIR, "Prueba_001.csv")
    digest = _md5(csv)

    print("[1] Proyecto + fuente + pipeline (incluye ICA)")
    proj = Project.create(tempfile.mkdtemp(), "e2e")
    sid = proj.add_source(csv)["id"]
    proj.add_pipeline_step("bandpass", {"low": 1.0, "high": 45.0, "order": 4})
    proj.add_pipeline_step("ica", {"n_components": 0, "kurt_threshold": 5.0})
    proj.add_pipeline_step("car")
    rec = proj.get_recording(sid)

    print("[2] Segmentos (2 clases) a partir de las épocas")
    for i, ep in enumerate(rec.epoch_ids[:8]):
        a, b = rec.epoch_range(ep)
        proj.add_segment(sid, a, b, label=f"clase_{i % 2}")

    print("[3] Dataset de características → clásicos y MLP")
    feat = ds.build_dataset(proj)
    print(f"    características X={feat.X.shape}")
    for name, params in (("random_forest", None), ("svm", {"kernel": "linear"})):
        r = clf.train(feat, name, clf_params=params)
        print(f"    {name}: CV={r.cv_mean:.2f}")
    mlp_cfg = neuralnet.default_config("mlp"); mlp_cfg["epochs"] = 15
    r_mlp = clf.train(feat, "nn_mlp", nn_config=mlp_cfg)
    assert r_mlp.input_kind == "features"

    print("[4] Dataset crudo → EEGNet y Riemann/CSP")
    raw = ds.build_raw_dataset(proj, window_samples=128)
    print(f"    crudo X={raw.X.shape}")
    eeg_cfg = neuralnet.default_config("eegnet"); eeg_cfg["epochs"] = 12; eeg_cfg["window_samples"] = 128
    r_eeg = clf.train_raw(raw, "nn_eegnet", nn_config=eeg_cfg)
    assert r_eeg.input_kind == "raw"
    for name in ("riemann_mdm", "csp_lda"):
        r = clf.train_riemann(raw, name)
        print(f"    {name}: CV={r.cv_mean:.2f}")

    print("[5] Predicción de una región (características y cruda)")
    data, fs = proj.segment_data({"source_id": sid, "start": 0, "stop": 512, "channels": None})
    vec, _ = extract_feature_vector(data, fs, True, True)
    assert clf.predict(r_mlp, vec.reshape(1, -1)).shape == (1,)
    win = fit_window(data, 128)[np.newaxis, ...]
    assert clf.predict(r_eeg, win).shape == (1,)
    print("    predicciones OK")

    print("[6] El CSV de origen no cambió")
    assert _md5(csv) == digest, "¡el CSV de origen fue modificado!"
    print("    MD5 idéntico ✓")

    print("\nFLUJO COMPLETO (E2E) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

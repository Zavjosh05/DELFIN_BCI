"""Ver/editar la configuración de un modelo y reentrenar; secciones de la imagen.

Sin pantalla (offscreen). Comprueba los editores de configuración (getters),
la persistencia de ``raw_window`` y que la imagen-informe respeta las secciones.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.core import classification  # noqa: E402
from eeg_studio.core.dataset import Dataset  # noqa: E402
from eeg_studio.ui import metrics_view, model_config  # noqa: E402


def _ds(n=40, d=6):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (n // 2, d)), rng.normal(3, 1, (n // 2, d))])
    y = np.array(["a"] * (n // 2) + ["b"] * (n // 2), dtype=object)
    return Dataset(X=X, y=y, feature_names=[f"f{i}" for i in range(d)],
                   segment_ids=[str(i) for i in range(n)])


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841

    print("[1] Editor RF prellena desde clf_params y devuelve los cambios")
    res = classification.train(_ds(), "random_forest",
                               clf_params={"n_estimators": 123, "criterion": "entropy"})
    box, getter = model_config._rf_editor(res.clf_params)   # noqa: SLF001
    params = getter()
    assert params["n_estimators"] == 123, params
    assert params["criterion"] == "entropy", params

    print("[2] Editor SVM prellena kernel/C y devuelve los cambios")
    _, sgetter = model_config._svm_editor({"kernel": "linear", "C": 4.0})   # noqa: SLF001
    sp = sgetter()
    assert sp["kernel"] == "linear" and sp["C"] == 4.0, sp

    print("[3] Reentrenar clásico con nuevos parámetros (mismo flujo que el controlador)")
    new = classification.train(_ds(), "random_forest", clf_params=params)
    assert new.model.named_steps["clf"].n_estimators == 123
    assert classification.predict(new, _ds().X[:3]).shape == (3,)

    print("[4] raw_window se conserva al serializar (para reentrenar Riemann)")
    r = classification.TrainingResult(model=None, classifier_name="riemann_ts",
                                      classes=["a", "b"], feature_names=[], raw_window=384)
    r2 = classification.result_from_bytes(classification.result_to_bytes(r))
    assert r2.raw_window == 384, r2.raw_window
    _, rget = model_config._riemann_editor(r2)   # noqa: SLF001
    assert rget() == 384, rget()

    print("[5] Editor NN edita escalares y deja las capas fijas")
    from eeg_studio.core import neuralnet
    cfg = neuralnet.default_config("mlp")
    cfg["epochs"] = 7
    _, nget = model_config._nn_editor(cfg)   # noqa: SLF001
    ncfg = nget()
    assert ncfg["epochs"] == 7 and ncfg["type"] == "mlp"
    assert isinstance(ncfg["layers"], list), "las capas deben conservarse"

    print("[6] La imagen-informe respeta las secciones elegidas")
    m = res.metrics
    if m and metrics_view.matplotlib_available():
        full = metrics_view.build_report_figure(m, "h", sections={
            "confusion": True, "f1": True, "per_class": True, "global": True})
        only_cm = metrics_view.build_report_figure(m, "h", sections={
            "confusion": True, "f1": False, "per_class": False, "global": False})
        assert len(full.axes) > len(only_cm.axes), (len(full.axes), len(only_cm.axes))
        assert only_cm.get_figheight() < full.get_figheight(), "menos secciones = más compacta"
        print(f"    completa={len(full.axes)} ejes / solo matriz={len(only_cm.axes)} ejes; "
              f"altura {full.get_figheight():.1f}in -> {only_cm.get_figheight():.1f}in")
    else:
        print("    (sin matplotlib o sin métricas; se omite)")

    print("\nCONFIGURACIÓN/REENTRENAMIENTO DE MODELOS OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

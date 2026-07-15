"""Estrategia multiclase: reducción a clasificadores binarios (OvO / OvR).

Con muchas clases (las 6 acciones Delfin) la literatura BCI sugiere descomponer el
problema multiclase en varios binarios. Cubre:

  * «nativa» (por defecto) NO envuelve: ``named_steps["clf"]`` sigue siendo el
    estimador de siempre (compatibilidad con el código y las pruebas existentes).
  * OvO entrena N·(N−1)/2 binarios y OvR entrena N; ambos predicen y conservan las
    clases.
  * Un problema binario no se envuelve aunque se pida (sería el mismo modelo).
  * La estrategia viaja en ``clf_params``: se persiste al guardar/cargar el modelo.
  * Riemann/CSP: con OvO/OvR el **CSP va DENTRO de cada binario** (cada uno aprende
    sus propios filtros espaciales), que es el enfoque estándar en MI multiclase.
  * La interfaz expone el selector para clásicos y Riemann, pero no para redes.
"""
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
from sklearn.ensemble import RandomForestClassifier

from eeg_studio.core import classification as C


class _DS:
    """Dataset de características mínimo (lo que espera ``train``)."""

    def __init__(self, n_classes: int, n=60, n_feat=8, seed=0):
        rng = np.random.default_rng(seed)
        self.X = rng.normal(0, 1, (n, n_feat))
        self.y = np.array([f"c{i % n_classes}" for i in range(n)])
        # Señal separable: desplaza cada clase para que el modelo aprenda algo.
        for i, lab in enumerate(sorted(set(self.y))):
            self.X[self.y == lab] += i * 1.5
        self.feature_names = [f"f{i}" for i in range(n_feat)]


def main() -> int:
    print("[1] «nativa» no envuelve (compatibilidad: clf sigue siendo el estimador)")
    pipe = C.build_pipeline("random_forest", {"n_estimators": 10}, n_classes=6)
    assert isinstance(pipe.named_steps["clf"], RandomForestClassifier)
    # Sin la clave tampoco (config antigua) y con estrategia explícita tampoco.
    pipe = C.build_pipeline("random_forest", {"multiclass": "nativa"}, n_classes=6)
    assert isinstance(pipe.named_steps["clf"], RandomForestClassifier)
    print("    named_steps['clf'] intacto ✓")

    print("[2] El nº de binarios sale de las clases DEL DATASET (nada fijado)")
    # Se comprueba con varios N, no solo con las 6 clases Delfin: OvO = N·(N−1)/2
    # y OvR = N, sea cual sea el dataset que reciba el clasificador.
    for n_cls in (3, 4, 5, 6, 7):
        for strat in ("ovo", "ovr"):
            expected = n_cls * (n_cls - 1) // 2 if strat == "ovo" else n_cls
            res = C.train(_DS(n_cls), "lda", cv=2, clf_params={"multiclass": strat})
            n_bin = len(res.model.named_steps["clf"].estimators_)
            assert n_bin == expected, (n_cls, strat, n_bin, expected)
            assert res.classes == [f"c{i}" for i in range(n_cls)]
        print(f"    {n_cls} clases → OvO: {n_cls * (n_cls - 1) // 2} binarios · "
              f"OvR: {n_cls} binarios ✓")

    print("[2b] Predice y guarda la estrategia elegida")
    for strat in ("ovo", "ovr"):
        res = C.train(_DS(6), "lda", cv=2, clf_params={"multiclass": strat})
        pred = C.predict(res, _DS(6).X[:5])
        assert pred.shape == (5,) and set(pred) <= set(res.classes)
        assert res.clf_params["multiclass"] == strat, "la estrategia no se guardó"
    print("    predicción y persistencia OK")

    print("[3] Un problema BINARIO no se envuelve (sería el mismo modelo)")
    pipe = C.build_pipeline("svm", {"multiclass": "ovo"}, n_classes=2)
    assert not hasattr(pipe.named_steps["clf"], "estimators_"), "no debería envolver"
    res2 = C.train(_DS(2), "lda", cv=2, clf_params={"multiclass": "ovr"})
    assert not hasattr(res2.model.named_steps["clf"], "estimators_")
    print("    con 2 clases se deja el estimador único ✓")

    print("[4] La estrategia sobrevive a guardar/cargar el modelo (.joblib)")
    res = C.train(_DS(6), "lda", cv=2, clf_params={"multiclass": "ovr"})
    back = C.result_from_bytes(C.result_to_bytes(res))
    assert back.clf_params["multiclass"] == "ovr", back.clf_params
    assert len(back.model.named_steps["clf"].estimators_) == 6
    print("    round-trip conserva la estrategia y los 6 binarios ✓")

    print("[5] Riemann/CSP: el CSP va DENTRO de cada binario")
    if C.riemann_available():
        p = C.build_riemann_pipeline("csp_lda", {"multiclass": "ovr"}, n_classes=6)
        inner = p.named_steps["clf"].estimator      # lo que se replica por binario
        assert "csp" in inner.named_steps and "lda" in inner.named_steps, \
            "CSP debe reaprenderse por cada problema binario"
        # Nativa: sin envolver (el CSP multiclase de pyriemann, como antes).
        p0 = C.build_riemann_pipeline("csp_lda", None, n_classes=6)
        assert not hasattr(p0.named_steps["clf"], "estimator")
        print("    csp+lda envueltos juntos en OvR; nativa sin envolver ✓")
    else:
        print("    pyriemann no disponible: se omite (esperado en ese caso)")

    print("[6] Interfaz: selector visible en clásicos y Riemann, NO en redes")
    from PyQt6.QtWidgets import QApplication
    from eeg_studio.core.project import Project
    from eeg_studio.ui.main_window import MainWindow
    app = QApplication(sys.argv)                   # noqa: F841
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "mc")
    panel = win.clf_panel
    strategies = [panel.multiclass_combo.itemData(i)
                  for i in range(panel.multiclass_combo.count())]
    assert strategies == ["nativa", "ovo", "ovr"], strategies

    def _select(key):
        panel.clf_combo.setCurrentIndex(panel.clf_combo.findData(key))

    _select("lda")
    assert panel.multiclass_box.isVisibleTo(panel), "clásico debería mostrarlo"
    assert panel.lda_params()["multiclass"] == "nativa"
    _select("csp_lda")
    assert panel.multiclass_box.isVisibleTo(panel), "Riemann debería mostrarlo"
    assert panel.riemann_params() == {"multiclass": "nativa"}
    _select("nn_mlp")
    assert not panel.multiclass_box.isVisibleTo(panel), "las redes NO lo usan"
    print("    visible en LDA y CSP+LDA, oculto en la red ✓")

    print("[7] La estrategia elegida llega a los parámetros y a la config")
    _select("lda")
    panel.multiclass_combo.setCurrentIndex(panel.multiclass_combo.findData("ovo"))
    assert panel.lda_params()["multiclass"] == "ovo"
    cfg = panel.current_config_dict()
    assert cfg["clf_params"]["multiclass"] == "ovo", cfg
    _select("csp_lda")
    panel.multiclass_combo.setCurrentIndex(panel.multiclass_combo.findData("ovr"))
    cfg = panel.current_config_dict()
    assert cfg["clf_params"]["multiclass"] == "ovr" and cfg["raw_window"], cfg
    # Y se puede volver a cargar en los campos.
    panel.multiclass_combo.setCurrentIndex(panel.multiclass_combo.findData("nativa"))
    panel.apply_config_dict(cfg)
    assert panel.multiclass_strategy() == "ovr", "apply_config_dict no la aplicó"
    print("    va y vuelve por current_config_dict/apply_config_dict ✓")

    win.acq_panel.shutdown()
    print("\nESTRATEGIA MULTICLASE (OvO / OvR) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

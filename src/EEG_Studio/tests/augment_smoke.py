"""Aumento de datos: técnicas, integración y —lo crítico— que NO falsee la validación.

Lo que más importa aquí: el aumento debe aplicarse SOLO al pliegue de
entrenamiento. Si tocara el de validación, la exactitud subiría midiendo memoria
en vez de aprendizaje, que es justo lo contrario de lo que se busca. Offscreen.
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

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.core import augment, classification as C  # noqa: E402
from eeg_studio.core.dataset import Dataset  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


def _all_on(copies=2, prob=1.0):
    cfg = augment.default_config()
    cfg.update(enabled=True, copies=copies, probability=prob)
    cfg["techniques"] = {k: True for k in augment.TECHNIQUES}
    return cfg


def _ds(n=60, d=8):
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (n // 2, d)), rng.normal(1.2, 1, (n // 2, d))])
    y = np.array(["a"] * (n // 2) + ["b"] * (n // 2), dtype=object)
    return Dataset(X=X, y=y, feature_names=[f"f{i}" for i in range(d)],
                   segment_ids=[str(i) for i in range(n)])


def main() -> int:
    app = QApplication(sys.argv)          # noqa: F841

    print("[1] Señal cruda: multiplica los ensayos y conserva los originales")
    X = np.random.default_rng(0).normal(0, 1, (12, 14, 256))
    y = np.array(["arriba", "abajo"] * 6, dtype=object)
    Xa, ya = augment.augment(X, y, _all_on())
    assert Xa.shape == (36, 14, 256) and len(ya) == 36, Xa.shape
    assert np.allclose(Xa[:12], X), "los originales deben conservarse tal cual"
    assert list(ya[:12]) == list(y) == list(ya[12:24]), "etiquetas descuadradas"
    assert not np.allclose(Xa[12:24], X), "las copias deberían diferir del original"
    print(f"    {X.shape} -> {Xa.shape}, originales intactos")

    print("[2] Características (2D): la traslación temporal se ignora, el resto aplica")
    F = np.random.default_rng(1).normal(0, 1, (10, 6))
    yf = np.array(["a", "b"] * 5, dtype=object)
    Fa, _ = augment.augment(F, yf, _all_on())
    assert Fa.shape == (30, 6) and np.allclose(Fa[:10], F), Fa.shape
    print("    (10, 6) -> (30, 6) sin romper por no tener eje temporal")

    print("[3] Reproducible, y apagado no toca nada")
    a, _ = augment.augment(X, y, _all_on(), np.random.default_rng(7))
    b, _ = augment.augment(X, y, _all_on(), np.random.default_rng(7))
    assert np.allclose(a, b), "misma semilla debería dar el mismo resultado"
    Xo, yo = augment.augment(X, y, augment.default_config())     # por defecto: apagado
    assert Xo.shape == X.shape and np.allclose(Xo, X), "apagado no debe alterar nada"
    assert augment.describe(augment.default_config()) == "sin aumento"
    print("    misma semilla = mismo resultado · apagado = sin cambios")

    print("[4] Mixup NO cruza clases (si no, la etiqueta dejaría de ser válida)")
    X2 = np.zeros((4, 2, 8))
    X2[0:2] = 1.0                                    # clase A = 1.0, clase B = 0.0
    y2 = np.array(["A", "A", "B", "B"], dtype=object)
    cfg = augment.default_config()
    cfg.update(enabled=True, copies=1, probability=1.0)
    cfg["techniques"] = {k: (k == "mixup") for k in augment.TECHNIQUES}
    Xm, _ = augment.augment(X2, y2, cfg, np.random.default_rng(0))
    assert np.allclose(Xm[4:6], 1.0), "una copia de A se mezcló con B"
    assert np.allclose(Xm[6:8], 0.0), "una copia de B se mezcló con A"
    print("    las copias de cada clase siguen dentro de su clase")

    print("[5] CRÍTICO: solo se aumenta el TRAIN; el test se evalúa con datos reales")
    ds = _ds()
    tam_train, tam_test = [], []
    orig_fit, orig_pred = C.Pipeline.fit, C.Pipeline.predict

    def spy_fit(self, X_, y_=None, **k):
        tam_train.append(len(X_))
        return orig_fit(self, X_, y_, **k)

    def spy_pred(self, X_, **k):
        tam_test.append(len(X_))
        return orig_pred(self, X_, **k)

    C.Pipeline.fit, C.Pipeline.predict = spy_fit, spy_pred
    try:
        res = C.train(ds, "lda", cv=5, augment_config=_all_on(copies=2))
    finally:
        C.Pipeline.fit, C.Pipeline.predict = orig_fit, orig_pred
    assert all(t == 144 for t in tam_train[:5]), f"train sin aumentar: {tam_train[:5]}"
    assert all(t == 12 for t in tam_test), f"¡el TEST se aumentó! {tam_test}"
    assert tam_train[-1] == 180, f"el modelo final debe usar todo + aumento: {tam_train[-1]}"
    print(f"    train 48→144 por pliegue · test 12 intacto · final 60→180")

    print("[6] La configuración queda en el modelo y sobrevive al .joblib")
    back = C.result_from_bytes(C.result_to_bytes(res))
    assert back.augment_config and back.augment_config["copies"] == 2, back.augment_config
    sin = C.train(ds, "lda", cv=5)                    # sin aumento: como siempre
    assert sin.augment_config is None, "sin aumento no debe inventarse una config"
    print(f"    {augment.describe(back.augment_config)} · sin aumento -> None")

    print("[6b] TODAS las vías de entrenamiento aceptan y USAN el aumento")
    # Regresión: train_raw() se quedó sin el parámetro, así que la llamada desde la
    # interfaz reventaba con TypeError dentro del hilo -> diálogo de error modal ->
    # la app (y la prueba) se quedaban colgadas esperando un clic.
    import inspect
    for fn in (C.train, C.train_raw, C.train_riemann, C._train_nn):
        assert "augment_config" in inspect.signature(fn).parameters, \
            f"{fn.__name__} no acepta augment_config"
    # Y no basta con aceptarlo: hay que pasarlo hacia dentro. Se comprueba viendo
    # que el aumento llega al fit de la red.
    seen = {}

    class _FakeRaw:
        def __init__(self, X, y):
            self.X, self.y = X, y

    Xr = np.random.default_rng(3).normal(0, 1, (8, 3, 32))
    yr = np.array(["a", "b"] * 4, dtype=object)
    orig_nn = C._train_nn

    def spy_nn(X_, y_, name, kind, cfg, feature_names, progress=None,
               augment_config=None):
        seen["aug"] = augment_config
        return "ok"
    C._train_nn = spy_nn
    try:
        C.train_raw(_FakeRaw(Xr, yr), "nn_cnn", {"type": "cnn"},
                    augment_config=_all_on(copies=1))
    finally:
        C._train_nn = orig_nn
    assert seen.get("aug") and seen["aug"]["enabled"], \
        "train_raw aceptó el aumento pero NO lo pasó a la red"
    print("    train / train_raw / train_riemann / _train_nn: aceptan y propagan")

    print("[7] La interfaz arranca apagada y su config viaja con la receta")
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "aug")
    win.refresh_all()
    panel = win.clf_panel
    assert panel.augment_config()["enabled"] is False, "no debe activarse solo"
    panel.aug_box.setChecked(True)
    panel.aug_copies.setValue(3)
    panel._aug_checks["mixup"].setChecked(True)
    entry = panel.current_config_dict()
    assert entry["augment_config"]["copies"] == 3, entry["augment_config"]
    panel.apply_config_dict(entry)                    # ida y vuelta
    assert panel.augment_config()["copies"] == 3
    # Una configuración ANTIGUA (sin la clave) deja el aumento apagado.
    panel.apply_config_dict({"classifier_name": panel.classifier_key,
                             "clf_params": panel.classic_params()})
    assert panel.augment_config()["enabled"] is False, "config vieja -> apagado"
    assert panel.default_config_dict("random_forest")["augment_config"]["enabled"] is False
    win.acq_panel.shutdown()
    print("    apagado por defecto · viaja en la receta · compatible con las viejas")

    print("\nAUMENTO DE DATOS OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

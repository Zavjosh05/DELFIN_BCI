"""Control del brazo desde un archivo grabado (offscreen).

Cubre las tres piezas de la función «controlar desde una grabación»:
  * ``FilePlaybackSource`` reproduce un CSV como si fuera señal en vivo.
  * ``classify_recording`` clasifica la grabación completa (voto mayoritario +
    exactitud frente a la verdad-terreno).
  * la sección «Controlar desde archivo grabado» del panel de Control clasifica
    el archivo y mueve el brazo simulado, mostrando esperado (del nombre) vs
    predicho.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402

from eeg_studio.acquisition import CSVRecorder, FilePlaybackSource  # noqa: E402
from eeg_studio.core import classification, stim as stim_core  # noqa: E402
from eeg_studio.core.dataset import Dataset  # noqa: E402
from eeg_studio.core.processing import extract_feature_vector  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.inference import classify_recording  # noqa: E402


def _signal(kind: str, ch: int = 14, T: int = 256, seed: int = 0) -> np.ndarray:
    """Ventana separable por clase: 'abajo' = 8 Hz, 'arriba' = 20 Hz + ruido."""
    rng = np.random.default_rng(seed)
    t = np.arange(T) / 128.0
    f = 8.0 if kind == "abajo" else 20.0
    return np.sin(2 * np.pi * f * t)[None, :] * np.ones((ch, 1)) + rng.normal(0, 0.3, (ch, T))


def _train_model():
    X, y = [], []
    for i in range(40):
        kind = "abajo" if i % 2 == 0 else "arriba"
        vec, _ = extract_feature_vector(_signal(kind, seed=i), 128.0, True, True)
        X.append(vec); y.append(kind)
    ds = Dataset(np.vstack(X), np.array(y, dtype=object),
                 feature_names=[f"f{i}" for i in range(len(X[0]))],
                 segment_ids=[str(i) for i in range(40)])
    return classification.train(ds, "random_forest")


def _write_recording(path: str, kind: str, n: int = 640) -> None:
    """Escribe una grabación sintética de una sola clase en formato OpenViBE."""
    sig = _signal(kind, T=n, seed=999)
    rec = CSVRecorder(path, n_channels=14, sample_rate=128.0)
    rec.write(sig)
    rec.close()


def main() -> int:
    tmp = tempfile.mkdtemp()

    print("[1] class_from_filename saca la clase esperada del nombre")
    assert stim_core.class_from_filename("Sujeto001_Abajo.csv") == "abajo"
    print("    Sujeto001_Abajo -> abajo")

    print("[2] FilePlaybackSource reproduce el CSV como fuente en vivo")
    rec_path = os.path.join(tmp, "Sujeto001_Abajo.csv")
    _write_recording(rec_path, "abajo", n=640)
    src = FilePlaybackSource(rec_path, speed=60.0)   # rápido para no esperar en tiempo real
    src.start()
    got = 0
    names = None
    for _ in range(400):
        chunk = src.read()
        if chunk is not None:
            got += chunk.shape[1]
            names = src.channel_names
        if src.finished and src.read() is None:
            break
        time.sleep(0.01)
    src.stop()
    assert names is not None and len(names) == 14, names
    assert got >= 600, f"reprodujo muy pocas muestras: {got}"
    assert src.finished, "la fuente no terminó la reproducción"
    print(f"    emitió {got} muestras de 14 canales y terminó ✓")

    print("[3] classify_recording: voto mayoritario + exactitud vs verdad-terreno")
    model = _train_model()
    proj = Project.create(os.path.join(tmp, "P.eegproj"), "P")
    from eeg_studio.core.csv_loader import load_recording
    rc = load_recording(rec_path)
    gt = np.full(rc.n_samples, "abajo", dtype=object)
    summary = classify_recording(model, proj, rc.data, rc.sample_rate, window=256, ground_truth=gt)
    print(f"    label={summary['label']}  n_ventanas={summary['n_windows']}  "
          f"exactitud={summary['accuracy']}")
    assert summary["label"] == "abajo", summary["counts"]
    assert summary["n_windows"] >= 2, "debería evaluar varias ventanas"
    assert summary["accuracy"] is not None and summary["accuracy"] > 0.9, summary["accuracy"]
    # Una grabación de la otra clase da la otra etiqueta.
    other_path = os.path.join(tmp, "Sujeto001_Arriba.csv")
    _write_recording(other_path, "arriba", n=512)
    ro = load_recording(other_path)
    s2 = classify_recording(model, proj, ro.data, ro.sample_rate, window=256)
    assert s2["label"] == "arriba", s2["counts"]
    print("    'abajo' -> abajo (100%), 'arriba' -> arriba ✓")

    print("[4] Panel de Control: clasifica el archivo y mueve el brazo simulado")
    from PyQt6.QtWidgets import QApplication
    from eeg_studio.ui.main_window import MainWindow
    app = QApplication(sys.argv)  # noqa: F841
    win = MainWindow()
    win.project = proj
    win.models["rf_abajo"] = model
    win.active_model_name = "rf_abajo"
    cp = win.control_panel
    cp.refresh()
    cp.profile_combo.setCurrentIndex(1)      # perfil «Brazo simulado» (movimiento visible)
    cp._sim_arm.reset()
    home_q = cp._sim_arm.q.copy()

    # _classify_file usa controller._spawn (async); lo hacemos SÍNCRONO para el test.
    win._spawn = lambda fn, on_done=None, on_error=None, *a, **k: (
        on_done(fn()) if on_done else fn())
    cp._file_path = rec_path
    cp.file_edit.setText(rec_path)
    cp._classify_file()

    assert not np.allclose(cp._sim_arm.q, home_q), "el brazo no se movió tras clasificar el archivo"
    txt = cp.file_result.text()
    print(f"    resultado: {txt.splitlines()[0]}")
    assert "abajo" in txt and "✓" in txt, txt
    win.acq_panel.shutdown()

    print("\nCONTROL DESDE ARCHIVO GRABADO OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

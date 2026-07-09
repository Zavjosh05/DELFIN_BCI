"""Prueba de humo del núcleo (sin interfaz gráfica).

Ejercita el flujo completo: cargar CSV, crear proyecto, definir pipeline,
crear segmentos, construir dataset (multiproceso) y entrenar un modelo.
Verifica además que el CSV de origen no se modifica.

Uso:
    python -m tests.smoke_test ../EEG/Prueba_001.csv
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile

# La consola de Windows usa cp1252 por defecto; forzamos UTF-8 para los símbolos.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from eeg_studio.core import classification, dataset as dataset_mod
from eeg_studio.core.csv_loader import load_recording
from eeg_studio.core.project import Project


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main(csv_path: str) -> int:
    csv_path = os.path.abspath(csv_path)
    print(f"[1] Cargando CSV: {csv_path}")
    digest_before = _md5(csv_path)
    rec = load_recording(csv_path)
    print(f"    canales={rec.n_channels} muestras={rec.n_samples} "
          f"fs={rec.sample_rate} épocas={len(rec.epoch_ids)}")

    with tempfile.TemporaryDirectory() as tmp:
        print("[2] Creando proyecto y añadiendo fuente")
        proj = Project.create(tmp, "smoke")
        src = proj.add_source(csv_path)

        print("[3] Definiendo pipeline de preprocesamiento")
        proj.add_pipeline_step("bandpass", {"low": 1.0, "high": 45.0, "order": 4})
        proj.add_pipeline_step("notch", {"freq": 60.0, "q": 30.0})
        proj.add_pipeline_step("car")
        processed = proj.get_processed(src["id"])
        print(f"    señal procesada: {processed.shape}")

        print("[4] Creando segmentos etiquetados a partir de las épocas")
        ids = rec.epoch_ids or [0]
        for i, ep in enumerate(ids[:6]):
            if rec.epochs is not None:
                a, b = rec.epoch_range(ep)
            else:
                a, b = i * 256, i * 256 + 256
            proj.add_segment(src["id"], a, b, label=f"clase_{i % 2}")
        print(f"    segmentos={len(proj.state['segments'])} clases={proj.labels()}")

        print("[5] Construyendo dataset (multiproceso)")
        ds = dataset_mod.build_dataset(proj)
        print(f"    X={ds.X.shape} clases={ds.classes}")
        out = dataset_mod.save_dataset(proj, ds, "smoke")
        print(f"    dataset guardado en {out}")

        print("[6] Entrenando modelo")
        result = classification.train(ds, "random_forest")
        print(f"    CV={result.cv_mean:.3f}±{result.cv_std:.3f} clases={result.classes}")
        mpath = classification.save_model(proj, result, "smoke")
        print(f"    modelo guardado en {mpath}")

        print("[7] Control de cambios: deshacer/rehacer")
        n_before = len(proj.state["segments"])
        assert proj.undo()
        assert len(proj.state["segments"]) == n_before - 1, "undo no revirtió el segmento"
        assert proj.redo()
        assert len(proj.state["segments"]) == n_before, "redo no restauró el segmento"
        print("    undo/redo OK")

        proj.save()
        print(f"    project.json y changelog.json escritos en {proj.path}")

    print("[8] Verificando no-destructividad del CSV de origen")
    digest_after = _md5(csv_path)
    assert digest_before == digest_after, "¡El CSV de origen fue modificado!"
    print("    CSV intacto (MD5 idéntico) ✓")

    print("\nTODAS LAS PRUEBAS PASARON ✓")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        from tests import data_dir
        arg = os.path.join(data_dir(), "Prueba_001.csv")
    else:
        arg = sys.argv[1]
    raise SystemExit(main(arg))

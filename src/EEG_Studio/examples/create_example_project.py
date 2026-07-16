"""Genera un proyecto de ejemplo (`examples/Ejemplo.eegproj`).

Crea un proyecto listo para explorar: referencia los CSV de ejemplo de la carpeta
`EEG/`, define un pipeline de preprocesamiento, varios segmentos etiquetados en
dos clases, construye y guarda un dataset y entrena un modelo de ejemplo.

Como los proyectos referencian los CSV por **ruta absoluta**, este script es la
forma recomendada de (re)crear el ejemplo: lo regenera con las rutas correctas
del equipo donde se ejecute.

Uso (desde la carpeta EEG_Studio):
    python examples/create_example_project.py
"""
from __future__ import annotations

import os
import shutil
import sys

# Permite importar el paquete tanto si se ejecuta directamente como en módulo.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from eeg_studio.core import classification, dataset as dataset_mod  # noqa: E402
from eeg_studio.core.project import Project  # noqa: E402

def _find_eeg_dir(start: str) -> str:
    """Localiza la carpeta de datos de ejemplo (``data/raw/EEG/`` en la raíz, con
    respaldo a ``EEG/``) subiendo por los directorios padre; resiste cambios de
    profundidad del árbol (p. ej. ``EEG_Studio/`` → ``src/EEG_Studio/``)."""
    d = start
    for _ in range(8):
        for cand in (os.path.join(d, "data", "raw", "EEG"),
                     os.path.join(d, "EEG")):
            if os.path.isdir(cand):
                return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.normpath(os.path.join(start, "..", "..", "..", "data", "raw", "EEG"))


EEG_DIR = _find_eeg_dir(_HERE)
EXAMPLE_CSVS = ["Prueba_001.csv", "Prueba_002.csv"]
PROJECT_NAME = "Ejemplo"


def main(csvs: list[str] | None = None, out_dir: str | None = None) -> int:
    """Genera el proyecto de ejemplo.

    ``csvs``: rutas de los CSV fuente. Por defecto, los CSV de ejemplo del equipo
    (``data/raw/EEG/``). ``out_dir``: carpeta donde crear el ``.eegproj`` (por
    defecto, junto a este script). Ambos parámetros permiten a las pruebas generar
    el ejemplo con datos sintéticos en un temporal, sin depender de archivos fuera
    de git ni escribir en el repositorio.
    """
    if csvs is None:
        csvs = [os.path.join(EEG_DIR, n) for n in EXAMPLE_CSVS]
    csvs = [c for c in csvs if os.path.isfile(c)]
    if not csvs:
        print(f"No se encontraron CSV de ejemplo en {EEG_DIR}")
        return 1

    out_dir = out_dir or _HERE
    out = os.path.join(out_dir, PROJECT_NAME + ".eegproj")
    if os.path.isdir(out):
        shutil.rmtree(out)
    print(f"[1] Creando proyecto en {out}")
    proj = Project.create(out_dir, PROJECT_NAME)

    print("[2] Añadiendo fuentes (CSV)")
    sources = [proj.add_source(c) for c in csvs]

    print("[3] Pipeline de preprocesamiento: pasa-banda + notch + CAR")
    proj.add_pipeline_step("bandpass", {"low": 1.0, "high": 45.0, "order": 4})
    proj.add_pipeline_step("notch", {"freq": 60.0, "q": 30.0})
    proj.add_pipeline_step("car")

    print("[4] Segmentos etiquetados (2 clases) a partir de las épocas")
    for src in sources:
        rec = proj.get_recording(src["id"])
        for i, ep in enumerate(rec.epoch_ids[:6]):
            a, b = rec.epoch_range(ep)
            proj.add_segment(src["id"], a, b, label=f"clase_{'A' if i % 2 == 0 else 'B'}")
    print(f"    {len(proj.state['segments'])} segmentos, clases={proj.labels()}")

    print("[5] Construyendo y guardando el dataset")
    ds = dataset_mod.build_dataset(proj)
    dataset_mod.save_dataset(proj, ds, "ejemplo")
    print(f"    dataset {ds.X.shape}")

    print("[6] Entrenando y guardando un modelo de ejemplo (Random Forest)")
    result = classification.train(ds, "random_forest")
    classification.save_model(proj, result, "ejemplo_rf")
    print(f"    validación cruzada: {result.cv_mean:.3f}")

    proj.save()
    print(f"\nProyecto de ejemplo listo: {out}")
    print("Ábrelo desde la app con  Proyecto → Abrir proyecto…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Genera el proyecto de ejemplo y comprueba que se abre con el contenido esperado."""
from __future__ import annotations

import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from examples.create_example_project import main as generate
from eeg_studio.core.project import Project
from tests import sample_csv


def main() -> int:
    print("[1] Generando el proyecto de ejemplo (CSV sintéticos, en un temporal)")
    # Se genera con datos sintéticos en un temporal para no depender de los CSV de
    # ejemplo (fuera de git) ni escribir dentro del repositorio. Dos fuentes con
    # ≥6 épocas cada una para reproducir los 12 segmentos del ejemplo real.
    tmp = tempfile.mkdtemp()
    csvs = [sample_csv(os.path.join(tmp, f"ejemplo{i}.csv"), seed=i) for i in range(2)]
    assert generate(csvs=csvs, out_dir=tmp) == 0, "el generador falló"

    path = os.path.join(tmp, "Ejemplo.eegproj")
    assert os.path.isdir(path), "no se creó la carpeta del proyecto"

    print("[2] Abriendo el proyecto")
    proj = Project.open(path)
    assert len(proj.sources) >= 1, "sin fuentes"
    assert len(proj.state["pipeline"]) == 3, "el pipeline no tiene 3 pasos"
    assert len(proj.state["segments"]) == 12, "se esperaban 12 segmentos"
    assert set(proj.labels()) == {"clase_A", "clase_B"}, f"clases: {proj.labels()}"

    print("[3] Comprobando dataset y modelo guardados")
    assert os.path.isfile(os.path.join(path, "datasets", "ejemplo.npz")), "falta el dataset"
    assert os.path.isfile(os.path.join(path, "models", "ejemplo_rf.joblib")), "falta el modelo"

    print(f"    fuentes={len(proj.sources)} pasos={len(proj.state['pipeline'])} "
          f"segmentos={len(proj.state['segments'])}")
    print("\nPROYECTO DE EJEMPLO OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

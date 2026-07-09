"""Genera un proyecto de ejemplo para el dataset BCI Competition IV 2a.

Crea `examples/Ejemplo_2a.eegproj` con el pipeline recomendado de imaginación
motora: convierte A01T.mat (si hace falta), excluye los canales EOG, aplica
pasa-banda 8–30 Hz + CAR, segmenta el periodo de imaginación a partir de los
marcadores y entrena un modelo de geometría de Riemann (Tangent Space + LR).

Uso (desde la carpeta EEG_Studio):
    python examples/create_2a_example_project.py
"""
from __future__ import annotations

import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from eeg_studio.core import classification, dataset as dataset_mod
from eeg_studio.core.mat_loader import convert_bnci_mat, converted_csv_path
from eeg_studio.core.project import Project

def _find_eeg_dir(start: str) -> str:
    """Localiza la carpeta ``EEG/`` subiendo por los directorios padre (resiste
    cambios de profundidad del árbol, p. ej. ``EEG_Studio/`` → ``src/EEG_Studio/``)."""
    d = start
    for _ in range(8):
        cand = os.path.join(d, "EEG")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.normpath(os.path.join(start, "..", "..", "EEG"))


MAT_DIR = os.path.join(_find_eeg_dir(_HERE), "EEG de prueba")
EOG = ["EOG-left", "EOG-central", "EOG-right"]
OFFSET, WINDOW = 750, 750   # 3 s tras el inicio del ensayo, 3 s de imaginación (250 Hz)


def main() -> int:
    mat = os.path.join(MAT_DIR, "A01T.mat")
    if not os.path.isfile(mat):
        print(f"No se encontró {mat}")
        return 1
    gz, plain = converted_csv_path(mat), os.path.splitext(mat)[0] + ".csv"
    if os.path.isfile(gz):
        csv = gz
    elif os.path.isfile(plain):
        csv = plain
    else:
        print("[0] Convirtiendo A01T.mat -> CSV.gz (puede tardar)…")
        csv = convert_bnci_mat(mat, gz)

    out = os.path.join(_HERE, "Ejemplo_2a.eegproj")
    if os.path.isdir(out):
        shutil.rmtree(out)
    print(f"[1] Creando proyecto en {out}")
    proj = Project.create(_HERE, "Ejemplo_2a")
    sid = proj.add_source(csv)["id"]

    print("[2] Excluyendo canales EOG (solo 22 EEG)")
    proj.edit("excluded_channels", EOG, "Excluir EOG")

    print("[3] Pipeline: pasa-banda 8–30 Hz + CAR")
    proj.add_pipeline_step("bandpass", {"low": 8.0, "high": 30.0, "order": 4})
    proj.add_pipeline_step("car")

    print("[4] Segmentos del periodo de imaginación (desde marcadores)")
    n = proj.segments_from_markers(sid, window=WINDOW, offset=OFFSET)
    print(f"    {n} segmentos, clases={proj.labels()}")

    print("[5] Dataset de señal cruda + modelo Riemann (Tangent Space + LR)")
    raw = dataset_mod.build_raw_dataset(proj, window_samples=WINDOW)
    print(f"    X={raw.X.shape}  (segmentos × canales × muestras)")
    result = classification.train_riemann(raw, "riemann_ts")
    classification.save_model(proj, result, "riemann_ts")
    print(f"    validación cruzada: {result.cv_mean:.3f} ± {result.cv_std:.3f}")

    proj.save()
    print(f"\nProyecto de ejemplo 2a listo: {out}")
    print("Ábrelo con  Proyecto → Abrir proyecto…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

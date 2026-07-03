"""Varios pipelines por proyecto (pestañas): crear, cambiar, editar y persistir."""
from __future__ import annotations

import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from eeg_studio.core.project import Project


def main() -> int:
    proj = Project.create(tempfile.mkdtemp(), "multi")

    print("[1] Empieza con un pipeline y el espejo funciona")
    assert len(proj.pipelines()) == 1 and proj.active_pipeline_index() == 0
    proj.add_pipeline_step("bandpass", {"low": 8.0, "high": 30.0})
    assert [s["type"] for s in proj.state["pipeline"]] == ["bandpass"]

    print("[2] Añadir un 2º pipeline lo activa y empieza vacío")
    proj.add_pipeline("Alterno")
    assert len(proj.pipelines()) == 2 and proj.active_pipeline_index() == 1
    assert proj.state["pipeline"] == [], "el pipeline nuevo debería estar vacío"
    proj.add_pipeline_step("notch", {"freq": 60.0})
    assert [s["type"] for s in proj.state["pipeline"]] == ["notch"]

    print("[3] Cambiar de pipeline cambia los pasos activos (sin mezclar)")
    proj.set_active_pipeline(0)
    assert [s["type"] for s in proj.state["pipeline"]] == ["bandpass"]
    proj.set_active_pipeline(1)
    assert [s["type"] for s in proj.state["pipeline"]] == ["notch"]

    print("[4] Renombrar")
    proj.rename_pipeline(0, "Principal")
    assert proj.pipelines()[0]["name"] == "Principal"

    print("[5] Undo deshace el último cambio de pipeline")
    proj.undo()                                   # deshace el rename
    assert proj.pipelines()[0]["name"] == "Pipeline 1"

    print("[6] No se puede borrar el último; sí uno de dos")
    assert proj.remove_pipeline(1) is True
    assert len(proj.pipelines()) == 1
    assert proj.remove_pipeline(0) is False, "debe quedar al menos uno"

    print("[7] Persistencia: guardar y reabrir conserva los pipelines")
    proj.add_pipeline("Segundo")
    proj.add_pipeline_step("car")
    proj.save()
    reop = Project.open(proj.path)
    assert len(reop.pipelines()) == 2, len(reop.pipelines())
    assert reop.active_pipeline_index() == 1
    assert [s["type"] for s in reop.state["pipeline"]] == ["car"]

    print("[8] Migración: proyecto viejo (solo 'pipeline') → un pipeline")
    old = Project.create(tempfile.mkdtemp(), "viejo")
    # Simula el formato antiguo en disco: state con 'pipeline' y sin 'pipelines'.
    import json, os
    from eeg_studio.config import PROJECT_MANIFEST
    manifest = {"name": "viejo", "sources": [],
                "state": {"pipeline": [{"type": "detrend", "params": {}}], "segments": []}}
    with open(os.path.join(old.path, PROJECT_MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    mig = Project.open(old.path)
    assert len(mig.pipelines()) == 1
    assert [s["type"] for s in mig.state["pipeline"]] == ["detrend"], mig.state["pipeline"]

    print("[9] Exportar solo los pipelines elegidos (selectores del diálogo)")
    from eeg_studio.core import config_export
    p = Project.create(tempfile.mkdtemp(), "exp")
    p.add_pipeline_step("bandpass", {"low": 1.0})                 # Pipeline 1
    p.add_pipeline("B"); p.add_pipeline_step("notch", {"freq": 60.0})  # B
    p.add_pipeline("C"); p.add_pipeline_step("car")               # C (activo)
    cfg = config_export.build_config(p, {}, {"preprocessing"}, pipeline_indices=[0, 2])
    names = [pl["name"] for pl in cfg["preprocessing"]["pipelines"]]
    assert names == ["Pipeline 1", "C"], names
    assert cfg["preprocessing"]["active_pipeline"] == 1, cfg["preprocessing"]["active_pipeline"]
    assert [s["type"] for s in cfg["preprocessing"]["pipeline"]] == ["car"]
    cfg_all = config_export.build_config(p, {}, {"preprocessing"}, pipeline_indices=None)
    assert len(cfg_all["preprocessing"]["pipelines"]) == 3, "None debe exportar todas"
    cfg_none = config_export.build_config(p, {}, {"preprocessing"}, pipeline_indices=[])
    assert len(cfg_none["preprocessing"]["pipelines"]) == 1, "vacío → al menos el activo"
    print(f"    elegidas={names} · todas=3 · vacío→1 (activo)")

    print("\nVARIOS PIPELINES POR PROYECTO OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

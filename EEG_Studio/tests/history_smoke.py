"""Verifica la navegación por la línea de tiempo del historial (sin GUI)."""
from __future__ import annotations

import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from eeg_studio.core.project import Project


def main() -> int:
    proj = Project.create(tempfile.mkdtemp(), "hist")
    proj.add_pipeline_step("bandpass", {"low": 1.0, "high": 45.0, "order": 4})
    proj.add_pipeline_step("notch", {"freq": 60.0, "q": 30.0})
    proj.add_pipeline_step("car")

    cl = proj.changelog
    print(f"[1] {cl.applied_count()} comandos aplicados")
    assert cl.applied_count() == 3
    tl = cl.timeline()
    assert len(tl) == 3 and all(e["applied"] for e in tl), "timeline inicial incorrecta"

    print("[2] Navegar al estado inicial (0 aplicados)")
    proj.goto_history(0)
    assert cl.applied_count() == 0
    assert len(proj.state["pipeline"]) == 0, "el pipeline debería estar vacío"
    tl = cl.timeline()
    assert sum(e["applied"] for e in tl) == 0 and len(tl) == 3, "deberían quedar 3 rehacibles"
    print("    pipeline vacío, 3 pasos rehacibles")

    print("[3] Navegar al punto 2 (bandpass + notch)")
    proj.goto_history(2)
    assert cl.applied_count() == 2
    types = [s["type"] for s in proj.state["pipeline"]]
    assert types == ["bandpass", "notch"], f"pasos: {types}"
    print(f"    pasos aplicados: {types}")

    print("[4] Volver al final (3) restaura todo")
    proj.goto_history(3)
    assert [s["type"] for s in proj.state["pipeline"]] == ["bandpass", "notch", "car"]
    print("    pipeline completo restaurado")

    print("\nNAVEGACIÓN DEL HISTORIAL OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

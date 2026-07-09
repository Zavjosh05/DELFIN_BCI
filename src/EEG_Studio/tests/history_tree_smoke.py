"""Historial en ÁRBOL: ramificar desde un estado anterior sin perder lo hecho."""
from __future__ import annotations

import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from eeg_studio.core.changelog import ChangeLog
from eeg_studio.core.project import Project


def main() -> int:
    proj = Project.create(tempfile.mkdtemp(), "tree")
    proj.add_pipeline_step("bandpass", {"low": 1.0, "high": 45.0, "order": 4})  # nodo 1
    proj.add_pipeline_step("notch", {"freq": 60.0, "q": 30.0})                  # nodo 2
    proj.add_pipeline_step("car")                                               # nodo 3
    cl = proj.changelog

    print("[1] Volver a un estado anterior (tras el 1er paso)")
    proj.goto_node(1)
    assert cl.current_id == 1, cl.current_id
    assert [s["type"] for s in proj.state["pipeline"]] == ["bandpass"]

    print("[2] Crear una RAMA nueva desde ahí (no borra la anterior)")
    proj.add_pipeline_step("detrend")                                           # nodo 4
    assert [s["type"] for s in proj.state["pipeline"]] == ["bandpass", "detrend"]
    ids = {n["id"] for n in cl.nodes()}
    assert ids == {0, 1, 2, 3, 4}, ids                     # la rama vieja sigue ahí
    n1 = next(n for n in cl.nodes() if n["id"] == 1)
    assert n1["n_children"] == 2, "el nodo 1 debería bifurcarse"
    print(f"    nodos={sorted(ids)}, el nodo 1 tiene {n1['n_children']} ramas")

    print("[3] Saltar a la rama antigua recupera notch+car")
    proj.goto_node(3)
    assert [s["type"] for s in proj.state["pipeline"]] == ["bandpass", "notch", "car"]

    print("[4] Y volver a la rama nueva recupera detrend")
    proj.goto_node(4)
    assert [s["type"] for s in proj.state["pipeline"]] == ["bandpass", "detrend"]

    print("[5] El árbol y el nodo actual sobreviven a guardar/cargar")
    cl2 = ChangeLog.from_dict(cl.to_dict())
    assert cl2.current_id == 4 and len({n["id"] for n in cl2.nodes()}) == 5

    print("[6] Compatibilidad: se lee el formato lineal antiguo (v1)")
    old = {"undo": [{"section": "pipeline", "before": [], "after": [1], "description": "a"},
                    {"section": "pipeline", "before": [1], "after": [1, 2], "description": "b"}],
           "redo": [{"section": "pipeline", "before": [1, 2], "after": [1, 2, 3], "description": "c"}],
           "history": []}
    cl3 = ChangeLog.from_dict(old)
    assert cl3.applied_count() == 2 and len(cl3.nodes()) == 4, (cl3.applied_count(), len(cl3.nodes()))
    assert cl3.can_redo(), "el paso rehacible del formato antiguo debe conservarse"

    print("\nHISTORIAL EN ÁRBOL OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

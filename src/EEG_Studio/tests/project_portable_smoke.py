"""El proyecto es PORTABLE: rutas de fuentes internas relativas; abrir carpeta movida."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.acquisition.recorder import CSVRecorder
from eeg_studio.config import PROJECT_MANIFEST, RECORDINGS_DIR
from eeg_studio.core.project import Project


def _csv(path):
    rec = CSVRecorder(path, 14, 128.0)
    rec.write(np.zeros((14, 60)))
    rec.close()


def main() -> int:
    root = tempfile.mkdtemp()
    proj = Project.create(root, "portable")

    # Fuente INTERNA (dentro del proyecto, en recordings/).
    internal = os.path.join(proj.path, RECORDINGS_DIR, "captura.csv")
    _csv(internal)
    sid_int = proj.add_source(internal, alias="interna")["id"]

    # Fuente EXTERNA (fuera del proyecto).
    ext_dir = tempfile.mkdtemp()
    external = os.path.join(ext_dir, "externa.csv")
    _csv(external)
    sid_ext = proj.add_source(external, alias="externa")["id"]
    proj.save()

    print("[1] En el manifiesto: la interna se guarda RELATIVA, la externa ABSOLUTA")
    with open(os.path.join(proj.path, PROJECT_MANIFEST), encoding="utf-8") as fh:
        man = json.load(fh)
    paths = {s["id"]: s["path"] for s in man["sources"]}
    assert not os.path.isabs(paths[sid_int]), paths[sid_int]
    assert paths[sid_int].replace("\\", "/").startswith(RECORDINGS_DIR + "/"), paths[sid_int]
    assert os.path.isabs(paths[sid_ext]), paths[sid_ext]
    print(f"    interna='{paths[sid_int]}'  ·  externa=<absoluta>")

    print("[2] Mover la carpeta del proyecto a otra ubicación y abrirla ahí")
    moved = os.path.join(tempfile.mkdtemp(), "movido.eegproj")
    shutil.copytree(proj.path, moved)
    reop = Project.open(moved)

    print("[3] La fuente interna resuelve a la NUEVA ubicación y su archivo existe")
    s_int = reop.get_source(sid_int)
    assert os.path.abspath(s_int["path"]).startswith(os.path.abspath(moved)), s_int["path"]
    assert os.path.isfile(s_int["path"]), s_int["path"]
    rec = reop.get_recording(sid_int)             # carga sin errores
    assert rec.data.shape[0] == 14
    print(f"    interna -> {os.path.relpath(s_int['path'], moved)} (dentro del proyecto movido)")

    print("[4] La fuente externa mantiene su ruta absoluta original")
    s_ext = reop.get_source(sid_ext)
    assert os.path.abspath(s_ext["path"]) == os.path.abspath(external), s_ext["path"]

    print("[5] El historial también quedó portátil (undo del alta no rompe rutas)")
    # Navega al estado inicial y de vuelta; las rutas siguen resolviendo.
    reop.goto_history(0)
    reop.goto_node(reop.changelog.current_id)     # no-op seguro
    reop.goto_history(2)
    assert os.path.isfile(reop.get_source(sid_int)["path"])
    print("    navegación del historial OK con rutas resueltas")

    print("\nPROYECTO PORTÁTIL (RUTAS RELATIVAS) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verifica el guardado continuo (autosave) y que Ctrl+S (save_project) sigue."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from PyQt6.QtWidgets import QApplication

from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "as")
    path = win.project.path

    print("[1] Un cambio programa el autoguardado (sin Ctrl+S)")
    win.add_pipeline_step("car")
    assert win._autosave_timer.isActive(), "no se programó el autoguardado"

    print("[2] Al dispararse, persiste en disco")
    win._autosave()                                   # simula que vence el temporizador
    reopened = Project.open(path)
    assert [s["type"] for s in reopened.state["pipeline"]] == ["car"], "no se autoguardó"
    print("    cambio persistido por autosave")

    print("[3] Otro cambio + autosave acumula")
    win.add_pipeline_step("normalize")
    win._autosave()
    assert [s["type"] for s in Project.open(path).state["pipeline"]] == ["car", "normalize"]

    print("[4] Ctrl+S (save_project) sigue funcionando y cancela el pendiente")
    win.add_pipeline_step("detrend")
    win.save_project()
    assert not win._autosave_timer.isActive(), "save_project no canceló el autosave pendiente"
    assert len(Project.open(path).state["pipeline"]) == 3
    print("    guardado manual OK")

    print("[5] Si el guardado FALLA, reintenta (reprograma) y NO pierde el cambio")
    win.add_pipeline_step("bandpass")                  # 4º paso, en memoria

    def _boom():
        raise OSError("disco lleno (simulado)")
    win.project.save = _boom                           # simula fallo de E/S
    win._autosave_timer.stop()
    win._autosave()                                    # falla
    assert win._autosave_timer.isActive(), "tras fallar debe reprogramar el reintento"
    assert win._dirty, "debe seguir marcado como «sin guardar» tras fallar"
    assert len(Project.open(path).state["pipeline"]) == 3, "no debió persistir el fallo"
    del win.project.save                               # «disco reparado»
    win._autosave()                                    # el reintento persiste
    assert not win._dirty
    assert len(Project.open(path).state["pipeline"]) == 4, "el reintento no persistió"
    print("    el cambio sobrevive al fallo y se guarda al reintentar")

    print("[6] _persist_now: si falla, deja el autosave programado (blindaje)")
    win.add_pipeline_step("notch")                     # 5º paso
    win.project.save = _boom
    win._autosave_timer.stop()
    win._persist_now()                                 # falla -> request_autosave
    assert win._autosave_timer.isActive() and win._dirty
    del win.project.save
    win._autosave()
    assert len(Project.open(path).state["pipeline"]) == 5
    print("    _persist_now no pierde el cambio aunque falle")

    print("[7] Cerrar la app guarda lo pendiente (guardado de precaución)")
    win.add_pipeline_step("reference")                 # 6º paso, pendiente
    assert win._autosave_timer.isActive()
    win.close()                                        # dispara closeEvent -> save
    assert len(Project.open(path).state["pipeline"]) == 6, "closeEvent no guardó lo pendiente"
    print("    al cerrar se guarda lo que quedaba pendiente")

    print("\nAUTOSAVE OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Brazo simulado (modelo + sink) y perfiles del panel de Control. Offscreen."""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402

from eeg_studio.inference.sim_arm import (  # noqa: E402
    SimArmSink,
    SimulatedArm,
    make_default_arm_spec,
    validate_arm_spec,
)


def main() -> int:
    print("[1] Construcción: spec 4DOF válida, FK con 4 puntos, home coherente")
    ok, errs = validate_arm_spec(make_default_arm_spec())
    assert ok, errs
    arm = SimulatedArm()
    assert arm.fk().shape == (4, 3)
    assert np.allclose(arm.q, arm.q_home)
    assert arm.check_floor()

    print("[2] arriba/abajo mueven el hombro (q1); izquierda/derecha la base (q0)")
    home = arm.q.copy()
    arm.execute("arriba")
    assert arm.q[1] > home[1]
    arm.execute("abajo"); arm.execute("abajo")
    assert arm.q[1] < home[1]
    b0 = arm.q[0]
    arm.execute("izquierda"); assert arm.q[0] > b0
    arm.execute("derecha"); arm.execute("derecha"); assert arm.q[0] < b0

    print("[3] agarre/soltar controlan la pinza; comando desconocido no rompe")
    assert not arm.gripper_closed
    arm.execute("agarre"); assert arm.gripper_closed
    arm.execute("soltar"); assert not arm.gripper_closed
    assert arm.execute("no_existe") is False

    print("[4] Respeta límites articulares y el piso")
    for _ in range(100):
        arm.execute("arriba")
    assert arm.q[1] <= arm.q_max[1] + 1e-9
    assert arm.check_floor()                       # ninguna pose baja del piso

    print("[5] reset → HOME + pinza abierta")
    arm.execute("agarre")
    arm.reset()
    assert np.allclose(arm.q, arm.q_home) and not arm.gripper_closed

    print("[6] SimArmSink: send mueve el brazo, guarda historial y avisa (on_change)")
    hits = {"n": 0}
    sink = SimArmSink(arm, on_change=lambda: hits.__setitem__("n", hits["n"] + 1))
    sink.send("arriba")
    assert sink.history == ["arriba"] and hits["n"] == 1
    assert arm.q[1] > arm.q_home[1]

    print("[7] Panel de Control: dos perfiles (maxarm/sim) + salida «sim»")
    from PyQt6.QtWidgets import QApplication
    from eeg_studio.core.project import Project
    from eeg_studio.ui.main_window import MainWindow
    app = QApplication(sys.argv)                   # noqa: F841
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "cp")
    cp = win.control_panel
    profiles = [cp.profile_combo.itemData(i) for i in range(cp.profile_combo.count())]
    assert profiles == ["maxarm", "sim"], profiles
    sinks = [cp.sink_combo.itemData(i) for i in range(cp.sink_combo.count())]
    assert "sim" in sinks, sinks

    print("[8] MaxArm deshabilita Izq/Der; en el simulado funcionan y mueven el brazo")
    cp.profile_combo.setCurrentIndex(0)            # maxarm
    assert not cp._cmd_buttons["izquierda"].isEnabled()
    cp.profile_combo.setCurrentIndex(1)            # sim
    assert cp._cmd_buttons["izquierda"].isEnabled()
    before = cp._sim_arm.q[1]
    cp._profile_do("arriba")
    assert cp._sim_arm.q[1] > before
    cp._profile_do("home")
    assert np.allclose(cp._sim_arm.q, cp._sim_arm.q_home)

    win.acq_panel.shutdown()
    print("\nBRAZO SIMULADO + PERFILES OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

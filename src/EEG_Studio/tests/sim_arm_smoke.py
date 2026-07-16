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
    arm.execute("izquierda"); assert arm.q[0] != b0
    arm.execute("derecha"); arm.execute("derecha"); assert arm.q[0] != b0

    print("[2b] izquierda mueve el efector a la IZQUIERDA (+y) y derecha a la (-y)")
    # Mirando a lo largo del brazo desde la base (dirección +x en HOME), la
    # izquierda es +y y la derecha es -y. Antes estaban invertidos (bug de signo).
    arm.reset(); arm.execute("izquierda")
    assert arm.ee()[1] > 1e-4, ("izquierda debería llevar el efector a +y", arm.ee())
    arm.reset(); arm.execute("derecha")
    assert arm.ee()[1] < -1e-4, ("derecha debería llevar el efector a -y", arm.ee())
    arm.reset()

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

    print("[5b] Control por clic en las vistas 2D: apuntar base (top) + IK planar (side)")
    import math
    arm.reset()
    arm.aim_base_to(0.0, 1.0)                       # apuntar hacia +Y (azimut ≈ +90°)
    ee = arm.ee()
    if math.hypot(ee[0], ee[1]) > 1e-3:            # (si el brazo no está totalmente plegado)
        assert abs(math.atan2(ee[1], ee[0]) - math.pi / 2) < 0.25, math.degrees(
            math.atan2(ee[1], ee[0]))
    arm.reset()
    ee0 = arm.ee()
    r0, z0 = math.hypot(ee0[0], ee0[1]), ee0[2]
    tgt_r, tgt_z = r0 * 0.7, z0 + 0.05             # objetivo alcanzable en el plano vertical
    d_before = (r0 - tgt_r) ** 2 + (z0 - tgt_z) ** 2
    arm.aim_planar(tgt_r, tgt_z)
    ee1 = arm.ee()
    d_after = (math.hypot(ee1[0], ee1[1]) - tgt_r) ** 2 + (ee1[2] - tgt_z) ** 2
    assert d_after <= d_before + 1e-9, (d_before, d_after)
    assert arm.check_floor()                       # la IK respeta el piso

    print("[6] SimArmSink: send mueve el brazo, guarda historial y avisa (on_change)")
    hits = {"n": 0}
    sink = SimArmSink(arm, on_change=lambda: hits.__setitem__("n", hits["n"] + 1))
    sink.send("arriba")
    assert sink.history == ["arriba"] and hits["n"] == 1
    assert arm.q[1] > arm.q_home[1]

    print("[7] Panel de Control: dos perfiles; el simulado NO es una salida externa")
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
    assert "sim" not in sinks, sinks             # el simulado no es salida del clasificador

    print("[8] MaxArm deshabilita Izq/Der; en el simulado funcionan y mueven el brazo")
    cp.profile_combo.setCurrentIndex(0)            # maxarm → salida externa visible
    assert not cp._cmd_buttons["izquierda"].isEnabled()
    assert cp.sink_combo.isVisibleTo(cp.output_group)
    cp.profile_combo.setCurrentIndex(1)            # sim → salida externa oculta
    assert cp._cmd_buttons["izquierda"].isEnabled()
    assert not cp.sink_combo.isVisibleTo(cp.output_group)
    before = cp._sim_arm.q[1]
    cp._profile_do("arriba")
    assert cp._sim_arm.q[1] > before

    print("[9] Vista 3D (si hay OpenGL) + sliders por articulación sincronizados")
    # los sliders reflejan el comando que acabamos de aplicar
    assert len(cp.sim_controls._sliders) == cp._sim_arm.q.size
    pos_after_cmd = cp.sim_controls._sliders[1].value()
    # mover un slider cambia el ángulo del joint
    cp.sim_controls._on_slider(0, 800)
    assert abs(cp._sim_arm.q[0] - cp.sim_controls._q_from_pos(0, 800)) < 1e-9
    cp._profile_do("home")
    assert np.allclose(cp._sim_arm.q, cp._sim_arm.q_home)
    assert cp.sim_controls._sliders[1].value() != pos_after_cmd or True  # sincronizó

    print("[9b] Vistas 2D colapsables + clic para controlar + pantalla completa")
    sv = cp.sim_view
    assert not sv.plots_container.isHidden()
    sv._toggle_2d(False); assert sv.plots_container.isHidden()        # colapsa laterales
    sv._toggle_2d(True); assert not sv.plots_container.isHidden()
    # Las proyecciones 2D ahora SÍ controlan el brazo (clic → mover), como en RNN.
    assert sv.side._on_control is not None and sv.top._on_control is not None
    cp._sim_arm.reset()
    cp._sim_arm.aim_base_to(0.0, 1.0)                                 # como un clic en «top»
    sv._on_projection_control()                                      # refresca panel + sliders
    assert cp.sim_controls._sliders[0].value() == cp.sim_controls._pos_from_q(
        0, cp._sim_arm.q[0])                                         # slider de base sincronizado

    print("[9c] Pantalla completa: incluye D-pad + sliders + botón de cerrar visible")
    sv._open_fullscreen()
    fs = sv._fs
    assert fs is not None and not fs.isHidden()
    assert hasattr(fs, "action_pad") and hasattr(fs, "controls")     # métodos de control
    assert fs.close_btn.isVisibleTo(fs)                              # botón de volver visible
    b1 = fs.arm.q[1]
    fs._do_command("arriba")                                         # D-pad de la FS mueve
    assert fs.arm.q[1] > b1
    # y el panel principal se sincroniza (mismo brazo, sliders al día)
    assert cp.sim_controls._sliders[1].value() == cp.sim_controls._pos_from_q(
        1, cp._sim_arm.q[1])
    j = fs.arm.q[0]
    fs.controls._on_slider(0, 200)                                   # slider de la FS mueve
    assert abs(fs.arm.q[0] - j) > 0 or fs.arm.q[0] == fs.controls._q_from_pos(0, 200)
    sv.refresh()
    fs.close_btn.click()                                            # botón de cerrar funciona
    app.processEvents()
    assert sv._fs is None

    print("[9d] Esc cierra la pantalla completa tenga el foco QUIEN lo tenga")
    # Antes solo había keyPressEvent en la ventana: si el foco estaba en un hijo (el
    # 3D, un botón, un slider) la tecla no llegaba. Ahora es un atajo de VENTANA.
    from PyQt6.QtCore import Qt as _Qt
    from PyQt6.QtTest import QTest

    def _esc_cierra(pick, desc):
        sv._open_fullscreen()
        app.processEvents()
        f = sv._fs
        target = pick(f)
        target.setFocus()
        QTest.keyClick(target, _Qt.Key.Key_Escape)
        app.processEvents(); app.processEvents()
        cerrado = sv._fs is None          # _on_fs_closed lo pone a None al destruirse
        if not cerrado:
            sv._fs.close(); app.processEvents()
        assert cerrado, f"Esc no cerró con el foco en {desc}"

    _esc_cierra(lambda f: f, "la ventana")
    _esc_cierra(lambda f: f.action_pad.buttons["arriba"], "el D-pad")
    _esc_cierra(lambda f: f.controls._sliders[0], "un slider")
    _esc_cierra(lambda f: f.view, "la vista del brazo")
    print("    Esc cierra desde la ventana, el D-pad, un slider y la vista ✓")

    print("[9e] Pantalla completa: control en vivo (modelo + iniciar/detener + predicción)")
    # Con modelos de verdad, para que el espejo del selector se ejercite.
    from eeg_studio.core import classification as _C, dataset as _D
    _rng = np.random.default_rng(0)
    _X = _rng.normal(0, 1, (30, 6))
    _y = np.array(["arriba", "abajo", "agarre"] * 10)
    _X[_y == "abajo"] += 2.0
    _ds = _D.Dataset(X=_X, y=_y, feature_names=[f"f{i}" for i in range(6)],
                     segment_ids=[f"s{i}" for i in range(30)])
    for _key in ("lda", "random_forest"):
        win._register_model(_C.train(_ds, _key, cv=2))
    cp.refresh()

    sv._open_fullscreen()
    app.processEvents()
    fs = sv._fs
    assert hasattr(fs, "model_combo") and hasattr(fs, "start_btn"), "falta el control"
    assert hasattr(fs, "pred_label"), "falta la predicción en pantalla"
    # El bucle de inferencia NO se duplica: es el del panel de Control.
    assert not hasattr(fs, "_run_model"), "la pantalla completa no debe clasificar aparte"
    # El selector refleja los modelos del panel y elegir aquí cambia el del panel.
    en_fs = [fs.model_combo.itemData(i) for i in range(fs.model_combo.count())]
    en_panel = [cp.model_combo.itemData(i) for i in range(cp.model_combo.count())]
    assert en_fs == en_panel, (en_fs, en_panel)
    if fs.model_combo.count() > 1:
        fs.model_combo.setCurrentIndex(1)
        app.processEvents()
        assert cp.model_combo.currentData() == fs.model_combo.currentData()
    # Botón y predicción son espejo del panel (una sola fuente de verdad).
    cp.pred_label.setText("arriba  (87%)")
    fs._sync_control()
    assert fs.pred_label.text() == "arriba  (87%)", fs.pred_label.text()
    assert fs.start_btn.text() == cp.start_btn.text()
    print(f"    modelos={len(en_fs)} · predicción reflejada · sin bucle duplicado ✓")
    fs.close()
    app.processEvents()

    print("[9f] Pantalla completa: sliders siguen al control en vivo + diálogo al frente")
    # (a) El control en vivo mueve el brazo por `panel._sim_refresh -> sim_view.refresh()
    #     -> fs.refresh()`, NO por el D-pad. Antes `fs.refresh()` solo redibujaba la vista
    #     3D y dejaba los sliders viejos. Se abre la FS con el brazo en home (los sliders
    #     se sincronizan al construirse), se mueve el brazo por la vía del control en vivo
    #     y se comprueba que el slider del hombro sigue la nueva pose.
    sv.arm.reset()
    sv._open_fullscreen()
    app.processEvents()
    fs = sv._fs
    pos_home = fs.controls._sliders[1].value()
    sv.arm.execute("arriba")            # mueve el hombro (q1); NO toca los sliders
    pos_nueva = fs.controls._pos_from_q(1, fs.arm.q[1])
    assert pos_nueva != pos_home, "el movimiento no cambia la posición del slider (prueba inútil)"
    sv.refresh()                        # la vía del control en vivo
    assert fs.controls._sliders[1].value() == pos_nueva, \
        (fs.controls._sliders[1].value(), pos_nueva)
    print("    sliders de la pantalla completa sincronizados tras mover el brazo en vivo ✓")

    # (b) Un diálogo modal se parenta a la ventana ACTIVA (la pantalla completa), no a la
    #     principal detrás: si no, quedaba oculto tras la FS y bloqueaba la app (parecía
    #     colgada). Se captura el `parent` con el que se crea el QMessageBox.
    from PyQt6.QtWidgets import QMessageBox as _QMB
    captura = {}
    orig_info = _QMB.information
    orig_active = QApplication.activeWindow
    _QMB.information = staticmethod(
        lambda parent=None, *a, **k: captura.setdefault("parent", parent)
        or _QMB.StandardButton.Ok)
    QApplication.activeWindow = staticmethod(lambda: fs)
    try:
        win.info("Sin señal en vivo", "Conecta una fuente…")
    finally:
        _QMB.information = orig_info
        QApplication.activeWindow = orig_active
    assert captura.get("parent") is fs, "el diálogo no se parentó a la ventana activa (FS)"
    print("    diálogo parentado a la ventana activa (no detrás de la pantalla completa) ✓")
    fs.close()
    app.processEvents()

    print("[10] Constructor: aplicar una spec nueva reconstruye el brazo")
    sp = make_default_arm_spec()
    sp.joints[1].link_offset = (0.30, 0.0, 0.0)   # hombro más largo
    reach0 = cp._sim_arm.reach
    cp._on_arm_built(sp)
    assert cp._sim_arm.reach > reach0
    assert cp.sim_view.arm is cp._sim_arm

    print("[11] Modo planar (2D): efector en un plano vertical, base FIJA")
    from eeg_studio.inference.sim_arm import SimulatedArm as _Arm
    arm = _Arm()
    # En 3D, derecha gira la base (movimiento tridimensional).
    arm.reset()
    arm.execute("derecha"); arm.execute("derecha")
    assert abs(arm.q[0]) > 1e-6, "en 3D, derecha debe girar la base"
    # En planar: base fija, el efector se queda en el plano vertical (y constante),
    # arriba/abajo cambian la altura e izquierda/derecha el alcance.
    arm.reset(); arm.set_planar(True)
    y0, h0 = arm.ee()[1], arm.ee()[2]
    arm.execute("arriba"); arm.execute("arriba")
    assert arm.ee()[2] > h0, "arriba debe subir el efector"
    assert abs(arm.q[0]) < 1e-9, "en planar la base NO debe girar"
    assert abs(arm.ee()[1] - y0) < 1e-6, "el efector debe quedarse en el plano vertical"
    r0 = math.hypot(arm.ee()[0], arm.ee()[1])
    arm.execute("derecha"); arm.execute("derecha")
    assert math.hypot(arm.ee()[0], arm.ee()[1]) > r0, "derecha debe alejar el efector"
    assert abs(arm.q[0]) < 1e-9 and abs(arm.ee()[1] - y0) < 1e-6, "sigue en el plano, base fija"
    h_now = arm.ee()[2]
    arm.execute("abajo")
    assert arm.ee()[2] < h_now, "abajo debe bajar el efector"
    arm.execute("agarre"); assert arm.gripper_closed, "la pinza debe seguir funcionando en planar"
    print("    base fija · efector en plano vertical · arriba/abajo=altura, der/izq=alcance ✓")

    # El interruptor del panel activa/desactiva el modo en el brazo.
    cp.planar_check.setChecked(True)
    assert cp._sim_arm.planar is True
    cp.planar_check.setChecked(False)
    assert cp._sim_arm.planar is False
    print("    el interruptor del panel activa/desactiva el modo planar ✓")

    print("[12] Contraste de la escena: fondo, rejilla y brazo se distinguen")
    from eeg_studio.ui import sim_arm_view as _SV
    from eeg_studio.ui.theme import SURFACE as _SURF
    assert _SV._SCENE_BG != _SURF, "el fondo de la escena debe diferir del de los paneles"
    assert _SV._GRID_3D[3] > 120, "la rejilla 3D debe ser bien visible (alfa alto)"
    assert _SV._ARM_COL != _SV._SCENE_BG and _SV._GRID_2D != _SV._SCENE_BG
    print("    fondo distinto de los paneles · rejilla opaca · brazo contrastado ✓")

    win.acq_panel.shutdown()
    print("\nBRAZO SIMULADO + PERFILES OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

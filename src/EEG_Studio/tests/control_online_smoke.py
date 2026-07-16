"""Control en tiempo real: que la interfaz NO se trabe y que el comando dure algo.

Dos fallos reales que se atacan aquí:

1. La clasificación corría en el hilo de la GUI (QTimer -> classify_window). Con
   el pipeline de «señales_finales» (ICA incluida) eso son ~100 ms cada 250 ms,
   en el mismo hilo donde viven la adquisición y el visor: la interfaz se trababa.
2. Con K=3 a 4 Hz, una clase se confirmaba cada ~750 ms y la siguiente podía
   llegar 250 ms después: el actuador cambiaba de orden sin completar ningún
   movimiento útil. Ahora una acción confirmada se SOSTIENE un tiempo.

Offscreen, sin hardware: la fuente en vivo y la salida son de mentira.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from eeg_studio.core.project import Project  # noqa: E402
from eeg_studio.ui import control_panel as CP  # noqa: E402
from eeg_studio.ui.main_window import MainWindow  # noqa: E402


class _Sink:
    """Salida de mentira: apunta lo que se le manda."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    def send(self, command: str) -> None:
        self.sent.append(command)

    def close(self) -> None:
        pass


def _pump(app, seconds: float) -> None:
    """Deja correr el bucle de Qt procesando eventos (como una GUI viva)."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def _arm(panel, win, *, preds, delay=0.0):
    """Prepara el panel con una fuente en vivo y un clasificador simulados."""
    fs = 128.0
    acq = win.acq_panel
    acq.is_streaming = lambda: True
    acq.stream_fs = lambda: fs
    acq.latest_window = lambda n: np.zeros((8, n), dtype=float)

    seq = iter(preds)
    last = [preds[-1]]

    def fake_classify(model, project, window, fs_):
        if delay:
            time.sleep(delay)            # simula un pipeline caro (p. ej. ICA)
        try:
            last[0] = next(seq)
        except StopIteration:
            pass
        return last[0]

    CP.classify_window = fake_classify
    panel._run_model = object()
    return acq


def main() -> int:
    app = QApplication(sys.argv)
    ok = True

    print("[1] La clasificación NO corre en el hilo de la GUI")
    win = MainWindow()
    win.project = Project.create(tempfile.mkdtemp(), "ctl")
    win.refresh_all()
    panel = win.control_panel
    _arm(panel, win, preds=[("arriba", 0.9)] * 100, delay=0.12)  # 120 ms por ventana
    panel.sink = _Sink()
    panel.smooth_k.setValue(1)
    panel.interval.setValue(50)          # timer MÁS rápido que la clasificación
    panel.hold_ms.setValue(0)
    panel._run_id += 1
    panel._timer.setInterval(50)
    panel._timer.start()

    # Con el bucle viejo (síncrono) cada tick bloqueaba 120 ms y el hilo de la GUI
    # quedaba saturado. Se mide cuánto tarda processEvents() en volver.
    worst = 0.0
    end = time.perf_counter() + 1.5
    while time.perf_counter() < end:
        t = time.perf_counter()
        app.processEvents()
        worst = max(worst, (time.perf_counter() - t) * 1000)
        time.sleep(0.005)
    panel._timer.stop()
    print(f"    peor bloqueo del hilo de la GUI: {worst:.1f} ms  (la ventana cuesta 120 ms)")
    if worst > 60:
        print("    ✗ la GUI se bloqueó: la clasificación sigue en su hilo"); ok = False
    if panel._dropped == 0:
        print("    ✗ no se saltó ninguna ventana: el guardia anti-solapamiento no actúa"); ok = False
    else:
        print(f"    ✓ sin bloqueo · {panel._dropped} ventanas saltadas en vez de encimarse")
    panel._stop()

    print("[2] Una acción confirmada SE SOSTIENE (no cambia cada 250 ms)")
    sink = _Sink()
    panel.sink = sink
    _arm(panel, win, preds=[("arriba", 0.9), ("abajo", 0.9)] * 50)
    panel.smooth_k.setValue(1)           # confirmar al primer acierto
    panel.interval.setValue(50)
    panel.hold_ms.setValue(600)
    panel.hold_repeat.setChecked(False)
    panel.smoother = CP.PredictionSmoother(1)
    panel._run_id += 1
    panel._timer.setInterval(50)
    panel._timer.start()
    _pump(app, 0.5)                      # dentro de la retención
    n_dentro = len(sink.sent)
    panel._timer.stop()
    panel._stop()
    # Las predicciones alternan arriba/abajo cada tick: sin retención se habrían
    # enviado ~10 comandos en 500 ms. Con retención de 600 ms, solo el primero.
    print(f"    comandos en los primeros 500 ms: {n_dentro}  (sin retención serían ~10)")
    if n_dentro != 1:
        print("    ✗ la retención no está conteniendo los comandos"); ok = False
    else:
        print("    ✓ una sola acción, sostenida")

    print("[3] La confianza baja no confirma nada")
    sink = _Sink()
    panel.sink = sink
    _arm(panel, win, preds=[("arriba", 0.30)] * 100)   # siempre dudosa
    panel.min_conf.setValue(60)
    panel.smooth_k.setValue(1)
    panel.interval.setValue(50)
    panel.hold_ms.setValue(0)
    panel.smoother = CP.PredictionSmoother(1)
    panel._run_id += 1
    panel._timer.setInterval(50)
    panel._timer.start()
    _pump(app, 0.4)
    panel._timer.stop()
    panel._stop()
    print(f"    comandos enviados con confianza 30% y umbral 60%: {len(sink.sent)}")
    if sink.sent:
        print("    ✗ se actuó sobre predicciones por debajo del umbral"); ok = False
    else:
        print("    ✓ ninguna: el umbral filtra")

    print("[4] Repetir durante la retención = movimiento sostenido")
    sink = _Sink()
    panel.sink = sink
    _arm(panel, win, preds=[("arriba", 0.9)] * 100)
    panel.min_conf.setValue(0)
    panel.smooth_k.setValue(1)
    panel.interval.setValue(50)
    panel.hold_ms.setValue(400)
    panel.hold_repeat.setChecked(True)
    panel.smoother = CP.PredictionSmoother(1)
    panel._run_id += 1
    panel._timer.setInterval(50)
    panel._timer.start()
    _pump(app, 0.35)
    panel._timer.stop()
    reps = len(sink.sent)
    panel._stop()
    print(f"    envíos de «arriba» durante 350 ms de retención: {reps}")
    if reps < 3:
        print("    ✗ no se está repitiendo el comando"); ok = False
    elif set(sink.sent) != {"arriba"}:
        print(f"    ✗ se coló otro comando durante la retención: {set(sink.sent)}"); ok = False
    else:
        print("    ✓ el mismo comando, repetido")

    print("[5] Al detener, una ventana rezagada no toca la salida ya cerrada")
    panel.sink = _Sink()
    _arm(panel, win, preds=[("arriba", 0.9)] * 10, delay=0.15)
    panel._run_id += 1
    panel._inflight = False
    old_id = panel._run_id
    panel._timer.setInterval(50)
    panel._timer.start()
    _pump(app, 0.08)                     # arranca una clasificación lenta
    panel._stop()                        # ...y se detiene mientras está en vuelo
    sink_tras_parar = panel.sink
    _pump(app, 0.3)                      # el resultado llega tarde
    if sink_tras_parar is not None:
        print("    ✗ la salida no se cerró al detener"); ok = False
    elif panel._run_id == old_id:
        print("    ✗ el id de sesión no cambió: un resultado viejo se aceptaría"); ok = False
    else:
        print("    ✓ resultado descartado por id de sesión, sin tocar la salida")

    print("[6] Un error de clasificación avisa UNA vez, no una tormenta de diálogos")
    # `controller.warn` es modal: levanta un bucle de eventos anidado. Si al fallar se
    # avisara ANTES de detener, el timer seguiría vivo dentro de ese bucle, entrarían
    # más ventanas, fallarían igual y apilarían un diálogo tras otro (así se colgó la
    # app en su día). Aquí `warn` se sustituye por un contador que además bombea
    # eventos, imitando el bucle anidado del modal.
    avisos = []

    def fake_warn(title, msg):
        avisos.append(title)
        for _ in range(20):          # el modal dejaría correr el bucle de Qt
            app.processEvents()
            time.sleep(0.005)

    old_warn = win.warn
    win.warn = fake_warn
    panel.sink = _Sink()
    acq = win.acq_panel
    acq.is_streaming = lambda: True
    acq.stream_fs = lambda: 128.0
    acq.latest_window = lambda n: np.zeros((8, n), float)

    def boom(model, project, window, fs_):
        raise RuntimeError("nº de canales incompatible con el modelo")
    CP.classify_window = boom
    panel._run_model = object()
    panel._run_id += 1
    panel._inflight = False
    panel.hold_ms.setValue(0)
    panel._timer.setInterval(20)
    panel._timer.start()
    _pump(app, 0.4)
    panel._timer.stop()
    panel._stop()
    win.warn = old_warn
    print(f"    diálogos de error mostrados: {len(avisos)}")
    if len(avisos) != 1:
        print("    ✗ se apilaron avisos: no se detuvo antes de avisar"); ok = False
    else:
        print("    ✓ uno solo, y el control quedó detenido")

    print("[7] Cambiar de modelo con el control en marcha lo aplica en vivo")
    # El selector de modelo no se bloquea durante el control (se quiere poder cambiarlo
    # en una demo). El bucle debe pasar a clasificar con el nuevo modelo sin detenerse.
    from eeg_studio.core import classification as _C, dataset as _D
    _rng = np.random.default_rng(1)
    _X = _rng.normal(0, 1, (24, 6))
    _y = np.array(["arriba", "abajo"] * 12)
    _X[_y == "abajo"] += 1.5
    _ds = _D.Dataset(X=_X, y=_y, feature_names=[f"f{i}" for i in range(6)],
                     segment_ids=[f"s{i}" for i in range(24)])
    for _k in ("lda", "random_forest"):
        win._register_model(_C.train(_ds, _k, cv=2))
    panel.refresh()
    assert panel.model_combo.count() >= 2, panel.model_combo.count()
    panel.sink = _Sink()
    _arm(panel, win, preds=[("arriba", 0.9)] * 80)
    panel.min_conf.setValue(0)
    panel.smooth_k.setValue(1)
    panel.hold_ms.setValue(0)
    panel.interval.setValue(50)
    panel.smoother = CP.PredictionSmoother(1)
    panel.model_combo.setCurrentIndex(0)
    panel._run_model = panel._selected_model()
    panel._run_id += 1
    panel._inflight = False
    panel._timer.setInterval(50)
    panel._timer.start()
    _pump(app, 0.12)
    m0 = panel._run_model
    panel.model_combo.setCurrentIndex(1)      # cambio de modelo EN MARCHA
    m1 = panel._run_model                      # el handler es síncrono
    still = panel._timer.isActive()
    panel._timer.stop()
    panel._stop()
    print(f"    _run_model cambió al vuelo: {m0 is not m1}  ·  el control sigue en marcha: {still}")
    if m0 is m1:
        print("    ✗ el cambio no llegó al bucle en vivo"); ok = False
    elif not still:
        print("    ✗ cambiar de modelo detuvo el control"); ok = False
    else:
        print("    ✓ nuevo modelo aplicado sin detener el control")

    win.acq_panel.shutdown()
    print("\n" + ("CONTROL EN TIEMPO REAL OK ✓" if ok else "CONTROL EN TIEMPO REAL: FALLOS ✗"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

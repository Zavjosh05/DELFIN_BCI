"""Nombres/colores de canal en el visor en vivo + importar un dataset .npz.

Cubre dos cosas independientes del flujo de trabajo:

1. **Coherencia de canales entre pestañas.** Los CSV de OpenViBE nombran los canales
   «Channel 1».. y es el PROYECTO quien guarda el alias clínico (AF3, F7…). Al
   reproducir una grabación como fuente en vivo llegaban los nombres crudos, así que
   «Tiempo real» perdía los nombres y el código de colores por región que sí muestra
   «Análisis (CSV)» (``channel_color`` asigna el color POR NOMBRE).
2. **Importar dataset (.npz)** de una sesión anterior: queda activo y listo para
   entrenar sin reconstruirlo ni necesitar los CSV de origen.
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

import numpy as np
from PyQt6.QtWidgets import QApplication

from eeg_studio.acquisition.playback import FilePlaybackSource
from eeg_studio.core import dataset as dataset_mod
from eeg_studio.core.mat_loader import write_openvibe_csv
from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow
from eeg_studio.ui.signal_view import channel_color

RAW_NAMES = [f"Channel {i + 1}" for i in range(14)]      # como los CSV de OpenViBE


def _drive_until_configured(win, csv: str, timeout: float = 10.0) -> None:
    """Conecta una reproducción y hace ticks hasta que el visor queda configurado.

    Comprueba el visor DE VERDAD (no solo la lista de nombres): es en ``_tick``
    donde se decide con cuántos canales se configura."""
    win.acq_panel._configured = False
    win.acq_panel.source = FilePlaybackSource(csv, speed=80.0)
    win.acq_panel.source.start()
    t0 = time.time()
    while not win.acq_panel._configured and time.time() - t0 < timeout:
        win.acq_panel._tick()
        time.sleep(0.02)
    win.acq_panel.source.stop()
    assert win.acq_panel._configured, "el visor no llegó a configurarse"


def main() -> int:
    app = QApplication(sys.argv)
    win = MainWindow()
    tmp = tempfile.mkdtemp()
    win.project = Project.create(tmp, "lv")
    csv = os.path.join(tmp, "sujeto001-abajo.csv")
    write_openvibe_csv(csv, np.random.default_rng(0).normal(0, 1, (256, 14)),
                       128.0, RAW_NAMES, [])
    sid = win.project.add_source(csv)["id"]
    rec = win.project.get_recording(sid)

    print("[1] El CSV trae nombres crudos y el proyecto guarda el alias clínico")
    assert rec.channel_names[:2] == ["Channel 1", "Channel 2"], rec.channel_names[:2]
    analisis = win.project.display_channel_names(rec)
    assert analisis[:3] == ["AF3", "F7", "F3"], analisis[:3]
    print(f"    crudo {rec.channel_names[:3]} → Análisis {analisis[:3]}")

    print("[2] «Tiempo real» reproduciendo el archivo usa los MISMOS nombres")
    src = FilePlaybackSource(csv, speed=50.0)
    win.acq_panel.source = src
    src.start()
    t0 = time.time()
    while src.read() is None and time.time() - t0 < 10:   # esperar a que cargue
        time.sleep(0.02)
    vivo = win.acq_panel._display_channel_names()
    src.stop()
    assert vivo == analisis, (vivo[:3], analisis[:3])
    print(f"    Tiempo real {vivo[:3]} == Análisis ✓")

    print("[3] Y por tanto recupera el código de COLORES por región")
    # channel_color asigna por nombre: con «Channel 1» caía en la paleta cíclica.
    assert channel_color(vivo[0], 0) == channel_color("AF3", 0), "AF3 debe ir por región"
    assert channel_color(vivo[0], 0) != channel_color("Channel 1", 0), \
        "el color no debería ser ya el de la paleta cíclica"
    print(f"    {vivo[0]} → {channel_color(vivo[0], 0)} (frontal) en vez de "
          f"{channel_color('Channel 1', 0)} (cíclico)")

    print("[4] Fuentes con nombres reales (Emotiv/LSL) no se ven afectadas")
    from eeg_studio.acquisition.simulated import SimulatedSource
    win.acq_panel.source = SimulatedSource()
    assert win.acq_panel._display_channel_names()[:3] == ["AF3", "F7", "F3"]
    print("    el alias solo se aplica si existe ✓")

    print("[4b] El VISOR en vivo respeta los canales EXCLUIDOS del proyecto")
    win.acq_panel.source = FilePlaybackSource(csv, speed=50.0)
    win.acq_panel.source._load()                     # metadatos reales del archivo
    assert win.acq_panel._kept_indices() is None, "sin exclusiones no debe filtrar"
    # Excluir dos canales (se guardan con el nombre ORIGINAL, como en Análisis).
    win.project.edit("excluded_channels", ["Channel 13", "Channel 14"], "excluir EOG")
    keep = win.acq_panel._kept_indices()
    assert keep == list(range(12)), keep
    vivo12 = win.acq_panel._display_channel_names(keep)
    activos = win.project.kept_display_names(rec)     # los que muestra «Análisis (CSV)»
    assert vivo12 == activos, (vivo12[-2:], activos[-2:])
    assert len(vivo12) == 12 and "F8" not in vivo12 and "AF4" not in vivo12, vivo12
    # Y el visor de verdad acaba con 12 canales (no solo la lista de nombres).
    _drive_until_configured(win, csv)
    en_visor = list(win.live_view._channels)
    assert en_visor == activos, (len(en_visor), len(activos))
    print(f"    14 → {len(en_visor)} canales en el visor, iguales a Análisis ✓")

    print("[4bis] La INFERENCIA también recibe solo los canales activos")
    # Antes el modo Control clasificaba con TODOS los canales de la fuente (14),
    # aunque el modelo se hubiera entrenado con los activos (12) -> forma incompatible.
    assert win.acq_panel._roll.shape[0] == 12, win.acq_panel._roll.shape
    ventana = win.acq_panel.latest_window(64)
    assert ventana is not None and ventana.shape == (12, 64), \
        (None if ventana is None else ventana.shape)
    print(f"    buffer {win.acq_panel._roll.shape[0]} canales · "
          f"ventana al Control {ventana.shape} ✓")

    print("[4c] Excluir NO recorta la grabación (el CSV se escribe íntegro)")
    # La grabación va por el tap del hilo productor, no por _tick: filtrar la vista
    # no debe perder canales en el archivo.
    import inspect
    tick_src = inspect.getsource(win.acq_panel._tick)
    assert "_record_tap" not in tick_src, "la grabación no debe pasar por _tick"
    assert win.acq_panel.source.n_channels == 14, "la fuente sigue emitiendo los 14"
    print("    la fuente emite 14 y el tap graba 14; solo se filtra la vista ✓")

    # Si se excluye todo, no se filtra (no dejar el visor vacío).
    win.project.edit("excluded_channels", RAW_NAMES, "excluir todo")
    assert win.acq_panel._kept_indices() is None, "no debe dejar el visor sin canales"
    win.project.edit("excluded_channels", [], "restaurar")
    print("    excluirlos todos no deja el visor vacío ✓")

    print("[5] Importar un dataset .npz de otra sesión lo deja activo")
    rng = np.random.default_rng(1)
    ds = dataset_mod.Dataset(X=rng.normal(0, 1, (12, 5)),
                             y=np.array(["a", "b"] * 6),
                             feature_names=[f"f{i}" for i in range(5)],
                             segment_ids=[f"s{i}" for i in range(12)])
    path = dataset_mod.save_dataset(win.project, ds, "de_otra_sesion")
    win.dataset = None
    win.import_dataset = _patched_import(win, path)       # evita el diálogo modal
    win.import_dataset()
    assert win.dataset is not None, "no se cargó el dataset"
    assert win.dataset.X.shape == (12, 5), win.dataset.X.shape
    assert sorted(set(win.dataset.y)) == ["a", "b"]
    info = win.dataset_panel.info_label.text()
    assert "IMPORTADO" in info and "12 muestras" in info, info
    print(f"    {win.dataset.n_samples} muestras × {win.dataset.n_features} "
          f"características, listo para entrenar")

    print("[6] Un archivo inválido avisa, no revienta ni pisa el dataset actual")
    bad = os.path.join(tmp, "no_es_dataset.npz")
    with open(bad, "wb") as fh:
        fh.write(b"esto no es un npz")
    warned: list = []
    win.warn = lambda *a, **k: warned.append(a)
    previo = win.dataset
    win.import_dataset = _patched_import(win, bad)
    win.import_dataset()
    assert warned, "debería avisar de que el archivo no es válido"
    assert win.dataset is previo, "no debe pisar el dataset bueno"
    print("    aviso mostrado y el dataset anterior intacto ✓")

    win.acq_panel.shutdown()
    print("\nCANALES EN VIVO + IMPORTAR DATASET OK ✓")
    return 0


def _patched_import(win, path: str):
    """``import_dataset`` con el diálogo de archivo resuelto (no abre ventana)."""
    from PyQt6.QtWidgets import QFileDialog
    real = type(win).import_dataset

    def run():
        orig = QFileDialog.getOpenFileName
        QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (path, ""))
        try:
            real(win)
        finally:
            QFileDialog.getOpenFileName = orig
    return run


if __name__ == "__main__":
    raise SystemExit(main())

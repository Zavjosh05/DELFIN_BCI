"""Panel de Fuentes: agrupado por sujeto (plegable) + buscador.

Con decenas de señales (`sujeto001-abajo`, `sujeto001-arriba`, … por cada sujeto)
la lista plana es inmanejable. Cubre:

  * ``source_group`` deduce el sujeto del nombre (tolera «-» y «_»).
  * el modo «Agrupado por sujeto» inserta filas-cabecera plegables que NO son
    seleccionables ni editables (no rompen selección ni renombrado).
  * plegar un grupo oculta sus señales; desplegarlo las vuelve a mostrar.
  * el buscador filtra por nombre y despliega lo que coincide aunque su grupo esté
    plegado, ocultando las cabeceras que se quedan sin nada.
  * en modo agrupado no se puede reordenar arrastrando (solo en «orden propio»).
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QAbstractItemView, QApplication

from eeg_studio.config import APP_NAME, ORG_NAME
from eeg_studio.core.mat_loader import write_openvibe_csv
from eeg_studio.core.project import Project
from eeg_studio.ui.main_window import MainWindow, source_group

SUJETOS = ("sujeto001", "sujeto002")
ACCIONES = ("abajo", "arriba", "agarre")


def _set_sort(win, mode: str) -> None:
    """Fija el modo de orden de forma determinista.

    Si el combo ya estaba en ese modo (p. ej. porque quedó guardado en QSettings)
    no emite señal, así que la prueba no puede depender de ella: se fija el estado
    y se refresca a mano."""
    win.source_sort_combo.setCurrentIndex(win.source_sort_combo.findData(mode))
    win._source_sort = mode
    win._apply_source_drag_mode()
    win._refresh_sources()


def _rows(win):
    """(texto, es_cabecera, oculto) de cada fila de la lista."""
    lst = win.sources_list
    out = []
    for i in range(lst.count()):
        it = lst.item(i)
        is_header = it.data(Qt.ItemDataRole.UserRole) is None
        out.append((it.text(), is_header, it.isHidden()))
    return out


def _visible_signals(win):
    return [t for t, hdr, hid in _rows(win) if not hdr and not hid]


def _header_for(win, group: str):
    """Cabecera de un grupo, buscada FRESCA: cada refresco reconstruye la lista,
    así que guardar la referencia dejaría un objeto C++ ya destruido."""
    lst = win.sources_list
    for i in range(lst.count()):
        it = lst.item(i)
        if it.data(Qt.ItemDataRole.UserRole) is None and group in it.text():
            return it
    raise AssertionError(f"no se encontró la cabecera de {group}")


def main() -> int:
    print("[1] source_group deduce el sujeto del nombre")
    assert source_group("sujeto007-abajo") == "sujeto007"
    assert source_group("sujeto006_soltar") == "sujeto006"     # tolera «_»
    assert source_group("Prueba 001") == "Prueba"              # tolera espacio
    assert source_group("sinseparador") == "Otras"             # sin grupo propio
    print("    «-», «_», espacio y sin separador ✓")

    app = QApplication(sys.argv)
    # El modo de orden se guarda en los QSettings REALES del usuario: se respalda
    # y se restaura al final para no dejarle la app cambiada.
    settings = QSettings(ORG_NAME, APP_NAME)
    backup = settings.value("source_sort", None)
    try:
        code = _run(app)
    finally:
        if backup is None:
            settings.remove("source_sort")
        else:
            settings.setValue("source_sort", backup)
    return code


def _run(app) -> int:
    win = MainWindow()
    tmp = tempfile.mkdtemp()
    win.project = Project.create(tmp, "grp")
    rng = np.random.default_rng(0)
    for suj in SUJETOS:                        # 2 sujetos × 3 acciones = 6 señales
        for acc in ACCIONES:
            csv = os.path.join(tmp, f"{suj}-{acc}.csv")
            write_openvibe_csv(csv, rng.normal(0, 1, (64, 2)), 128.0, ["C3", "C4"], [])
            win.project.add_source(csv, alias=f"{suj}-{acc}")

    print("[2] Modo agrupado: una cabecera por sujeto + sus señales")
    _set_sort(win, "group")
    rows = _rows(win)
    headers = [t for t, hdr, _ in rows if hdr]
    assert len(headers) == 2, headers
    assert all(s in " ".join(headers) for s in SUJETOS), headers
    assert "(3)" in headers[0], headers[0]                 # nº de señales del grupo
    assert len(_visible_signals(win)) == 6
    print(f"    cabeceras: {headers}")

    print("[3] Las cabeceras no son seleccionables ni editables")
    lst = win.sources_list
    hdr_item = next(lst.item(i) for i in range(lst.count())
                    if lst.item(i).data(Qt.ItemDataRole.UserRole) is None)
    assert not (hdr_item.flags() & Qt.ItemFlag.ItemIsSelectable), "no debe seleccionarse"
    assert not (hdr_item.flags() & Qt.ItemFlag.ItemIsEditable), "no debe renombrarse"
    assert hdr_item.flags() & Qt.ItemFlag.ItemIsEnabled, "debe poder clicarse"
    print("    cabecera clicable pero ni seleccionable ni editable ✓")

    print("[4] Plegar/desplegar un grupo oculta y muestra sus señales")
    win._on_source_item_clicked(_header_for(win, SUJETOS[0]))     # pliega sujeto001
    vis = _visible_signals(win)
    assert len(vis) == 3, vis                              # solo quedan las del otro
    assert all(SUJETOS[0] not in v for v in vis), vis
    assert "▸" in _header_for(win, SUJETOS[0]).text()      # flecha de plegado
    win._on_source_item_clicked(_header_for(win, SUJETOS[0]))     # despliega
    assert len(_visible_signals(win)) == 6
    assert "▾" in _header_for(win, SUJETOS[0]).text()
    print("    plegado → 3 visibles · desplegado → 6 ✓")

    print("[5] Buscador: filtra por nombre y oculta cabeceras vacías")
    win.source_filter.setText("arriba")
    vis = _visible_signals(win)
    assert len(vis) == 2 and all("arriba" in v for v in vis), vis
    vis_headers = [t for t, hdr, hid in _rows(win) if hdr and not hid]
    assert len(vis_headers) == 2, vis_headers              # ambos tienen un «arriba»
    win.source_filter.setText(f"{SUJETOS[0]}-agarre")      # una sola señal
    vis = _visible_signals(win)
    assert vis == [f"{SUJETOS[0]}-agarre"], vis
    vis_headers = [t for t, hdr, hid in _rows(win) if hdr and not hid]
    assert len(vis_headers) == 1, "la cabecera sin coincidencias debe ocultarse"
    print(f"    «{SUJETOS[0]}-agarre» → {vis} (1 cabecera visible)")

    print("[6] Con filtro, un grupo plegado se despliega para mostrar lo que coincide")
    win.source_filter.setText("")
    win._on_source_item_clicked(_header_for(win, SUJETOS[0]))     # pliega sujeto001
    assert len(_visible_signals(win)) == 3
    win.source_filter.setText("agarre")
    vis = _visible_signals(win)
    assert len(vis) == 2, vis                              # incluye el del grupo plegado
    assert any(SUJETOS[0] in v for v in vis), vis
    win.source_filter.setText("")
    assert len(_visible_signals(win)) == 3, "al limpiar vuelve a mandar el plegado"
    print("    el filtro ignora el plegado; al limpiarlo se respeta de nuevo ✓")

    print("[7] Agrupado no permite reordenar arrastrando (solo «orden propio»)")
    assert win.sources_list.dragDropMode() == QAbstractItemView.DragDropMode.NoDragDrop
    _set_sort(win, "custom")
    assert win.sources_list.dragDropMode() == QAbstractItemView.DragDropMode.InternalMove
    assert not any(hdr for _, hdr, _ in _rows(win)), "sin agrupar no hay cabeceras"
    print("    drag solo en «orden propio»; sin cabeceras fuera del modo agrupado ✓")

    win.acq_panel.shutdown()
    print("\nFUENTES: AGRUPADO POR SUJETO + BUSCADOR OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

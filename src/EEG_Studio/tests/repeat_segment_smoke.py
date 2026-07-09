"""Generar segmentos periódicos a partir del primero (ayudante «periodizar»)."""
from __future__ import annotations

import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from eeg_studio.core.project import Project


def main() -> int:
    proj = Project.create(tempfile.mkdtemp(), "periodic")
    sid = "abc123"                                # id de fuente ficticio (no hace falta CSV)
    fs = 128.0
    dur = int(5 * fs)                             # 5 s
    period = int(15 * fs)                         # 15 s (5 tarea + 10 reposo)

    seed = proj.add_segment(sid, 1584, 1584 + dur, "arriba")

    print("[1] Total=4 → genera 3 más, cada 15 s, misma duración y etiqueta")
    created = proj.repeat_segment(seed["id"], period, count=4, n_samples=8157)
    assert created == 3, created
    segs = sorted((s for s in proj.state["segments"] if s["source_id"] == sid),
                  key=lambda x: x["start"])
    starts = [s["start"] for s in segs]
    assert starts == [1584, 1584 + period, 1584 + 2 * period, 1584 + 3 * period], starts
    assert all(s["stop"] - s["start"] == dur and s["label"] == "arriba" for s in segs)
    print(f"    inicios={starts} (todos «arriba», {dur/fs:.0f}s)")

    print("[2] No duplica si se vuelve a ejecutar")
    again = proj.repeat_segment(seed["id"], period, count=4, n_samples=8157)
    assert again == 0, again
    assert len([s for s in proj.state["segments"] if s["source_id"] == sid]) == 4

    print("[3] Se detiene en el final de la señal (no crea fuera de rango)")
    p2 = Project.create(tempfile.mkdtemp(), "fill")
    seed2 = p2.add_segment(sid, 1000, 1000 + dur, "abajo")
    # señal de 6000 muestras: caben inicios en 1000, 2920, 4840 (6760 > 6000 se corta)
    n = p2.repeat_segment(seed2["id"], period, count=None, n_samples=6000)
    starts2 = sorted(s["start"] for s in p2.state["segments"])
    assert starts2 == [1000, 1000 + period, 1000 + 2 * period], starts2
    assert all(s["stop"] <= 6000 for s in p2.state["segments"])
    print(f"    «hasta el final»: inicios={starts2}, todos dentro de 6000")

    print("[4] Un solo commit → un solo Ctrl+Z lo deshace")
    p2.undo()
    assert len(p2.state["segments"]) == 1, "el undo debe quitar todos los generados de golpe"

    print("\nGENERAR SEGMENTOS PERIÓDICOS OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

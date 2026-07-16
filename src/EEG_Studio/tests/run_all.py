"""Ejecuta TODA la batería de pruebas de humo en paralelo.

Por qué existe: cada prueba corre en su propio proceso (una `QApplication` por
proceso, aislamiento total), y arrancar ese proceso —importar PyQt6, numpy,
sklearn, pyqtgraph…— cuesta varios segundos FIJOS. En serie, esos segundos se
pagan 80+ veces y la batería tarda muchos minutos. Lanzándolas en paralelo, el
arranque se reparte entre los núcleos y el tiempo de pared cae proporcionalmente.

Uso (desde ``src/EEG_Studio``):
    ./.venv/Scripts/python.exe -m tests.run_all             # auto (nº de núcleos)
    ./.venv/Scripts/python.exe -m tests.run_all -j 4        # 4 en paralelo
    ./.venv/Scripts/python.exe -m tests.run_all -k control  # solo las que casen

Imprime PASS/FAIL con el tiempo de cada una (más lentas arriba) y termina con
código != 0 si alguna falla. No necesita ningún archivo: las pruebas sintetizan
sus datos (ver ``tests.sample_csv``).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # src/EEG_Studio


def _discover(pattern: str | None) -> list[str]:
    """Nombres de módulo de prueba (``tests.<x>``), excluyéndose a sí mismo."""
    mods = []
    for path in sorted(glob.glob(os.path.join(_HERE, "*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in ("__init__", "run_all"):
            continue
        if pattern and pattern not in name:
            continue
        mods.append(name)
    return mods


# Cada prueba corre en su proceso; varias a la vez. Para que no se sobresuscriba la
# CPU (cada proceso podría abrir su propio pool de extracción y numpy/sklearn lanzan
# hilos BLAS por proceso), se fuerza a los hijos a un solo worker y a un solo hilo de
# cálculo. Sin esto, N pruebas × M núcleos de hilos internos hacen que la máquina se
# atasque y pruebas de segundos tarden minutos.
_CHILD_ENV = {
    "QT_QPA_PLATFORM": "offscreen",
    "PYTHONIOENCODING": "utf-8",
    "EEG_N_WORKERS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _run_one(name: str, timeout: int) -> tuple[str, bool, float, str]:
    env = dict(os.environ, **_CHILD_ENV)
    start = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-m", f"tests.{name}"],
                           cwd=_ROOT, env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        ok = p.returncode == 0
        tail = (p.stdout or "")[-1500:] if not ok else ""
    except subprocess.TimeoutExpired:
        ok, tail = False, f"TIMEOUT tras {timeout}s"
    return name, ok, time.perf_counter() - start, tail


def main() -> int:
    ap = argparse.ArgumentParser(description="Batería de pruebas en paralelo.")
    ap.add_argument("-j", "--jobs", type=int, default=0,
                    help="pruebas en paralelo (0 = nº de núcleos, máx 8).")
    ap.add_argument("-k", "--filter", default=None,
                    help="ejecutar solo las pruebas cuyo nombre contenga este texto.")
    ap.add_argument("--timeout", type=int, default=300,
                    help="límite por prueba en segundos (por defecto 300).")
    args = ap.parse_args()

    jobs = args.jobs or min(8, os.cpu_count() or 4)
    mods = _discover(args.filter)
    if not mods:
        print("No hay pruebas que casen.")
        return 1

    print(f"Ejecutando {len(mods)} pruebas, {jobs} en paralelo…\n")
    t0 = time.perf_counter()
    results: list[tuple[str, bool, float, str]] = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for res in ex.map(lambda m: _run_one(m, args.timeout), mods):
            name, ok, secs, _ = res
            print(f"  {'ok  ' if ok else 'FAIL'} {secs:6.1f}s  {name}", flush=True)
            results.append(res)

    wall = time.perf_counter() - t0
    fails = [r for r in results if not r[1]]
    results.sort(key=lambda r: r[2], reverse=True)
    print("\nMás lentas:")
    for name, ok, secs, _ in results[:8]:
        print(f"  {secs:6.1f}s  {'ok' if ok else 'FAIL'}  {name}")

    if fails:
        print(f"\n{len(fails)} FALLARON:")
        for name, _ok, _secs, tail in fails:
            print(f"\n===== {name} =====\n{tail}")
    print(f"\nPASA={len(results) - len(fails)}  FALLA={len(fails)}  "
          f"tiempo de pared={wall:.0f}s (suma en serie sería "
          f"{sum(r[2] for r in results):.0f}s)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

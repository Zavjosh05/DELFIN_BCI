"""Indicadores de calidad/ruido de la señal y diagnóstico del dongle (sin hardware)."""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.acquisition import quality


def main() -> int:
    rng = np.random.default_rng(0)
    n = 256
    good = 4200 + rng.normal(0, 5, n)                 # buen contacto: fluctúa poco
    flat = np.full(n, 4200.0)                         # plano: desconectado
    noise = 4200 + rng.normal(0, 1200, n)             # ruido: oscila muchísimo
    saturated = np.full(n, 60000.0)                   # a riel

    print("[1] Clasificación por canal")
    assert quality.channel_status(good) == "ok", quality.channel_status(good)
    assert quality.channel_status(flat) == "plano"
    assert quality.channel_status(noise) == "ruido", quality.channel_status(noise)
    assert quality.channel_status(saturated) == "saturado"
    print("    ok / plano / ruido / saturado detectados ✓")

    print("[2] Resumen: mayoría mala => is_noise=True")
    bad = quality.assess(np.vstack([good, flat, noise, saturated]))
    assert bad["n"] == 4 and bad["n_ok"] == 1 and bad["is_noise"] is True, bad
    okq = quality.assess(np.vstack([good, good, good, flat]))
    assert okq["n_ok"] == 3 and okq["is_noise"] is False, okq
    print(f"    mala: {bad['n_ok']}/4 OK (ruido)  ·  buena: {okq['n_ok']}/4 OK")

    print("[3] quick_diagnose sin dongle conectado")
    import eeg_studio.acquisition.emotiv as em
    if not em.emotiv_deps_available():
        print("    (sin deps; se omite)")
    else:
        saved = em.EmotivDongleSource._find_device
        try:
            em.EmotivDongleSource._find_device = staticmethod(lambda: None)
            r = em.quick_diagnose()
            assert r["ok"] is False and r["found"] is False, r
            assert "No se encontró" in r["summary"], r["summary"]
            print(f"    -> {r['summary'].splitlines()[0]}")
        finally:
            em.EmotivDongleSource._find_device = saved

    print("\nCALIDAD DE SEÑAL / DIAGNÓSTICO OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

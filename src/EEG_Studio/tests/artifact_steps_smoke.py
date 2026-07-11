"""Prueba de los pasos de preprocesamiento: ASR, rechazo por umbral y línea base.

Verifica que los tres pasos nuevos (ver `core/preprocessing.py`) están completos
en las cinco tablas del registro (STEP_REGISTRY/LABELS/DESCRIPTIONS/
PARAM_DESCRIPTIONS/DEFAULTS), que corrigen el comportamiento esperado sobre
señales sintéticas, y que funcionan encadenados dentro de `apply_pipeline`.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.core import preprocessing as pp

NEW_STEPS = ("asr", "threshold", "baseline")


def main() -> int:
    print("[1] Los 3 pasos nuevos están registrados en las 5 tablas")
    for stype in NEW_STEPS:
        assert stype in pp.STEP_REGISTRY, stype
        assert stype in pp.STEP_LABELS, stype
        assert stype in pp.STEP_DESCRIPTIONS, stype
        assert stype in pp.STEP_DEFAULTS, stype
        for key in pp.STEP_DEFAULTS[stype]:
            assert key in pp.PARAM_DESCRIPTIONS, f"{stype}.{key} sin descripción"
    print("    asr / threshold / baseline completos en las 5 tablas")

    fs = 128.0
    n = 1024
    t = np.arange(n) / fs

    print("[2] ASR: atenúa una ráfaga de artefacto conservando la señal limpia")
    rng = np.random.default_rng(0)
    n_ch = 4
    s1 = np.sin(2 * np.pi * 10 * t)
    s2 = np.sin(2 * np.pi * 2 * t)
    mix = rng.uniform(0.5, 1.5, (n_ch, 2))
    clean = mix @ np.vstack([s1, s2]) + rng.normal(0, 0.05, (n_ch, n))
    burst_slice = slice(400, 450)
    burst = np.zeros(n)
    burst[burst_slice] = 40.0                    # ráfaga muy grande y localizada
    data = clean + np.outer(np.ones(n_ch), burst)

    out = pp.asr_reconstruct(data, fs, window_sec=0.3, cutoff=4.0)
    assert out.shape == data.shape

    rms_before = np.sqrt(np.mean(data[:, burst_slice] ** 2))
    rms_after = np.sqrt(np.mean(out[:, burst_slice] ** 2))
    print(f"    RMS en la ráfaga: antes={rms_before:.1f}  después={rms_after:.1f}")
    assert rms_after < rms_before * 0.6, "ASR no atenuó la ráfaga"

    outside = np.r_[0:350, 500:n]
    corr = np.corrcoef(out[0, outside], clean[0, outside])[0, 1]
    print(f"    correlación con la señal limpia fuera de la ráfaga: {corr:.3f}")
    assert corr > 0.9, "ASR distorsionó la señal limpia fuera de la ráfaga"

    single = pp.asr_reconstruct(rng.normal(0, 1, (1, 200)), fs)
    assert single.shape == (1, 200), "un solo canal debe devolverse sin tocar"
    print("    ráfaga atenuada, señal limpia conservada, 1 canal no revienta")

    print("[3] Rechazo por umbral: manual recorta a un valor fijo")
    x = np.array([[1.0, -1.0, 150.0, -150.0, 5.0, 0.0]])
    out_manual = pp.threshold_reject(x, mode="manual", threshold=100.0)
    assert np.allclose(out_manual, [[1.0, -1.0, 100.0, -100.0, 5.0, 0.0]])
    print(f"    manual: {x.ravel()} -> {out_manual.ravel()}")

    print("[4] Rechazo por umbral: automático se adapta a cada canal")
    calm = rng.normal(0, 1.0, n)                  # canal tranquilo
    noisy = rng.normal(0, 1.0, n)
    noisy[10] = 200.0                             # un solo pico enorme
    two_ch = np.vstack([calm, noisy])
    out_auto = pp.threshold_reject(two_ch, mode="automatico", k=5.0)
    assert out_auto.shape == two_ch.shape
    assert out_auto[1, 10] < 200.0, "el pico del canal ruidoso no se recortó"
    assert abs(out_auto[1, 10]) < 50.0, "el recorte automático debería ser mucho menor a 200"
    typical_before = two_ch[:, :5]
    typical_after = out_auto[:, :5]
    assert np.allclose(typical_before, typical_after, atol=1e-6), \
        "el modo automático no debería tocar muestras dentro de rango normal"
    print(f"    pico 200.0 -> {out_auto[1, 10]:.1f}; muestras normales intactas")

    print("[5] Corrección de la línea base: ancla al tramo inicial")
    baseline_true = 30.0
    n_base = int(round(0.2 * fs))
    sig = np.full(n, baseline_true)
    sig[n_base:] += np.linspace(0, 5, n - n_base)   # deriva lenta tras la línea base
    sig = sig[None, :]
    out_bl = pp.baseline_correction(sig, fs, baseline_sec=0.2)
    base_mean_after = out_bl[0, :n_base].mean()
    print(f"    media de la línea base tras corregir: {base_mean_after:.4f}")
    assert abs(base_mean_after) < 1e-9, "la línea base no quedó en cero"
    assert abs(out_bl[0, -1] - 5.0) < 1e-9, "la deriva posterior no debe tocarse"

    # baseline_sec mayor que el segmento -> usa el segmento completo, no revienta.
    short = np.array([[1.0, 2.0, 3.0]])
    out_short = pp.baseline_correction(short, fs, baseline_sec=10.0)
    assert out_short.shape == short.shape
    print("    fallback con baseline_sec > duración del segmento no revienta")

    print("[6] apply_pipeline encadena los 3 pasos nuevos sin romperse")
    pipe = [
        {"type": "baseline", "params": {"baseline_sec": 0.2}},
        {"type": "threshold", "params": {"mode": "automatico", "k": 5.0}},
        {"type": "asr", "params": {"window_sec": 0.5, "cutoff": 5.0}},
    ]
    out_pipe = pp.apply_pipeline(data, fs, pipe)
    assert out_pipe.shape == data.shape and np.all(np.isfinite(out_pipe))
    print("    pipeline baseline -> threshold -> asr OK")

    print("\nPASOS NUEVOS (ASR / umbral / línea base) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

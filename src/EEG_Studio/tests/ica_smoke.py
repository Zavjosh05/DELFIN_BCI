"""Prueba del paso de eliminación de artefactos por ICA (sin GUI).

Inyecta un artefacto muy picudo (kurtosis alta) en un canal y comprueba que el
paso ICA lo atenúa, conservando la forma de la señal.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
from scipy.stats import kurtosis

from eeg_studio.core import preprocessing as pp


def main() -> int:
    assert "ica" in pp.STEP_REGISTRY, "el paso ICA no está registrado"
    rng = np.random.default_rng(0)
    n_ch, n = 14, 2000
    # Señal base + una fuente picuda mezclada en varios canales (parpadeo).
    data = rng.normal(0, 1, (n_ch, n))
    spike = np.zeros(n)
    spike[::200] = 30.0                      # picos periódicos -> kurtosis alta
    data += np.outer(rng.uniform(0.5, 1.0, n_ch), spike)

    k_before = float(np.max(np.abs(kurtosis(data, axis=1))))
    out = pp.apply_pipeline(data, 128.0,
                            [{"type": "ica", "params": {"n_components": 0, "kurt_threshold": 5.0}}])
    k_after = float(np.max(np.abs(kurtosis(out, axis=1))))

    print(f"    forma {data.shape} -> {out.shape}")
    print(f"    kurtosis máx antes={k_before:.1f}  después={k_after:.1f}")
    assert out.shape == data.shape, "la forma cambió"
    assert k_after < k_before, "la ICA no redujo el artefacto picudo"

    print("[2] Tras un CAR, la ICA NO debe destruir la señal")
    # Regresión: el CAR (referencia promedio común) resta la media entre canales,
    # así que los canales quedan linealmente dependientes -> rango n_canales-1.
    # Se le pedían a FastICA tantos componentes como canales: el blanqueado dividía
    # entre un autovalor ~0, la matriz de mezcla salía con cond ~1e17 (~1e33 en una
    # ventana corta) y `inverse_transform` devolvía ruido. Resultado: un 84-100% de
    # error de reconstrucción AUNQUE no se anulara ningún componente, tanto al
    # entrenar como en el control en vivo.
    car_pipe = [{"type": "car", "params": {}}]
    ica_step = [{"type": "ica", "params": {"n_components": 0, "kurt_threshold": 5.0}}]
    clean = rng.normal(0, 10, (8, 2048))     # sin artefactos: la ICA no debe anular nada
    car = pp.apply_pipeline(clean, 128.0, car_pipe)
    rank = int(np.linalg.matrix_rank(car.T))
    out = pp.apply_pipeline(car, 128.0, ica_step)
    err = float(np.linalg.norm(out - car) / np.linalg.norm(car))
    print(f"    tras el CAR: rango {rank}/{car.shape[0]} (deficiente por construcción)")
    print(f"    error de reconstrucción de la ICA: {err * 100:.2f}%")
    assert rank == car.shape[0] - 1, f"el CAR debería dejar rango n-1, dio {rank}"
    assert err < 0.05, f"la ICA destruyó la señal tras el CAR: {err * 100:.0f}% de error"

    print("[3] …y tampoco en una ventana corta (el caso del control en vivo)")
    # 256 muestras = 2 s a 128 Hz: la ventana por defecto de la inferencia online.
    win = pp.apply_pipeline(rng.normal(0, 10, (8, 256)), 128.0, car_pipe)
    out_w = pp.apply_pipeline(win, 128.0, ica_step)
    err_w = float(np.linalg.norm(out_w - win) / np.linalg.norm(win))
    amp = float(np.linalg.norm(out_w) / np.linalg.norm(win))
    print(f"    ventana de 256: error {err_w * 100:.2f}%  ·  amplitud salida/entrada {amp:.3f}")
    assert err_w < 0.05, f"la ICA destruyó la ventana corta: {err_w * 100:.0f}% de error"

    print("[4] El nº de componentes se recorta al rango, no al de canales")
    fit = pp._fit_ica(car, 0)
    assert fit is not None, "_fit_ica no debería rendirse con datos válidos"
    _, sources = fit
    print(f"    componentes ajustados: {sources.shape[1]} (canales {car.shape[0]}, rango {rank})")
    assert sources.shape[1] == rank, \
        f"se pidieron {sources.shape[1]} componentes con rango {rank}"

    print("\nICA (artefactos) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

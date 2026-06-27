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

    print("\nICA (artefactos) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pasa-banda/altas/bajas con diseño Butterworth (IIR) y FIR seleccionable."""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np

from eeg_studio.core import preprocessing as P


def main() -> int:
    fs = 250.0
    n = 1000
    t = np.arange(n) / fs
    inband = np.sin(2 * np.pi * 15 * t)       # 15 Hz: dentro de 8–30
    outband = np.sin(2 * np.pi * 70 * t)      # 70 Hz: fuera
    data = (inband + outband)[None, :]
    sl = slice(80, -80)                       # recortar transitorios de borde

    print("[1] Pasa-banda 8–30 con ambos diseños conserva 15 Hz y quita 70 Hz")
    outs = {}
    for design in ("butter", "fir"):
        out = P.bandpass(data, fs, 8.0, 30.0, design=design, numtaps=101)
        assert out.shape == data.shape, (design, out.shape)
        o = out[0, sl]
        ci = abs(np.corrcoef(o, inband[sl])[0, 1])
        co = abs(np.corrcoef(o, outband[sl])[0, 1])
        print(f"    {design:6s}: corr(15Hz)={ci:.2f}  corr(70Hz)={co:.2f}")
        assert ci > 0.9 and co < 0.2, (design, ci, co)
        outs[design] = o

    print("[2] Butterworth y FIR dan resultados distintos (la opción cambia algo)")
    assert not np.allclose(outs["butter"], outs["fir"]), "butter y fir idénticos"
    # El diseño por defecto debe ser Butterworth (compatibilidad).
    assert np.allclose(P.bandpass(data, fs, 8.0, 30.0)[0, sl], outs["butter"])
    print("    distintos, y el defecto = butter ✓")

    print("[3] Pasa-altas y pasa-bajas FIR funcionan")
    assert P.highpass(data, fs, 5.0, design="fir").shape == data.shape
    assert P.lowpass(data, fs, 40.0, design="fir").shape == data.shape
    print("    highpass/lowpass FIR OK")

    print("[4] _odd_numtaps: impar y acotado a la longitud")
    assert P._odd_numtaps(100, 100000) == 101            # par -> impar
    short = P._odd_numtaps(999, 90)                       # acotado para segmento corto
    assert short % 2 == 1 and short <= 90 // 3 + 1, short
    print(f"    100->101 ; corto(90)->{short}")

    print("[5] apply_pipeline con un pasa-banda FIR")
    pipe = [{"type": "bandpass",
             "params": {"low": 8.0, "high": 30.0, "design": "fir", "numtaps": 101}}]
    out = P.apply_pipeline(data, fs, pipe)
    assert out.shape == data.shape and np.all(np.isfinite(out))
    print("    pipeline con FIR OK")

    print("[6] Notch 60 Hz: 'iir' y 'fir' atenúan 60 Hz y conservan 15 Hz")
    n2 = 2000
    t2 = np.arange(n2) / fs
    s15 = np.sin(2 * np.pi * 15 * t2)
    s60 = np.sin(2 * np.pi * 60 * t2)
    d2 = (s15 + s60)[None, :]
    sl2 = slice(200, -200)
    base_co = abs(np.corrcoef((s15 + s60)[sl2], s60[sl2])[0, 1])   # ~0.71 sin filtrar
    for design in ("iir", "fir"):
        out = P.notch(d2, fs, freq=60.0, q=30.0, design=design, numtaps=257)[0, sl2]
        ci = abs(np.corrcoef(out, s15[sl2])[0, 1])
        co = abs(np.corrcoef(out, s60[sl2])[0, 1])
        print(f"    {design:4s}: corr(15Hz)={ci:.2f}  corr(60Hz)={co:.2f} (sin filtrar {base_co:.2f})")
        assert ci > 0.9, (design, ci)
        assert co < 0.35 and co < base_co / 2, (design, co)
    print("    notch IIR y FIR atenúan la red eléctrica ✓")

    print("\nDISEÑO DE FILTROS (Butterworth / FIR, incl. notch) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

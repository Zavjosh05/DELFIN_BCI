"""Verifica el parseo de líneas estilo CyKit en TCPSource (sin red ni hardware).

CyKit emite líneas de texto con valores separados por comas; esta prueba
comprueba que se extraen los 14 canales con distintas columnas de inicio.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from eeg_studio.acquisition import TCPSource


def main() -> int:
    # Línea con COUNTER, INTERPOLATE y 14 canales (formato sin 'nocounter').
    ch = [f"{4200 + i:.4f}" for i in range(14)]
    line_with_counter = ("3, 0, " + ", ".join(ch)).encode("utf-8")
    # Línea solo con los 14 canales (formato con 'nocounter').
    line_nocounter = (", ".join(ch)).encode("utf-8")

    print("[1] channel_start=2 (con COUNTER/INTERPOLATE)")
    src = TCPSource("127.0.0.1", 5151, n_channels=14, channel_start=2)
    out = src._parse_lines([line_with_counter, line_with_counter])
    assert out is not None and out.shape == (14, 2), f"forma inesperada: {None if out is None else out.shape}"
    assert abs(out[0, 0] - 4200.0) < 1e-6, f"primer canal mal: {out[0, 0]}"
    print(f"    forma={out.shape}, primer canal={out[0,0]:.1f}")

    print("[2] channel_start=0 (con 'nocounter')")
    src0 = TCPSource("127.0.0.1", 5151, n_channels=14, channel_start=0)
    out0 = src0._parse_lines([line_nocounter])
    assert out0 is not None and out0.shape == (14, 1), f"forma inesperada: {out0.shape if out0 is not None else None}"
    assert abs(out0[13, 0] - 4213.0) < 1e-6, f"último canal mal: {out0[13,0]}"
    print(f"    forma={out0.shape}, último canal={out0[13,0]:.1f}")

    print("[3] Líneas incompletas se ignoran")
    assert src._parse_lines([b"1, 2, 3", b""]) is None, "no debería parsear líneas cortas"
    print("    líneas incompletas ignoradas")

    print("\nPARSEO CYKIT/TCP OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

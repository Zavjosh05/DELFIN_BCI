"""La auto-detección de modo (14/16 bits) usa el byte contador (regresión del bug real).

Reproduce, sin hardware, lo observado con el dongle real: el casco era 14-bit pero
la heurística de µV elegía 16-bit. La selección por contador debe elegir el correcto.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def main() -> int:
    try:
        from Crypto.Cipher import AES
    except Exception:  # noqa: BLE001
        print("pycryptodome no disponible; se omite.")
        return 0

    from eeg_studio.acquisition.emotiv import build_key, counter_score

    serial = "UD2016030300214D"            # mismo formato que el casco real
    true_mode = "14bit"

    print("[1] Frames con contador limpio, cifrados con la clave 14bit")
    enc = AES.new(build_key(serial, true_mode), AES.MODE_ECB)
    cts = []
    for i in range(60):
        frame = bytearray([128]) * 0 + bytearray(32)
        frame[0] = i % 128                 # contador incremental
        for j in range(1, 32):
            frame[j] = 128
        cts.append(enc.encrypt(bytes(frame)))

    print("[2] counter_score distingue el modo correcto del incorrecto")
    scores = {}
    for m in ("16bit", "14bit"):
        dec = AES.new(build_key(serial, m), AES.MODE_ECB)
        scores[m] = counter_score([dec.decrypt(c) for c in cts])
    print(f"    score 14bit={scores['14bit']:.2f}  ·  16bit={scores['16bit']:.2f}")
    assert scores["14bit"] > 0.95, scores
    assert scores["16bit"] < 0.30, scores
    assert max(scores, key=scores.get) == true_mode

    print("[3] counter_score robusto ante listas vacías/cortas")
    assert counter_score([]) == 0.0
    assert counter_score([b"\x05" + b"\x00" * 31]) == 0.0

    print("\nAUTO-DETECCIÓN DE MODO EMOTIV OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

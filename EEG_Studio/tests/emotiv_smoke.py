"""Prueba del lector nativo Emotiv (parte determinista, sin hardware).

Valida la derivación de clave, el descifrado AES y la conversión a µV cifrando un
frame conocido y comprobando el camino inverso. Verifica además que, sin dongle,
la fuente falla de forma limpia (sin crashear).
"""
from __future__ import annotations

import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np
from Crypto.Cipher import AES

import eeg_studio.acquisition.emotiv as em
from eeg_studio.acquisition import EmotivDongleSource, emotiv_deps_available
from eeg_studio.acquisition.emotiv import build_key, convert_frame, frame_plausibility


def _plausible_plaintext() -> bytes:
    """Frame cuyo descifrado correcto da ~4200 µV (v2=128 → término fino 0)."""
    b = bytearray(32)
    b[0], b[1] = 10, 5            # contador, no-gyro
    for i in list(range(2, 16, 2)) + list(range(18, 32, 2)):
        b[i], b[i + 1] = 50, 128
    return bytes(b)


class _FakeDevice:
    def __init__(self, frames):
        self._frames = frames
        self._i = 0

    def read(self, n, timeout_ms=0):
        f = self._frames[self._i % len(self._frames)]
        self._i += 1
        return list(f)


def main() -> int:
    assert emotiv_deps_available(), "faltan hidapi/pycryptodome"
    serial = "SN201602AB1234X"  # 15 caracteres, como un serial real

    print("[1] Derivación de claves 16-bit y 14-bit")
    k16 = build_key(serial, "16bit")
    k14 = build_key(serial, "14bit")
    assert len(k16) == 16 and len(k14) == 16, "la clave debe ser de 16 bytes"
    assert k16 != k14, "las claves de 14 y 16 bits deberían diferir"
    print(f"    k16={k16.hex()}  k14={k14.hex()}")

    print("[2] Cifrar un frame conocido y recuperar los 14 canales")
    plain = bytes((i * 7 + 3) % 256 for i in range(32))  # frame de prueba
    plain = bytes([plain[0], 5]) + plain[2:]              # byte[1]=5 (no es gyro)
    expected = convert_frame(plain)

    cipher = AES.new(k16, AES.MODE_ECB)
    encrypted = cipher.encrypt(plain)
    decoded = AES.new(k16, AES.MODE_ECB).decrypt(encrypted)
    recovered = convert_frame(decoded)

    assert np.allclose(recovered, expected), "el camino descifrar→convertir no recupera el frame"
    assert recovered.shape == (14,), "deben ser 14 canales"
    # La línea base debe rondar ~4200 µV (constante de EPOC+).
    assert 3000 < recovered.mean() < 5500, f"valores fuera de rango: {recovered.mean():.1f}"
    print(f"    14 canales recuperados, media={recovered.mean():.1f} µV")

    print("[3] Plausibilidad: clave correcta puntúa más que la incorrecta")
    plain = _plausible_plaintext()
    good = frame_plausibility(convert_frame(plain))
    wrong = frame_plausibility(convert_frame(bytes((i * 53 + 7) % 256 for i in range(32))))
    assert good > wrong, "la métrica de plausibilidad no discrimina"
    print(f"    correcta={good:.1f} > aleatoria={wrong:.1f}")

    print("[4] Autodetección de modo (14/16-bit) sobre frames cifrados")
    for mode in ("16bit", "14bit"):
        ct = AES.new(build_key(serial, mode), AES.MODE_ECB).encrypt(plain)
        src = EmotivDongleSource(mode="auto", serial=serial)
        src._running.set()
        chosen, _cipher = src._select_mode(_FakeDevice([ct] * 50), serial)
        src._running.clear()
        assert chosen == mode, f"esperaba {mode}, eligió {chosen}"
        print(f"    cifrado en {mode} → detectado {chosen} ✓")

    print("[5] Detección de dispositivo por nombre y por vendor id")
    saved = em.hid.enumerate
    try:
        em.hid.enumerate = lambda: [{"product_string": "x", "vendor_id": 0x1},
                                    {"product_string": "EEG Signals", "path": b"p1"}]
        assert EmotivDongleSource._find_device()["path"] == b"p1", "no detectó por nombre"
        em.hid.enumerate = lambda: [{"product_string": "?", "vendor_id": 0x21A1, "path": b"p2"}]
        assert EmotivDongleSource._find_device()["path"] == b"p2", "no detectó por vendor id"
        em.hid.enumerate = lambda: [{"product_string": "otro", "vendor_id": 0x999}]
        assert EmotivDongleSource._find_device() is None, "no debería detectar nada"
    finally:
        em.hid.enumerate = saved
    print("    detección por nombre y por vendor id OK")

    print("[6] Sin dongle conectado: la fuente debe fallar limpiamente")
    src = EmotivDongleSource(mode="auto")
    src.start()
    time.sleep(0.5)
    src.stop()
    assert src.error is not None, "debería reportar que no encontró el dispositivo"
    print(f"    error esperado: {src.error.splitlines()[0]}")

    print("\nEMOTIV (descifrado/conversión/autodetección) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

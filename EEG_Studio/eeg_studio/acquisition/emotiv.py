"""Lector nativo del Emotiv EPOC+ por USB (sin CyKit ni OpenViBE).

Reimplementa, dentro de la app, el método de adquisición que usa CyKit: abre el
dispositivo HID del dongle, descifra cada reporte con AES-ECB (clave derivada del
número de serie) y convierte los bytes a microvoltios.

Dependencias (opcionales, import protegido): ``hidapi`` y ``pycryptodome``.
Si faltan, la app sigue funcionando y solo se deshabilita esta fuente.

.. note::
   La capa de descifrado y conversión es determinista y está cubierta por
   pruebas. La **lectura USB** depende del hardware (modo 14/16 bits, formato del
   reporte) y debe afinarse con el casco conectado.
"""
from __future__ import annotations

import numpy as np

from ..config import EPOC_CHANNELS
from .base import StreamSource

try:
    import hid
    _HID_OK = True
except Exception:  # noqa: BLE001
    hid = None
    _HID_OK = False

try:
    from Crypto.Cipher import AES
    _CRYPTO_OK = True
except Exception:  # noqa: BLE001
    AES = None
    _CRYPTO_OK = False


def emotiv_deps_available() -> bool:
    return _HID_OK and _CRYPTO_OK


# Nombres de producto HID que usa Emotiv (los mismos que prueba CyKit).
EMOTIV_PRODUCTS = {"EPOC+", "EEG Signals", "Emotiv RAW DATA", "FLEX", "00000000000"}
# Vendor IDs conocidos de los receptores Emotiv (respaldo si el nombre no coincide).
EMOTIV_VENDOR_IDS = {0x1234, 0x21A1}

# Posiciones de los 14 canales dentro del frame de 32 bytes descifrado
# (bytes 0-1 = contador/estado, 16-17 = giroscopio; el resto, EEG).
_CHANNEL_BYTES = list(range(2, 16, 2)) + list(range(18, 32, 2))

# Línea base aproximada del EPOC+ en µV (constante de conversión de CyKit).
_BASELINE_UV = 4201.0


def frame_plausibility(values: np.ndarray) -> float:
    """Puntúa cuán "real" parece un frame ya convertido (mayor = mejor).

    Con la clave correcta, los 14 canales se agrupan en torno a ~4200 µV; con una
    clave errónea, AES devuelve bytes aleatorios y los valores se disparan. Sirve
    para autodetectar el modo (14 vs 16 bits) de forma automática.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or not np.all(np.isfinite(v)):
        return -1e18
    if v.min() < -50000 or v.max() > 50000:
        return -1e18
    return -float(np.mean(np.abs(v - _BASELINE_UV)))


def build_key(serial: str, mode: str = "16bit") -> bytes:
    """Deriva la clave AES de 16 bytes a partir del número de serie del casco.

    ``mode`` = ``"16bit"`` (EPOC+ Consumer, modelo 6 de CyKit) o ``"14bit"``
    (EPOC+ en modo 14 bits, modelo 7).
    """
    sn = [ord(c) for c in serial]
    if len(sn) < 4:
        raise ValueError("Número de serie del casco demasiado corto.")
    if mode == "14bit":
        k = [sn[-1], 0, sn[-2], 21, sn[-3], 0, sn[-4], 12,
             sn[-3], 0, sn[-2], 68, sn[-1], 0, sn[-2], 88]
    else:  # 16bit
        k = [sn[-1], sn[-2], sn[-2], sn[-3], sn[-3], sn[-3], sn[-2], sn[-4],
             sn[-1], sn[-4], sn[-2], sn[-2], sn[-4], sn[-4], sn[-2], sn[-1]]
    return bytes(k)


def convert_frame(frame: bytes) -> np.ndarray:
    """Convierte un frame de 32 bytes descifrado en 14 valores en µV."""
    vals = []
    for i in _CHANNEL_BYTES:
        v1, v2 = frame[i], frame[i + 1]
        vals.append((v1 * 0.128205128205129 + 4201.02564096001) + ((v2 - 128) * 32.82051289))
    return np.asarray(vals, dtype=np.float64)


class EmotivDongleSource(StreamSource):
    """Lector nativo del EPOC+ (la lógica de CyKit, integrada en la app)."""

    display_name = "Emotiv EPOC+ (lector integrado)"

    def __init__(self, mode: str = "auto", serial: str | None = None,
                 sample_rate: float = 128.0, skip_gyro: bool = True) -> None:
        super().__init__(EPOC_CHANNELS, sample_rate)
        self._mode = mode                  # "auto" | "16bit" | "14bit"
        self._serial_override = serial
        self._skip_gyro = skip_gyro
        self._info = ""

    @property
    def info(self) -> str:
        """Descripción del dispositivo detectado y el modo elegido."""
        return self._info

    @staticmethod
    def _find_device() -> dict | None:
        devices = hid.enumerate()
        for d in devices:                                  # 1) por nombre de producto
            if (d.get("product_string") or "") in EMOTIV_PRODUCTS:
                return d
        for d in devices:                                  # 2) por vendor id conocido
            if d.get("vendor_id") in EMOTIV_VENDOR_IDS:
                return d
        return None

    @staticmethod
    def _read_block(device) -> bytes | None:
        report = device.read(33, timeout_ms=200)
        if not report:
            return None
        raw = bytes(report)
        # pywinusb/hidapi pueden incluir o no el byte de report-id inicial.
        block = raw[1:33] if len(raw) >= 33 else raw[:32]
        return block if len(block) >= 32 else None

    def _select_mode(self, device, serial: str) -> tuple[str, "AES"]:
        """Autodetecta 14 vs 16 bits puntuando frames con cada clave."""
        modes = ["16bit", "14bit"] if self._mode == "auto" else [self._mode]
        ciphers = {m: AES.new(build_key(serial, m), AES.MODE_ECB) for m in modes}
        if len(modes) == 1:
            return modes[0], ciphers[modes[0]]
        scores = {m: [] for m in modes}
        collected = 0
        while self._running.is_set() and collected < 40:
            block = self._read_block(device)
            if block is None:
                continue
            for m in modes:
                scores[m].append(frame_plausibility(convert_frame(ciphers[m].decrypt(block))))
            collected += 1
        best = max(modes, key=lambda m: (np.mean(scores[m]) if scores[m] else -1e18))
        return best, ciphers[best]

    def _run(self) -> None:
        if not emotiv_deps_available():
            raise RuntimeError("Faltan dependencias: pip install hidapi pycryptodome")
        info = self._find_device()
        if info is None:
            raise RuntimeError(
                "No se encontró el dongle Emotiv. ¿Está conectado y emparejado el receptor USB?"
            )
        device = hid.device()
        device.open_path(info["path"])
        try:
            serial = self._serial_override or info.get("serial_number") \
                or device.get_serial_number_string()
            mode, cipher = self._select_mode(device, serial)
            product = info.get("product_string") or "Emotiv"
            self._info = f"{product} · serie {serial} · modo {mode}"
            while self._running.is_set():
                block = self._read_block(device)
                if block is None:
                    continue
                frame = cipher.decrypt(block)
                if self._skip_gyro and frame[1] == 32:
                    continue
                self._emit(convert_frame(frame).reshape(self.n_channels, 1))
        finally:
            try:
                device.close()
            except Exception:  # noqa: BLE001
                pass

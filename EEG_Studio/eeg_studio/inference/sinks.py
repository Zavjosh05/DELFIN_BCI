"""Salidas de comandos hacia un controlador externo (brazo robótico, carrito…).

Cada clase detectada se traduce a un comando (texto) y se envía por el canal
elegido. ``pyserial`` es opcional: si no está, la salida por puerto serie queda
deshabilitada y el resto funciona igual.
"""
from __future__ import annotations

import socket

try:
    import serial  # pyserial
    _SERIAL_OK = True
except Exception:  # noqa: BLE001
    serial = None
    _SERIAL_OK = False


def serial_available() -> bool:
    return _SERIAL_OK


class CommandSink:
    """Interfaz de salida de comandos."""

    def send(self, command: str) -> None:  # pragma: no cover - interfaz
        raise NotImplementedError

    def close(self) -> None:
        pass


class LogSink(CommandSink):
    """Solo registra el comando (para pruebas o monitorización)."""

    def __init__(self) -> None:
        self.last: str | None = None
        self.history: list[str] = []

    def send(self, command: str) -> None:
        self.last = command
        self.history.append(command)


class UdpSink(CommandSink):
    """Envía el comando por UDP a ``host:port`` (un controlador que escuche ahí)."""

    def __init__(self, host: str, port: int) -> None:
        self._addr = (host, int(port))
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, command: str) -> None:
        self._sock.sendto((command + "\n").encode("utf-8"), self._addr)

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:  # noqa: BLE001
            pass


class SerialSink(CommandSink):
    """Envía el comando por puerto serie (p. ej. un Arduino que controla el motor)."""

    def __init__(self, port: str, baud: int = 9600) -> None:
        if not _SERIAL_OK:
            raise RuntimeError("pyserial no está instalado (pip install pyserial).")
        self._ser = serial.Serial(port, int(baud), timeout=0.1)

    def send(self, command: str) -> None:
        self._ser.write((command + "\n").encode("utf-8"))

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:  # noqa: BLE001
            pass


def make_sink(kind: str, **params) -> CommandSink:
    """Crea la salida indicada: ``"log"``, ``"udp"`` o ``"serial"``."""
    if kind == "udp":
        return UdpSink(params.get("host", "127.0.0.1"), params.get("port", 9001))
    if kind == "serial":
        return SerialSink(params.get("port", "COM3"), params.get("baud", 9600))
    return LogSink()

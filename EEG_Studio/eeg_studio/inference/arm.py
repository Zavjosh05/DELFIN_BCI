"""Control del brazo robótico **MaxArm** (Hiwonder + ESP32) por HTTP.

La ESP32 es un punto de acceso WiFi (``MaxArm_IPN`` / ``maxarm2024``) con un
servidor HTTP en ``http://192.168.4.1``:

* ``GET /cmd?id=<servo>&v=<-1..1>`` — mueve un servo a velocidad continua
  (``v=0`` lo detiene). Servos: **1**=Hombro (sube/baja), **3**=Codo,
  **4**=Rotación de la pinza. La base (id 2) está deshabilitada en el firmware.
* ``GET /pump?on=1|0`` — bomba de succión (agarrar / soltar).
* ``GET /reset`` — posición HOME.

Los 6 comandos del proyecto Delfin se mapean a acciones del brazo en
:data:`ARM_COMMANDS` (editable). Como ``/cmd`` es velocidad continua, un comando
discreto se traduce en un **pulso**: mover un ratito y luego detener.
"""
from __future__ import annotations

import threading
import time
import urllib.request

DEFAULT_HOST = "192.168.4.1"
DEFAULT_PORT = 80
DEFAULT_PULSE_MS = 400

# Comando (clase del clasificador) -> acción del brazo. Ajustable.
#   ("move", id_servo, v): pulso de movimiento del servo en la dirección v (±1)
#   ("pump", bool):        bomba de succión on/off
#   ("reset", None):       posición HOME
ARM_COMMANDS: dict[str, tuple] = {
    "arriba":    ("move", 1, +1.0),   # Hombro sube
    "abajo":     ("move", 1, -1.0),   # Hombro baja
    "derecha":   ("move", 4, +1.0),   # Rotación de la pinza (der.)
    "izquierda": ("move", 4, -1.0),   # Rotación de la pinza (izq.)
    "agarre":    ("pump", True),      # succión ON
    "soltar":    ("pump", False),     # succión OFF
}
ARM_COMMAND_NAMES = list(ARM_COMMANDS)


class ArmClient:
    """Cliente HTTP mínimo para el MaxArm (peticiones GET al firmware)."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout: float = 1.5) -> None:
        self.host = host or DEFAULT_HOST
        self.port = int(port)
        self.timeout = float(timeout)

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as resp:
            return resp.read()

    def move(self, servo_id: int, value: float) -> None:
        self._get(f"/cmd?id={int(servo_id)}&v={float(value):.3f}")

    def pump(self, on: bool) -> None:
        self._get(f"/pump?on={'1' if on else '0'}")

    def reset(self) -> None:
        self._get("/reset")

    def ping(self) -> bool:
        """True si el brazo responde al servidor HTTP."""
        try:
            self._get("/")
            return True
        except Exception:  # noqa: BLE001
            return False

    def pulse(self, servo_id: int, value: float, duration_ms: int) -> None:
        """Mueve un servo en ``value`` durante ``duration_ms`` y luego lo detiene."""
        self.move(servo_id, value)
        time.sleep(max(0, int(duration_ms)) / 1000.0)
        self.move(servo_id, 0.0)

    def execute(self, command: str, pulse_ms: int = DEFAULT_PULSE_MS) -> bool:
        """Ejecuta un comando (nombre de clase). Devuelve si se reconoció."""
        spec = ARM_COMMANDS.get(command)
        if spec is None:
            return False
        kind = spec[0]
        if kind == "move":
            self.pulse(spec[1], spec[2], pulse_ms)
        elif kind == "pump":
            self.pump(spec[1])
        elif kind == "reset":
            self.reset()
        return True

    def execute_async(self, command: str, pulse_ms: int = DEFAULT_PULSE_MS,
                      on_error=None) -> None:
        """Ejecuta el comando en un hilo (no bloquea la interfaz ni el bucle)."""
        def _run():
            try:
                self.execute(command, pulse_ms)
            except Exception as exc:  # noqa: BLE001
                if on_error is not None:
                    on_error(exc)
        threading.Thread(target=_run, daemon=True).start()


class ArmHttpSink:
    """Salida del modo de control: envía cada clase detectada al brazo por HTTP.

    No bloquea el bucle de inferencia (cada comando se ejecuta en un hilo)."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 pulse_ms: int = DEFAULT_PULSE_MS) -> None:
        self._client = ArmClient(host, port)
        self._pulse = int(pulse_ms)
        self.last: str | None = None
        self.history: list[str] = []

    def send(self, command: str) -> None:
        self.last = command
        self.history.append(command)
        self._client.execute_async(command, self._pulse)

    def close(self) -> None:
        pass

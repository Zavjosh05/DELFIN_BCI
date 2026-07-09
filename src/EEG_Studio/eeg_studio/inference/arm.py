"""Control del brazo robótico **MaxArm** (Hiwonder + ESP32) por HTTP.

La ESP32 es un punto de acceso WiFi (``MaxArm_IPN`` / ``maxarm2024``) con un
servidor HTTP en ``http://192.168.4.1``. Según el firmware de pruebas que sí
funciona (``Brazo/Codigo de pruebas.txt``):

* ``GET /cmd?<id>=<-1..1>`` — mueve el servo ``<id>`` a **velocidad continua**
  (0 lo detiene). Se pueden combinar varios: ``/cmd?1=0.5&3=-0.2``. Servos:
  **1**=Hombro (sube/baja), **3**=Codo, **4**=Rotación de la pinza. La **Base**
  (id 2) está **deshabilitada** en el firmware (sin servicio).
* ``GET /pump?on=1|0`` — bomba de succión: ``1`` enciende (agarrar), ``0``
  apaga (soltar).
* ``GET /reset`` — posición HOME.

Los 6 comandos del proyecto Delfin se mapean a acciones del brazo en
:data:`ARM_COMMANDS` (editable). Como ``/cmd`` es velocidad continua, un comando
discreto se traduce en un **pulso**: mover un ratito y luego detener.

**Izquierda/derecha** irían a la base giratoria (servo 2), que está sin servicio,
así que quedan como ``"disabled"``: la interfaz muestra sus botones pero
inhabilitados y el clasificador simplemente los ignora.
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
#   ("disabled", None):    comando reconocido pero sin acción (hardware sin servicio)
ARM_COMMANDS: dict[str, tuple] = {
    "arriba":    ("move", 1, +1.0),   # Hombro sube
    "abajo":     ("move", 1, -1.0),   # Hombro baja
    "izquierda": ("disabled", None),  # Base giratoria (servo 2): sin servicio
    "derecha":   ("disabled", None),  # Base giratoria (servo 2): sin servicio
    "agarre":    ("pump", True),      # bomba de succión ON (encender)
    "soltar":    ("pump", False),     # bomba de succión OFF (apagar)
}
ARM_COMMAND_NAMES = list(ARM_COMMANDS)
# Comandos reconocidos pero sin acción (p. ej. servo sin servicio en el firmware).
ARM_DISABLED = frozenset(
    name for name, spec in ARM_COMMANDS.items() if spec[0] == "disabled")


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
        # El firmware espera la query "<id>=<valor>" (no "id=&v="): /cmd?1=1.000
        self._get(f"/cmd?{int(servo_id)}={float(value):.3f}")

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
        else:  # "disabled" u otro: reconocido pero sin acción
            return False
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

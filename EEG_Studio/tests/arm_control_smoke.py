"""Control del brazo MaxArm por HTTP (cliente + mapeo + sink), sin hardware.

Levanta un servidor HTTP local que captura las peticiones, así comprobamos que se
envían las URLs correctas sin necesidad del ESP32.
"""
from __future__ import annotations

import http.server
import socketserver
import sys
import threading
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from eeg_studio.inference import make_sink
from eeg_studio.inference.arm import (
    ARM_COMMANDS,
    ARM_DISABLED,
    ArmClient,
    ArmHttpSink,
)

_LOG: list[str] = []


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        _LOG.append(self.path)
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, *a):  # silencia el log del servidor
        pass


def _wait_for(pred, timeout=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.02)
    return False


def main() -> int:
    srv = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = ArmClient("127.0.0.1", port, timeout=2.0)

    print("[1] ping responde al servidor")
    assert client.ping() is True

    print("[2] move/pump/reset usan el formato del firmware (/cmd?<id>=<v>)")
    _LOG.clear()
    client.move(1, 1.0); client.pump(True); client.reset()
    assert "/cmd?1=1.000" in _LOG, _LOG
    assert "/pump?on=1" in _LOG and "/reset" in _LOG, _LOG

    print("[3] execute('arriba') = pulso (mover Hombro +1, luego 0)")
    _LOG.clear()
    client.execute("arriba", pulse_ms=30)
    cmds = [p for p in _LOG if p.startswith("/cmd")]
    assert cmds == ["/cmd?1=1.000", "/cmd?1=0.000"], cmds

    print("[4] execute('agarre') = bomba ON; 'soltar' = bomba OFF")
    _LOG.clear(); client.execute("agarre")
    assert _LOG == ["/pump?on=1"], _LOG
    _LOG.clear(); client.execute("soltar")
    assert _LOG == ["/pump?on=0"], _LOG

    print("[5] izquierda/derecha están deshabilitados (no envían nada)")
    assert ARM_DISABLED == {"izquierda", "derecha"}, ARM_DISABLED
    for cmd in ("izquierda", "derecha"):
        _LOG.clear()
        assert client.execute(cmd) is False, cmd
        assert _LOG == [], (cmd, _LOG)

    print("[6] make_sink('arm') → ArmHttpSink; send() envía en segundo plano")
    sink = make_sink("arm", host="127.0.0.1", port=port, pulse_ms=30)
    assert isinstance(sink, ArmHttpSink)
    _LOG.clear()
    sink.send("abajo")
    assert sink.history == ["abajo"]
    assert _wait_for(lambda: len([p for p in _LOG if p.startswith("/cmd")]) >= 2), _LOG
    cmds = [p for p in _LOG if p.startswith("/cmd")]
    assert cmds[:2] == ["/cmd?1=-1.000", "/cmd?1=0.000"], cmds

    print("[7] Los 6 comandos de Delfin están mapeados")
    assert set(ARM_COMMANDS) == {"arriba", "abajo", "izquierda", "derecha",
                                 "agarre", "soltar"}, set(ARM_COMMANDS)

    print("[8] Comando desconocido no rompe (execute devuelve False)")
    assert client.execute("no_existe") is False

    srv.shutdown()
    print("\nCONTROL DEL BRAZO MAXARM (HTTP) OK ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

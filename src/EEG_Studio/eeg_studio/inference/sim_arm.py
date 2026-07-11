"""Brazo robótico **simulado** (perfil de control sin hardware).

Extraído/adaptado del proyecto de referencia ``Proyecto_RNN`` (módulos de
**construcción**, **cinemática directa** y **control** del brazo; se omiten la
cinemática inversa y las series temporales). Es un brazo 4DOF tipo SCARA:

* ``q0`` base (yaw, giro horizontal en XY),
* ``q1`` hombro (elevación en el plano vertical),
* ``q2`` codo, ``q3`` muñeca.

Se controla con los **mismos 6 comandos** del brazo real (``inference/arm.py``):
arriba/abajo mueven el hombro, izquierda/derecha giran la base (aquí sí funciona,
al contrario que en el MaxArm), y agarre/soltar abren/cierran la pinza. Cada
comando es un «empujón» (jog) del ángulo, respetando los límites articulares y el
piso (z ≥ 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def _rot(axis, theta: float) -> np.ndarray:
    """Matriz de rotación (Rodrigues) con la **convención histórica** del
    proyecto: ``R(+y, +θ)`` lleva ``+x`` hacia ``+z`` (un ángulo positivo
    «levanta» el brazo). Equivale a negar el seno respecto a la regla estándar."""
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    c = math.cos(theta)
    s = -math.sin(theta)          # seno negado (convención histórica)
    C = 1.0 - c
    x, y, z = a
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])

# Comando (clase del clasificador) -> acción del brazo simulado.
#   ("joint", indice, signo): empuja el ángulo del joint en esa dirección
#   ("gripper", bool):        abre (False) / cierra (True) la pinza
#
# Marco de referencia de izquierda/derecha: de pie en la base y mirando a lo largo
# del brazo (dirección +x, la que apunta en HOME), la IZQUIERDA es +y y la DERECHA
# es -y (regla de la mano derecha con +z hacia arriba). OJO: por la convención
# histórica de _rot() (seno negado), un yaw POSITIVO gira el brazo hacia -y (la
# derecha) y uno NEGATIVO hacia +y (la izquierda); por eso 'izquierda' usa signo
# -1 y 'derecha' signo +1 (invertir estos signos vuelve a cruzar los controles).
SIM_ARM_COMMANDS: dict[str, tuple] = {
    "arriba":    ("joint", 1, +1.0),   # hombro sube
    "abajo":     ("joint", 1, -1.0),   # hombro baja
    "izquierda": ("joint", 0, -1.0),   # base gira a la izquierda (efector hacia +y)
    "derecha":   ("joint", 0, +1.0),   # base gira a la derecha  (efector hacia -y)
    "agarre":    ("gripper", True),    # cerrar pinza
    "soltar":    ("gripper", False),   # abrir pinza
}
SIM_ARM_COMMAND_NAMES = list(SIM_ARM_COMMANDS)

DEFAULT_STEP = 0.16          # rad por comando (~9°)


@dataclass
class JointSpec:
    """Especificación de un joint revoluto (construcción del brazo)."""
    name: str = "joint"
    axis: tuple = (0.0, 0.0, 1.0)          # eje de rotación local
    link_offset: tuple = (0.0, 0.0, 0.0)   # eslabón que sigue al joint
    q_min: float = -math.pi
    q_max: float = +math.pi
    mass: float = 0.0                      # masa del eslabón (informativa)


@dataclass
class ArmSpec:
    """Especificación completa del brazo: joints + pose home + piso."""
    name: str = "unnamed"
    description: str = ""
    joints: list[JointSpec] = field(default_factory=list)
    q_home: tuple = ()
    floor_z: float = 0.0

    @property
    def n_joints(self) -> int:
        return len(self.joints)


def make_default_arm_spec() -> ArmSpec:
    """Brazo 4DOF por defecto: base (yaw) + hombro + codo + muñeca (pitch),
    eslabones de 0.18 m."""
    L = 0.18
    return ArmSpec(
        name="Default 4DOF",
        description="Base yaw + 3 joints en plano vertical (hombro, codo, muñeca).",
        joints=[
            JointSpec("base_yaw", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0),
                      -math.pi, math.pi, 0.0),
            JointSpec("shoulder", (0.0, 1.0, 0.0), (L, 0.0, 0.0),
                      -0.1, math.pi * 0.9, 2.5),
            JointSpec("elbow", (0.0, 1.0, 0.0), (L, 0.0, 0.0),
                      -math.pi * 0.9, math.pi * 0.9, 1.8),
            JointSpec("wrist", (0.0, 1.0, 0.0), (L, 0.0, 0.0),
                      -math.pi * 0.9, math.pi * 0.9, 1.0),
        ],
        q_home=(0.0, 0.6, -0.4, 0.0),
        floor_z=0.0,
    )


def validate_arm_spec(spec: ArmSpec) -> tuple[bool, list[str]]:
    """Valida una ``ArmSpec``. Devuelve ``(ok, errores)``."""
    errors: list[str] = []
    if spec.n_joints < 1:
        errors.append("El brazo necesita al menos 1 joint.")
    if len(spec.q_home) != spec.n_joints:
        errors.append(f"q_home tiene {len(spec.q_home)} valores pero hay "
                      f"{spec.n_joints} joints.")
    for i, j in enumerate(spec.joints):
        if j.q_min >= j.q_max:
            errors.append(f"Joint {i} ({j.name}): q_min >= q_max.")
        elif i < len(spec.q_home) and not (j.q_min <= spec.q_home[i] <= j.q_max):
            errors.append(f"Joint {i} ({j.name}): q_home fuera de límites.")
    return (not errors), errors


class SimulatedArm:
    """Brazo 4DOF simulado: construcción (desde ``ArmSpec``), cinemática directa
    y control por comandos discretos (jog de articulaciones + pinza)."""

    def __init__(self, spec: ArmSpec | None = None, step: float = DEFAULT_STEP) -> None:
        self.step = float(step)
        self.gripper_closed = False
        self._floor_margin = 1e-3
        self.apply_spec(spec or make_default_arm_spec())

    def apply_spec(self, spec: ArmSpec) -> None:
        """(Re)construye el brazo desde una ``ArmSpec`` y lo lleva a HOME.

        Permite «construir un brazo desde cero»: cambiar joints/límites/eslabones
        y refrescar toda la cinemática."""
        self.spec = spec
        offs = np.array([j.link_offset for j in spec.joints], dtype=float)
        self._link_lengths = np.linalg.norm(offs, axis=1) if len(offs) else np.zeros(0)
        self.reach = float(np.sum(self._link_lengths)) or 0.54
        self.q_min = np.array([j.q_min for j in spec.joints], dtype=float)
        self.q_max = np.array([j.q_max for j in spec.joints], dtype=float)
        self.q_home = np.array(spec.q_home, dtype=float)
        if self.q_home.size != spec.n_joints:
            self.q_home = np.zeros(spec.n_joints)
        self.q = self.q_home.copy()
        self.FLOOR = spec.floor_z

    @property
    def joint_names(self) -> list[str]:
        pretty = {"base_yaw": "q₁ Base (yaw)", "shoulder": "q₂ Hombro",
                  "elbow": "q₃ Codo", "wrist": "q₄ Muñeca"}
        return [pretty.get(j.name, f"q{i + 1} {j.name}")
                for i, j in enumerate(self.spec.joints)]

    def set_q(self, idx: int, value: float) -> None:
        """Fija el ángulo de un joint (control manual por slider), acotado a sus
        límites articulares."""
        if 0 <= idx < self.q.size:
            self.q[idx] = float(np.clip(value, self.q_min[idx], self.q_max[idx]))

    # --- Cinemática directa (cadena general de transformaciones) ----------
    def fk(self, q=None) -> np.ndarray:
        """Puntos de la cadena (base + final de cada eslabón) como array (M, 3).

        Cadena genérica: soporta cualquier nº de joints revolutos con ejes y
        eslabones arbitrarios (para el constructor). Los eslabones de longitud
        cero (p. ej. la base yaw) no añaden un punto propio."""
        q = self.q if q is None else np.asarray(q)
        R = np.eye(3)
        p = np.zeros(3)
        pts = [p.copy()]
        n = len(self.spec.joints)
        for i, j in enumerate(self.spec.joints):
            ang = float(q[i]) if i < len(q) else 0.0
            R = R @ _rot(j.axis, ang)
            off = np.asarray(j.link_offset, dtype=float)
            p = p + R @ off
            if np.linalg.norm(off) > 1e-9 or i == n - 1:
                pts.append(p.copy())
        return np.array(pts)

    def ee(self, q=None) -> np.ndarray:
        """Posición del efector (end-effector)."""
        return self.fk(q)[-1]

    def check_floor(self, q=None) -> bool:
        """True si ninguna articulación baja del piso (z ≥ 0)."""
        pts = self.fk(q)
        return bool(np.all(pts[:, 2] >= self.FLOOR - self._floor_margin))

    # --- Control por comandos ---------------------------------------------
    def execute(self, command: str, step: float | None = None) -> bool:
        """Aplica un comando (nombre de clase). Devuelve si se reconoció.

        Los movimientos respetan los límites articulares y el piso: si el paso
        violaría el piso, no se aplica."""
        spec = SIM_ARM_COMMANDS.get(command)
        if spec is None:
            return False
        kind = spec[0]
        if kind == "gripper":
            self.gripper_closed = bool(spec[1])
            return True
        if kind == "joint":
            idx, sign = spec[1], spec[2]
            delta = sign * (self.step if step is None else float(step))
            q_try = self.q.copy()
            q_try[idx] = float(np.clip(q_try[idx] + delta,
                                       self.q_min[idx], self.q_max[idx]))
            if self.check_floor(q_try):
                self.q = q_try
            return True
        return False

    def reset(self) -> None:
        """Vuelve a la pose inicial (HOME) y abre la pinza."""
        self.q = self.q_home.copy()
        self.gripper_closed = False

    # --- Control interactivo desde las vistas 2D (clic para apuntar) -------
    def _pitch_indices(self) -> list[int]:
        """Joints que mueven el brazo en el plano vertical (eje ≈ y): hombro,
        codo, muñeca. La base (eje z) se excluye."""
        return [i for i, j in enumerate(self.spec.joints)
                if abs(np.asarray(j.axis, dtype=float)[2]) < 0.5]

    def aim_base_to(self, x: float, y: float) -> None:
        """Gira SOLO la base (yaw) para apuntar el brazo hacia ``(x, y)`` del
        plano horizontal (clic en la vista superior). No mueve el resto."""
        if abs(x) < 1e-9 and abs(y) < 1e-9:
            return
        idx = next((i for i, j in enumerate(self.spec.joints)
                    if abs(np.asarray(j.axis, dtype=float)[2]) > 0.5), None)
        if idx is None:
            return
        self.set_q(idx, -math.atan2(y, x))     # convención histórica (seno negado)

    def aim_planar(self, radius: float, height: float, iters: int = 80) -> bool:
        """Acerca el efector al objetivo ``(radio, altura)`` del plano vertical
        (clic en la vista lateral) moviendo hombro/codo/muñeca — IK aproximada
        por descenso de coordenadas, respetando límites articulares y el piso.
        La base (yaw) no se toca. Devuelve ``True`` si mejoró la distancia."""
        idxs = self._pitch_indices()
        if not idxs:
            return False

        def cost(q) -> float:
            ee = self.ee(q)
            return (math.hypot(ee[0], ee[1]) - radius) ** 2 + (ee[2] - height) ** 2

        q = self.q.copy()
        cur = start = cost(q)
        step = 0.15
        for _ in range(int(iters)):
            improved = False
            for i in idxs:
                for s in (step, -step):
                    q_try = q.copy()
                    q_try[i] = float(np.clip(q_try[i] + s, self.q_min[i], self.q_max[i]))
                    if not self.check_floor(q_try):
                        continue
                    c = cost(q_try)
                    if c < cur - 1e-7:
                        q, cur, improved = q_try, c, True
            if not improved:
                step *= 0.5
                if step < 1e-3:
                    break
        self.q = q
        return cur < start - 1e-9


class SimArmSink:
    """Salida del modo de control hacia el brazo **simulado**.

    Cada clase detectada mueve el brazo simulado (en memoria). ``on_change`` se
    llama tras cada comando para refrescar la vista. El bucle de inferencia corre
    en el hilo de la GUI, así que actualizar la vista aquí es seguro."""

    def __init__(self, arm: SimulatedArm, on_change=None) -> None:
        self.arm = arm
        self._on_change = on_change
        self.last: str | None = None
        self.history: list[str] = []

    def send(self, command: str) -> None:
        self.last = command
        self.history.append(command)
        self.arm.execute(command)
        if self._on_change is not None:
            self._on_change()

    def close(self) -> None:
        pass

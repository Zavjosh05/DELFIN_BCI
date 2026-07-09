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

# Comando (clase del clasificador) -> acción del brazo simulado.
#   ("joint", indice, signo): empuja el ángulo del joint en esa dirección
#   ("gripper", bool):        abre (False) / cierra (True) la pinza
SIM_ARM_COMMANDS: dict[str, tuple] = {
    "arriba":    ("joint", 1, +1.0),   # hombro sube
    "abajo":     ("joint", 1, -1.0),   # hombro baja
    "izquierda": ("joint", 0, +1.0),   # base gira a la izquierda (yaw +)
    "derecha":   ("joint", 0, -1.0),   # base gira a la derecha  (yaw -)
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
                      -math.pi, math.pi),
            JointSpec("shoulder", (0.0, 1.0, 0.0), (L, 0.0, 0.0),
                      -0.1, math.pi * 0.9),
            JointSpec("elbow", (0.0, 1.0, 0.0), (L, 0.0, 0.0),
                      -math.pi * 0.9, math.pi * 0.9),
            JointSpec("wrist", (0.0, 1.0, 0.0), (L, 0.0, 0.0),
                      -math.pi * 0.9, math.pi * 0.9),
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
        self.spec = spec or make_default_arm_spec()
        self.step = float(step)
        offs = np.array([j.link_offset for j in self.spec.joints], dtype=float)
        self._link_lengths = np.linalg.norm(offs, axis=1)
        self.reach = float(np.sum(self._link_lengths)) or 0.54
        # Longitudes de los 3 eslabones móviles (hombro, codo, muñeca).
        self.L2, self.L3, self.L4 = (float(self._link_lengths[i])
                                     if i < len(self._link_lengths) else 0.18
                                     for i in (1, 2, 3))
        self.q_min = np.array([j.q_min for j in self.spec.joints], dtype=float)
        self.q_max = np.array([j.q_max for j in self.spec.joints], dtype=float)
        self.q_home = np.array(self.spec.q_home, dtype=float)
        if self.q_home.size != self.spec.n_joints:
            self.q_home = np.zeros(self.spec.n_joints)
        self.q = self.q_home.copy()
        self.gripper_closed = False
        self.FLOOR = self.spec.floor_z
        self._floor_margin = 1e-3

    # --- Cinemática directa (forma cerrada 4DOF) --------------------------
    def fk(self, q=None) -> np.ndarray:
        """Puntos ``(base, hombro, codo, efector)`` como array (4, 3)."""
        q = self.q if q is None else np.asarray(q)
        phi = q[0]
        cx, cy = math.cos(phi), math.sin(phi)
        t2 = q[1]
        t23 = t2 + q[2]
        t234 = t23 + q[3]
        x1 = self.L2 * math.cos(t2);  z1 = self.L2 * math.sin(t2)
        x2 = x1 + self.L3 * math.cos(t23); z2 = z1 + self.L3 * math.sin(t23)
        x3 = x2 + self.L4 * math.cos(t234); z3 = z2 + self.L4 * math.sin(t234)
        return np.array([
            [0.0, 0.0, 0.0],
            [x1 * cx, x1 * cy, z1],
            [x2 * cx, x2 * cy, z2],
            [x3 * cx, x3 * cy, z3],
        ])

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

"""Clasificación en línea de la señal en vivo y salida a un controlador.

Permite, con un modelo ya entrenado, clasificar ventanas de la señal entrante
(tratadas por el mismo preprocesamiento del proyecto) y enviar la clase detectada
a un controlador externo (brazo robótico, carrito…) por UDP, puerto serie o
simplemente registrarla.
"""
from .arm import (
    ARM_COMMAND_NAMES,
    ARM_COMMANDS,
    ARM_DISABLED,
    ArmClient,
    ArmHttpSink,
)
from .online import PredictionSmoother, classify_window
from .sim_arm import (
    SIM_ARM_COMMAND_NAMES,
    SIM_ARM_COMMANDS,
    ArmSpec,
    JointSpec,
    SimArmSink,
    SimulatedArm,
    make_default_arm_spec,
)
from .sinks import (
    CommandSink,
    LogSink,
    SerialSink,
    UdpSink,
    make_sink,
    serial_available,
)

__all__ = [
    "classify_window",
    "PredictionSmoother",
    "CommandSink",
    "LogSink",
    "UdpSink",
    "SerialSink",
    "make_sink",
    "serial_available",
    "ArmClient",
    "ArmHttpSink",
    "ARM_COMMANDS",
    "ARM_COMMAND_NAMES",
    "ARM_DISABLED",
    "SimulatedArm",
    "SimArmSink",
    "SIM_ARM_COMMANDS",
    "SIM_ARM_COMMAND_NAMES",
    "ArmSpec",
    "JointSpec",
    "make_default_arm_spec",
]

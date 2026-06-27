"""Control de cambios no destructivo.

Toda modificación que el usuario hace en un proyecto (añadir un filtro, definir
un segmento, reetiquetar, renombrar canales...) se representa como un
:class:`EditCommand` basado en *instantáneas* (snapshots) de una sección del
estado del proyecto. Esto da:

* **Undo/redo** completo dentro de la sesión.
* Una **bitácora de auditoría** persistente (``changelog.json``) con el historial.
* Garantía de que el CSV original nunca se toca: los cambios viven en el estado
  del proyecto, en archivos locales.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field


@dataclass
class EditCommand:
    """Cambio sobre una sección del estado del proyecto.

    section: clave del estado afectada (p.ej. ``"pipeline"``, ``"segments"``).
    before / after: instantáneas serializables del valor antes y después.
    description: texto legible para la bitácora y la interfaz.
    timestamp: epoch en segundos.
    """

    section: str
    before: object
    after: object
    description: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "section": self.section,
            "before": self.before,
            "after": self.after,
            "description": self.description,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EditCommand":
        return cls(
            section=d["section"],
            before=d.get("before"),
            after=d.get("after"),
            description=d.get("description", ""),
            timestamp=d.get("timestamp", time.time()),
        )


class ChangeLog:
    """Pila de comandos con undo/redo y registro histórico completo."""

    def __init__(self) -> None:
        self._undo: list[EditCommand] = []
        self._redo: list[EditCommand] = []
        self._history: list[dict] = []  # auditoría persistente (no se vacía con undo)

    # --- API principal ----------------------------------------------------
    def push(self, cmd: EditCommand) -> None:
        """Registra un comando ya aplicado al estado."""
        self._undo.append(cmd)
        self._redo.clear()
        self._history.append({**cmd.to_dict(), "event": "do"})

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> EditCommand | None:
        """Devuelve el comando a deshacer (el llamador restaura ``before``)."""
        if not self._undo:
            return None
        cmd = self._undo.pop()
        self._redo.append(cmd)
        self._history.append({**cmd.to_dict(), "event": "undo"})
        return cmd

    def redo(self) -> EditCommand | None:
        """Devuelve el comando a rehacer (el llamador restaura ``after``)."""
        if not self._redo:
            return None
        cmd = self._redo.pop()
        self._undo.append(cmd)
        self._history.append({**cmd.to_dict(), "event": "redo"})
        return cmd

    # --- Navegación por la línea de tiempo --------------------------------
    def applied_count(self) -> int:
        """Nº de comandos aplicados (posición actual en la línea de tiempo)."""
        return len(self._undo)

    def timeline(self) -> list[dict]:
        """Comandos en orden cronológico: los aplicados y los rehacibles.

        Cada entrada: ``{description, timestamp, section, applied}``. El nº de
        entradas con ``applied=True`` coincide con :meth:`applied_count`.
        """
        out = [
            {"description": c.description, "timestamp": c.timestamp,
             "section": c.section, "applied": True}
            for c in self._undo
        ]
        out += [
            {"description": c.description, "timestamp": c.timestamp,
             "section": c.section, "applied": False}
            for c in reversed(self._redo)
        ]
        return out

    # --- Auditoría / persistencia ----------------------------------------
    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def descriptions(self) -> list[str]:
        return [c.description for c in self._undo]

    def to_dict(self) -> dict:
        return {
            "history": self._history,
            "undo": [c.to_dict() for c in self._undo],
            "redo": [c.to_dict() for c in self._redo],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChangeLog":
        log = cls()
        log._history = list(d.get("history", []))
        log._undo = [EditCommand.from_dict(x) for x in d.get("undo", [])]
        log._redo = [EditCommand.from_dict(x) for x in d.get("redo", [])]
        return log


def snapshot(value: object) -> object:
    """Copia profunda serializable de un valor de estado."""
    return copy.deepcopy(value)

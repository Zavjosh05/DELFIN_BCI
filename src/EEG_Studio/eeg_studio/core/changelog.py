"""Control de cambios no destructivo, en **árbol**.

Toda modificación que el usuario hace en un proyecto (añadir un filtro, definir
un segmento, reetiquetar, renombrar canales...) se representa como un
:class:`EditCommand` basado en *instantáneas* (snapshots) de una sección del
estado del proyecto. El historial no es una simple línea: es un **árbol** de
estados. Esto da:

* **Undo/redo** dentro de la sesión (moverse por la rama actual).
* **Ramas**: si vuelves a un estado anterior y haces un cambio nuevo, se crea
  una rama en lugar de borrar lo que habías hecho después. Nada se pierde.
* Una **bitácora de auditoría** persistente (``changelog.json``).
* Garantía de que el CSV original nunca se toca: los cambios viven en el estado
  del proyecto, en archivos locales.

Cada **nodo** es un estado; la **arista** hacia su padre guarda el comando
(``before``/``after``) que los separa. Navegar entre dos nodos cualesquiera es
deshacer hasta el ancestro común y rehacer hacia el destino.
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
    """Árbol de estados con undo/redo, ramas y registro histórico completo."""

    ROOT_ID = 0

    def __init__(self) -> None:
        # Cada nodo: {id, parent, children[], cmd|None, description, timestamp}.
        self._nodes: dict[int, dict] = {
            self.ROOT_ID: {"id": self.ROOT_ID, "parent": None, "children": [],
                           "cmd": None, "description": "Estado inicial",
                           "timestamp": time.time()},
        }
        self._current = self.ROOT_ID
        self._next = self.ROOT_ID + 1
        self._history: list[dict] = []   # auditoría persistente (nunca se vacía)

    # --- API principal ----------------------------------------------------
    def push(self, cmd: EditCommand) -> int:
        """Registra un comando ya aplicado creando un nodo hijo del actual.

        Si el nodo actual ya tenía hijos (habías vuelto atrás), esto crea una
        **rama nueva** sin borrar las anteriores.
        """
        nid = self._next
        self._next += 1
        self._nodes[nid] = {"id": nid, "parent": self._current, "children": [],
                            "cmd": cmd, "description": cmd.description,
                            "timestamp": cmd.timestamp}
        self._nodes[self._current]["children"].append(nid)
        self._current = nid
        self._history.append({**cmd.to_dict(), "event": "do"})
        return nid

    def can_undo(self) -> bool:
        return self._nodes[self._current]["parent"] is not None

    def can_redo(self) -> bool:
        return bool(self._nodes[self._current]["children"])

    def undo(self) -> EditCommand | None:
        """Sube al nodo padre. Devuelve el comando (el llamador restaura ``before``)."""
        node = self._nodes[self._current]
        if node["parent"] is None:
            return None
        cmd = node["cmd"]
        self._current = node["parent"]
        self._history.append({**cmd.to_dict(), "event": "undo"})
        return cmd

    def redo(self) -> EditCommand | None:
        """Baja a la rama más reciente. Devuelve el comando (restaura ``after``)."""
        children = self._nodes[self._current]["children"]
        if not children:
            return None
        nid = children[-1]                          # rama creada más recientemente
        cmd = self._nodes[nid]["cmd"]
        self._current = nid
        self._history.append({**cmd.to_dict(), "event": "redo"})
        return cmd

    # --- Navegación por el árbol ------------------------------------------
    @property
    def current_id(self) -> int:
        return self._current

    def _ancestors(self, nid: int) -> list[int]:
        """Camino raíz→``nid`` (incluidos ambos extremos)."""
        path = []
        n: int | None = nid
        while n is not None:
            path.append(n)
            n = self._nodes[n]["parent"]
        path.reverse()
        return path

    def steps_to(self, target: int) -> list[tuple[str, EditCommand]]:
        """Pasos ``("undo"|"redo", cmd)`` para ir del nodo actual a ``target``.

        Deshace hasta el ancestro común más bajo y rehace hacia el destino.
        No mueve el nodo actual (eso lo hace :meth:`set_current`).
        """
        if target not in self._nodes or target == self._current:
            return []
        cur_path = self._ancestors(self._current)
        tgt_path = self._ancestors(target)
        cur_set = set(cur_path)
        lca = self.ROOT_ID
        for n in tgt_path:                          # ancestro común más profundo
            if n in cur_set:
                lca = n
        steps: list[tuple[str, EditCommand]] = []
        n = self._current                           # deshacer hasta el LCA
        while n != lca:
            steps.append(("undo", self._nodes[n]["cmd"]))
            n = self._nodes[n]["parent"]
        down: list[tuple[str, EditCommand]] = []    # rehacer del LCA al destino
        n = target
        while n != lca:
            down.append(("redo", self._nodes[n]["cmd"]))
            n = self._nodes[n]["parent"]
        down.reverse()
        steps.extend(down)
        return steps

    def set_current(self, target: int) -> None:
        if target in self._nodes:
            if target != self._current:
                self._history.append({"event": "goto", "target": target,
                                      "timestamp": time.time()})
            self._current = target

    def applied_count(self) -> int:
        """Profundidad del nodo actual (nº de cambios aplicados en la rama actual)."""
        return len(self._ancestors(self._current)) - 1

    def timeline(self) -> list[dict]:
        """Comandos de la **rama actual**: aplicados (ancestros) + rehacibles.

        En un historial lineal coincide con la lista de siempre. Cada entrada:
        ``{description, timestamp, section, applied}``.
        """
        out = []
        for nid in self._ancestors(self._current)[1:]:      # aplicados (sin raíz)
            c = self._nodes[nid]["cmd"]
            out.append({"description": c.description, "timestamp": c.timestamp,
                        "section": c.section, "applied": True})
        n = self._current                                    # rehacibles (rama reciente)
        while self._nodes[n]["children"]:
            n = self._nodes[n]["children"][-1]
            c = self._nodes[n]["cmd"]
            out.append({"description": c.description, "timestamp": c.timestamp,
                        "section": c.section, "applied": False})
        return out

    def nodes(self) -> list[dict]:
        """Árbol completo en orden (DFS) para pintarlo con sangría.

        Cada entrada: ``{id, depth, description, section, timestamp, is_current,
        on_path, is_root, n_children}``. ``on_path`` = está en la rama actual
        (ancestro o el propio nodo actual).
        """
        on_path = set(self._ancestors(self._current))
        out: list[dict] = []

        def walk(nid: int, depth: int) -> None:
            node = self._nodes[nid]
            out.append({
                "id": nid, "depth": depth,
                "description": node["description"],
                "section": node["cmd"].section if node["cmd"] else "",
                "timestamp": node["timestamp"],
                "is_current": nid == self._current,
                "on_path": nid in on_path,
                "is_root": node["parent"] is None,
                "n_children": len(node["children"]),
            })
            for c in node["children"]:
                walk(c, depth + 1)

        walk(self.ROOT_ID, 0)
        return out

    # --- Auditoría / persistencia ----------------------------------------
    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def descriptions(self) -> list[str]:
        """Descripciones de los comandos aplicados en la rama actual."""
        return [self._nodes[n]["cmd"].description for n in self._ancestors(self._current)[1:]]

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "current": self._current,
            "next": self._next,
            "nodes": {
                str(nid): {
                    "parent": n["parent"],
                    "children": list(n["children"]),
                    "cmd": n["cmd"].to_dict() if n["cmd"] else None,
                    "description": n["description"],
                    "timestamp": n["timestamp"],
                }
                for nid, n in self._nodes.items()
            },
            "history": self._history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChangeLog":
        log = cls()
        log._history = list(d.get("history", []))
        if "nodes" in d:                            # formato árbol (v2)
            log._nodes = {}
            for k, n in d["nodes"].items():
                nid = int(k)
                cmd = EditCommand.from_dict(n["cmd"]) if n.get("cmd") else None
                log._nodes[nid] = {
                    "id": nid, "parent": n.get("parent"),
                    "children": [int(c) for c in n.get("children", [])],
                    "cmd": cmd,
                    "description": n.get("description", cmd.description if cmd else "Estado inicial"),
                    "timestamp": n.get("timestamp", time.time()),
                }
            log._current = int(d.get("current", cls.ROOT_ID))
            log._next = int(d.get("next", max(log._nodes) + 1 if log._nodes else 1))
            if cls.ROOT_ID not in log._nodes:       # salvaguarda
                log.__init__()
        else:                                       # formato lineal antiguo (v1)
            chain = [EditCommand.from_dict(x) for x in d.get("undo", [])]
            chain += [EditCommand.from_dict(x) for x in reversed(d.get("redo", []))]
            applied = len(d.get("undo", []))
            prev = cls.ROOT_ID
            for i, cmd in enumerate(chain):
                nid = log._next
                log._next += 1
                log._nodes[nid] = {"id": nid, "parent": prev, "children": [],
                                   "cmd": cmd, "description": cmd.description,
                                   "timestamp": cmd.timestamp}
                log._nodes[prev]["children"].append(nid)
                prev = nid
                if i + 1 == applied:
                    log._current = nid
            if applied == 0:
                log._current = cls.ROOT_ID
        return log


def snapshot(value: object) -> object:
    """Copia profunda serializable de un valor de estado."""
    return copy.deepcopy(value)

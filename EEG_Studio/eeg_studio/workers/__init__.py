"""Ejecución concurrente de tareas pesadas (filtrado, extracción, entrenamiento)."""
from .worker import Worker, run_async

__all__ = ["Worker", "run_async"]

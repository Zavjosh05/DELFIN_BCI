"""Constantes y configuración global de EEG Studio.

Valores por defecto pensados para el casco Emotiv EPOC+ (14 canales, 128 Hz)
y para los CSV exportados por OpenViBE Designer.
"""
from __future__ import annotations

APP_NAME = "DELFIN EEG Studio"
APP_VERSION = "0.1.0"
ORG_NAME = "DELFIN_BCI"

# --- Adquisición -----------------------------------------------------------
DEFAULT_SAMPLE_RATE = 128.0  # Hz, EPOC+

# Orden de canales del EPOC+ según el montaje 10-20.
# El CSV de OpenViBE los nombra "Channel 1".."Channel 14"; aquí damos el alias
# clínico por defecto (configurable por proyecto).
EPOC_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

# Columnas estructurales del CSV de OpenViBE (no son canales EEG).
TIME_COLUMN_PREFIX = "Time:"        # p.ej. "Time:128Hz"
EPOCH_COLUMN = "Epoch"
EVENT_COLUMNS = ["Event Id", "Event Date", "Event Duration"]

# --- Bandas de frecuencia EEG ---------------------------------------------
# (límite inferior, límite superior) en Hz.
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# --- Estructura del proyecto en disco -------------------------------------
PROJECT_EXT = ".eegproj"
PROJECT_MANIFEST = "project.json"
CHANGELOG_FILE = "changelog.json"
CACHE_DIR = "cache"
DATASETS_DIR = "datasets"
MODELS_DIR = "models"
RECORDINGS_DIR = "recordings"  # CSV capturados en vivo (formato OpenViBE)

# --- Adquisición en tiempo real (opcional) --------------------------------
# La interfaz NO necesita conectarse a ningún dispositivo: la adquisición es
# una función opcional. Estos son los valores por defecto de cada fuente.
LIVE_WINDOW_SECONDS = 5.0          # ventana visible del visor en vivo
LIVE_REFRESH_MS = 33               # cadencia de refresco del visor (~30 fps)

# OpenViBE Acquisition Server -> salida LSL.
LSL_SIGNAL_NAME = "openvibeSignal"
LSL_MARKERS_NAME = "openvibeMarkers"

# CyKit (CyKIT.py en modo 'generic') -> servidor TCP.
CYKIT_HOST = "127.0.0.1"
CYKIT_PORT = 5555
# Columna donde empiezan los 14 canales en cada línea de CyKit. Con la salida
# por defecto (COUNTER, INTERPOLATED, AF3, ...) los canales empiezan en la 2.
CYKIT_CHANNEL_START = 2

# --- Rendimiento -----------------------------------------------------------
# Nº de procesos para la extracción de características en lote.
# 0 => usar os.cpu_count().
N_WORKERS = 0

# Cachear en disco (carpeta cache/) la señal procesada por el pipeline, para que
# reabrir un proyecto no tenga que recalcular filtros costosos (p. ej. ICA).
DISK_CACHE_PROCESSED = True

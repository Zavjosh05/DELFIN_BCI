# Changelog

Todos los cambios notables de **DELFIN EEG Studio** se documentan en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).
Las fechas usan el formato `AAAA-MM-DD`. Mientras no haya versiones publicadas, los
cambios se agrupan por fecha de trabajo.

> Nota: este `CHANGELOG.md` documenta el **código del repositorio**. Cada proyecto
> `.eegproj` tiene además su propia bitácora (`changelog.json`) con el historial de
> ediciones del usuario (undo/redo y línea de tiempo) — son cosas distintas.

---

## [2026-07-01]

### Añadido
- Al **crear un proyecto nuevo**, se pregunta si quieres **importar un bundle
  existente** (`.eegbundle`) para arrancar con pipeline + dataset + modelos.
- **Métricas globales**: tabla resumen del modelo en general (exactitud, precisión/
  recall/F1 macro y F1 ponderado, muestras totales), además de las métricas por clase.
- La **imagen guardada de las métricas** ahora incluye **todo el informe**: matriz de
  confusión + F1 por clase + tabla de scores por clase + tabla de métricas globales
  (se captura el informe completo en un PNG).

### Cambiado
- Botones con texto largo **acortados** (con tooltip con la descripción completa)
  para que **no se recorten** en pantallas/resoluciones pequeñas.
- Se documenta y verifica que el **bundle nunca incluye imágenes/gráficos**: la
  matriz de confusión y demás se **regeneran al importar** desde las métricas
  numéricas del modelo, así que no aumentan el tamaño del archivo.

## [2026-06-29]

Gran tanda de funcionalidades sobre la base del 27. Commits de git de esta fecha:
`452419a`, `cccd9f0`, `3b9782e`, `545bf9d`, `e482ee1` (+ cambios sin commitear).

### Añadido
- **Adquisición en vivo del Emotiv EPOC+ sin OpenViBE ni CyKit**: lector nativo por
  USB (HID + descifrado AES con clave derivada del nº de serie), con autodetección
  de modo 14/16-bit. Fuentes adicionales: Simulado, OpenViBE-LSL y CyKit/TCP (respaldo).
- **Botón "Probar dongle Emotiv"** en la pestaña Tiempo real: diagnóstico de
  detección, datos, modo y calidad sin necesidad de conectar.
- **Indicadores de calidad/ruido** de la señal en vivo (canal ok / plano / saturado
  / ruido; aviso global verde/ámbar/rojo).
- **Redes neuronales (PyTorch)**: MLP, CNN 1D, LSTM y EEGNet, con configuración por
  capa (unidades, activación, dropout, kernel, optimizador, épocas…).
- **Métodos de la literatura**: eliminación de artefactos por ICA, geometría de
  Riemann (MDM y Tangent Space + LR) y CSP + LDA.
- **Modo de control en tiempo real**: clasifica ventanas en vivo y envía la clase a
  un controlador (robot/carro) por UDP, puerto serie o registro.
- **Importación de datasets**: `.mat` (BCI IV 2a / BNCI) → CSV, y `.fif`/`.edf`/`.gdf`/
  BrainVision/EEGLAB vía MNE. Etiquetado de `.fif` a partir de los `.mat` originales.
- **Exclusión de canales** (p. ej. EOG) no destructiva; al importar `.mat` se
  **excluyen los EOG por defecto conservando las etiquetas** (opción configurable).
- **Varios clasificadores por proyecto** con métricas (matriz de confusión, f1 por
  clase) y exportación/importación de modelos entre proyectos.
- **Visor de características** (mapa de calor de potencias por banda y temporales).
- **Segmentos desde marcadores** (incl. "todas las fuentes"); superposición de
  segmentos por clase y marcadores como ayuda visual.
- **Guardado continuo (autosave)** estilo PyCharm, manteniendo Ctrl+S.
- **Proyectos recientes**, **pantalla de bienvenida**, **tema oscuro**, **barra de
  herramientas** y título de ventana con el proyecto + indicador de cambios sin guardar.
- **Varias señales a la vez** en ventanas independientes.
- **Activar/desactivar pasos del pipeline** con casilla (sin borrarlos), además del
  botón Eliminar.
- **Diseño de filtros FIR** seleccionable (pasa-banda/altas/bajas y notch) junto al
  Butterworth (IIR).
- **Barras de progreso** al filtrar y al entrenar (progreso por época en redes).
- **Vaciar todos los segmentos** de una vez y **eliminar archivos del proyecto**
  (nunca de la carpeta de origen).
- **Botón unificado** "Añadir o importar señal" (CSV + datasets en un solo paso).
- **Métricas embellecidas**: la matriz de confusión se dibuja como **mapa de calor
  con matplotlib** (conteos anotados + barra de color) junto a un gráfico de **F1 por
  clase**, y los scores se muestran en una **tabla con color**. La figura se puede
  **guardar como imagen** (PNG/PDF/SVG) y se conserva el informe de **texto** («Ver
  texto…»). Sin matplotlib, se usa el texto de siempre.
- **Edición de señal en el visor**: **recortar (eliminar) tramos seleccionados** (no
  destructivo, reversible con Ctrl+Z, sombreados en gris y excluidos del dataset) y
  **borrar los segmentos etiquetados de la selección** directamente desde el visor.
- **Botón "Abrir carpeta del proyecto"** en el explorador de archivos del sistema.
- **Exportar/importar bundle** `.eegbundle` (ZIP autónomo): un diálogo con casillas
  permite elegir qué incluir (preprocesamiento / dataset / modelos). El bundle
  guarda la **configuración** (pipeline, canales, segmentos, recortes), los
  **datasets** (`.npz`) y los **modelos entrenados** (`.joblib`), de modo que en
  otra máquina **Importar configuración/bundle…** reconstruye pipeline + dataset +
  modelos de un solo archivo. Opcionalmente incluye las **señales de origen (CSV)**
  conservando el id de cada fuente (los segmentos siguen válidos). El bundle **no
  incluye la caché** (regenerable) y comprime los archivos, por lo que **suele pesar
  bastante menos que la carpeta del proyecto** (el export informa del tamaño y
  avisa si el archivo acabara siendo mayor que el proyecto, sugiriendo excluir las
  señales de origen). El
  explorador se abre por defecto en la carpeta del proyecto. También existe el export
  ligero solo-configuración `.eegcfg` (JSON). Los hiperparámetros del clasificador
  clásico se guardan ahora en el modelo.
- **`CHANGELOG.md`** del repositorio (este archivo) + enlace desde el `README`.

### Cambiado
- Los filtros pasa-banda/altas/bajas y el notch ahora aceptan diseño **Butterworth
  (IIR)** o **FIR**, con `numtaps` configurable.
- Las conversiones de import se guardan **dentro del proyecto** (`imported/`), nunca
  en la carpeta de datos de origen; salida comprimida `.csv.gz`.
- Construir el dataset reúne **todos** los segmentos actuales y es robusto ante
  fuentes faltantes (las omite e informa).
- Mejoras de documentación en la interfaz: unidades de desfase/ventana, descripción
  del filtro CAR y del resto de parámetros, ventana (muestras) de los modelos.
- `threadpoolctl` listado explícitamente en `requirements.txt`.
- **Optimización:** la **lectura del CSV** al importar/añadir fuentes se hace en el
  **hilo de trabajo** (antes bloqueaba la GUI al añadir archivos grandes); se pasa la
  grabación ya cargada a `add_source`.

### Corregido
- Autodetección del modo del Emotiv: usaba el nivel en µV (fallaba sin contacto de
  electrodos y elegía 16-bit); ahora usa la **monotonía del byte contador** y detecta
  correctamente el modo (14-bit en el dongle del usuario).
- Robustez ante **fuentes cuyo archivo falta** (no se cae; ofrece reubicar/quitar).
- `FastICA did not converge` (ConvergenceWarning): más iteraciones, tolerancia y
  silenciado controlado.
- `UserWarning` de EEGNet por `padding='same'` con kernel par (relleno explícito).
- `DtypeWarning` al leer CSV en la columna «Event Id» (`low_memory=False`).
- Aviso de geometría de ventana en pantallas 1080p (paneles desplazables + tamaño
  mínimo seguro); la interfaz se ve bien en 1080p y 1440p.

## [2026-06-27]

Primera iteración de la aplicación. Commits: `aed5166`, `9528271`.

### Añadido
- **Aplicación de escritorio PyQt6** para visualizar, preprocesar, construir datasets
  y clasificar señales EEG (Emotiv EPOC+ desde CSV de OpenViBE).
- **Proyecto `.eegproj` no destructivo**: el CSV original nunca se modifica; los
  cambios viven en el estado del proyecto (archivos locales).
- **Control de cambios** con undo/redo y bitácora persistente (`changelog.json`) +
  dock de historial navegable.
- **Visor de señal** (pyqtgraph) con marcadores y selección de regiones.
- **Pipeline de preprocesamiento**: detrend, pasa-banda/altas/bajas, notch, CAR,
  referencia a canal y normalización, aplicado sobre copias.
- **Extracción de características**: potencias por banda (Welch) y temporales
  (RMS, pico-a-pico, longitud de línea, parámetros de Hjorth…).
- **Construcción de datasets** a partir de segmentos etiquetados.
- **Clasificadores clásicos**: Random Forest, SVM y LDA, con validación cruzada y
  métricas.
- **Concurrencia**: hilos (QThread) para la GUI y multiproceso para la extracción de
  características, con caché en disco de la señal procesada.

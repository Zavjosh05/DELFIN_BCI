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
- **Pausar y descartar la grabación**: botón **⏸ Pausar/Reanudar** (la señal en
  vivo sigue; en pausa no se escribe) y **✕ Descartar** (detiene y borra el archivo
  y sus marcas, **pidiendo confirmación**).
- **Archivo lateral de marcas** (`<csv>.marks.json`): los segmentos de cada
  grabación se guardan **junto al CSV mientras grabas** (escritura atómica), no
  solo en el proyecto. Si la app se cierra o falla, las marcas quedan en disco y se
  **recuperan** al añadir/reabrir la grabación. Blindaje contra la pérdida de marcas.
- **Renombrar señales desde la lista de Fuentes**: **clic izquierdo** sobre la
  señal ya seleccionada (o **F2**, o clic derecho → **Renombrar…**) la edita en el
  sitio. Cambia el nombre mostrado y, si el archivo es **interno** al proyecto,
  también **renombra el CSV en disco** (conserva la extensión `.csv`/`.csv.gz`,
  con sufijo si hay colisión); las fuentes **externas** solo cambian el nombre
  mostrado (el archivo de origen no se toca). Reversible con Ctrl+Z. «Abrir en
  ventana nueva» pasó al **menú contextual** (clic derecho).
- **Exportar CSV (descomprimido)** y **visor de datos numérico** en el menú
  contextual de *Fuentes* (clic derecho): «Exportar CSV (descomprimido)…» guarda
  el CSV **en texto plano** en la ubicación que elijas (descomprime los `.csv.gz`
  para poder abrirlos en VS Code u otros editores que no leen comprimidos), y
  «Ver datos (tabla numérica)…» abre una **tabla eficiente** (virtualizada, apta
  para grabaciones grandes) con los valores: nº de muestra, tiempo, cada canal y
  el `Event Id`. El visor incluye un botón para exportar y un «ir a muestra».
- **Editar segmentos desde el visor de la señal (clic derecho)**: sobre un
  segmento etiquetado, un menú permite **Reetiquetar** (cambiar su clase, con la
  lista de clases existentes) o **Eliminar** el segmento. Si hay segmentos
  solapados, actúa sobre el **más específico** (el de menor duración bajo el
  cursor). Funciona en el visor de análisis y en las ventanas de señal
  desacopladas; reversible con Ctrl+Z.
- **Nombrar la grabación**: campo «Nombre» en el panel de adquisición. Se usa como
  nombre del CSV (saneado y con sufijo `_2`, `_3`… si ya existe) y como **alias**
  de la fuente al añadirla. Si se deja vacío, se usa la fecha/hora
  (`rec_AAAAMMDD_HHMMSS.csv`). La grabación en vivo se guarda como **`.csv`** (sin
  comprimir) en la carpeta `recordings/` del proyecto.
- **Segmentos en vivo (inicio/fin) durante la grabación**: además de la marca de
  **instante**, un botón/atajo marca el **inicio** de un segmento y otro clic el
  **fin** (con la etiqueta indicada). Al añadir la grabación como fuente, esos
  tramos se crean como **segmentos etiquetados** del proyecto (listos para el
  dataset). Atajos: **F3** = marca instantánea, **F4** = iniciar/terminar
  segmento. Un segmento que quede abierto se cierra al detener la grabación.
- **Marca de duración fija**: selector de **Duración** (s) + botón/atajo (**F5**)
  que crea un segmento de esa duración **desde el instante actual** (p. ej. 5 s de
  una clase). Si la grabación termina antes de completarse, el segmento se
  **recorta** a lo grabado.
- **Varios pipelines por proyecto**, como **pestañas de navegador** en el panel
  de Preprocesamiento: se pueden **crear** (`＋`), **renombrar** (doble clic),
  **cambiar** y **eliminar** pipelines independientes. Para eliminar hay un botón
  dedicado **🗑** (con confirmación, reversible con Ctrl+Z) además del cierre de
  la propia pestaña; siempre queda al menos un pipeline. El pipeline **activo** es
  el que se aplica a la señal, al dataset y a los modelos. Con **undo/redo** y
  **persistencia**; los proyectos antiguos (un solo pipeline) **migran** solos.
- Al **exportar el bundle** se puede elegir **qué pipelines incluir**, con una
  casilla por pipeline y un selector global **«Todas las pipelines»**. El bundle
  reconstruye el pipeline activo dentro de la selección exportada.
- **Centro multi-fuente con pestañas**: al abrir varias fuentes se ven como
  **pestañas** (estilo navegador) en una sola vista; se cambia de señal con la
  pestaña, se pueden **cerrar y reabrir**, y se mantiene **"Abrir en ventana
  nueva"** para desacoplar una fuente en su propia ventana.
- **Historial en árbol**: el historial deja de ser una línea. Si vuelves a un
  estado anterior y haces un cambio, se crea una **rama nueva** en lugar de
  borrar lo que habías hecho después — nada se pierde. El dock *Historial*
  muestra el árbol con sangría (marcando bifurcaciones y la rama actual) y un
  clic salta a cualquier nodo, aunque esté en otra rama. Retrocompatible: los
  `changelog.json` lineales antiguos se leen y migran a árbol.
- **Aislar un canal** en el visor, tanto al revisar un CSV como en tiempo real:
  un selector "Canal" muestra **solo ese canal a escala real** y una fila con
  sus **medidas** (mín, máx, media, σ y rango pico-a-pico en µV) para saber
  entre qué valores varía la señal.
- **Configuración de un modelo**: botón "Configuración…" para **ver los
  hiperparámetros** con los que se entrenó y **editarlos**; al aceptar, el
  modelo **se reentrena** conservando su nombre. Cubre clásicos (Random Forest,
  SVM), los escalares de las redes (épocas, batch, learning rate, ventana…) y la
  ventana de muestras de Riemann/CSP (que ahora se guarda en el modelo).
- **Menú "Ver"** con casillas para mostrar/ocultar los paneles (Fuentes,
  Herramientas, Historial) y "Restaurar paneles": **arregla** que, al cerrar un
  panel acoplado, no hubiera forma de volver a abrirlo.
- **Dataset**: el panel muestra el **total de muestras** y el **desglose por clase**
  (segmentos etiquetados y, al construir, muestras del dataset por clase).
- **Modelos**: se indica **con cuántos datos se entrenó y se evaluó**, adaptado al
  método y con porcentajes — clásicos/Riemann: validación cruzada de *k* pliegues
  (≈(k-1)/k entrena, ≈1/k evalúa por pliegue); redes: holdout 75/25. Aparece en el
  diálogo de métricas, en la imagen guardada y en el tooltip de la lista de modelos.
- Al **crear un proyecto nuevo**, se pregunta si quieres **importar un bundle
  existente** (`.eegbundle`) para arrancar con pipeline + dataset + modelos.
- **Métricas globales**: tabla resumen del modelo en general (exactitud, precisión/
  recall/F1 macro y F1 ponderado, muestras totales), además de las métricas por clase.
- **Matriz de confusión normalizable**: casilla para verla en **porcentajes por fila**
  (por defecto: conteos). El estado se refleja también en la imagen guardada.
- La **imagen guardada de las métricas** ahora incluye **todo el informe**: matriz de
  confusión + F1 por clase + tabla de scores por clase + tabla de métricas globales.
  Se compone con matplotlib para que **todas las filas** de las tablas se vean (sin
  barras de desplazamiento); en la interfaz las tablas siguen igual.

### Corregido
- **Grabaciones que se perdían al cerrar** (blindado): al añadir una grabación como
  fuente no se guardaba, así que si cerrabas sin otro cambio, las altas se perdían.
  Ahora **se guarda de inmediato** (no depende del temporizador), hay **guardado de
  precaución al cerrar** la app (siempre que haya proyecto), y las marcas quedan
  además en su **archivo lateral**. Al abrir un proyecto se detectan las grabaciones
  de `recordings/` no añadidas y se ofrece incorporarlas **con sus marcas**; también
  desde el menú contextual → «Buscar grabaciones sueltas…».
- **Adquisición Emotiv más robusta**: el lector **tolera fallos transitorios** de
  lectura USB (un hipo del dongle ya no tumba toda la sesión; solo se rinde si
  persisten ~10 s), y si la fuente se detiene se **muestra el motivo** en el panel.
- **Proyecto portátil**: las rutas de las fuentes **internas** (dentro de la
  carpeta `.eegproj`: `recordings/`, `imported/`…) se guardan **relativas al
  proyecto** en `project.json` y en `changelog.json`, y se **resuelven contra la
  ubicación actual** al abrir. Así, **mover, copiar o renombrar** la carpeta del
  proyecto (u otra máquina/disco) ya no rompe los enlaces. Las fuentes externas
  siguen guardándose con ruta absoluta.
- **Visor en vivo**: el **eje de tiempo ahora avanza** con la señal (muestra el
  tiempo transcurrido real, `[t−ventana, t]`), en vez de quedarse fijo en
  `[−ventana, 0]`.
- **Visor de CSV**: se **resta el offset DC por canal solo para visualizar** (p.
  ej. la línea base ~4200 µV del EPOC+). Antes las señales salían aplastadas y
  descolocadas respecto a su etiqueta (la escala «empezaba en cero»); ahora cada
  canal se **centra** y la escala refleja la **amplitud real**. No altera los
  datos; las medidas del canal aislado siguen mostrando los valores reales.

### Cambiado
- **Árbol de cambios más navegable**: el historial pasa de una lista con sangría a
  un **árbol colapsable** (cada nodo cuelga de su padre), con botones **Expandir /
  Colapsar ramas**, la rama actual resaltada (▶) y las bifurcaciones marcadas (⑂).
- **Guardado más robusto**: el proyecto se escribe de forma **atómica** (a un
  temporal + reemplazo con `fsync`), de modo que un fallo/corte a mitad de guardado
  **no corrompe** `project.json`/`changelog.json`; y el **autoguardado reintenta**
  si una escritura falla.
- **La grabación se añade automáticamente** como fuente al terminar (antes
  preguntaba «¿Añadirla como fuente?»). Se guarda enseguida, con su nombre y sus
  segmentos.
- **Panel de adquisición reordenado**: el **estado y la calidad de la señal
  (canales detectados)** se muestran **arriba** (siempre a la vista), y los botones
  de marca/segmento se **compactan en rejilla** (marca · segmento · marca fija) para
  que no queden amontonados.
- **Visor en vivo — escala seleccionable**: nuevo modo **«Fija (µV)»** (por
  defecto, estilo OpenViBE) con escala en microvoltios **constante y ajustable**
  (selector «µV/canal»), para que la escala **no cambie sola** y las amplitudes
  sean comparables. Se conserva el modo **«Auto (normalizada)»** (cada canal por
  su desviación) como opción.
- **Estilo**: pestañas con aspecto de navegador (esquinas redondeadas y acento en
  la activa); las barras de pestañas (fuentes y pipelines) **eliden** el texto y
  usan **botones de desplazamiento** para no desbordar en pantallas 1080p.
- **Imagen de las métricas más compacta**: las secciones (matriz de confusión,
  F1, tablas) salen **lo más juntas posible** y la figura se dimensiona a las
  secciones elegidas. Al **guardar**, un diálogo pregunta **qué métricas
  incluir** (matriz / F1 / tabla por clase / tabla global) y si normalizar.
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

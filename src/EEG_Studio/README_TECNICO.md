# DELFIN EEG Studio — Referencia técnica

Interfaz de escritorio (PyQt6) para **visualizar, preprocesar, procesar y
clasificar** señales EEG capturadas con el casco **Emotiv EPOC+** y exportadas a
CSV desde **OpenViBE Designer**.

> **Ubicación (tras la reestructuración de julio 2026):** el código de la app vive
> en `src/EEG_Studio/` (este directorio) y los datos de ejemplo para pruebas en
> `data/raw/EEG/`. Este documento es la **referencia técnica de desarrollo**
> (arquitectura, métodos, pruebas); el `README.md` de la raíz del repositorio es la
> presentación oficial del proyecto.

> El historial de cambios del código está en [`CHANGELOG.md`](../../CHANGELOG.md).

Permite reunir varios CSV en un mismo proyecto, **aislar y agrupar** segmentos de
señal para construir un **dataset etiquetado**, y entrenar un **modelo de
clasificación** sobre él. Todo el procesamiento es **no destructivo**: el CSV de
origen nunca se modifica; los cambios viven en los archivos locales del proyecto
y disponen de **deshacer/rehacer** e historial.

---

## Características

- **Visualización multicanal** (14 canales EPOC+ a 128 Hz) con pyqtgraph:
  ganancia, normalización de vista, navegación temporal y selección de regiones.
- **Preprocesamiento** como *pipeline* reproducible: eliminar tendencia, filtros
  pasa-banda / altas / bajas, notch (red eléctrica), referencia promedio común
  (CAR), referencia a canal, normalización y **eliminación de artefactos por ICA**
  (rechazo automático de componentes por kurtosis). Cada filtro y cada parámetro
  muestran en la interfaz **qué hacen y qué efecto tiene modificarlos**.
- **Selección de canales**: activa/desactiva canales (p. ej. excluir los EOG de
  un dataset ajeno); afecta a CAR, características y modelos.
- **Procesamiento / extracción de características**: potencias por banda
  (delta, theta, alpha, beta, gamma) y características temporales (RMS, longitud
  de línea, parámetros de Hjorth…).
- **Construcción de datasets**: aísla segmentos de varios CSV, etiquétalos y
  genera una matriz de características (extracción **en multiproceso**).
- **Clasificación**: **Random Forest** (nº de árboles, profundidad, criterio…),
  **SVM** (kernel seleccionable: lineal, RBF, polinomial o sigmoide, con
  C/gamma/grado) o LDA; **geometría de Riemann** (MDM,
  *tangent space* + regresión logística) y **CSP + LDA** sobre señal cruda
  (scikit-learn / pyriemann); y **redes neuronales** (PyTorch) configurables en
  detalle — MLP, CNN 1D, LSTM y **EEGNet** sobre señal cruda, con activación por
  capa, nº de capas, dropout, kernel/bidireccional, optimizador y épocas.
  Se pueden entrenar **varios modelos por proyecto**, ver sus **métricas**
  (exactitud, matriz de confusión, precisión/recall/F1 por clase), **activar** uno,
  **exportarlo/importarlo** (.joblib) y predecir la región seleccionada.
- **Control de cambios no destructivo**: cada edición se registra con
  deshacer/rehacer y queda en `changelog.json`. El CSV original es de solo lectura.
- **Adquisición en tiempo real (opcional)**: visor en vivo y grabación a CSV
  local desde fuentes intercambiables: **Simulado** (sin hardware), **OpenViBE
  Acquisition Server vía LSL**, **Emotiv EPOC+ (lector integrado)** y **CyKit/TCP**.
  La app funciona igual sin conectarse a nada (solo procesar CSV).
- **Control en tiempo real (opcional)**: con un modelo entrenado, clasifica la
  señal en vivo (tratada por el preprocesamiento) y envía la clase detectada a un
  **controlador externo** (brazo robótico, carrito…) por **UDP**, **puerto serie**
  (Arduino) o registro, con suavizado para evitar oscilaciones.
- **Rendimiento**: tareas pesadas en hilos (`QThread`), preprocesado de varias
  fuentes en paralelo y extracción de características con `ProcessPoolExecutor`
  (multiprocessing). Ver la sección *Rendimiento*.

## Estructura del proyecto en disco

Un proyecto es una carpeta `nombre.eegproj/`:

```
nombre.eegproj/
├── project.json     # manifiesto: fuentes (rutas a CSV), pipeline, segmentos, dataset
├── changelog.json   # historial de cambios (undo/redo + auditoría)
├── cache/           # señales procesadas en caché (.npz)
├── datasets/        # datasets exportados (.npz)
└── models/          # modelos entrenados (.joblib)
```

> Los CSV **no se copian** dentro del proyecto: se referencian por ruta y se leen
> en modo solo lectura. El **CSV es el formato de intercambio** (compatible con
> OpenViBE, inspeccionable, con marcadores incluidos); los archivos convertidos de
> `.mat`/`.fif` se guardan **comprimidos** (`.csv.gz`) y los datos pesados internos
> del proyecto ya son binarios (`cache/` y `datasets/` en `.npz`, modelos en
> `.joblib`). Para recomprimir fuentes ya añadidas: **Proyecto → Comprimir fuentes
> a .csv.gz…** (mismo contenido, ~3-6× más pequeño; opción de borrar los originales).

## Instalación

```powershell
# Desde la carpeta EEG_Studio
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Ejecución

```powershell
.\.venv\Scripts\python.exe run.py
```

## Proyecto de ejemplo

Para explorar la app sin partir de cero, genera un proyecto de ejemplo (referencia
los CSV de `EEG/`, define un pipeline, segmentos etiquetados, un dataset y un
modelo entrenado):

```powershell
.\.venv\Scripts\python.exe examples\create_example_project.py
```

Esto crea `examples\Ejemplo.eegproj`. Ábrelo en la app con **Proyecto → Abrir
proyecto…**. Como los proyectos referencian los CSV por ruta absoluta, este
script es la forma recomendada de (re)crearlo con las rutas correctas del equipo.

## Importar datasets externos (.mat / .fif / .edf… → CSV)

Para trabajar con otros dispositivos/datasets, **Proyecto → Importar dataset…**
convierte a un CSV en formato OpenViBE y lo añade como fuente. Soporta:

- **`.mat`** del dataset **BCI Competition IV 2a / BNCI 001‑2014** (ver abajo).
- **`.fif`** (formato nativo de MNE) y otros formatos profesionales que lee MNE:
  **`.edf`/`.bdf`, `.gdf`, BrainVision `.vhdr`, EEGLAB `.set`**. Conserva los
  canales de datos (descarta los de estímulo/auxiliares), reescala a µV y toma
  los marcadores de las **anotaciones** o de un canal de estímulo con códigos de
  evento. Requiere `mne` (opcional).

> **Etiquetar un `.fif` sin etiquetas:** si un `.fif` trae la señal pero no los
> eventos, `mne_loader.label_fif_from_mat(fif, mat)` (o
> `python examples/label_fif_from_mat.py`) crea una **copia** `*_etiquetado.fif`
> con las clases del `.mat` original como anotaciones, **sin modificar la señal**
> (requiere que el `.fif` sea la concatenación de los runs del `.mat`).

Para el `.mat` de 2a, además:

- Concatena los *runs* de imaginación motora (250 Hz, **25 canales**: 22 EEG + 3
  EOG) y coloca un **marcador** al inicio de cada ensayo con el nombre de su clase
  (`left_hand`, `right_hand`, `feet`, `tongue`) — 288 marcadores por sesión.
- Los **nombres reales de canal** (Fz, C3, Cz…) se conservan: la app no fuerza el
  montaje del EPOC+ cuando el CSV ya trae nombres (admite cualquier nº de canales).

> Los archivos convertidos se guardan **comprimidos** (`.csv.gz`, ~3-4× más
> pequeños: una sesión 2a pasa de ~107 MB a ~33 MB) y pandas los lee de forma
> transparente. La conversión corre en segundo plano y se reutiliza si ya existe.

### Pipeline recomendado para el dataset 2a (imaginación motora, 4 clases)

1. **Canales:** *Preprocesamiento → Seleccionar canales…* → **Excluir EOG**
   (deja solo los 22 EEG). La exclusión afecta a CAR, características y modelos.
2. **Preprocesamiento:** *Filtro pasa-banda* **8–30 Hz** (ritmos μ y β de la
   imaginación motora) + *CAR*. (Opcional: *notch* 50 Hz.)
3. **Etiquetado:** *Dataset → Segmentos desde marcadores* con **Desfase ≈ 750**
   (3 s) y **Ventana ≈ 750** (3 s) → captura el periodo de imaginación [3–6 s].
   Cada marcador aporta su clase (4 clases).
4. **Modelo:** **Riemann — Tangent Space + LR** o **CSP + LDA** (de referencia en
   imaginación motora) sobre señal cruda; o **EEGNet** si prefieres red neuronal.

> **Proyecto de ejemplo 2a:** `python examples/create_2a_example_project.py` crea
> `examples/Ejemplo_2a.eegproj` con todo este pipeline ya montado y un modelo
> Riemann entrenado (en el sujeto A01 da ~80 % de validación cruzada, 4 clases).

## Flujo de trabajo típico

1. **Proyecto → Nuevo proyecto** y elige carpeta/nombre.
2. **Proyecto → Añadir CSV…** para integrar una o varias grabaciones (CSV de
   OpenViBE). El repositorio incluye archivos de ejemplo en la carpeta `EEG/`.
3. Selecciona una fuente en el panel izquierdo para verla.
4. Pestaña **Preprocesamiento**: añade pasos (p.ej. *pasa-banda 1–45 Hz* +
   *notch 60 Hz* + *CAR*). La vista se actualiza con la señal procesada.
5. En el visor, arrastra la **región** sobre el tramo de interés y pulsa
   **«Crear segmento de la selección»**; asígnale una etiqueta (clase).
   Repite con distintos CSV y clases.
6. Pestaña **Dataset**: **Construir dataset** (multiproceso) y, opcionalmente,
   **Guardar dataset**.
7. Pestaña **Clasificación**: elige el modelo, **Entrenar**, revisa la validación
   cruzada y **Clasificar selección actual** o **Guardar modelo**.

El proyecto se **guarda solo** (autoguardado) poco después de cada cambio, sin
necesidad de **Ctrl+S** (que sigue disponible para guardar al instante). Al salir
se vuelca cualquier cambio pendiente.

Usa **Editar → Deshacer/Rehacer** (Ctrl+Z / Ctrl+Y) para revertir cualquier
cambio. El **Historial** (panel inferior) muestra la línea de tiempo con iconos
y hora: el punto **actual** va resaltado y los pasos rehacibles atenuados. Haz
**clic en cualquier punto** para navegar directamente a ese estado.

> **Ganancia** (control del visor): es una amplificación **solo visual** de la
> señal (x2 = el doble de alta en pantalla). No modifica los datos; sirve para
> ver mejor fluctuaciones pequeñas o evitar que señales grandes se solapen.

## Etapas: cómo se usan y qué aportan

### 1. Preprocesamiento
- **Qué es:** una lista de pasos que se aplican en orden a la señal, de forma no
  destructiva (el CSV nunca cambia).
- **Cómo se usa:** pestaña *Preprocesamiento* → elige un paso → **Añadir paso**.
  Selecciónalo en la lista para ver su descripción y editar sus parámetros (cada
  uno explica qué hace y qué efecto tiene); reordena con ▲▼ o elimínalo. El paso
  en edición queda **resaltado** y la selección **sigue al paso** al moverlo, para
  saber siempre cuál se está modificando. La vista *Procesada* se actualiza al
  instante.
- **Qué ganas:** señal más limpia y comparable. *Pasa-banda 1–45 Hz* quita deriva
  y ruido alto; *notch 50/60 Hz* elimina la red eléctrica; *CAR* resta el ruido
  común; **ICA** elimina parpadeos/músculo automáticamente → mejor exactitud.
- **Marcadores y segmentos legibles:** en grabaciones largas con muchos marcadores
  (p. ej. 288 ensayos), el visor solo dibuja los **visibles en el rango actual**
  (atenuados, con etiqueta al acercar) y arranca en una ventana de ~30 s, en vez de
  saturar la pantalla. Usa la rueda/arrastre para navegar.

### 2. Extracción de características
- **Qué es:** convierte cada segmento en un vector (potencias por banda δθαβγ +
  medidas temporales: RMS, longitud de línea, Hjorth).
- **Cómo se usa:** pestaña *Dataset* → casillas *Potencias por banda* y/o
  *Características temporales*.
- **Qué ganas:** una representación compacta e interpretable que funciona bien
  con clasificadores clásicos y MLP, incluso con pocos datos.

### 3. Construcción del dataset
- **Qué es:** reúne los segmentos etiquetados de uno o varios CSV en una matriz
  lista para entrenar.
- **Cómo se usa:** en el visor, los **marcadores** (Event Id) se dibujan como
  líneas con su etiqueta (interruptor *Marcadores*), como **referencia visual**
  para decidir dónde segmentar. Arrastra la región y **Crear segmento** con su
  etiqueta (repite con varias clases/CSV); en *Dataset* pulsa **Construir
  dataset** y, opcionalmente, **Guardar dataset**. (Atajo opcional: *Segmentos
  desde marcadores* crea segmentos automáticamente a partir de los marcadores —
  de la **fuente abierta** o, marcando *«Todas las fuentes»*, de **todo el
  proyecto** de una vez.)
- **Ayuda visual:** los segmentos ya creados se **sombrean sobre la señal** con
  un color por clase y su etiqueta (interruptor *Segmentos*), para ver de un
  vistazo qué tramos ya están etiquetados mientras trabajas. Al **pasar el ratón**
  por un segmento se muestra su detalle (clase, rango en segundos y nº de muestras).
- **Qué ganas:** integras varias grabaciones y clases en un conjunto único y
  reproducible; la extracción aprovecha varios núcleos (ver *Rendimiento*).

### 4. Clasificación
- **Cómo se usa:** pestaña *Clasificación* → elige el tipo (aparece su
  configuración) → **Entrenar y añadir al proyecto**. Cada modelo entrenado se
  añade a la lista **«Modelos entrenados»**, donde puedes **Activar** uno,
  ver **Métricas…** (matriz de confusión + por clase), **Exportar…**/**Importar…**
  (.joblib) o **Eliminar**. **Clasificar selección actual** usa el modelo activo.
  Los modelos se guardan con el proyecto (carpeta `models/`).
- **Qué familia conviene:**
  - **Clásicos (RF / SVM / LDA)** sobre características: rápidos y robustos, buena
    línea base. Al seleccionarlos aparece su configuración: el **Random Forest**
    expone nº de árboles, profundidad máxima, mínimo para dividir, nº de
    características y criterio; el **SVM** permite elegir kernel (lineal/RBF/poly/
    sigmoide) y C/gamma/grado.
  - **Riemann (MDM / Tangent+LR) y CSP+LDA** sobre señal cruda: muy efectivos en
    imaginación motora y multiclase, robustos al ruido y con pocos datos.
  - **Redes (MLP / CNN / LSTM / EEGNet)**: aprenden los filtros de los datos;
    **EEGNet** es compacto y específico de EEG. Configurables capa a capa, con la
    capa de entrada y salida indicadas en pantalla.

## Rendimiento (hilos y multiprocessing)

La app reparte el trabajo pesado para no bloquear la interfaz y aprovechar varios
núcleos:

| Operación | Mecanismo | Ganancia |
|---|---|---|
| Preprocesar la vista, construir dataset, entrenar | **Hilos** (`QThread`) | La interfaz sigue fluida durante el cómputo |
| Señal procesada de varias fuentes | **Hilos en paralelo** (scipy/numpy liberan el GIL) | Datasets **multi-CSV** más rápidos |
| Extracción de características | **Multiprocessing** (`ProcessPoolExecutor`) cuando hay ≥ 12 segmentos | Acelera datasets grandes; por debajo va en serie para no pagar el coste de crear procesos |
| Adquisición en vivo | **Hilo** productor + buffer; la UI solo lee | Captura estable sin congelar la pantalla |
| Señal procesada | **Caché en disco** (`cache/`, por firma del pipeline) | Reabrir un proyecto no recalcula filtros costosos (p. ej. ICA) |

**Qué ganas:** respuesta inmediata de la interfaz y menores tiempos de
construcción y entrenamiento en equipos multinúcleo.

## Adquisición en tiempo real (opcional)

Pestaña **Tiempo real** (dock derecho) + vista **Tiempo real** (centro). Eliges
la fuente, **Conectar**, y opcionalmente **Iniciar grabación** (escribe un CSV
nuevo en `recordings/` del proyecto, formato OpenViBE, que puedes añadir como
fuente al terminar).

**Marcadores.** Durante la grabación, escribe una etiqueta y pulsa **Marcar**: el
marcador se guarda en la columna `Event Id`. Esos marcadores se **dibujan sobre
la señal** en el visor (interruptor *Marcadores*) como **ayuda visual** para
etiquetar manualmente las regiones de interés. Como atajo opcional, en la pestaña
**Dataset** → **«Segmentos desde marcadores»** puedes crear segmentos
automáticamente a partir de ellos (ventana en muestras, 0 = hasta el siguiente
marcador). También funciona con CSV de OpenViBE que ya traigan estimulaciones.

> La captura es opcional: si solo quieres procesar CSV, ignora esta pestaña.

### Fuente 1 — Simulado
No requiere nada. Genera 14 canales sintéticos a 128 Hz para probar la interfaz
sin el casco.

### Fuente 2 — OpenViBE Acquisition Server (LSL)
No hace falta el **Designer**; basta el **Acquisition Server** como driver:
1. Abrir **OpenViBE Acquisition Server**.
2. **Preferences** → sección **LSL** → `LSL_EnableLSLOutput = true`. Anotar el
   nombre del stream de señal (por defecto `openvibeSignal`).
3. **Driver:** el del dispositivo (p. ej. Emotiv) → **Connect** → **Play**.
4. En EEG Studio: fuente *OpenViBE (LSL)*, nombre del stream, **Conectar**.

Requiere `pylsl` (incluido en `requirements.txt`). Entrega EEG **crudo** sin
licencia Cortex, reutilizando un driver ya configurado en el Acquisition Server.

### Fuente 3 — Emotiv EPOC+ (lector integrado) · *CyKit dentro de la app*
Es **CyKit integrado en la app**: la misma técnica (descifrado HID con AES y
clave derivada del nº de serie), pero reimplementada en Python 3.13 y ejecutada
**en el propio proceso**. Así se **elimina el requisito de un Python antiguo** y
no hace falta ni OpenViBE ni ejecutar CyKit aparte.

1. Conecta el receptor USB y empareja el casco.
2. En EEG Studio: fuente *Emotiv EPOC+ (lector integrado)* → modo **Auto** (o
   fija 14/16 bits) → **Conectar**. El modo *Auto* prueba ambas claves y elige
   la que produce señal coherente; al conectar muestra el dispositivo detectado.

Reconoce el dongle por nombre de producto (`EPOC+`, `EEG Signals`…) o por vendor
id. Requiere `hidapi` y `pycryptodome` (en `requirements.txt`). Es la opción
indicada cuando la versión de OpenViBE instalada **no incluye driver Emotiv**
(Emotiv retiró su SDK gratuito) y no se dispone de licencia Cortex.

### Fuente 4 — CyKit / TCP (respaldo, directo del dongle, sin OpenViBE)
`CyKIT.py` lee el EPOC+ del dongle USB y emite por TCP. Es una **alternativa**
completa a OpenViBE (no un complemento). Lánzalo en su propio Python, aparte:

```
python CyKIT.py 127.0.0.1 5151 6 generic+nocounter+noheader+nobattery
```

(modelo `6` = EPOC+ Consumer 16-bit; `5` para Premium). En EEG Studio: fuente
*CyKit / TCP*, con el mismo host y puerto en los que escucha CyKit, y 14 canales.
Ajustar «Columna inicial» si la combinación de banderas cambia el formato de
cada línea (con `nocounter`, los canales empiezan en la columna 0).

**Configurador de CyKit incluido:** en la fuente *CyKit / TCP*, el botón
**«Configurar / lanzar CyKit…»** abre una ventana para activar/desactivar cada
bandera (`openvibe`, `generic`, `nocounter`, `noheader`, `nobattery`, `float`,
`integer`, `info`…) y ajustar las cantidades (`ovdelay`, `ovsamples`); construye
el comando en vivo, lo **copia**, **lanza CyKit** (mostrando su salida) y
**aplica** host/puerto a la fuente (con `nocounter`, fija la columna inicial en 0).
Un ejemplo de comando con todas esas banderas:

```
python CyKIT.py 127.0.0.1 5151 6 openvibe+generic+nocounter+noheader+nobattery+float+ovdelay:100+ovsamples:004
```

| | Necesita OpenViBE | Necesita CyKit aparte | Licencia | Dificultad |
|---|---|---|---|---|
| **Emotiv lector integrado** | No | No | No | Media (drivers USB) |
| **LSL (AS)** | Solo el Acquisition Server | No | No | Baja |
| **CyKit/TCP** | No | Sí | No | Media |
| **Simulado** | No | No | No | Nula (pruebas) |

> En versiones recientes de OpenViBE (p. ej. 2.2.0) **ya no existe driver Emotiv**
> (Emotiv retiró su SDK gratuito) y el driver *LSL* es de entrada (necesita que
> otro programa publique el stream). Sin licencia Cortex Raw EEG, leer el dongle
> exige el descifrado de CyKit — que el **lector integrado** ya incorpora en
> Python 3.13, eliminando los intermediarios y el requisito de un Python antiguo.
> El **lanzador de CyKit** original sigue disponible como respaldo (avisa si el
> intérprete seleccionado es ≥3.10 y autodetecta uno 3.7–3.9).

## Clasificación con redes neuronales (PyTorch)

En la pestaña **Clasificación**, además de Random Forest / SVM / LDA, puedes
elegir tres redes y **configurarlas a detalle**:

- **MLP (características)** — red densa sobre el vector de características.
- **CNN 1D (señal cruda)** — convolucional sobre ventanas de la señal.
- **LSTM (señal cruda)** — recurrente sobre ventanas de la señal.
- **EEGNet (señal cruda)** — CNN compacta específica de EEG (conv temporal →
  conv espacial *depthwise* → conv separable); parámetros F1, D, F2, kernel y
  dropout (doi:10.1088/1741-2552/aace8c).

Además, modelos de **señal cruda no neuronales**: *Riemann — MDM*, *Riemann —
Tangent Space + LR* y *CSP + LDA*, con su propio control de **tamaño de ventana**.

Al elegir una red aparece su editor: **lista de capas** (añadir/quitar), con
**unidades/filtros**, **función de activación por capa**, **dropout** y, según el
tipo, **kernel** (CNN) o **bidireccional** (LSTM); más **épocas**, **batch
size**, **learning rate**, **optimizador** y **tamaño de ventana** (CNN/LSTM). El
editor muestra arriba la **capa de entrada** (con su nº de neuronas) y abajo la
**capa de salida** (= nº de clases), recalculadas según los datos.

Para **SVM**, al seleccionarlo aparece un cuadro con el **kernel** (lineal, RBF,
polinomial, sigmoide) y sus parámetros (C, gamma, grado).

Las redes CNN/LSTM construyen su dataset de **señal cruda** automáticamente a
partir de los segmentos (ventanas de tamaño fijo, recorte/relleno centrado). El
MLP usa el mismo dataset de características que los clasificadores clásicos. La
validación se reporta por *holdout* (entrenar k-fold una red sería muy lento).

### Capas de entrada y salida

- **Capa de salida** (las tres redes): `Linear(…, nº de clases)` con
  `CrossEntropyLoss`; las neuronas de salida = nº de clases del dataset y se
  ajustan solas. Las probabilidades salen de aplicar *softmax* a los logits.
- **Capa de entrada**, según el tipo:
  - **MLP** → vector de características aplanado: primera capa
    `Linear(nº_características, …)`. El nº de características lo fija el dataset
    (potencias de banda + medidas temporales por canal; p. ej. 5 bandas × 14
    canales + 8 medidas × 14 = **182**).
  - **CNN 1D / LSTM** → ventana de señal cruda de forma **(canales, muestras) =
    (14, T)**. La CNN convoluciona a lo largo del tiempo con los 14 canales como
    canales de entrada; la LSTM recorre las T muestras con los 14 canales como
    características por paso.

### Dimensión de las CNN (y si conviene cambiarla)

Se trabaja con **(14, T)**: 14 canales y T muestras temporales.

- **Canales (14):** fijos. `build_raw_dataset` usa siempre todos los canales para
  que todos los segmentos tengan la misma forma.
- **T (tamaño de ventana):** **configurable** en el editor de la red. A 128 Hz:
  `T=512 ≈ 4 s`, `T=256 ≈ 2 s`, `T=128 ≈ 1 s`. Cada segmento se recorta o rellena
  (centrado) hasta T.

¿Conviene cambiar T? Depende del fenómeno:
- Hazlo coherente con la duración de los eventos/épocas. Si el patrón dura ~1 s,
  `T=128–256` basta.
- **Demasiado grande** respecto a los segmentos = mucho relleno con ceros (ruido).
- **Demasiado pequeño** = se pierde contexto temporal.
- El **kernel** de cada capa también es configurable: mayor capta ritmos lentos;
  menor, detalles finos. La red reduce el tiempo con *pooling* y termina en
  `AdaptiveAvgPool1d`, así que T no tiene que ser potencia de 2.

Regla práctica: empieza con T = duración típica del segmento en muestras (p. ej.
una época de 512) y ajusta si la validación no mejora.

> Requiere `torch`. Si no está instalado, la app funciona y solo se deshabilitan
> las opciones de red neuronal. Instalación CPU en Windows:
> `pip install torch --index-url https://download.pytorch.org/whl/cpu`

## Control en tiempo real (clasificación en línea → controlador)

Con un modelo entrenado puedes clasificar la señal **en vivo** y enviar la clase
detectada a un controlador externo (brazo robótico, carrito…). Pestaña **Control**:

1. Entrena o importa uno o varios modelos (pestaña *Clasificación*).
2. Conecta una fuente en la pestaña **Tiempo real** (Simulado, LSL, Emotiv, CyKit).
3. En **Control**: elige el **modelo** (si el proyecto tiene varios), ajusta la
   **ventana** (muestras a clasificar), el **intervalo**
   y la **confirmación K** (nº de predicciones iguales seguidas para confirmar una
   clase — evita que el controlador oscile).
4. Define el **comando por clase** (texto que se envía por cada clase; por defecto
   el nombre de la clase).
5. Elige la **salida**: *Registro* (solo mostrar), *UDP* (host/puerto) o *Puerto
   serie* (Arduino). Pulsa **Iniciar control**.

Cada ventana entrante se procesa con el **mismo pipeline** del proyecto y se
clasifica; cuando una clase se confirma (K iguales seguidas) se envía su comando
una sola vez (al cambiar de clase estable). En pantalla ves la predicción actual,
su confianza y el último comando enviado.

> El controlador externo (tu robot/carrito) solo tiene que **escuchar** los
> comandos: por UDP, un script o microcontrolador que reciba en ese puerto; por
> serie, un Arduino que lea líneas. La salida serie requiere `pyserial` (opcional).

## Métodos basados en la literatura

Algunas técnicas implementadas y las referencias en que se apoyan:

| Etapa | Método (en la app) | Referencia |
|---|---|---|
| Preprocesamiento | Eliminación de artefactos por **ICA** (rechazo por kurtosis) | Revisión de eliminación de artefactos, doi:10.18280/isi.290124; preprocesamiento y extracción, doi:10.3390/app152212075 |
| Características | Potencias por banda + Hjorth | doi:10.3390/app152212075 |
| Clasificación | **Geometría de Riemann** (MDM, tangent space) | Revisión Riemann, arXiv:2407.20250; filtrado espacial de Riemann (RSF), doi:10.1145/3691521.3691529 |
| Clasificación | **CSP + LDA** | Revisión DL en MI, doi:10.1016/j.neucom.2024.128577; control multiclase EPOC X, doi:10.3389/fninf.2025.1625279 |
| Deep learning | **EEGNet** (CNN compacta) | Lawhern et al., doi:10.1088/1741-2552/aace8c |

**Hoja de ruta** (inspirada en las mismas fuentes, aún no implementada):
EEGNet con atención — AMEEGNet (doi:10.3389/fnbot.2025.1540033), CIACNet
(doi:10.3389/fnins.2025.1543508); **Transformers** — TCFormer
(doi:10.1038/s41598-025-16219-7), SATrans-Net (doi:10.1038/s41598-025-30806-8),
revisión (doi:10.3389/s25051293); **transfer learning** ConvoReleNet
(doi:10.3389/fnins.2025.1691929); reconocimiento de emociones
(doi:10.3390/math13020254); lazo cerrado para neurorehabilitación
(doi:10.2196/72218); control robótico en tiempo real (doi:10.1038/s41467-025-61064-x).

## Arquitectura

```
eeg_studio/
├── config.py            # constantes (canales EPOC+, bandas, rutas)
├── app.py               # arranque de QApplication
├── core/                # lógica de dominio (sin dependencias de UI)
│   ├── csv_loader.py     # lectura de CSV de OpenViBE
│   ├── mat_loader.py     # conversión .mat (BNCI 2a) -> CSV
│   ├── mne_loader.py     # conversión .fif/.edf/.gdf… (MNE) -> CSV
│   ├── recording.py      # grabación inmutable (fuente de solo lectura)
│   ├── preprocessing.py  # pipeline de filtros/referencia/normalización
│   ├── processing.py     # extracción de características (picklable)
│   ├── dataset.py        # datasets de características y de señal cruda
│   ├── classification.py # entrenamiento/predicción (clásicos + redes)
│   ├── neuralnet.py      # redes configurables MLP/CNN/LSTM (PyTorch)
│   ├── changelog.py      # control de cambios (undo/redo + auditoría)
│   └── project.py        # modelo de proyecto (orquesta todo lo anterior)
├── acquisition/         # captura en vivo (opcional)
│   ├── base.py           # StreamSource (productor/consumidor seguro para Qt)
│   ├── simulated.py      # fuente sintética (sin hardware)
│   ├── lsl.py            # OpenViBE Acquisition Server vía LSL (pylsl)
│   ├── emotiv.py         # lector nativo EPOC+ por USB (HID+AES, sin OpenViBE/CyKit)
│   ├── tcp.py            # CyKit / socket TCP genérico (respaldo)
│   └── recorder.py       # grabación a CSV formato OpenViBE
├── inference/           # control en tiempo real (clasificación en línea)
│   ├── online.py         # clasificar ventana + suavizado de predicciones
│   └── sinks.py          # salida de comandos: registro / UDP / serie
├── workers/             # ejecución asíncrona (QThread)
└── ui/                  # widgets PyQt6 (visor, paneles, ventana principal)
    ├── nn_config.py      # editor de arquitectura de redes neuronales
    └── cykit_launcher.py # configurador/lanzador de CyKit (banderas + cantidades)
```

## Pruebas

Pruebas de humo que cubren núcleo, interfaz (modo *offscreen*), adquisición,
redes, Riemann/CSP, ICA y concurrencia. Para ejecutar una:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"   # para las pruebas de interfaz
.\.venv\Scripts\python.exe -m tests.smoke_test
```

Suites disponibles (`tests/*.py`): `smoke_test`, `gui_smoke`, `acq_smoke`,
`acq_gui_smoke`, `lsl_smoke`, `params_smoke`, `nn_smoke`, `nn_gui_smoke`,
`emotiv_smoke`, `cykit_smoke`, `svm_smoke`, `ui_extras_smoke`, `riemann_smoke`,
`ica_smoke`, `concurrency_smoke`, `diskcache_smoke`, `tcp_parser_smoke`,
`pipeline_select_smoke`, `example_smoke`, `markers_smoke`, `marker_view_smoke`,
`segment_view_smoke`, `rf_params_smoke`, `online_smoke`, `control_gui_smoke`,
`mat_smoke`, `fif_smoke`, `fif_label_smoke`, `channels_smoke`, `models_smoke`,
`compress_smoke`, `marker_cull_smoke`, `autosave_smoke`, `history_smoke`,
`features_view_smoke`, `missing_source_smoke`, `project_files_smoke`,
`dataset_build_smoke`, `theme_smoke`, `train_progress_smoke`,
`pipeline_toggle_smoke`, `emotiv_mode_smoke`, `quality_smoke`,
`filter_design_smoke`, `eog_exclude_smoke`, `signal_edit_smoke`,
`preload_smoke`, `config_export_smoke`, `config_import_smoke`,
`metrics_plot_smoke`, `split_report_smoke`, `channel_isolate_smoke`,
`model_config_smoke`, `history_tree_smoke`, `signal_tabs_smoke`,
`multi_pipeline_smoke`, `pipeline_ui_smoke`, `live_scale_smoke`,
`live_segment_smoke`, `signal_scale_smoke`, `project_portable_smoke`,
`segment_edit_smoke`, `csv_export_view_smoke`, `source_rename_smoke`,
`orphan_recordings_smoke`, `recording_control_smoke`, `repeat_segment_smoke`,
`arm_control_smoke` (control del brazo MaxArm por HTTP, sin hardware),
`source_panel_smoke` (ordenar la lista de fuentes e indicadores de contenido),
`e2e_smoke` (flujo completo de extremo a extremo).

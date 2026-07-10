# Changelog

Todos los cambios notables de **DELFIN EEG Studio** se documentan en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).
Las fechas usan el formato `AAAA-MM-DD`. Mientras no haya versiones publicadas, los
cambios se agrupan por fecha de trabajo.

> Nota: este `CHANGELOG.md` documenta el **código del repositorio**. Cada proyecto
> `.eegproj` tiene además su propia bitácora (`changelog.json`) con el historial de
> ediciones del usuario (undo/redo y línea de tiempo) — son cosas distintas.

---

## [2026-07-10]

### Añadido
- **Aviso de señal retrasada (lag)** en «Tiempo real»: si la señal en vivo se cuelga
  (deja de llegar) o su tasa efectiva cae muy por debajo de la nominal —típico del
  EPOC+ con la batería baja—, se muestra una **advertencia** (estado en rojo + barra
  de estado), sin bloquear ni interrumpir la grabación.
- **Código de colores por región en el visor de señales**, con **dos tonos por
  zona** (como el gorro EPOC+): **azul** frontal (AF3/AF4 más fuerte, F7/F8 más
  claro), **rojo** central/temporal (F3/F4 vino, FC5/FC6/T7/T8 salmón) y **verde**
  parieto-occipital (P7/P8 claro, O1/O2 oscuro). Los canales con nombres desconocidos
  usan la paleta cíclica. El **visor en vivo (Tiempo real)** usa ahora **el mismo
  código de colores por región** que el visor de señales.
- **Recientes de la pantalla de bienvenida: renombrar y quitar**. Clic derecho sobre
  un proyecto reciente para **renombrarlo** (mueve la carpeta `.eegproj` y actualiza
  el nombre interno del proyecto) o **quitarlo de la lista** (solo lo olvida de
  recientes; **no** borra nada del disco).
- **Brazo simulado a pantalla completa: ahora con controles**. La ventana de pantalla
  completa ya no muestra solo el brazo: incluye a la derecha el **D-pad de acciones**
  (arriba/abajo, izquierda/derecha, agarre/soltar + HOME) y los **sliders por
  articulación**, y un **botón «✕ Cerrar (Esc)»** visible para volver (antes solo se
  salía con Esc, sin indicación). Mover el brazo ahí **sincroniza** el panel principal.
- **Control por clic en las vistas 2D del brazo** (como en `Proyecto_RNN`): clic en la
  **vista superior** gira la **base** para apuntar a ese punto; clic en la **vista
  lateral** acerca el efector a esa **altura/distancia** moviendo hombro/codo/muñeca
  (IK aproximada por descenso de coordenadas, respetando límites y piso). Funciona
  tanto en el panel como en pantalla completa.
- **Barra de paneles a la izquierda (estilo PyCharm)**: una barra vertical con un
  **botón por panel** (Fuentes, Herramientas, Historial) que lo **despliega o
  colapsa** con un clic; el botón queda marcado cuando el panel está visible.
- **Escalas de los ejes en el visor de señales**: nuevo apartado «Ejes» para fijar
  a mano el **rango X** (tiempo: «desde» + «ventana») y el **rango Y** (amplitud:
  min/max), más un botón **Auto (ajustar)**. Los campos reflejan el rango actual al
  hacer pan/zoom con el ratón.
- **Indicador de segmento durante el estímulo**: mientras se reproduce el video, se
  resalta un aviso (**«● SEGMENTO: ‹clase›»**) sobre el reproductor a pantalla
  completa —y en el estado de la interfaz— cada vez que el instante actual cae dentro
  de un segmento etiquetado.
- **Brazo simulado**: las **vistas laterales 2D** (lateral + superior) ahora son
  **colapsables** (botón para ocultarlas y ganar espacio para el 3D), y un botón
  **⛶ Pantalla completa** abre **solo el brazo** a pantalla completa para mejor
  visualización (Esc para volver).
- **Línea de tiempo del estímulo**:
  - **Repetir un segmento periódicamente**: selecciona un segmento y elige el
    **periodo** y el **nº de repeticiones**; se generan las copias espaciadas (sin
    salirse del video).
  - **F6** funciona como **inicio/fin de segmento** (mismo efecto que el botón).

### Cambiado
- **Importar `.eegbundle` no duplica lo ya presente**: al importar un bundle, las
  **fuentes** ya existentes (por id o nombre de archivo) y los **segmentos/etiquetas**
  repetidos se **omiten**; solo se traen los que faltan (antes los segmentos se
  sobrescribían por completo).
- **Estímulo: un video nuevo empieza SIN marcas/segmentos automáticos** (antes se
  prellenaban una marca y un segmento por defecto). Ahora el usuario coloca todo a
  mano en la línea de tiempo.
- **Importar configuración de estímulos pregunta ante repetidos**: si al importar un
  JSON de estímulos encuentra configuraciones **iguales a las ya presentes** (misma
  etiqueta y archivo de video, o mismo id), pregunta si **sobrescribir** o **ignorar**
  las repetidas (o cancelar); las nuevas se importan siempre. Antes se duplicaban.
- **Los paneles se re-adaptan al desplegarse**: al **ocultar** un panel (Fuentes /
  Herramientas / Historial) el visor central recupera su espacio, y al **volver a
  mostrarlo** recupera un tamaño usable (antes Qt podía restaurarlo colapsado).

### Corregido / reforzado
- **Creación de `.eegbundle` blindada**: el bundle se escribe primero en un archivo
  temporal (`.part`) y solo al final se **reemplaza atómicamente** el destino — un
  fallo a mitad ya **no deja un bundle corrupto** en su sitio. Cada binario
  (modelo/dataset/fuente) se empaqueta de forma **tolerante**: si uno falla (archivo
  bloqueado, ilegible o ausente) se **omite y se anota** en vez de abortar todo el
  export, y al terminar se **verifica la integridad** del ZIP. Al leer, se rechaza con
  un error claro cualquier archivo que no sea un bundle válido (no-ZIP o sin
  `bundle.json`). La exportación avisa de los elementos omitidos.
- **El visor de señal ya no impone un ancho mínimo enorme**: las dos filas de
  controles van ahora en un desplazamiento horizontal, así el visor **se puede
  encoger** (antes su ancho mínimo aplastaba los demás paneles). Su ancho mínimo pasó
  de ~900 px a ~76 px.
- **Filtros pasa-banda/altas/bajas + notch blindados**: verificado que atenúan las
  frecuencias correctas (respuesta en frecuencia). Además, ya **no revientan** con
  parámetros inválidos (p. ej. `low > high` en el pasa-banda, `cutoff ≤ 0`) ni con
  **segmentos muy cortos** (se ajusta el `padlen` de `filtfilt`/`sosfiltfilt`; antes
  lanzaban `ValueError`).

---

## [2026-07-09]

### Añadido
- **Estimulación sincronizada** (nuevo módulo en «Tiempo real», bajo la grabación):
  reproducir un **video de estímulo** dispara **automáticamente** la grabación EEG y
  coloca **segmentos exactos** en los tiempos definidos — elimina el error humano al
  etiquetar. Los 6 videos de `data/videos` se **mapean solos** a las 6 clases Delfin
  (arriba/abajo/izquierda/derecha/agarre/soltar). Al configurar un video se abre una
  **línea de tiempo** (con vista previa) para fijar las **marcas** (instantes) y los
  **segmentos** (lapsos); la configuración **se guarda en el proyecto**. Los estímulos
  ya configurados aparecen en la sección con **▶ Reproducir**: solo pide el nombre de
  la grabación y procede solo. El reproductor se lanza **a pantalla completa en un
  monitor externo** (si lo hay; si no, en la principal, sin cerrar la interfaz) y, al
  terminar, se **asegura de guardar todo** y coloca los segmentos exactos (calculados
  desde la línea de tiempo, descontando el desfase inicio-grabación/video). Los videos
  se referencian desde `data/videos` (no hay `ffmpeg` para comprimir, así que se toman
  del origen). No sustituye la grabación manual existente. Requiere QtMultimedia
  (incluido en PyQt6); si falta, la sección se muestra deshabilitada.
  - **General (cualquier proyecto)**: al añadir un estímulo se abre un **explorador
    de archivos** (arranca en `data/videos` pero sirve para videos en cualquier
    ubicación). Las **clases se toman del proyecto** (segmentos ya etiquetados +
    estímulos), no de una lista fija; el nombre Delfin solo autodetecta la clase por
    el archivo. Un video puede llevar **varias clases** (marca/segmento con su clase).
  - **Línea de tiempo estilo editor de video**: barra con *playhead*, marcas y
    segmentos dibujados, y el **instante exacto bajo el cursor** al pasar el ratón;
    clic/arrastre para moverse a un punto y fijar ahí la marca o el segmento.
  - **Exportar / importar** la configuración de estímulos (JSON). Al importar en otro
    proyecto/equipo, si no encuentra un video **pregunta su ubicación** y reubica por
    nombre.
  - **Selector de monitor**: elige en qué pantalla se despliega el video del estímulo
    (a pantalla completa); la interfaz principal se queda en la suya. Lista los
    monitores conectados (con un botón para actualizarla) y recuerda la elección; por
    defecto usa un monitor externo si lo hay.
  - **Vista previa del frame en la línea de tiempo**: al moverte por la barra (o al
    cargar el video) la vista muestra el **frame correspondiente** a esa posición, y
    un campo **«Ir a (s)»** manda el cursor a un **instante exacto** (escribes el
    segundo y pulsas Ir/Enter); el campo también refleja el tiempo actual.
- **Perfiles de control + brazo simulado** en la pestaña **Control**. El control del
  actuador ahora es un **perfil** seleccionable: **«Brazo MaxArm (real)»** (el de
  antes, por HTTP) y **«Brazo simulado»** (nuevo, sin hardware). El brazo simulado es
  un **4DOF** (base, hombro, codo, muñeca) extraído/adaptado del proyecto de
  referencia `Proyecto_RNN` (módulos de **construcción**, **cinemática directa** y
  **control**; se omiten la cinemática inversa y las series temporales), dibujado en
  **2D con pyqtgraph** (vista lateral + superior, sin dependencias nuevas). Se
  controla con los **mismos 6 comandos**: arriba/abajo mueven el hombro,
  izquierda/derecha giran la base (aquí sí funcionan), agarre/soltar la pinza. El
  **D-pad** manual y el **clasificador en tiempo real** pueden moverlo — «controlar
  con la mente» sin necesidad del robot físico. El perfil simulado **no usa salida
  externa**: al iniciar el control, el clasificador mueve el brazo directamente.
  - **Vista 3D** del brazo (OpenGL vía `pyqtgraph.opengl`) además de las 2D; si no
    hay **PyOpenGL**, degrada con elegancia a solo las proyecciones 2D.
  - **Control por articulación**: un **slider por joint** (base/hombro/codo/muñeca)
    con la lectura del ángulo y un botón de **HOME**; se sincroniza con los comandos.
  - **Constructor de brazo**: pestaña «Construir brazo» con **preset** y una **tabla**
    de joints (nombre, eje, eslabón LinkX/Y/Z, masa, límites) para **elegir o construir
    el brazo desde cero**; al aplicar, reconstruye la cinemática (FK general para
    cualquier cadena de joints). Extraído/adaptado de `Proyecto_RNN`.
- **Longitud de la selección por tiempo** en el visor de la señal: nuevo campo
  **«Long.»** (segundos) junto a la selección que fija la **duración exacta** de la
  región marcada, manteniendo el inicio (si no cabe hasta el final, corre el inicio
  hacia atrás). Se **sincroniza en ambos sentidos**: al arrastrar la región, el
  campo refleja su longitud; al escribir un valor, la región se ajusta. Útil para
  marcar ventanas de duración exacta (p. ej. tareas de 5 s del paradigma Delfin).
- **Medidor de batería de la diadema** (Emotiv) en «Tiempo real»: muestra el % de
  batería y **avisa cuando baja de un umbral configurable** (por defecto **70%**,
  porque la diadema vieja suele fallar por debajo). El umbral se ajusta en la propia
  interfaz y se recuerda entre sesiones. Se decodifica del propio flujo del casco;
  las fuentes que no reportan batería ocultan el medidor.

### Corregido
- **Crash al recolocar/flotar el panel «Herramientas»**: la vista 3D del brazo
  (`GLViewWidget`) se reparenta al mover el dock y su contexto OpenGL se recreaba,
  lo que hacía **crashear la app**. Se activan **contextos OpenGL compartidos**
  (`AA_ShareOpenGLContexts`) antes de crear la aplicación, así el widget 3D sobrevive
  a la reubicación del panel.
- **Crash al cancelar la línea de tiempo de un estímulo**: al pulsar Cancelar, el
  reproductor de video (`QMediaPlayer`) quedaba vivo al destruir el diálogo y la
  interfaz se caía. Ahora se **libera el video** (stop + soltar salida) tanto al
  aceptar como al cancelar (y también en el reproductor a pantalla completa).
- **Bug gráfico de la vista 3D del brazo simulado**: se dibujaba cada eslabón como
  un item OpenGL suelto (líneas erráticas) y el efector se solapaba con las
  articulaciones (mancha blanca). Ahora el brazo es **una sola polilínea**, con las
  articulaciones y el efector como marcadores aparte, y la cámara se ajusta al alcance.
- **Distribución del perfil «Brazo simulado»** reorganizada: antes el control estaba
  repartido en pestañas (vista / sliders / constructor) y era difícil de manejar.
  Ahora la **simulación (3D + 2D) y los sliders por articulación están juntos** en
  una sola vista (se ve el brazo mientras se controla), y el constructor pasa a un
  **diálogo** («Construir / elegir brazo…»).

### Verificado / reforzado
- **Recepción en segundo plano (dos monitores) y blindaje de la grabación**: la
  grabación ya **no depende del temporizador de la GUI** (que Windows estrangula
  cuando la app no tiene el foco, lo que truncaba grabaciones). Ahora se escribe en
  el **hilo productor** (un «tap» sobre el flujo de muestras), con **flush + fsync
  periódico** (cada ~1 s) para que un cierre o fallo no pierda lo grabado; el timer
  de la vista en vivo pasa a `PreciseTimer`. Prueba `recording_robust_smoke` (captura
  completa sin consumir la cola, volcado a disco, aviso de batería).
- **Auditoría del guardado automático**: revisados todos los disparadores (cada
  mutación del proyecto llama a `_after_state_change` → autosave con debounce de
  800 ms, o a `_persist_now` inmediato para lo crítico como nuevas grabaciones),
  el guardado atómico (`tmp`+`fsync`+`os.replace`), el reintento ante fallo, el
  guardado de precaución al cerrar y el sidecar `.marks.json` de las grabaciones.
  `autosave_smoke` ampliado con los casos de **fallo→reintento**, `_persist_now`
  con fallo y guardado al cerrar. Sin cambios de código necesarios: está sólido.

---

## [2026-07-08]

### Añadido
- **Ordenar el panel de Fuentes** con un selector arriba de la lista: **orden
  propio** (arrastrar para reordenar, se guarda en el proyecto y es reversible con
  Ctrl+Z), **alfabético (A→Z)**, **fecha de creación** y **última modificación**
  (por el archivo en disco). El modo elegido se recuerda entre sesiones.
- **Indicador de contenido por archivo** en el panel de Fuentes: un **punto
  pequeño y discreto** a la derecha de cada señal indica si tiene datos
  etiquetados — **verde** si tiene **segmentos**, **ámbar** si solo tiene
  **marcadores** (Event Id). No modifica el tamaño de la fila ni el nombre; el
  recuento se calcula en segundo plano y el *tooltip* detalla cuántos hay.
- **Estilo del panel de Fuentes** renovado, inspirado en el árbol de proyecto de
  PyCharm: cabecera de orden discreta, filas planas con **selección redondeada**
  y resaltado al pasar el cursor, a juego con el tema oscuro.
- **Control del brazo robótico MaxArm** (Hiwonder + ESP32) desde la pestaña
  **Control**. Sección de **prueba manual** con botones para los 6 comandos del
  proyecto Delfin (**Arriba, Abajo, Izquierda, Derecha, Agarre, Soltar**), además de
  **HOME** y **Probar conexión**. Envía peticiones **HTTP** al firmware (punto de
  acceso `MaxArm_IPN`, `http://192.168.4.1`): `/cmd?<id>=<valor>` mueve los servos
  (**1=Hombro** → arriba/abajo) mediante **pulsos** de velocidad continua, y
  `/pump?on=` activa la bomba de succión (**agarre**=encender, **soltar**=apagar).
  Nueva **salida «Brazo MaxArm (HTTP)»** en el modo en tiempo real, para que el
  **clasificador mueva el brazo** con las clases detectadas («controlar con la
  mente»). El envío es **no bloqueante** (cada comando en su propio hilo) y el mapeo
  clase→acción vive en `inference/arm.py` (editable). Cliente HTTP nuevo
  (`ArmClient`) + salida (`ArmHttpSink`), con `make_sink("arm", …)`.

### Corregido
- **Formato de `/cmd` del brazo MaxArm**: se enviaba `/cmd?id=<n>&v=<x>`, pero el
  firmware espera la clave del servo directamente (`/cmd?<n>=<x>`, p. ej.
  `/cmd?1=1.000`); por eso solo respondía HOME. Ahora los movimientos funcionan.
- **Izquierda/Derecha** apuntaban a la **base giratoria (servo 2)**, que está
  **sin servicio** en el firmware: sus botones siguen visibles pero **inhabilitados**
  (y el clasificador los ignora). Arriba/Abajo controlan el Hombro; Agarre/Soltar,
  la bomba.
- **Ruta de los datos de ejemplo tras la reestructuración a `src/`**: las pruebas
  y los generadores de ejemplo buscaban la carpeta `EEG/` con una ruta relativa
  fija (`../../EEG`) que dejó de resolver al mover el árbol un nivel más adentro
  (`EEG_Studio/` → `src/EEG_Studio/`). Ahora la localizan subiendo por los
  directorios padre (`tests.data_dir()`), resistente a futuros cambios de nivel.

### Organización del repositorio
- Los **datos de ejemplo** de las pruebas se movieron a **`data/raw/EEG/`**
  (siguiendo la nueva estructura), **gitignored** para no subir señales al repo.
- El entorno virtual (`.venv`) vive ahora junto al código en **`src/EEG_Studio/`**.
- Los **proyectos de prueba locales** se movieron bajo `data/`
  (`data/Proyectos_prueba/`, `data/Prueba_interfaz/`), **gitignored** (no se suben
  al repo). Incluye el proyecto recuperado `Señales_Delfin.eegproj`.
- La **referencia técnica de desarrollo** (arquitectura, métodos, pruebas) se
  conserva en **`src/EEG_Studio/README_TECNICO.md`**, aparte del `README.md`
  oficial de la raíz.

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
- **Generar segmentos periódicos** (clic derecho sobre un segmento → «Repetir
  periódicamente…»): marcas el **primero** y la app crea los demás hacia adelante a
  un **intervalo regular** (p. ej. 5 s de tarea cada 15 s), con la misma duración y
  etiqueta, hasta un total o hasta el final de la señal. Ideal para protocolos
  repetitivos (tarea/reposo). En **un solo paso deshacible**.
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

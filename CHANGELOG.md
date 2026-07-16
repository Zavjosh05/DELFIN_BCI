# Changelog

Todos los cambios notables de **DELFIN EEG Studio** se documentan en este archivo.

El formato se basa en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).
Las fechas usan el formato `AAAA-MM-DD`. Mientras no haya versiones publicadas, los
cambios se agrupan por fecha de trabajo.

> Nota: este `CHANGELOG.md` documenta el **código del repositorio**. Cada proyecto
> `.eegproj` tiene además su propia bitácora (`changelog.json`) con el historial de
> ediciones del usuario (undo/redo y línea de tiempo) — son cosas distintas.

---

## [2026-07-16]

### Añadido
- **Modo planar (2D) del brazo simulado**: casilla que hace que el efector se mueva sobre
  un **plano vertical** (ortogonal al plano de soporte de la base), con la **base fija**:
  arriba/abajo = altura, izquierda/derecha = alcance (acercar/alejar). Pensado para
  etiquetas 2D como las de `señales_finales` (arriba/abajo/izquierda/derecha), donde girar
  la base en 3D no casa con un movimiento bidimensional. Sin marcar (3D), el
  comportamiento es el de antes (izquierda/derecha giran la base). Usa la IK planar ya
  existente (base fija). En modo planar, la vista 3D **dibuja ese plano ortogonal** (una
  rejilla vertical naranja alineada con el brazo, sobre la zona de trabajo), para ver a
  dónde puede moverse el efector. La casilla está en el panel y en la pantalla completa,
  delegando en un único estado. Cubierto por `sim_arm_smoke` [11].
- **La interfaz indica que el porcentaje de un clasificador es la exactitud (accuracy)**:
  nota bajo la lista de «Modelos entrenados» y etiqueta explícita en el detalle
  («Exactitud (validación cruzada)» en clásicos/Riemann, «Exactitud (holdout)» en redes).
  Es la fracción de segmentos bien clasificados (media de la validación cruzada, o del
  holdout en redes). Cubierto por `split_report_smoke`.
- **Batería de pruebas en paralelo (`tests/run_all.py`)**: lanza las ~86 pruebas de humo,
  cada una en su proceso aislado, repartidas entre los núcleos. En un portátil hace la
  batería completa en **~2-3 min** (en serie eran ~10, porque cada proceso paga varios
  segundos FIJOS de arranque —PyQt6, numpy, sklearn, pyqtgraph— y en serie eso se paga
  86 veces). Imprime PASS/FAIL con tiempos (las más lentas arriba) y termina en != 0 si
  alguna falla. Filtro por nombre con `-k`. Fuerza a los procesos hijos a un solo worker
  y un solo hilo de cálculo (`EEG_N_WORKERS`, `OMP_NUM_THREADS`…) para que N pruebas × M
  hilos internos no sobresuscriban la CPU.
- **Generador de grabaciones sintéticas para las pruebas (`tests.sample_csv`)**: escribe
  un CSV OpenViBE pequeño pero válido (14 canales, épocas, señal reproducible). Ahora
  **ninguna prueba depende de archivos concretos**: antes 13 cargaban `Prueba_001.csv` /
  `Prueba_002.csv` de `data/raw/EEG/` —que está en `.gitignore`—, así que **fallaban en
  un clon limpio** (otra máquina, un compañero). Además son más rápidas al parsear datos
  pequeños en vez de CSV de varios MB. El generador del proyecto de ejemplo acepta ahora
  CSV y carpeta de salida, para generarse en un temporal con datos sintéticos.

### Cambiado
- **El modelo se puede cambiar con el control en tiempo real EN MARCHA**: elegir otro
  modelo (en el panel o en el selector de la pantalla completa del brazo) ahora se aplica
  al bucle en vivo —reinicia el suavizador y corta la retención— en lugar de quedar el
  modelo fijado al arranque mientras la interfaz mostraba otro. `EEG_N_WORKERS` permite
  fijar por entorno el nº de procesos de extracción de características.

### Corregido / reforzado
- **La exclusión de canales también vale para fuentes EN VIVO**: los canales excluidos
  se guardan con el nombre ORIGINAL del CSV (`Channel 13`), pero una fuente en vivo
  (Emotiv/LSL/Simulado) reporta el nombre clínico (`F8`). Al comparar solo contra el
  nombre crudo, en vivo no coincidía ninguno y **no se excluía nada**: el visor y, sobre
  todo, el clasificador recibían todos los canales, de modo que un modelo entrenado con
  los activos (p. ej. 12) fallaba con la señal en vivo (14). Ahora la exclusión compara
  contra ambas formas del nombre (cruda y alias), así que visor e inferencia ven los
  mismos canales activos que «Análisis (CSV)». La grabación sigue guardando la señal
  íntegra. Cubierto por `live_names_import_smoke` [4ter].
- **Los diálogos modales ya no quedan detrás de la pantalla completa del brazo**: `warn`
  y `info` se parentaban siempre a la ventana principal. Con la pantalla completa delante,
  un aviso (p. ej. iniciar el control sin fuente en vivo, o un error de clasificación)
  quedaba oculto tras ella y —siendo modal— bloqueaba la app sin que se viera: parecía un
  cuelgue. Ahora se parentan a la ventana **activa** (la pantalla completa si está abierta).
- **Los sliders de la pantalla completa siguen al control en vivo**: durante el control, el
  brazo se mueve por `SimArmView.refresh() → _fs.refresh()`, y ese `refresh()` solo
  redibujaba la vista 3D, así que los sliders por articulación de al lado se quedaban en la
  pose anterior. Ahora también se sincronizan. Ambos cubiertos por `sim_arm_smoke` [9f].
- **Más contraste en la escena del brazo simulado**: el fondo de los paneles (`SURFACE`) y
  la rejilla (`BORDER`) eran casi el mismo tono, y la rejilla 3D era muy transparente, así
  que el fondo, el plano de soporte y el brazo se confundían. Ahora la escena (2D y 3D)
  usa un fondo propio más oscuro, la rejilla del plano de soporte es clara y opaca, y los
  eslabones van más gruesos. Guarda de regresión en `sim_arm_smoke` [12].
- **El runner de la batería es robusto ante la carga**: reintenta en serie las pruebas que
  fallen en la corrida paralela (una prueba sensible al tiempo puede quedarse sin CPU con
  8 procesos y fallar sin ser un fallo real); si pasan al reintentar, se marcan «flaky», no
  como fallo. Además reconfigura su propio stdout a UTF-8 (imprimía la salida de las
  pruebas —con «→», «✓»— y reventaba en la consola cp1252 de Windows).

---

## [2026-07-15]

### Añadido
- **Control en tiempo real desde la pantalla completa del brazo simulado**: la
  ventana a pantalla completa incluye ahora **selector de modelo**, botón
  **Iniciar/Detener control** y el **comando predicho en grande** (con su confianza),
  para poder hacer una demo sin volver al panel. **Delega** en el panel de *Control*
  (mismo modelo, mismo botón, mismo bucle): no hay un segundo clasificador, y elegir
  el modelo en cualquiera de los dos sitios lo cambia en el otro.
- **Aumento de datos (*data augmentation*)**: nueva sección **«Aumento de datos (solo al
  entrenar)»** en el panel de *Clasificación*. Genera **copias perturbadas** de los
  ensayos para que el modelo aprenda el patrón y no el ensayo concreto — útil con pocos
  ensayos por clase, que es lo normal en imaginación motora. Técnicas (con su nivel
  ajustable y descripción en la interfaz):
  - **Ruido gaussiano** proporcional a la desviación de cada canal.
  - **Perturbación de amplitud** (escalado aleatorio del ensayo).
  - **Traslación circular** en el tiempo (solo modelos de señal cruda).
  - **Mixup**: combina dos ensayos **de la misma clase**, así la etiqueta sigue siendo
    válida.
  - **Aumentación automática**: cada técnica activa se aplica **al azar** (probabilidad
    configurable), de modo que cada copia es una combinación distinta.
  - Se aplica **SOLO al pliegue de entrenamiento** (y al *holdout* de las redes): la
    validación se mide siempre con los ensayos reales. Si se aumentara antes de partir,
    copias del mismo ensayo caerían en *train* y *test* y la exactitud subiría midiendo
    **memoria**, no aprendizaje. El modelo final sí se entrena con todo + el aumento.
  - Viene **apagado por defecto** (aumentar no siempre ayuda: a LDA con shrinkage o a
    Riemann les aporta poco), es **reproducible** (semilla), viaja con las
    **configuraciones de modelo** y los **bundles**, y queda guardado en el `.joblib`
    del modelo. Los modelos y configuraciones anteriores no cambian.
  - Núcleo reutilizable: `core/augment.py`.
- **Importar dataset (.npz)** en la pestaña *Dataset*: carga un dataset ya construido
  en **otra sesión** (o que te pasó otra persona) y lo deja **activo y listo para
  entrenar**, sin volver a extraer características ni necesitar los CSV de origen.
  Arranca en la carpeta `datasets/` del proyecto, muestra el resumen por clase y, si
  el archivo no es un dataset válido, **avisa sin pisar** el que ya hubiera cargado.
  Los segmentos del proyecto no se tocan (con «Construir dataset» se regenera).
- **Pares de clases más confundibles** en la ventana de **Métricas…**: una tabla,
  ordenada de peor a mejor, que dice **dónde** falla el modelo — qué dos acciones no
  sabe separar (p. ej. «izquierda ↔ derecha: 55 %»), en vez de solo la exactitud
  global. Para cada par mide: de los ensayos que **realmente** son una de las dos
  clases y se predijeron como una de las dos, qué fracción se acertó; muestra además
  las confusiones en cada sentido (`izquierda→derecha: 5`) y el nº de ensayos. El par
  se colorea (rojo = se confunden, verde = bien separadas).
  - Se calcula a partir de la **matriz de confusión de la validación cruzada**, así
    que **no hay que reentrenar** ni el modelo tiene que ser OvO: sirve para
    cualquier familia (clásicos, Riemann/CSP o redes). Con menos de 3 clases no
    aparece (el único par sería la exactitud global).
  - También sale en el informe de **texto** (`Ver texto…`, y el respaldo cuando no
    hay matplotlib). Núcleo reutilizable: `classification.pairwise_confusion()`.
- **Estabilidad del comando en el control en tiempo real**: nueva sección en la pestaña
  *Control*. Confirmar una clase no bastaba para controlar nada: con K=3 a 4 Hz se
  confirma una cada ~750 ms y la siguiente puede llegar 250 ms después, así que el
  actuador cambiaba de orden **sin completar ningún movimiento útil**. Dos ajustes:
  - **Confianza mínima** (60 % por defecto): las predicciones dudosas se ignoran *y
    cortan la racha*, para que el ruido no sume hacia la K. Si el modelo no da
    probabilidades el filtro no aplica; 0 = aceptar todas.
  - **Duración de la acción** (1500 ms por defecto): una vez confirmada, la acción se
    **mantiene** ese tiempo sin atender predicciones nuevas. Con **repetir** activado el
    comando se reenvía en cada intervalo mientras dura: como cada envío es un
    pulso/incremento del actuador, repetirlo convierte una orden suelta en un
    **movimiento sostenido**. Al terminar, la misma clase puede volver a confirmarse, así
    que mantener la imaginación motora encadena acciones; 0 = comportamiento anterior.
  - Los tres se pueden tocar **con el control en marcha** (se leen en cada ventana), que
    es justo cuando se quieren afinar. El panel muestra además el coste por ventana y
    cuántas se saltaron. Cubierto por `control_online_smoke`.

### Corregido / reforzado
- **Esc cierra la pantalla completa del brazo simulado**: solo funcionaba el botón
  **✕** (el ratón no necesita foco); la tecla no hacía nada. Eran dos cosas: la
  ventana se abría **sin activarse**, así que el teclado seguía yendo a la ventana
  principal; y, aunque llegara, dependía del `keyPressEvent` de la ventana, que no se
  dispara si el foco está en un hijo (la vista 3D, un botón, un slider). Ahora la
  ventana se **activa** al abrirse y Esc es un **atajo de ventana**: cierra tenga el
  foco quien lo tenga.
- **La ICA ya no destruye la señal cuando el pipeline lleva un CAR** (afecta al
  entrenamiento *y* al control en vivo). El **CAR** (referencia promedio común) resta la
  media entre canales, así que los deja linealmente dependientes: el **rango** baja a
  `nº de canales − 1`. Se le pedían a FastICA tantos componentes como canales, de modo
  que el blanqueado dividía entre un autovalor ≈ 0, la matriz de mezcla salía mal
  condicionada (`cond` ≈ 10¹⁷ en una grabación, ≈ 10³³ en una ventana de 2 s) y
  `inverse_transform` reconstruía **ruido**: un **84–100 % de error** *aunque no se
  anulara ningún componente*. Ahora el nº de componentes se recorta al **rango real** de
  los datos, con lo que el error de reconstrucción cae al **0 %** (`cond` ≈ 5) y, de
  paso, la ICA es **~40× más rápida** (ya no agota `max_iter` persiguiendo una dirección
  degenerada). Impacto: los datasets y modelos entrenados con un pipeline que combine
  **CAR + ICA** se construyeron sobre señal corrupta y **conviene reconstruirlos y
  reentrenarlos**. Cubierto por `ica_smoke` [2]–[4].
- **La caché de señal procesada caduca al corregir el preprocesamiento**: su firma solo
  dependía de la *configuración* del pipeline, así que un arreglo como el anterior —que
  cambia la **salida** sin cambiar la configuración— habría seguido sirviendo desde disco
  la señal vieja. Ahora la firma incluye `PROCESSING_VERSION`, que se sube con cada
  arreglo que altere el resultado de `apply_pipeline`.
- **El control en tiempo real ya no traba la interfaz**: la clasificación de cada ventana
  corría en el **hilo de la interfaz** (un `QTimer` llamando directamente a
  `classify_window`). Con un pipeline con ICA eso son ~100 ms cada 250 ms en el mismo
  hilo donde viven la adquisición y el visor, así que al iniciar el control **todo se
  trababa**. Ahora el temporizador solo toma la ventana y despacha el trabajo a un hilo;
  si una ventana tarda más que el intervalo se **salta** la siguiente en vez de
  encimarlas, y al detener se descartan las ventanas rezagadas (control por id de sesión)
  para que su resultado no intente escribir en una salida ya cerrada.
- **Reentrenar un modelo de Riemann/CSP ya no pierde su estrategia multiclase**: al
  reentrenarlo (o al entrenar su configuración desde un bundle) no se le pasaban los
  `clf_params`, así que un modelo OvO/OvR volvía a **«nativa»** en silencio y no era el
  mismo modelo. Los clásicos y las redes no estaban afectados.
- **El visor «Tiempo real» conserva los nombres y colores de los canales**: al
  conectar una **grabación** (fuente «Reproducir grabación»), los canales aparecían
  como `Channel 1`…`Channel 14` y con la paleta cíclica, perdiendo los nombres (AF3,
  F7…) y el **código de colores por región** que sí muestra «Análisis (CSV)». Causa:
  los CSV de OpenViBE nombran los canales `Channel N` y es el **proyecto** quien
  guarda el alias clínico, pero el visor en vivo recibía los nombres crudos del
  archivo (y `channel_color` asigna el color **por nombre**). Ahora el visor en vivo
  traduce los nombres con los alias del proyecto, así que ambas pestañas coinciden.
  Las fuentes que ya reportan nombres reales (Emotiv, LSL) no cambian.
- **El visor «Tiempo real» respeta los canales excluidos del proyecto**: si excluyes
  canales (p. ej. los EOG), el visor en vivo mostraba igualmente **todos**, mientras
  que «Análisis (CSV)» solo muestra los activos. Ahora ambos enseñan los mismos. La
  **grabación no se toca**: el CSV se sigue escribiendo íntegro desde el hilo
  productor — excluir es cosa del análisis, no de la captura. Si se excluyeran todos,
  no se filtra nada (no se deja el visor vacío).
- **El modo «Control» en vivo también usa solo los canales activos**: si excluías
  canales y entrenabas un modelo con los activos (p. ej. 12), la clasificación en
  vivo le pasaba **todos** los de la fuente (14) y fallaba por forma incompatible —
  el entrenamiento sí aplica la exclusión (`get_processed`) pero la inferencia no.
  Ahora el buffer que alimenta al clasificador lleva los mismos canales activos, así
  que entrenamiento e inferencia coinciden.
- **Riemann/CSP con «Uno contra Uno» (OvO) ya entrena**: elegir OvO en *CSP + LDA* o
  en *Riemann — MDM* reventaba con `Found array with dim 3, while dim <= 2 is
  required by OneVsOneClassifier`. Causa: esos modelos consumen **matrices de
  covarianza (3D)**, pero el meta-clasificador OvO de scikit-learn valida la entrada
  y exige 2D (OvR no la valida, por eso sí funcionaba). Ahora las covarianzas se
  **aplanan** para atravesar el meta-clasificador y se **reconstruyen dentro de cada
  binario** (el reshape es exacto), así que cada problema binario sigue viendo las
  covarianzas reales y aprendiendo **sus propios filtros CSP**. La estrategia
  «nativa» vuelve al pipeline de siempre (`cov → csp → lda`), sin envolver.
- **La prueba de multiclase ahora ENTRENA de verdad**: `multiclass_smoke` solo
  comprobaba la *estructura* del pipeline de Riemann/CSP y por eso no detectó el
  fallo anterior (solo aparece al hacer `fit()`). Ahora entrena y predice las **9
  combinaciones** (CSP+LDA / MDM / Tangent Space × nativa / OvO / OvR).

---

## [2026-07-14]

### Añadido
- **Estrategia multiclase: reducción a clasificadores binarios (OvO / OvR)**. Con
  muchas clases (las 6 acciones Delfin) la literatura BCI sugiere no entrenar un
  único modelo multiclase, sino descomponer el problema en varios **binarios** y
  decidir por votación. Nuevo selector **«Estrategia multiclase»** en el panel de
  *Clasificación* y en el diálogo **«Configuración…»** (para reentrenar), con la
  descripción de cada opción y el nº de modelos que implica:
  - **Uno contra Uno (OvO)**: un binario por cada **par** de clases (6 clases → 15)
    y mayoría de votos. Cada binario ve solo los datos de sus 2 clases.
  - **Uno contra el Resto (OvR)**: un binario por clase (6 clases → 6). Usa todos
    los datos, pero cada problema queda desbalanceado (1 vs N−1): conviene
    combinarlo con **«Peso de clases» = balanced**.
  - **Nativa** (por defecto): la de cada clasificador, como hasta ahora — el
    comportamiento y los modelos ya guardados no cambian.
  - Aplica a **RF, SVM y LDA** y también a **Riemann/CSP**. En **CSP+LDA** es donde
    más aporta: el CSP es **binario por naturaleza**, así que con OvO/OvR se envuelve
    **CSP+LDA juntos** y cada problema binario aprende **sus propios filtros
    espaciales** (enfoque estándar en imaginación motora multiclase). Nota: el SVM ya
    usaba OvO internamente, así que ahí el cambio es menor.
- **Panel de Fuentes: agrupado por sujeto y buscador**. Con decenas de señales
  (`sujeto001-abajo`, `sujeto001-arriba`…) la lista plana era inmanejable:
  - Nuevo modo de orden **«Agrupado por sujeto»**: inserta una **cabecera plegable**
    por sujeto (deducido del prefijo del nombre, tolerando `-`, `_` o espacio) con el
    nº de señales; un clic **despliega/pliega** el grupo.
  - Nuevo **buscador** sobre la lista: filtra por nombre y **despliega solo lo que
    coincide**, aunque su grupo esté plegado, ocultando las cabeceras que se quedan
    sin nada. Al limpiarlo vuelve a mandar el estado de plegado.

- **Mapas topográficos de los componentes ICA**: al seleccionar el paso
  *Eliminar artefactos (ICA)* en el preprocesamiento aparece el botón **«Ver mapas
  espaciales (ICA)…»**, que abre una ventana con un **mapa por componente** dibujado
  sobre un esquema de la cabeza (vista superior, nariz arriba): **rojo** = peso
  positivo, **azul** = negativo. Los componentes de **kurtosis alta** (candidatos a
  artefacto, los que ICA elimina) se **resaltan con ⚠**, para ver de un vistazo en
  qué zonas del cuero cabelludo surgen los artefactos (p. ej. frontal = parpadeos).
  Se calcula sobre la fuente abierta aplicando antes los pasos previos del pipeline
  (para que coincida con lo que ICA ve), en un hilo aparte; se puede **guardar como
  imagen** (PNG/PDF/SVG). Requiere `matplotlib` (opcional); si falta, se avisa.
  - Nuevo `core/montage.py` con las **posiciones 2D** de los 14 canales EPOC+ y
    `core/preprocessing.ica_decompose()` (descompone y devuelve la matriz de mezcla
    + kurtosis + flags, con el mismo ajuste que `ica_artifact`: lo que se ve = lo
    que se elimina).
  - **Paleta más legible**: los mapas usan un degradado divergente propio con
    **rojo y azul base más claros y vivos** (en vez del `RdBu` de matplotlib, cuyos
    extremos vino/azul marino oscuros costaba diferenciar entre tonos).

- **Configuraciones de modelo guardables, SIN entrenar**. En el panel de
  *Clasificación*, nueva sección **«Configuraciones de modelo (sin entrenar)»**: ajusta
  los hiperparámetros de cualquier clasificador y **guárdalos con un nombre** en el
  proyecto sin entrenar nada; cárgalos cuando quieras (**Cargar / Guardar actual… /
  Eliminar**). Cada clasificador ofrece siempre **«· Valores por defecto ·»** para
  volver a los valores originales del programa. Las configuraciones viven en el
  proyecto (`model_configs`) con **deshacer/rehacer** e historial, como el resto de
  ediciones; los proyectos anteriores siguen abriendo sin problema.
- **Entrenar todas las configuraciones guardadas de una vez**: botón que recorre las
  configuraciones del proyecto y entrena un modelo por cada una, una tras otra
  (nombrando cada modelo como su configuración). Avisa de cuántas entrenará y **omite
  las que necesiten datos que aún no existen** (dataset o segmentos).
- **Reentrenar todos los modelos entrenados**: botón en la caja de modelos que vuelve a
  entrenar cada modelo con **sus mismos hiperparámetros** pero con los **datos actuales**
  del proyecto, conservando su nombre — pensado para cuando **cambia el dataset** o los
  segmentos. Avisa antes de sustituirlos.
- **Los bundles/configuraciones llevan los hiperparámetros de los modelos y se pueden
  reutilizar con TUS datos**. Al **importar** un `.eegbundle` (o un `.eegcfg`), si trae
  configuraciones de clasificador, la app las **detecta** y ofrece **entrenarlas sobre
  los datos de este proyecto**: aparece una lista con cada configuración (modelo,
  clasificador y resumen de sus hiperparámetros) para elegir cuáles usar. Los modelos
  resultantes se **añaden** (sufijo `_local`) y **no sustituyen** a los modelos
  importados, así puedes comparar «el modelo de otro» con «sus parámetros sobre mis
  datos». Las configuraciones que necesitan datos que aún no existen (un dataset
  construido, o segmentos etiquetados para Riemann/CSP/redes) se muestran
  **deshabilitadas indicando qué falta**.
- El bundle incluye ahora también la **ventana de señal cruda** (`raw_window`) de cada
  modelo, necesaria para poder reentrenar Riemann/CSP/redes en otro proyecto.
- **Las configuraciones de modelo sin entrenar también se exportan/importan**: nueva
  casilla **«Configuraciones de modelo sin entrenar (N)»** al exportar el bundle (son
  solo texto, no pesan). Al importar se **añaden al proyecto** las que falten —las ya
  presentes, por nombre, no se pisan— y aparecen junto a los modelos entrenados en la
  oferta de **entrenar con los datos locales**.
- **Al importar se puede elegir QUÉ importar**: nueva ventana con una casilla por cada
  parte que trae el archivo (preprocesamiento, dataset, modelos entrenados,
  configuraciones sin entrenar y señales de origen), con el **número de elementos** de
  cada una y **todo marcado por defecto**. Recuerda además dónde queda cada cosa dentro
  del proyecto. Antes se importaba todo sin preguntar.

### Corregido / reforzado
- **Importar una configuración de estímulos localiza los videos sola**. Al traer una
  configuración de **otro equipo**, la ruta guardada no existe aquí, y la app pedía la
  carpeta **aunque los videos estuvieran en `data/videos`** (su sitio de siempre, que
  la app ya sabe encontrar). Ahora `relocate_video` busca por orden: la ruta original →
  la carpeta que indiques → **`data/videos`**; en el caso normal no pregunta nada.
  Además:
  - Si un video **no aparece en ningún lado**, antes se guardaba la ruta inexistente
    **sin avisar** y el estímulo quedaba inservible sin que se notara. Ahora **avisa**
    de cuáles faltan (y el estado lo cuenta como «N sin video»), conservando igualmente
    las marcas y segmentos de la configuración.
  - Si aun así no aparece, **sigues pudiendo indicar la carpeta**: la que elijas se
    recuerda para los demás videos, y si alguno vive en **otra** carpeta se te vuelve a
    preguntar por ese. Si **cancelas**, no se insiste con el resto.
- **Renombrar una señal ya no pierde sus marcas ni rompe la fuente al deshacer**. Al
  renombrar un CSV interno del proyecto pasaban dos cosas:
  - Las **marcas** viven en un archivo lateral llamado según el CSV
    (`<csv>.marks.json`) y **no se renombraba con él**: quedaba huérfano y las marcas
    de la grabación **se perdían**. Ahora el lateral se mueve junto al CSV.
  - **Deshacer** devolvía la ruta anterior en el proyecto pero **no renombraba el
    archivo de vuelta**, así que la fuente quedaba apuntando a un archivo inexistente
    (y daba errores). Ahora el movimiento del archivo va atado al cambio: deshacer y
    rehacer dejan el CSV, su lateral y el proyecto siempre coherentes.
  - Los **segmentos** no estaban afectados (referencian el id de la fuente, no su
    nombre) y siguen igual. Los archivos **externos** al proyecto se siguen sin tocar.
- **Los CSV importados de un bundle se llaman como la señal**. Antes se guardaban con
  el id delante (`<id>__señal.csv`), así que el archivo **no coincidía** con el nombre
  que se ve en la interfaz; y como al re-exportar se le añadía **otro** prefijo, cada
  ciclo importar→exportar encadenaba uno más (`id__id__id__señal.csv`) hasta volverlos
  ilegibles. Ahora el prefijo se usa **solo dentro del ZIP** (para que no choquen dos
  señales homónimas) y **al extraer se quita**: el archivo se llama como la señal. Si
  ese nombre ya estuviera ocupado, se guarda con un sufijo (`señal_2.csv`).
  - En consecuencia, las fuentes ya presentes se omiten **solo por id** (que es lo que
    identifica de verdad a una fuente, y se conserva entre proyectos). Antes también se
    omitían por nombre de archivo, lo que ahora habría descartado en silencio
    grabaciones **distintas** que casualmente se llamaran igual.
  - Los archivos ya importados con el nombre antiguo **no se tocan**: siguen funcionando.
- **Importar un bundle ya no borra tus pipelines**. Al traer el preprocesamiento se
  **reemplazaban TODOS** los pipelines del proyecto por los del bundle, así que perdías
  los tuyos (y recuperarlos exigía deshacer varias veces, porque una importación son
  muchos pasos). Ahora los pipelines del bundle **se añaden** a los que ya tienes
  —renombrando los que repitan nombre y omitiendo los idénticos—, **sin cambiar tu
  pipeline activo**; solo se sustituyen si el proyecto está en blanco (un único
  pipeline vacío), que es lo natural al empezar. Los **canales excluidos** y los
  **alias de canal** también se **fusionan** en vez de pisarse (los tuyos mandan).
- **Importar varios bundles ya no borra en silencio lo del anterior**. Todos los
  bundles traen su dataset con el mismo nombre (`dataset.npz`) y suelen repetir los
  nombres de modelo (`rf_1`), así que al importar un **segundo** bundle en el mismo
  proyecto **se sobrescribían sin aviso** el dataset y los modelos del primero (tanto
  en memoria como en `datasets/` y `models/`), y parecía que lo importado antes había
  desaparecido. Ahora **nada se pisa**: si el nombre está ocupado, lo nuevo se guarda
  con un sufijo (`dataset_2.npz`, `rf_1_2`) y el resumen de importación indica qué se
  renombró. Las fuentes y los segmentos ya se fusionaban bien y no cambian.
- **Barra de controles del visor reacomodable (señal y tiempo real)**: antes, con
  muchos botones/configuraciones, la barra superior del visor se **recortaba o
  desbordaba** y no se veían todos los controles. Ahora los controles se reparten
  en **varias filas** según el ancho de la ventana (nuevo `ui/flow_layout.py`), con
  un botón **⤢** para **expandir/compactar** la barra (verlos todos de golpe o
  dejarla compacta con desplazamiento). Cada etiqueta va pegada a su control para no
  separarse al reacomodarse, y los campos numéricos son más compactos. Afecta al
  visor de señal (CSV) y al visor en vivo (tiempo real).

---

## [2026-07-13]

### Añadido
- **Más hiperparámetros para los clasificadores clásicos**, disponibles tanto en el
  panel de **Clasificación** (al entrenar) como en el diálogo **«Configuración…»** de
  cada modelo (para reentrenar):
  - **Random Forest**: **«Mín. por hoja»** (`min_samples_leaf`) y **«Peso de clases»**
    (`class_weight`: ninguno / *balanced*).
  - **SVM**: **`coef0`** (término independiente para los kernels *poly*/*sigmoide*) y
    **«Peso de clases»** (*balanced*).
  - **LDA**: pasa de «sin parámetros» a ser configurable con **solver** (svd / lsqr /
    eigen) y **shrinkage** (ninguno / *auto*, Ledoit-Wolf). El *shrinkage* solo se
    habilita con *lsqr*/*eigen* (el *svd* no lo admite) y ayuda con **pocas muestras y
    muchas características**, situación típica en EEG. El **«Peso de clases» balanced**
    compensa un número desigual de ensayos por clase.

---

## [2026-07-11]

### Añadido
- **Controlar el brazo desde una grabación (sin diadema)**. Ahora se puede
  alimentar la clasificación con un archivo grabado en vez de la señal en vivo, de
  dos formas complementarias:
  - **Fuente «Reproducir grabación (archivo)»** (pestaña *Tiempo real*): reproduce
    un CSV como si fuera la diadema (a 1×, una sola pasada). Al conectarla, el visor
    en vivo y el modo *Control* funcionan igual que con el EPOC+ — útil para
    demostraciones y para revisar la señal sin hardware.
  - **«Controlar desde archivo grabado»** (pestaña *Control*): eliges una grabación
    (p. ej. `Sujeto001_Abajo.csv`), se clasifica **completa** por ventanas
    deslizantes (voto mayoritario) y se ejecuta **un movimiento** en el actuador del
    perfil activo (brazo simulado o MaxArm). Muestra la **clase esperada deducida
    del nombre del archivo** frente a la predicha, la confianza y —comparando contra
    esa verdad-terreno— la **exactitud** ventana a ventana, con **aviso de
    compatibilidad** si el archivo no encaja con el modelo o la clase no corresponde
    a un movimiento del brazo.
  - Núcleo reutilizable: `acquisition/playback.py::FilePlaybackSource` y
    `inference/online.py::classify_recording` (resumen con etiqueta ganadora,
    conteos por clase, confianza, exactitud y predicción por ventana).
- **Preprocesamiento: 3 pasos nuevos**, con la misma descripción integrada (qué hace
  el paso y qué efecto tiene cada parámetro) que ya tenían los filtros existentes:
  - **Reconstrucción de Subespacios de Artefactos (ASR)**: descompone la señal en
    componentes espaciales (PCA entre canales) y atenúa, ventana a ventana, los
    componentes cuya energía supera un umbral (`cutoff`) respecto a su nivel
    habitual — corrige ráfagas de artefacto (saltos de electrodo, movimiento) sin
    descartar la ventana ni tocar los componentes con señal limpia. Parámetros:
    `window_sec` (duración de la ventana de análisis) y `cutoff` (umbral de energía).
  - **Rechazo por umbral (Manual/Automático)**: recorta (clip) las muestras que
    superan un límite de amplitud, canal a canal. En modo `manual` usa un valor fijo
    (`threshold` en µV); en `automatico` calcula un umbral propio por canal a partir
    de sus datos (mediana ± `k`×MAD), adaptándose a la amplitud típica de cada canal.
  - **Corrección de la línea base**: resta a cada canal la media de una ventana de
    referencia al inicio del segmento (`baseline_sec`), típico en análisis tipo ERP
    para alinear al nivel previo a un evento; a diferencia de «Eliminar tendencia»
    (que ajusta toda la señal), aquí solo se usa el tramo inicial como referencia.
  - Nota: la **normalización** que se pedía junto con estos pasos ya existía como
    paso «Normalizar» (zscore/minmax); no se duplicó.

### Corregido / reforzado
- **Brazo simulado: se corrige la inversión izquierda/derecha**. Los comandos
  `izquierda`/`derecha` (D-pad, sliders del panel de Control y clase del
  clasificador) movían la base al lado contrario: por la convención histórica de la
  matriz de rotación (seno negado), un yaw positivo gira el brazo hacia la derecha,
  no hacia la izquierda como asumía el mapeo. Ahora `izquierda` lleva el efector a
  la izquierda (+y) y `derecha` a la derecha (−y), mirando a lo largo del brazo
  desde la base. El control por clic en las vistas 2D ya era correcto y no cambia.

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

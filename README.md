# 🧠 BCI (Brain-Computer Interface) - EEG Studio 🦾

## 📖 Descripción del Proyecto
Este repositorio contiene **DELFIN EEG Studio**, una plataforma de escritorio desarrollada en PyQt6 para la adquisición, procesamiento y clasificación de señales electroencefalográficas (EEG) mediante el casco Emotiv EPOC+. 

El objetivo final del sistema es el despliegue de modelos de Machine Learning y Deep Learning para la inferencia en tiempo real y el control de hardware externo (brazo robótico). Desarrollado durante el Verano de Investigación Delfín 2026 en el Centro de Investigación en Computación (CIC) del IPN.

### 🏛️ Encargado del Proyecto
* Dr. Juan Humberto Sossa Azuela

### 🎓 Asesores Académicos
* M. en C. Abel Alejandro Rubín Alvarado
* M. en C. Víctor Adonis Hernández Almendra
* M. en C. Jonathan Axel Cruz Vázquez

### 👨‍💻 Equipo de Ingeniería
* Leonardo Lagos López
* Joshua Iván Zavaleta Guerrero
* Matthew Steve Robbin Ruiz Pacheco

---

## ⚙️ Características Técnicas del Sistema

El software opera como un *pipeline* integral no destructivo (los datos crudos jamás se sobrescriben), abarcando cuatro fases:

1. **📡 Adquisición Híbrida:** Soporte para lectura *offline* (conversión nativa de `.csv`, `.mat`, `.fif`, `.edf`) y telemetría en tiempo real vía OpenViBE (LSL), CyKit/TCP, y un lector USB nativo integrado para el EPOC+ (sin depender de licencias Cortex).
2. **🧹 Preprocesamiento de Señales:** Aplicación de filtros pasa-banda/notch, Referencia Promedio Común (CAR) y eliminación automática de artefactos biomédicos mediante Análisis de Componentes Independientes (ICA) apoyado en pyqtgraph.
3. **🤖 Extracción y Clasificación (IA):** 
   * **Modelos Clásicos:** Random Forest, SVM y LDA usando características temporales y potencias por banda de frecuencia.
   * **Deep Learning (PyTorch):** Redes MLP, CNN 1D, LSTM y la arquitectura especializada **EEGNet**.
   * **Geometría de Riemann:** MDM y Tangent Space.
4. **🕹️ Control de Hardware (Inferencia Online):** Ejecución del modelo en vivo con suavizado paramétrico para emitir comandos estables hacia microcontroladores vía UDP o Puerto Serie (Arduino/Brazo Robótico).

---

## 📁 Arquitectura del Repositorio (Estándar Modular)
Para mantener la integridad del proyecto y evitar deuda técnica, el entorno está estrictamente modularizado:

* `🖨️ /assets/` - Modelos 3D y archivos de diseño de hardware (STL, 3MF, etc.).
* `📊 /data/` - Datasets EEG en crudo (`.eegbundle`, `.csv`) y configuraciones locales (`.eegproj`).
* `📝 /docs/` - Documentación gráfica, diagramas, métricas de rendimiento y manuales de interfaz.
* `💻 /src/` - Núcleo del código fuente de EEG Studio.
* `🐍 venv/` - Entorno virtual de Python (Ignorado por control de versiones).

---

## 🚀 Entorno de Ejecución y Despliegue

El sistema está diseñado para operar bajo un entorno virtual aislado optimizado para operaciones en hilos (`QThread`) y multiprocesamiento (`ProcessPoolExecutor`).

**1. 🏗️ Preparación del Entorno (Python 3.14 recomendado)**
```bash
python -m venv venv
source venv/bin/activate  # En Linux/macOS
# .\venv\Scripts\activate # En Windows
```
**2. 📦 Inyección de Dependencias**
```Bash
pip install -r src/EEG_Studio/requirements.txt
pip install pyqtgraph  # Motor de renderizado científico
```

**3. ▶️ Despliegue de la Interfaz**
```Bash
python src/EEG_Studio/run.py
```

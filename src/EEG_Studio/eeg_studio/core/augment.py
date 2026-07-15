"""Aumento de datos (*data augmentation*) para los ensayos de entrenamiento.

Con pocos ensayos por clase —lo normal en imaginación motora— los modelos de
mucha capacidad sobreajustan. Estas técnicas generan **copias perturbadas** de los
ensayos de entrenamiento para que el modelo aprenda el patrón y no el ensayo
concreto.

REGLA DE ORO (y el motivo de que esto viva aquí y no en la construcción del
dataset): **solo se aumenta el pliegue de ENTRENAMIENTO**, nunca el de
validación. Si una copia del mismo ensayo cayera en train y en test, el modelo lo
reconocería y la exactitud subiría sin que el modelo haya aprendido nada — se
mediría memoria, no aprendizaje. Por eso :func:`augment` se llama **dentro** de
cada pliegue de la validación cruzada (ver ``classification._cv_eval``).

Todas las técnicas **conservan la forma** de la señal, así que valen tanto para la
señal cruda ``(n, canales, muestras)`` (Riemann/CSP/redes) como para vectores de
características ``(n, k)`` (RF/SVM/LDA) — salvo las que necesitan el eje temporal,
que se ignoran si la entrada no es cruda.

Referencias de apoyo:

* F. Lotte, "Signal processing approaches to minimize or suppress calibration time
  in oscillatory activity-based brain-computer interfaces", *Proc. IEEE* 103(6),
  2015 — ensayos artificiales para MI con pocos datos.
* H. Zhang et al., "mixup: Beyond empirical risk minimization", *ICLR*, 2018 —
  combinaciones convexas de ejemplos y etiquetas.
* R. T. Schirrmeister et al., *Human Brain Mapping* 38, 2017.
  doi:10.1002/hbm.23730 — recorte de ventanas (pendiente: ver nota más abajo).

.. note::
   La **ventana deslizante** (*cropped training*) no está aquí a propósito: no
   conserva la forma (acorta la ventana) y exige (1) que la validación agrupe por
   ensayo y (2) votar entre los recortes al predecir. Hecha a medias **infla** la
   exactitud en vez de mejorarla, que es justo lo contrario de lo que se busca.
"""
from __future__ import annotations

import numpy as np

# Técnicas disponibles: clave -> (etiqueta, descripción para la interfaz).
TECHNIQUES: dict[str, str] = {
    "noise": "Ruido gaussiano",
    "amplitude": "Perturbación de amplitud",
    "shift": "Traslación circular (tiempo)",
    "mixup": "Mixup (mezcla de ensayos)",
}

TECHNIQUE_DESCRIPTIONS: dict[str, str] = {
    "noise": "Suma ruido gaussiano proporcional a la desviación de cada canal. "
             "Enseña al modelo a tolerar interferencias. «Nivel» = fracción de la "
             "desviación típica (0.05 = 5 %).",
    "amplitude": "Escala la señal por un factor aleatorio. Simula variaciones "
                 "naturales de intensidad entre ensayos y sesiones (contacto del "
                 "electrodo, atención). «Nivel» = variación máxima (0.1 = ±10 %).",
    "shift": "Desplaza la señal en el tiempo de forma circular (lo que sale por un "
             "extremo entra por el otro). Enseña que el patrón no depende del "
             "instante exacto. «Nivel» = desplazamiento máximo como fracción de la "
             "ventana. Solo aplica a señal cruda.",
    "mixup": "Combina linealmente DOS ensayos de la MISMA clase (y por tanto su "
             "etiqueta no cambia). Rellena el espacio entre ejemplos reales. "
             "«Nivel» = cuánto puede pesar el segundo ensayo (0.4 = hasta 40 %).",
}

# Nivel por defecto de cada técnica (deliberadamente conservador: perturbar de más
# destruye el ERD/ERS que se quiere clasificar).
DEFAULT_LEVELS: dict[str, float] = {
    "noise": 0.05,        # 5 % de la desviación del canal
    "amplitude": 0.10,    # ±10 %
    "shift": 0.10,        # ±10 % de la ventana
    "mixup": 0.40,        # el 2º ensayo pesa hasta 40 %
}


def default_config() -> dict:
    """Configuración por defecto: **desactivado**.

    Aumentar no siempre ayuda (a LDA con shrinkage o a Riemann, poco), así que no
    se activa a espaldas del usuario ni cambia lo que ya tenía entrenado."""
    return {
        "enabled": False,
        "copies": 1,          # copias aumentadas por cada ensayo original
        "techniques": {k: (k in ("noise", "amplitude")) for k in TECHNIQUES},
        "levels": dict(DEFAULT_LEVELS),
        "probability": 0.5,   # prob. de aplicar CADA técnica activa a cada copia
        "seed": 0,
    }


def describe(config: dict | None) -> str:
    """Resumen corto para la interfaz/informes."""
    cfg = config or {}
    if not cfg.get("enabled"):
        return "sin aumento"
    active = [TECHNIQUES[k] for k, on in (cfg.get("techniques") or {}).items()
              if on and k in TECHNIQUES]
    if not active:
        return "sin aumento (ninguna técnica activa)"
    return f"×{int(cfg.get('copies', 1)) + 1} · " + ", ".join(active)


def _apply_noise(x: np.ndarray, level: float, rng) -> np.ndarray:
    """Ruido gaussiano proporcional a la desviación de cada canal."""
    axis = -1
    sigma = np.std(x, axis=axis, keepdims=True)
    return x + rng.normal(0.0, 1.0, x.shape) * sigma * float(level)


def _apply_amplitude(x: np.ndarray, level: float, rng) -> np.ndarray:
    """Escalado global del ensayo por un factor en ``1 ± level``."""
    factor = 1.0 + rng.uniform(-float(level), float(level))
    return x * factor


def _apply_shift(x: np.ndarray, level: float, rng) -> np.ndarray:
    """Traslación circular en el eje temporal (solo señal cruda)."""
    if x.ndim < 2:                     # vector de características: no hay tiempo
        return x
    n_samples = x.shape[-1]
    max_shift = int(n_samples * float(level))
    if max_shift < 1:
        return x
    k = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(x, k, axis=-1) if k else x


def _mix_partner(x: np.ndarray, other: np.ndarray, level: float, rng) -> np.ndarray:
    """Combinación convexa con otro ensayo de la MISMA clase.

    Al mezclar solo dentro de la clase, la etiqueta no cambia y no hace falta
    tocar ``y`` (mixup «suave»: no mezcla etiquetas)."""
    lam = rng.uniform(0.0, float(level))
    return (1.0 - lam) * x + lam * other


def augment(X: np.ndarray, y: np.ndarray, config: dict | None = None,
            rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve ``(X_aug, y_aug)`` = los originales **más** las copias generadas.

    ``X`` puede ser señal cruda ``(n, canales, muestras)`` o características
    ``(n, k)``; la forma de cada ensayo se conserva. Los originales **siempre** se
    conservan: el aumento añade, no sustituye.

    Pensada para llamarse con el pliegue de ENTRENAMIENTO únicamente.
    """
    cfg = config or {}
    X = np.asarray(X)
    y = np.asarray(y)
    copies = int(cfg.get("copies", 1))
    techs = {k: bool(v) for k, v in (cfg.get("techniques") or {}).items()}
    if (not cfg.get("enabled") or copies < 1 or X.shape[0] == 0
            or not any(techs.values())):
        return X, y

    rng = rng or np.random.default_rng(int(cfg.get("seed", 0)))
    levels = {**DEFAULT_LEVELS, **(cfg.get("levels") or {})}
    prob = float(cfg.get("probability", 0.5))
    # Índices por clase, para que mixup mezcle solo dentro de la misma clase.
    by_class: dict = {}
    for i, lab in enumerate(y):
        by_class.setdefault(lab, []).append(i)

    out_X = [X]
    out_y = [y]
    for _ in range(copies):
        new = np.empty_like(X, dtype=np.float64)
        for i in range(X.shape[0]):
            xi = X[i].astype(np.float64, copy=True)
            applied = False
            # «Aumentación automática»: cada técnica activa se aplica al azar, así
            # cada copia es una combinación distinta.
            for key in ("noise", "amplitude", "shift", "mixup"):
                if not techs.get(key) or rng.random() > prob:
                    continue
                if key == "noise":
                    xi = _apply_noise(xi, levels["noise"], rng)
                elif key == "amplitude":
                    xi = _apply_amplitude(xi, levels["amplitude"], rng)
                elif key == "shift":
                    xi = _apply_shift(xi, levels["shift"], rng)
                elif key == "mixup":
                    pool = by_class.get(y[i], [])
                    if len(pool) > 1:
                        j = int(rng.choice([p for p in pool if p != i]))
                        xi = _mix_partner(xi, X[j].astype(np.float64),
                                          levels["mixup"], rng)
                    else:
                        continue           # sin pareja de su clase: no se mezcla
                applied = True
            if not applied:      # ninguna técnica salió sorteada: perturba mínimo
                xi = _apply_noise(xi, levels["noise"], rng)
            new[i] = xi
        out_X.append(new)
        out_y.append(y)
    return np.concatenate(out_X, axis=0), np.concatenate(out_y, axis=0)

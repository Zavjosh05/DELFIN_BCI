"""Entrenamiento y predicción de modelos de clasificación EEG.

Usa pipelines de scikit-learn (imputación + escalado + clasificador). Los
modelos se guardan con ``joblib`` dentro de ``<proyecto>/models``.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import MODELS_DIR
from . import neuralnet

try:
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    from pyriemann.spatialfilters import CSP
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    _PYRIEMANN_OK = True
except Exception:  # noqa: BLE001
    _PYRIEMANN_OK = False


def riemann_available() -> bool:
    return _PYRIEMANN_OK

SVM_KERNELS = {
    "linear": "Lineal",
    "rbf": "RBF (gaussiano)",
    "poly": "Polinomial",
    "sigmoid": "Sigmoide",
}


def _class_weight(value):
    """Normaliza el parámetro ``class_weight`` de la interfaz.

    La interfaz lo pasa como texto: ``"balanced"`` compensa clases desbalanceadas
    (útil en MI, donde no siempre hay el mismo nº de ensayos por clase);
    cualquier otro valor (``"none"``/``None``) = sin ponderar."""
    return "balanced" if value == "balanced" else None


def _make_classifier(name: str, params: dict | None):
    """Construye el estimador clásico, aplicando parámetros si los hay."""
    params = params or {}
    if name == "random_forest":
        max_depth = params.get("max_depth", 0)
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=(int(max_depth) if max_depth else None),  # 0 = sin límite
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            max_features=params.get("max_features", "sqrt"),
            criterion=params.get("criterion", "gini"),
            class_weight=_class_weight(params.get("class_weight")),
            random_state=0,
        )
    if name == "svm":
        # Sin probability=True (obsoleto en sklearn 1.9); predict_proba devuelve
        # None para SVM y la interfaz lo maneja mostrando solo la clase.
        return SVC(
            kernel=params.get("kernel", "rbf"),
            C=float(params.get("C", 1.0)),
            gamma=params.get("gamma", "scale"),
            degree=int(params.get("degree", 3)),
            coef0=float(params.get("coef0", 0.0)),       # término libre (poly/sigmoide)
            class_weight=_class_weight(params.get("class_weight")),
            random_state=0,
        )
    if name == "lda":
        solver = params.get("solver", "svd")
        shrinkage = params.get("shrinkage")
        # El shrinkage (regularización, útil con pocas muestras y muchas
        # características, típico en EEG) solo es válido con lsqr/eigen, NO con svd.
        if solver == "svd" or shrinkage in ("none", "", None):
            shrinkage = None
        return LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
    raise ValueError(f"Clasificador desconocido: {name}")


# Nombres de los clasificadores clásicos disponibles.
CLASSIFIERS = {"random_forest": None, "svm": None, "lda": None}

CLASSIFIER_LABELS = {
    "random_forest": "Random Forest",
    "svm": "SVM (RBF)",
    "lda": "Análisis discriminante lineal",
    # Geometría de Riemann / CSP (señal cruda).
    "riemann_mdm": "Riemann — MDM (señal cruda)",
    "riemann_ts": "Riemann — Tangent Space + LR (señal cruda)",
    "csp_lda": "CSP + LDA (señal cruda)",
    # Redes neuronales (PyTorch).
    "nn_mlp": "Red neuronal — MLP (características)",
    "nn_cnn": "Red neuronal — CNN 1D (señal cruda)",
    "nn_lstm": "Red neuronal — LSTM (señal cruda)",
    "nn_eegnet": "Red neuronal — EEGNet (señal cruda)",
}

# Familia de cada modelo y tipo de entrada que necesita.
MODEL_FAMILY = {
    "random_forest": "classic", "svm": "classic", "lda": "classic",
    "riemann_mdm": "riemann", "riemann_ts": "riemann", "csp_lda": "riemann",
    "nn_mlp": "nn", "nn_cnn": "nn", "nn_lstm": "nn", "nn_eegnet": "nn",
}
NN_NET_TYPE = {"nn_mlp": "mlp", "nn_cnn": "cnn", "nn_lstm": "lstm", "nn_eegnet": "eegnet"}
INPUT_KIND = {
    "random_forest": "features", "svm": "features", "lda": "features",
    "nn_mlp": "features", "nn_cnn": "raw", "nn_lstm": "raw", "nn_eegnet": "raw",
    "riemann_mdm": "raw", "riemann_ts": "raw", "csp_lda": "raw",
}


def is_nn(name: str) -> bool:
    return MODEL_FAMILY.get(name) == "nn"


def is_riemann(name: str) -> bool:
    return MODEL_FAMILY.get(name) == "riemann"


def requires_raw(name: str) -> bool:
    return INPUT_KIND.get(name) == "raw"


def net_type(name: str) -> str:
    return NN_NET_TYPE.get(name, "mlp")


# --- Métricas --------------------------------------------------------------
def _metrics_dict(y_true, y_pred, labels) -> dict:
    """Matriz de confusión y métricas por clase a partir de predicciones."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {
        "labels": list(labels),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion": cm.tolist(),
        "precision": prec.tolist(),
        "recall": rec.tolist(),
        "f1": f1.tolist(),
        "support": [int(s) for s in support],
    }


def _cv_eval(make_estimator, X, y, folds: int):
    """Validación cruzada estratificada: scores por *fold* y predicciones OOF."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    scores = []
    y_pred = np.empty(len(y), dtype=object)
    for tr, te in skf.split(X, y):
        est = make_estimator()
        est.fit(X[tr], y[tr])
        pred = est.predict(X[te])
        y_pred[te] = pred
        scores.append(accuracy_score(y[te], pred))
    return np.array(scores), y_pred


@dataclass
class TrainingResult:
    model: object                      # Pipeline (clásico) o TorchClassifier (NN)
    classifier_name: str
    classes: list[str]
    feature_names: list[str]
    cv_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    input_kind: str = "features"       # "features" o "raw"
    nn_config: dict | None = None      # config de la red (si es NN)
    metrics: dict | None = None        # matriz de confusión + métricas por clase
    clf_params: dict | None = None     # hiperparámetros del clasificador clásico
    raw_window: int = 0                 # ventana (muestras) usada por Riemann/CSP (0 = n/a)
    n_samples: int = 0                 # muestras totales del dataset
    n_train: int = 0                   # muestras de entrenamiento (holdout, o por pliegue en CV)
    n_eval: int = 0                    # muestras de evaluación (holdout, o por pliegue en CV)
    eval_method: str = ""              # "holdout" | "cross_val" | ""
    cv_folds: int = 0                  # nº de pliegues (si es validación cruzada)

    @property
    def cv_mean(self) -> float:
        return float(self.cv_scores.mean()) if self.cv_scores.size else float("nan")

    def split_report(self) -> str:
        """Texto de con cuántos datos se entrenó y evaluó (adaptado al método)."""
        n = self.n_samples
        if not n:
            return "Datos de entrenamiento/evaluación: no disponible."
        if self.eval_method == "holdout":
            tr, ev = self.n_train, self.n_eval
            return (f"Entrenamiento (holdout): {tr} muestras ({100 * tr / n:.0f}%)  ·  "
                    f"Evaluación: {ev} muestras ({100 * ev / n:.0f}%). "
                    f"El modelo final se reentrena con las {n} muestras (100%).")
        if self.eval_method == "cross_val" and self.cv_folds >= 2:
            k = self.cv_folds
            ev = n // k
            tr = n - ev
            return (f"Validación cruzada de {k} pliegues sobre {n} muestras: por pliegue "
                    f"~{tr} entrenan ({100 * tr / n:.0f}%) y ~{ev} evalúan ({100 * ev / n:.0f}%); "
                    f"cada muestra se evalúa 1 vez ({n} en total). "
                    f"El modelo final usa las {n} muestras (100%).")
        return f"Entrenado con {n} muestras (sin validación: datos insuficientes)."

    @property
    def cv_std(self) -> float:
        return float(self.cv_scores.std()) if self.cv_scores.size else float("nan")

    @property
    def is_nn(self) -> bool:
        return is_nn(self.classifier_name)

    @property
    def score_label(self) -> str:
        return "Exactitud (holdout)" if self.is_nn else "Validación cruzada"


def build_pipeline(classifier_name: str, clf_params: dict | None = None) -> Pipeline:
    if classifier_name not in CLASSIFIERS:
        raise ValueError(f"Clasificador desconocido: {classifier_name}")
    return Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
        ("clf", _make_classifier(classifier_name, clf_params)),
    ])


def train(dataset, classifier_name: str = "random_forest", cv: int = 5,
          nn_config: dict | None = None, clf_params: dict | None = None,
          progress=None) -> TrainingResult:
    """Entrena con un dataset de **características** (clásicos y MLP)."""
    X, y = dataset.X, dataset.y
    if X.shape[0] < 2:
        raise ValueError("Se necesitan al menos 2 segmentos para entrenar.")

    if is_nn(classifier_name):
        return _train_nn(X, y, classifier_name, "features", nn_config,
                         feature_names=list(dataset.feature_names), progress=progress)

    classes, counts = np.unique(y, return_counts=True)
    labels = sorted(classes.tolist())
    cv_scores, metrics, folds = _classic_cv(
        lambda: build_pipeline(classifier_name, clf_params),
        X, y, labels, counts.min(), cv)

    pipe = build_pipeline(classifier_name, clf_params)
    pipe.fit(X, y)
    return TrainingResult(
        model=pipe,
        classifier_name=classifier_name,
        classes=labels,
        feature_names=list(dataset.feature_names),
        cv_scores=cv_scores,
        input_kind="features",
        metrics=metrics,
        clf_params=dict(clf_params) if clf_params else None,
        n_samples=int(X.shape[0]),
        eval_method="cross_val" if folds >= 2 else "",
        cv_folds=folds,
    )


def _classic_cv(make_estimator, X, y, labels, min_count, cv):
    """Validación cruzada + métricas para modelos clásicos/Riemann (si procede).

    Devuelve ``(scores, metrics, folds)``; ``folds=0`` si no se pudo validar.
    """
    if len(labels) >= 2 and min_count >= 2:
        folds = int(min(cv, min_count))
        if folds >= 2:
            scores, y_pred = _cv_eval(make_estimator, X, y, folds)
            return scores, _metrics_dict(y, y_pred, labels), folds
    return np.array([]), None, 0


def train_raw(raw_dataset, classifier_name: str, nn_config: dict | None = None,
              progress=None) -> TrainingResult:
    """Entrena una red CNN/LSTM con un dataset de **señal cruda**."""
    X, y = raw_dataset.X, raw_dataset.y
    if X.shape[0] < 2:
        raise ValueError("Se necesitan al menos 2 segmentos para entrenar.")
    return _train_nn(X, y, classifier_name, "raw", nn_config, feature_names=[],
                     progress=progress)


def _train_nn(X, y, classifier_name: str, input_kind: str,
              nn_config: dict | None, feature_names: list[str],
              progress=None) -> TrainingResult:
    if not neuralnet.torch_available():
        raise RuntimeError(
            "PyTorch no está instalado. Instala 'torch' para usar redes neuronales."
        )
    config = nn_config or neuralnet.default_config(net_type(classifier_name))
    classes = np.unique(y)

    # Validación por holdout estratificado (k-fold sería muy lento entrenando red).
    cv_scores = np.array([])
    metrics = None
    labels = sorted(classes.tolist())
    counts = np.array([np.sum(y == c) for c in classes])
    n_train = n_eval = 0
    eval_method = ""
    if len(classes) >= 2 and counts.min() >= 2 and X.shape[0] >= 4:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=0, stratify=y
            )
            holdout = neuralnet.TorchClassifier(config)
            holdout.fit(X_tr, y_tr)
            y_pred = holdout.predict(X_te)
            cv_scores = np.array([accuracy_score(y_te, y_pred)])
            metrics = _metrics_dict(y_te, y_pred, labels)
            n_train, n_eval, eval_method = len(X_tr), len(X_te), "holdout"
        except Exception:  # noqa: BLE001
            cv_scores = np.array([])

    clf = neuralnet.TorchClassifier(config)
    clf.fit(X, y, progress=progress)
    return TrainingResult(
        model=clf,
        classifier_name=classifier_name,
        classes=labels,
        feature_names=feature_names,
        cv_scores=cv_scores,
        input_kind=input_kind,
        nn_config=config,
        metrics=metrics,
        n_samples=int(X.shape[0]),
        n_train=n_train,
        n_eval=n_eval,
        eval_method=eval_method,
    )


def build_riemann_pipeline(name: str) -> Pipeline:
    """Pipeline de geometría de Riemann / CSP sobre señal cruda (n, canales, T).

    Referencias: Barachant et al. (Riemannian geometry, arXiv:2407.20250);
    filtrado espacial de Riemann (doi:10.1145/3691521.3691529); CSP clásico.
    """
    if not _PYRIEMANN_OK:
        raise RuntimeError("pyriemann no está instalado (pip install pyriemann).")
    cov = Covariances(estimator="oas")
    if name == "riemann_mdm":
        return Pipeline([("cov", cov), ("mdm", MDM())])
    if name == "riemann_ts":
        return Pipeline([("cov", cov), ("ts", TangentSpace()),
                         ("lr", LogisticRegression(max_iter=500))])
    if name == "csp_lda":
        return Pipeline([("cov", cov), ("csp", CSP(nfilter=6, log=True)),
                         ("lda", LinearDiscriminantAnalysis())])
    raise ValueError(f"Modelo de Riemann desconocido: {name}")


def train_riemann(raw_dataset, classifier_name: str, cv: int = 5,
                  raw_window: int = 0) -> TrainingResult:
    """Entrena un modelo de Riemann/CSP con el dataset de señal cruda."""
    X, y = raw_dataset.X, raw_dataset.y
    if X.shape[0] < 2:
        raise ValueError("Se necesitan al menos 2 segmentos para entrenar.")
    classes, counts = np.unique(y, return_counts=True)
    labels = sorted(classes.tolist())
    try:
        cv_scores, metrics, folds = _classic_cv(
            lambda: build_riemann_pipeline(classifier_name), X, y, labels, counts.min(), cv)
    except Exception:  # noqa: BLE001
        cv_scores, metrics, folds = np.array([]), None, 0

    pipe = build_riemann_pipeline(classifier_name)
    pipe.fit(X, y)
    return TrainingResult(
        model=pipe,
        classifier_name=classifier_name,
        classes=labels,
        feature_names=[],
        cv_scores=cv_scores,
        input_kind="raw",
        metrics=metrics,
        raw_window=int(raw_window),
        n_samples=int(X.shape[0]),
        eval_method="cross_val" if folds >= 2 else "",
        cv_folds=folds,
    )


def predict(result: TrainingResult, X: np.ndarray) -> np.ndarray:
    return result.model.predict(X)


def predict_proba(result: TrainingResult, X: np.ndarray):
    if hasattr(result.model, "predict_proba"):
        try:
            return result.model.predict_proba(X)
        except Exception:
            return None
    return None


def _result_blob(result: TrainingResult) -> dict:
    return {
        "model": result.model,
        "classifier_name": result.classifier_name,
        "classes": result.classes,
        "feature_names": result.feature_names,
        "cv_scores": result.cv_scores,
        "input_kind": result.input_kind,
        "nn_config": result.nn_config,
        "metrics": result.metrics,
        "clf_params": getattr(result, "clf_params", None),
        "raw_window": getattr(result, "raw_window", 0),
        "n_samples": getattr(result, "n_samples", 0),
        "n_train": getattr(result, "n_train", 0),
        "n_eval": getattr(result, "n_eval", 0),
        "eval_method": getattr(result, "eval_method", ""),
        "cv_folds": getattr(result, "cv_folds", 0),
    }


def save_model(project, result: TrainingResult, name: str = "model") -> str:
    out_dir = os.path.join(project.path, MODELS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.joblib")
    return save_model_to(result, out_path)


def save_model_to(result: TrainingResult, path: str) -> str:
    """Guarda el modelo en una ruta arbitraria (para exportar)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(_result_blob(result), path)
    return path


def _blob_to_result(blob: dict) -> TrainingResult:
    return TrainingResult(
        model=blob["model"],
        classifier_name=blob.get("classifier_name", "unknown"),
        classes=blob.get("classes", []),
        feature_names=blob.get("feature_names", []),
        cv_scores=blob.get("cv_scores", np.array([])),
        input_kind=blob.get("input_kind", "features"),
        nn_config=blob.get("nn_config"),
        metrics=blob.get("metrics"),
        clf_params=blob.get("clf_params"),
        raw_window=blob.get("raw_window", 0),
        n_samples=blob.get("n_samples", 0),
        n_train=blob.get("n_train", 0),
        n_eval=blob.get("n_eval", 0),
        eval_method=blob.get("eval_method", ""),
        cv_folds=blob.get("cv_folds", 0),
    )


def load_model(path: str) -> TrainingResult:
    return _blob_to_result(joblib.load(path))


def result_to_bytes(result: TrainingResult) -> bytes:
    """Serializa un modelo a bytes (para incluirlo en un bundle .eegbundle)."""
    buf = io.BytesIO()
    joblib.dump(_result_blob(result), buf)
    return buf.getvalue()


def result_from_bytes(data: bytes) -> TrainingResult:
    return _blob_to_result(joblib.load(io.BytesIO(data)))


def metrics_report(result: TrainingResult) -> str:
    """Texto con la matriz de confusión y las métricas por clase."""
    m = result.metrics
    if not m:
        return "Sin métricas detalladas (pocos datos para validar)."
    labels = m["labels"]
    lines = [f"Exactitud: {m['accuracy'] * 100:.1f}%", "", "Por clase:",
             f"{'clase':<14}{'prec.':>7}{'recall':>8}{'f1':>7}{'n':>6}"]
    for i, lab in enumerate(labels):
        lines.append(f"{lab:<14}{m['precision'][i]:>7.2f}{m['recall'][i]:>8.2f}"
                     f"{m['f1'][i]:>7.2f}{m['support'][i]:>6}")
    lines += ["", "Matriz de confusión (filas=real, col=predicho):",
              "          " + " ".join(f"{l[:6]:>7}" for l in labels)]
    for i, lab in enumerate(labels):
        row = " ".join(f"{v:>7}" for v in m["confusion"][i])
        lines.append(f"{lab[:9]:<9} {row}")
    return "\n".join(lines)

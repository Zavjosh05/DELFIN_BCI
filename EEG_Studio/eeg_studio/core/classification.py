"""Entrenamiento y predicción de modelos de clasificación EEG.

Usa pipelines de scikit-learn (imputación + escalado + clasificador). Los
modelos se guardan con ``joblib`` dentro de ``<proyecto>/models``.
"""
from __future__ import annotations

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


def _make_classifier(name: str, params: dict | None):
    """Construye el estimador clásico, aplicando parámetros si los hay."""
    params = params or {}
    if name == "random_forest":
        max_depth = params.get("max_depth", 0)
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 200)),
            max_depth=(int(max_depth) if max_depth else None),  # 0 = sin límite
            min_samples_split=int(params.get("min_samples_split", 2)),
            max_features=params.get("max_features", "sqrt"),
            criterion=params.get("criterion", "gini"),
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
            random_state=0,
        )
    if name == "lda":
        return LinearDiscriminantAnalysis()
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

    @property
    def cv_mean(self) -> float:
        return float(self.cv_scores.mean()) if self.cv_scores.size else float("nan")

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
    cv_scores, metrics = _classic_cv(lambda: build_pipeline(classifier_name, clf_params),
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
    )


def _classic_cv(make_estimator, X, y, labels, min_count, cv):
    """Validación cruzada + métricas para modelos clásicos/Riemann (si procede)."""
    if len(labels) >= 2 and min_count >= 2:
        folds = int(min(cv, min_count))
        if folds >= 2:
            scores, y_pred = _cv_eval(make_estimator, X, y, folds)
            return scores, _metrics_dict(y, y_pred, labels)
    return np.array([]), None


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


def train_riemann(raw_dataset, classifier_name: str, cv: int = 5) -> TrainingResult:
    """Entrena un modelo de Riemann/CSP con el dataset de señal cruda."""
    X, y = raw_dataset.X, raw_dataset.y
    if X.shape[0] < 2:
        raise ValueError("Se necesitan al menos 2 segmentos para entrenar.")
    classes, counts = np.unique(y, return_counts=True)
    labels = sorted(classes.tolist())
    try:
        cv_scores, metrics = _classic_cv(
            lambda: build_riemann_pipeline(classifier_name), X, y, labels, counts.min(), cv)
    except Exception:  # noqa: BLE001
        cv_scores, metrics = np.array([]), None

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


def load_model(path: str) -> TrainingResult:
    blob = joblib.load(path)
    return TrainingResult(
        model=blob["model"],
        classifier_name=blob.get("classifier_name", "unknown"),
        classes=blob.get("classes", []),
        feature_names=blob.get("feature_names", []),
        cv_scores=blob.get("cv_scores", np.array([])),
        input_kind=blob.get("input_kind", "features"),
        nn_config=blob.get("nn_config"),
        metrics=blob.get("metrics"),
    )


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

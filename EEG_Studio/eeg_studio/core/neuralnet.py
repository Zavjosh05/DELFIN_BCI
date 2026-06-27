"""Redes neuronales configurables (PyTorch) para clasificación de EEG.

Soporta tres tipos de red, descritos por un ``config`` serializable:

* ``mlp``  — perceptrón multicapa sobre el **vector de características**.
* ``cnn``  — red convolucional 1D sobre la **señal cruda** ``(n_canales, T)``.
* ``lstm`` — red recurrente sobre la **señal cruda** ``(n_canales, T)``.

Cada capa se configura **en detalle** (unidades/filtros, función de activación,
dropout, tamaño de kernel...). :class:`TorchClassifier` envuelve el modelo con
la interfaz de scikit-learn (``fit``/``predict``/``predict_proba``) para que
encaje con el resto del flujo (validación, guardado, predicción).

El import de torch está protegido: si no está instalado, la app sigue
funcionando y solo las opciones de red neuronal quedan deshabilitadas.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def torch_available() -> bool:
    return TORCH_AVAILABLE


# --- Catálogo para la interfaz --------------------------------------------
ACTIVATIONS = ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "gelu", "identity"]
ACTIVATION_LABELS = {
    "relu": "ReLU", "tanh": "Tanh", "sigmoid": "Sigmoide",
    "leaky_relu": "Leaky ReLU", "elu": "ELU", "gelu": "GELU",
    "identity": "Lineal (sin activación)",
}
OPTIMIZERS = ["adam", "sgd", "rmsprop"]
NET_TYPES = {"mlp": "MLP (características)", "cnn": "CNN 1D (señal cruda)",
             "lstm": "LSTM (señal cruda)", "eegnet": "EEGNet (señal cruda)"}


def _activation(name: str):
    table = {
        "relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU, "elu": nn.ELU, "gelu": nn.GELU,
        "identity": nn.Identity,
    }
    return table.get(name, nn.ReLU)()


def default_config(net_type: str = "mlp") -> dict:
    """Configuración por defecto para cada tipo de red."""
    common = {"epochs": 80, "batch_size": 16, "learning_rate": 0.001, "optimizer": "adam"}
    if net_type == "mlp":
        return {**common, "type": "mlp", "layers": [
            {"units": 64, "activation": "relu", "dropout": 0.2},
            {"units": 32, "activation": "relu", "dropout": 0.0},
        ]}
    if net_type == "cnn":
        return {**common, "type": "cnn", "window_samples": 512, "layers": [
            {"units": 16, "kernel_size": 5, "activation": "relu", "dropout": 0.0, "pool": True},
            {"units": 32, "kernel_size": 3, "activation": "relu", "dropout": 0.25, "pool": True},
        ]}
    if net_type == "lstm":
        return {**common, "type": "lstm", "window_samples": 256, "layers": [
            {"units": 32, "activation": "tanh", "dropout": 0.0, "bidirectional": False},
        ]}
    if net_type == "eegnet":
        # Hiperparámetros del EEGNet (Lawhern et al., 2018; doi:10.1088/1741-2552/aace8c).
        return {**common, "type": "eegnet", "window_samples": 256,
                "F1": 8, "D": 2, "F2": 16, "kernel_length": 64, "dropout": 0.25}
    raise ValueError(f"Tipo de red desconocido: {net_type}")


# --- Módulo configurable ---------------------------------------------------
if TORCH_AVAILABLE:

    class ConfigurableNet(nn.Module):
        """Red cuya arquitectura se construye a partir de ``config``."""

        def __init__(self, config: dict, input_shape: tuple, n_classes: int) -> None:
            super().__init__()
            self.kind = config.get("type", "mlp")
            layers = config.get("layers", [])
            if not layers:
                layers = [{"units": 32, "activation": "relu"}]
            if self.kind == "mlp":
                self._build_mlp(layers, input_shape, n_classes)
            elif self.kind == "cnn":
                self._build_cnn(layers, input_shape, n_classes)
            elif self.kind == "lstm":
                self._build_lstm(layers, input_shape, n_classes)
            elif self.kind == "eegnet":
                self._build_eegnet(config, input_shape, n_classes)
            else:
                raise ValueError(f"Tipo de red desconocido: {self.kind}")

        def _build_mlp(self, layers, input_shape, n_classes) -> None:
            prev = int(np.prod(input_shape))
            mods = []
            for layer in layers:
                units = int(layer.get("units", 64))
                mods.append(nn.Linear(prev, units))
                mods.append(_activation(layer.get("activation", "relu")))
                drop = float(layer.get("dropout", 0.0))
                if drop > 0:
                    mods.append(nn.Dropout(drop))
                prev = units
            self.body = nn.Sequential(*mods)
            self.head = nn.Linear(prev, n_classes)

        def _build_cnn(self, layers, input_shape, n_classes) -> None:
            prev = int(input_shape[0])  # nº de canales
            mods = []
            for layer in layers:
                filt = int(layer.get("units", 16))
                k = int(layer.get("kernel_size", 3))
                mods.append(nn.Conv1d(prev, filt, k, padding=k // 2))
                mods.append(_activation(layer.get("activation", "relu")))
                if layer.get("pool", True):
                    mods.append(nn.MaxPool1d(2))
                drop = float(layer.get("dropout", 0.0))
                if drop > 0:
                    mods.append(nn.Dropout(drop))
                prev = filt
            self.body = nn.Sequential(*mods)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(prev, n_classes)

        def _build_lstm(self, layers, input_shape, n_classes) -> None:
            prev = int(input_shape[0])  # canales = nº de características por paso
            self.lstms = nn.ModuleList()
            self.acts = nn.ModuleList()
            self.drops = nn.ModuleList()
            for layer in layers:
                units = int(layer.get("units", 32))
                bidir = bool(layer.get("bidirectional", False))
                self.lstms.append(nn.LSTM(prev, units, batch_first=True, bidirectional=bidir))
                self.acts.append(_activation(layer.get("activation", "tanh")))
                drop = float(layer.get("dropout", 0.0))
                self.drops.append(nn.Dropout(drop) if drop > 0 else nn.Identity())
                prev = units * (2 if bidir else 1)
            self.head = nn.Linear(prev, n_classes)

        def _build_eegnet(self, config, input_shape, n_classes) -> None:
            """EEGNet: conv temporal → conv espacial depthwise → conv separable."""
            C, T = int(input_shape[0]), int(input_shape[1])
            F1 = int(config.get("F1", 8))
            D = int(config.get("D", 2))
            F2 = int(config.get("F2", F1 * D))
            k = int(config.get("kernel_length", 64))
            p = float(config.get("dropout", 0.25))

            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, k), padding="same", bias=False),
                nn.BatchNorm2d(F1),
            )
            self.depthwise = nn.Sequential(
                nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False),  # filtro espacial
                nn.BatchNorm2d(F1 * D), nn.ELU(),
                nn.AvgPool2d((1, 4)), nn.Dropout(p),
            )
            self.separable = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D, padding="same", bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(p),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, C, T)
                flat = self.separable(self.depthwise(self.block1(dummy))).flatten(1).shape[1]
            self.head = nn.Linear(flat, n_classes)

        def forward(self, x):
            if self.kind == "mlp":
                return self.head(self.body(x.flatten(1)))
            if self.kind == "cnn":
                x = self.body(x)               # (B, C, T)
                x = self.pool(x).flatten(1)    # (B, C)
                return self.head(x)
            if self.kind == "eegnet":
                x = x.unsqueeze(1)             # (B, 1, C, T)
                x = self.separable(self.depthwise(self.block1(x)))
                return self.head(x.flatten(1))
            # lstm: (B, C, T) -> (B, T, C)
            x = x.transpose(1, 2)
            for lstm, act, drop in zip(self.lstms, self.acts, self.drops):
                x, _ = lstm(x)
                x = drop(act(x))
            return self.head(x[:, -1, :])       # último paso temporal


def _make_optimizer(config: dict, params):
    name = config.get("optimizer", "adam")
    lr = float(config.get("learning_rate", 1e-3))
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    return torch.optim.Adam(params, lr=lr)


# --- Estimador compatible con scikit-learn --------------------------------
class TorchClassifier:
    """Clasificador de red neuronal con interfaz estilo scikit-learn.

    Acepta entrada 2D ``(n, features)`` (MLP) o 3D ``(n, canales, T)`` (CNN/LSTM).
    Normaliza la entrada internamente (z-score) y entrena con el optimizador y
    los hiperparámetros del ``config``.
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or default_config("mlp")

    # API mínima de sklearn para clonar (get/set params).
    def get_params(self, deep: bool = True) -> dict:
        return {"config": self.config}

    def set_params(self, **params):
        if "config" in params:
            self.config = params["config"]
        return self

    # --- Normalización ----------------------------------------------------
    def _fit_normalizer(self, X: np.ndarray) -> None:
        if X.ndim == 2:
            self._mean = X.mean(axis=0, keepdims=True)
            self._std = X.std(axis=0, keepdims=True)
        else:  # 3D: por canal, sobre (muestras, tiempo)
            self._mean = X.mean(axis=(0, 2), keepdims=True)
            self._std = X.std(axis=(0, 2), keepdims=True)
        self._std = np.where(self._std == 0, 1.0, self._std)

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self._mean) / self._std
        return np.nan_to_num(Xn, copy=False).astype(np.float32)

    # --- Entrenamiento ----------------------------------------------------
    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch no está instalado.")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y).astype(np.int64)

        self._fit_normalizer(X)
        Xn = self._normalize(X)
        input_shape = Xn.shape[1:]

        torch.manual_seed(0)
        self.module_ = ConfigurableNet(self.config, input_shape, len(self.classes_))

        Xt = torch.from_numpy(Xn)
        yt = torch.from_numpy(y_idx)
        ds = torch.utils.data.TensorDataset(Xt, yt)
        bs = max(1, int(self.config.get("batch_size", 16)))
        loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
        opt = _make_optimizer(self.config, self.module_.parameters())
        crit = nn.CrossEntropyLoss()
        epochs = int(self.config.get("epochs", 80))

        self.module_.train()
        for _ in range(epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = crit(self.module_(xb), yb)
                loss.backward()
                opt.step()
        return self

    # --- Inferencia -------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xn = self._normalize(X)
        self.module_.eval()
        with torch.no_grad():
            logits = self.module_(torch.from_numpy(Xn))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

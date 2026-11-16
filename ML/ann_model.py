"""
HealthGuard AI - Artificial Neural Network (ANN) Architecture
=============================================================
PyTorch Feedforward Multi-Layer Perceptron (MLP) with Batch Normalization,
Dropout, and class-weighted BCE loss for provider tabular fraud detection.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class PyTorchANNModule(nn.Module):
    """
    Three-layer MLP: 128 -> 64 -> 32 -> 1
    Includes Batch Normalization, ReLU activations, and Dropout.
    """

    def __init__(self, in_features: int = 51, dropout_rates: Tuple[float, float, float] = (0.3, 0.3, 0.2)):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rates[0])

        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rates[1])

        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(dropout_rates[2])

        self.out = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act1(self.bn1(self.layer1(x))))
        x = self.drop2(self.act2(self.bn2(self.layer2(x))))
        x = self.drop3(self.act3(self.bn3(self.layer3(x))))
        logits = self.out(x).squeeze(-1)
        return logits


class ANNFraudDetector:
    """ANN Fraud Classifier with PyTorch backend, class weighting, and early stopping."""

    def __init__(
        self,
        input_dim: int = 51,
        hidden_layers: Optional[List[int]] = None,
        dropout_rates: Tuple[float, float, float] = (0.3, 0.3, 0.2),
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        random_state: int = 42,
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.dropout_rates = dropout_rates
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

        self.model: Optional[PyTorchANNModule] = None
        self.scaler: Optional[RobustScaler] = None
        self.feature_names: List[str] = []
        self.training_history_: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_prauc": []}
        self.best_epoch_: int = 0
        self.best_val_prauc_: float = 0.0

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def _prepare_data(self, X: Any, fit_scaler: bool = False) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if not self.feature_names:
                self.feature_names = list(X.columns)
            arr = X[self.feature_names].values.astype(np.float32)
        elif isinstance(X, np.ndarray):
            arr = X.astype(np.float32)
        else:
            arr = np.array(X, dtype=np.float32)

        # Handle any remaining NaNs safely by imputing median of column
        if np.isnan(arr).any():
            col_medians = np.nanmedian(arr, axis=0)
            inds = np.where(np.isnan(arr))
            arr[inds] = np.take(col_medians, inds[1])
            if np.isnan(arr).any():
                np.nan_to_num(arr, copy=False, nan=0.0)

        if fit_scaler:
            self.scaler = RobustScaler()
            arr = self.scaler.fit_transform(arr)
        elif self.scaler is not None:
            arr = self.scaler.transform(arr)

        return arr

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        validation_data: Optional[Tuple[Any, Any]] = None,
        epochs: int = 60,
        batch_size: int = 64,
        patience: int = 15,
        calculate_pos_weight: bool = True,
    ) -> "ANNFraudDetector":
        """
        Train ANN model using class-weighted BCEWithLogitsLoss and PR-AUC early stopping.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_tr = self._prepare_data(X_train, fit_scaler=True)
        y_tr = np.asarray(y_train, dtype=np.float32)

        self.input_dim = X_tr.shape[1]
        self.model = PyTorchANNModule(in_features=self.input_dim, dropout_rates=self.dropout_rates)

        # Class imbalance weighting
        if calculate_pos_weight:
            n_neg = float((y_tr == 0).sum())
            n_pos = float((y_tr == 1).sum())
            pos_weight_val = n_neg / max(n_pos, 1.0)
        else:
            pos_weight_val = 1.0
        pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        train_dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Validation data setup
        has_val = validation_data is not None
        if has_val:
            X_val = self._prepare_data(validation_data[0], fit_scaler=False)
            y_val = np.asarray(validation_data[1], dtype=np.float32)
            X_val_t = torch.from_numpy(X_val)
            y_val_t = torch.from_numpy(y_val)

        best_score = -1.0
        best_state = None
        no_improve_epochs = 0

        self.training_history_ = {"train_loss": [], "val_loss": [], "val_prauc": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch_y)

            epoch_train_loss = running_loss / len(X_tr)
            self.training_history_["train_loss"].append(epoch_train_loss)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val_t)
                    val_loss = criterion(val_logits, y_val_t).item()
                    val_probs = torch.sigmoid(val_logits).cpu().numpy()

                val_prauc = average_precision_score(y_val, val_probs)
                self.training_history_["val_loss"].append(val_loss)
                self.training_history_["val_prauc"].append(val_prauc)

                if val_prauc > best_score:
                    best_score = val_prauc
                    self.best_epoch_ = epoch
                    self.best_val_prauc_ = val_prauc
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= patience:
                        break
            else:
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return fraud probabilities [0, 1]."""
        if self.model is None:
            raise RuntimeError(
                "ANNFraudDetector has not been trained yet. "
                "Fit the model using Phase 2 training routines before calling predict_proba()."
            )
        X_arr = self._prepare_data(X, fit_scaler=False)
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(X_arr)
            logits = self.model(x_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str) -> None:
        """Save PyTorch weights, scaler, and metadata."""
        if self.model is None:
            raise RuntimeError("Cannot save untrained model.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_path = path.with_suffix(".pt")
        torch.save(self.model.state_dict(), str(model_path))

        meta = {
            "input_dim": self.input_dim,
            "hidden_layers": self.hidden_layers,
            "dropout_rates": list(self.dropout_rates),
            "feature_names": self.feature_names,
            "best_epoch": self.best_epoch_,
            "best_val_prauc": self.best_val_prauc_,
            "random_state": self.random_state,
        }
        meta_path = path.with_name(path.stem + "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self.scaler is not None:
            import joblib
            scaler_path = path.with_name(path.stem + "_scaler.joblib")
            joblib.dump(self.scaler, str(scaler_path))

    def load(self, filepath: str) -> "ANNFraudDetector":
        """Load PyTorch weights, scaler, and metadata."""
        path = Path(filepath)
        model_path = path.with_suffix(".pt")
        meta_path = path.with_name(path.stem + "_meta.json")
        scaler_path = path.with_name(path.stem + "_scaler.joblib")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.input_dim = meta.get("input_dim", 51)
            self.hidden_layers = meta.get("hidden_layers", [128, 64, 32])
            self.dropout_rates = tuple(meta.get("dropout_rates", (0.3, 0.3, 0.2)))
            self.feature_names = meta.get("feature_names", [])
            self.best_epoch_ = meta.get("best_epoch", 0)
            self.best_val_prauc_ = meta.get("best_val_prauc", 0.0)

        if scaler_path.exists():
            import joblib
            self.scaler = joblib.load(str(scaler_path))

        self.model = PyTorchANNModule(in_features=self.input_dim, dropout_rates=self.dropout_rates)
        self.model.load_state_dict(torch.load(str(model_path), weights_only=True))
        self.model.eval()
        return self

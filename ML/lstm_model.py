"""
HealthGuard AI - Bidirectional LSTM Sequence Model Architecture
================================================================
Temporal modeling of 12-month provider claim sequences with
explicit validity masking to distinguish zero-claim months from padding.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class PyTorchBiLSTMModule(nn.Module):
    """
    Bidirectional LSTM with Masked Temporal Mean Pooling:
    - Input: (N, 12, 6) sequences, (N, 12) binary validity masks.
    - Bidirectional LSTM: 64 units per direction -> 128-dim hidden state per timestep.
    - Masked Pooling: calculates average over only valid timesteps (mask == 1).
    - Dense Projection: 128 -> 32 -> 1 logits.
    """

    def __init__(self, in_features: int = 6, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (N, 12, in_features), mask: (N, 12)
        lstm_out, _ = self.lstm(x)  # (N, 12, hidden_dim * 2)

        # Masked mean pooling across valid temporal steps
        mask_exp = mask.unsqueeze(-1)  # (N, 12, 1)
        masked_out = lstm_out * mask_exp
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (N, 1)
        pooled = masked_out.sum(dim=1) / lengths  # (N, 128)

        pooled_drop = self.dropout(pooled)
        logits = self.fc(pooled_drop).squeeze(-1)  # (N,)

        return logits, pooled


class LSTMFraudDetector:
    """
    Recurrent Neural Network for temporal billing pattern fraud detection.
    Takes 3D monthly claim sequences (n_providers, 12, 6) and 2D masks (n_providers, 12).
    """

    def __init__(
        self,
        seq_len: int = 12,
        n_features: int = 6,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        random_state: int = 42,
    ):
        self.seq_len = seq_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state

        self.model: Optional[PyTorchBiLSTMModule] = None
        self.scaler: Optional[RobustScaler] = None
        self.training_history_: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_prauc": []}
        self.best_epoch_: int = 0
        self.best_val_prauc_: float = 0.0

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

    def _normalize_sequences(self, sequences: np.ndarray, masks: np.ndarray, fit_scaler: bool = False) -> np.ndarray:
        """
        Normalize 3D sequence array using 2D scaler fitted strictly on valid (non-padded) timesteps.
        """
        N, T, D = sequences.shape
        seq_copy = sequences.copy().astype(np.float32)

        # Reshape to (N * T, D)
        flat = seq_copy.reshape(-1, D)
        flat_mask = masks.reshape(-1) > 0

        if fit_scaler:
            self.scaler = RobustScaler()
            valid_rows = flat[flat_mask]
            if len(valid_rows) > 0:
                self.scaler.fit(valid_rows)

        if self.scaler is not None:
            # Scale all rows, then zero out invalid masked positions
            scaled_flat = self.scaler.transform(flat)
            # Re-mask padded timesteps to clean zeros
            scaled_flat[~flat_mask] = 0.0
            seq_copy = scaled_flat.reshape(N, T, D)

        return seq_copy

    def fit(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        masks: Optional[np.ndarray] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 15,
        calculate_pos_weight: bool = True,
    ) -> "LSTMFraudDetector":
        """
        Train sequence model with temporal mask support and PR-AUC early stopping.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        if masks is None:
            masks = np.ones((len(sequences), self.seq_len), dtype=np.float32)

        X_tr = self._normalize_sequences(sequences, masks, fit_scaler=True)
        y_tr = np.asarray(labels, dtype=np.float32)
        m_tr = np.asarray(masks, dtype=np.float32)

        self.n_features = X_tr.shape[2]
        self.seq_len = X_tr.shape[1]
        self.model = PyTorchBiLSTMModule(
            in_features=self.n_features,
            hidden_dim=self.lstm_units,
            dropout=self.dropout_rate,
        )

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

        train_dataset = TensorDataset(
            torch.from_numpy(X_tr),
            torch.from_numpy(m_tr),
            torch.from_numpy(y_tr),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        has_val = validation_data is not None
        if has_val:
            val_seq, val_labels = validation_data[0], validation_data[1]
            val_masks = validation_data[2] if len(validation_data) > 2 and validation_data[2] is not None else np.ones((len(val_seq), self.seq_len), dtype=np.float32)
            X_val = self._normalize_sequences(val_seq, val_masks, fit_scaler=False)
            y_val = np.asarray(val_labels, dtype=np.float32)
            m_val = np.asarray(val_masks, dtype=np.float32)

            X_val_t = torch.from_numpy(X_val)
            m_val_t = torch.from_numpy(m_val)
            y_val_t = torch.from_numpy(y_val)

        best_score = -1.0
        best_state = None
        no_improve_epochs = 0

        self.training_history_ = {"train_loss": [], "val_loss": [], "val_prauc": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            for batch_x, batch_m, batch_y in train_loader:
                optimizer.zero_grad()
                logits, _ = self.model(batch_x, batch_m)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(batch_y)

            epoch_train_loss = running_loss / len(X_tr)
            self.training_history_["train_loss"].append(epoch_train_loss)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_logits, _ = self.model(X_val_t, m_val_t)
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

    def predict_proba(self, sequences: np.ndarray, masks: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict fraud probability from monthly sequences and valid masks."""
        if self.model is None:
            raise RuntimeError(
                "LSTMFraudDetector has not been trained yet. "
                "Fit the model using Phase 2 sequence training routines before calling predict_proba()."
            )
        if masks is None:
            masks = np.ones((len(sequences), self.seq_len), dtype=np.float32)

        X_norm = self._normalize_sequences(sequences, masks, fit_scaler=False)
        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X_norm)
            m_t = torch.from_numpy(masks.astype(np.float32))
            logits, _ = self.model(x_t, m_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def extract_temporal_embeddings(self, sequences: np.ndarray, masks: Optional[np.ndarray] = None) -> np.ndarray:
        """Extract dense 128-dimensional sequence representations for ensemble analysis."""
        if self.model is None:
            raise RuntimeError(
                "LSTMFraudDetector has not been trained yet. "
                "Fit the model using Phase 2 sequence training routines before extracting embeddings."
            )
        if masks is None:
            masks = np.ones((len(sequences), self.seq_len), dtype=np.float32)

        X_norm = self._normalize_sequences(sequences, masks, fit_scaler=False)
        self.model.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(X_norm)
            m_t = torch.from_numpy(masks.astype(np.float32))
            _, embeddings = self.model(x_t, m_t)
            emb_np = embeddings.cpu().numpy()
        return emb_np

    def predict(self, sequences: np.ndarray, masks: Optional[np.ndarray] = None, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(sequences, masks)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str) -> None:
        """Save PyTorch weights, sequence scaler, and metadata."""
        if self.model is None:
            raise RuntimeError("Cannot save untrained model.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_path = path.with_suffix(".pt")
        torch.save(self.model.state_dict(), str(model_path))

        meta = {
            "seq_len": self.seq_len,
            "n_features": self.n_features,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
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

    def load(self, filepath: str) -> "LSTMFraudDetector":
        """Load PyTorch weights, sequence scaler, and metadata."""
        path = Path(filepath)
        model_path = path.with_suffix(".pt")
        meta_path = path.with_name(path.stem + "_meta.json")
        scaler_path = path.with_name(path.stem + "_scaler.joblib")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.seq_len = meta.get("seq_len", 12)
            self.n_features = meta.get("n_features", 6)
            self.lstm_units = meta.get("lstm_units", 64)
            self.dropout_rate = meta.get("dropout_rate", 0.2)
            self.best_epoch_ = meta.get("best_epoch", 0)
            self.best_val_prauc_ = meta.get("best_val_prauc", 0.0)

        if scaler_path.exists():
            import joblib
            self.scaler = joblib.load(str(scaler_path))

        self.model = PyTorchBiLSTMModule(
            in_features=self.n_features,
            hidden_dim=self.lstm_units,
            dropout=self.dropout_rate,
        )
        self.model.load_state_dict(torch.load(str(model_path), weights_only=True))
        self.model.eval()
        return self

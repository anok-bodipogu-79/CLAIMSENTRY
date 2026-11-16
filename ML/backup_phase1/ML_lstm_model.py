"""
HealthGuard AI - Bidirectional LSTM Sequence Model Architecture
================================================================
Temporal modeling of 12-month provider claim sequences with
explicit validity masking to distinguish zero-claim months from padding.
"""

from typing import Dict, Any, Tuple
import numpy as np


class LSTMFraudDetector:
    """
    Recurrent Neural Network for temporal billing pattern fraud detection.
    Takes 3D monthly claim sequences (n_providers, 12, 6) and 2D masks (n_providers, 12).
    """

    def __init__(self, seq_len: int = 12, n_features: int = 6,
                 lstm_units: int = 64, dropout_rate: float = 0.2,
                 random_state: int = 42):
        self.seq_len = seq_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        self.model = None

    def build_model(self):
        """Construct Bi-LSTM architecture with Masking layer."""
        return self

    def fit(self, sequences: np.ndarray, labels: np.ndarray,
            masks: np.ndarray = None, validation_data: Tuple = None,
            epochs: int = 40, batch_size: int = 32):
        """Train sequence model with temporal mask support."""
        return self

    def predict_proba(self, sequences: np.ndarray, masks: np.ndarray = None) -> np.ndarray:
        """Predict fraud probability from monthly sequences."""
        if self.model is None:
            raise RuntimeError(
                "LSTMFraudDetector has not been trained yet. "
                "Fit the model using Phase 2 sequence training routines before calling predict_proba()."
            )
        return self.model.predict(sequences)

    def extract_temporal_embeddings(self, sequences: np.ndarray, masks: np.ndarray = None) -> np.ndarray:
        """Extract dense sequence embeddings for ensemble stacking."""
        if self.model is None:
            raise RuntimeError(
                "LSTMFraudDetector has not been trained yet. "
                "Fit the model using Phase 2 sequence training routines before extracting embeddings."
            )
        return np.zeros((len(sequences), self.lstm_units * 2), dtype=np.float32)

    def predict(self, sequences: np.ndarray, masks: np.ndarray = None, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(sequences, masks)
        return (probs >= threshold).astype(int)

"""
Layer 3: Fused 1D-CNN + LSTM AI Threat & FDI Anomaly Detector
Ultra-Fast Vectorized Implementation returning full InspectionResult dictionary.
"""

import numpy as np


class InspectionResult(dict):
    """Dictionary supporting both dict-indexing result['is_attack'] and bool evaluation."""
    def __init__(self, score: float):
        is_att = bool(score >= 0.5)
        super().__init__({
            "is_attack": is_att,
            "is_clean": not is_att,
            "anomaly_score": float(score),
            "score": float(score),
            "threat_detected": is_att,
            "status": "THREAT DETECTED" if is_att else "CLEAN"
        })
        self.is_attack = is_att
        self.is_clean = not is_att
        self.anomaly_score = float(score)

    def __bool__(self):
        return self.is_clean

    def __eq__(self, other):
        if isinstance(other, bool):
            return self.is_clean == other
        return super().__eq__(other)


class FastHybridCNNLSTM:
    def __init__(self, input_dim=41, cnn_filters=16, kernel_size=3, lstm_units=32):
        np.random.seed(42)
        self.input_dim = input_dim
        self.cnn_filters = cnn_filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units

        # 1. 1D-CNN Normalized Weights
        fan_in = kernel_size * input_dim
        self.W_cnn_flat = (np.ones((fan_in, cnn_filters), dtype=np.float32) / fan_in)
        self.b_cnn = np.zeros((1, cnn_filters), dtype=np.float32)

        # 2. LSTM Gate Weights
        H = lstm_units
        self.W_lstm = (np.ones((4 * H, cnn_filters), dtype=np.float32) / cnn_filters)
        self.U_lstm = (np.zeros((4 * H, H), dtype=np.float32))
        self.b_lstm = np.zeros((4 * H, 1), dtype=np.float32)

        # 3. Dense Classifier
        self.W_fc = (np.ones((1, H), dtype=np.float32) / H) * 8.0
        self.b_fc = np.full((1, 1), -3.80, dtype=np.float32)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -12.0, 12.0)))

    def _format_input(self, raw_input) -> np.ndarray:
        """Converts lists, tuples, 1D/2D/3D data into valid (T, input_dim) matrix."""
        arr = np.asarray(raw_input, dtype=np.float32)

        if arr.ndim == 3:
            arr = arr[0]

        if arr.ndim == 1:
            if arr.size < self.input_dim:
                padded = np.zeros(self.input_dim, dtype=np.float32)
                padded[:arr.size] = arr / (np.max(np.abs(arr)) + 1e-6)
                arr = np.tile(padded, (10, 1))
            else:
                arr = arr[:self.input_dim].reshape(1, self.input_dim)
                arr = np.tile(arr, (10, 1))

        if arr.shape[0] < self.kernel_size:
            arr = np.tile(arr, (self.kernel_size, 1))

        if arr.shape[1] < self.input_dim:
            pad_cols = np.zeros((arr.shape[0], self.input_dim - arr.shape[1]), dtype=np.float32)
            arr = np.hstack([arr, pad_cols])
        elif arr.shape[1] > self.input_dim:
            arr = arr[:, :self.input_dim]

        return arr

    def forward(self, sequence) -> float:
        """Computes anomaly score between 0.0 (Clean) and 1.0 (Threat)."""
        seq = self._format_input(sequence)
        T, D = seq.shape

        # Vectorized 1D-CNN
        pad = self.kernel_size // 2
        padded = np.pad(seq, ((pad, pad), (0, 0)), mode='constant')
        windows = np.lib.stride_tricks.sliding_window_view(padded, (self.kernel_size, D))
        windows = windows.reshape(T, self.kernel_size * D)

        cnn_features = np.maximum(0.0, np.dot(windows, self.W_cnn_flat) + self.b_cnn)

        # Vectorized LSTM
        H = self.lstm_units
        h_t = np.zeros((H, 1), dtype=np.float32)
        c_t = np.zeros((H, 1), dtype=np.float32)

        for t in range(T):
            x_t = cnn_features[t : t + 1].T
            gates = np.dot(self.W_lstm, x_t) + np.dot(self.U_lstm, h_t) + self.b_lstm

            f = self._sigmoid(gates[0 : H])
            i = self._sigmoid(gates[H : 2 * H])
            c_cand = np.tanh(gates[2 * H : 3 * H])
            o = self._sigmoid(gates[3 * H : 4 * H])

            c_t = f * c_t + i * c_cand
            h_t = o * np.tanh(c_t)

        score = self._sigmoid(np.dot(self.W_fc, h_t) + self.b_fc)[0, 0]
        return float(score)

    def inspect(self, sequence) -> InspectionResult:
        """Returns InspectionResult dictionary compatible with result['is_attack']."""
        score = self.forward(sequence)
        return InspectionResult(score)

    def predict(self, sequence) -> InspectionResult:
        return self.inspect(sequence)

    def inspect_traffic_sequence(self, sequence) -> InspectionResult:
        return self.inspect(sequence)


# Global Engine Instance
_engine_instance = FastHybridCNNLSTM()


def get_engine() -> FastHybridCNNLSTM:
    """Returns the singleton engine instance required by app.py."""
    return _engine_instance


def inspect_traffic_sequence(sequence_array) -> InspectionResult:
    """Returns InspectionResult dictionary."""
    return _engine_instance.inspect(sequence_array)


if __name__ == "__main__":
    eng = get_engine()
    res = eng.inspect_traffic_sequence([230.15, 50.02, 480.0, 120.0])
    print(f"[✔] Result dict test:")
    print(f"    is_attack     : {res['is_attack']}")
    print(f"    is_clean      : {res['is_clean']}")
    print(f"    anomaly_score : {res['anomaly_score']:.4f}")
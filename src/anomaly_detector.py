"""
anomaly_detector.py
-------------------
Convolutional autoencoder for structural chromosomal anomaly detection.

Architecture (as published in Scientific Reports):
  Input:   28 × 28 × 1  (grayscale)
  Encoder: Conv2D(128) → MaxPool → Conv2D(64) → MaxPool → Conv2D(32) → MaxPool  → latent 4×4×32
  Decoder: Conv2D(32)  → UpSamp  → Conv2D(64) → UpSamp  → Conv2D(128) → UpSamp → Conv2D(1, sigmoid)
  Loss:    MSE
  Threshold τ = 0.0148  (selected from validation error distribution)

Training data: NORMAL chromosomes ONLY from Dataset A.
Anomaly criterion: reconstruction error e(x) = ||x - x̂||²  > τ
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

# Published decision threshold (Section 8.5, Table 2)
DEFAULT_THRESHOLD: float = 0.0148
IMG_SIZE: tuple[int, int] = (28, 28)


def build_autoencoder():
    """
    Build the convolutional autoencoder exactly as described in the paper
    (Figure 4 / Section 6).

    Input/output: (None, 28, 28, 1) grayscale, sigmoid output.
    Latent code:  (None, 4, 4, 32)
    """
    from tensorflow.keras import layers, Model

    inp = layers.Input(shape=(28, 28, 1))

    # Encoder
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D(2, padding="same")(x)          # 14×14×128
    x = layers.Conv2D(64,  3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)          # 7×7×64
    x = layers.Conv2D(32,  3, activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D(2, padding="same")(x)    # 4×4×32

    # Decoder
    x = layers.Conv2D(32,  3, activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D(2)(x)                          # 8×8×32
    x = layers.Conv2D(64,  3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)                          # 16×16×64
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)                          # 32×32×128  (padded to 28)
    decoded = layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    # Crop back to 28×28 (UpSampling from 14→28 is exact; pad/crop not needed)
    # Note: 4→8→16→32 overshoots 28; we crop with a Lambda layer
    decoded = layers.Cropping2D(((2, 2), (2, 2)))(decoded)  # 32→28

    return Model(inp, decoded, name="autoencoder")


class AnomalyDetector:
    """
    Structural anomaly detector based on the convolutional autoencoder.

    Parameters
    ----------
    model_path : str | Path | None
        Path to saved .keras weights. If None, call build_and_train() first.
    threshold : float
        Reconstruction error threshold. Default τ = 0.0148 (paper value).
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        self.threshold = threshold
        self.model = None
        if model_path and Path(model_path).exists():
            from tensorflow.keras.models import load_model
            self.model = load_model(str(model_path))

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Convert any image (RGB or grayscale) to (28, 28, 1) float32 in [0, 1].
        """
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        resized = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0)[..., np.newaxis]  # (28,28,1)

    def reconstruction_error(self, image: np.ndarray) -> float:
        """MSE between input and reconstruction for one image."""
        x = self._preprocess(image)
        x_hat = self.model.predict(x[np.newaxis], verbose=0)[0]
        return float(np.mean((x - x_hat) ** 2))

    def reconstruction_errors_batch(self, images: list[np.ndarray]) -> list[float]:
        if not images:
            return []
        xs = np.stack([self._preprocess(img) for img in images])
        x_hats = self.model.predict(xs, verbose=0)
        return [float(np.mean((x - xh) ** 2)) for x, xh in zip(xs, x_hats)]

    def is_anomaly(self, image: np.ndarray) -> tuple[bool, float]:
        """Return (is_anomalous, reconstruction_error)."""
        error = self.reconstruction_error(image)
        return error > self.threshold, error

    def fit_threshold(
        self,
        normal_images: list[np.ndarray],
        percentile: float = 95.0,
    ) -> float:
        """Fit threshold from normal validation crops (percentile method)."""
        errors = self.reconstruction_errors_batch(normal_images)
        self.threshold = float(np.percentile(errors, percentile))
        return self.threshold

    def score_karyotype(
        self, crops_by_label: dict[str, list[np.ndarray]]
    ) -> dict[str, dict]:
        scores: dict[str, dict] = {}
        for label, imgs in crops_by_label.items():
            if not imgs:
                scores[label] = {"errors": [], "max_error": 0.0, "anomaly": False}
                continue
            errors = self.reconstruction_errors_batch(imgs)
            max_err = max(errors)
            scores[label] = {
                "errors": errors,
                "max_error": max_err,
                "anomaly": max_err > self.threshold,
            }
        return scores

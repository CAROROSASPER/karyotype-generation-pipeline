"""
resnet_classifier.py
--------------------
ResNet-50 chromosome pair classifier (Pipeline B).

Architecture (as published in Scientific Reports, Section 3.2):
  - Base: ResNet-50 pretrained on ImageNet
  - Top:  Dense(24, softmax)  [replacing the original 1000-class head]
  - Frozen layers: first ~50 layers
  - Optimizer: Adam  lr=0.001  (NOT AdamW)
  - Loss: categorical_crossentropy
  - Callbacks: EarlyStopping(patience=15), ModelCheckpoint(best val_loss),
                ReduceLROnPlateau(factor=0.02, patience=3, min_lr=1e-6)
  - Epochs: up to 100
  - Preprocessing: tf.keras.applications.resnet50.preprocess_input
  - Input size: 224 × 224

Training dataset: Dataset B — Q-band isolated chromosomes (Poletti et al., 5474 images, 24 classes).
Split: 75% train / 25% validation (inside TF ImageDataGenerator, NOT separate folders).
Augmentation: rotations ≤ 30°, zoom ≤ 20%, horizontal flips, shifts ≤ 10%.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np

KARYOTYPE_ORDER = [
    "A1", "A2", "A3",
    "B4", "B5",
    "C6", "C7", "C8", "C9", "C10", "C11", "C12",
    "D13", "D14", "D15",
    "E16", "E17", "E18",
    "F19", "F20",
    "G21", "G22",
    "X", "Y",
]
NUM_CLASSES = 24
IMG_SIZE = (224, 224)
FREEZE_UP_TO = 50  # number of layers to freeze (preserve generic ImageNet features)


def build_resnet50(num_classes: int = NUM_CLASSES):
    """
    Build ResNet-50 fine-tuned for chromosome pair classification.
    Matches the architecture described in Section 3.2 of the paper.
    """
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    base = ResNet50(weights="imagenet", include_top=True)

    # Replace final layer: take output before last dense (avgpool features)
    x = base.layers[-2].output
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=predictions)

    # Freeze first ~50 layers to preserve generic features
    for layer in base.layers[:FREEZE_UP_TO]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(checkpoint_path: str):
    """Return the three callbacks used during training (paper Section 3.2)."""
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    )
    return [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.02, patience=3, min_lr=1e-6, verbose=1),
    ]


def get_data_generators(dataset_b_dir: str, batch_size: int = 32):
    """
    Create TF ImageDataGenerator with augmentation as in paper Section 3.2.

    Parameters
    ----------
    dataset_b_dir : root directory of Dataset B (class sub-folders).
    batch_size    : batch size (paper uses 32).

    Returns
    -------
    (train_generator, val_generator)
    """
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        rotation_range=30,
        zoom_range=0.20,
        horizontal_flip=True,
        width_shift_range=0.10,
        height_shift_range=0.10,
        validation_split=0.25,  # 75% train / 25% val (paper)
    )

    train_gen = datagen.flow_from_directory(
        dataset_b_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
    )
    val_gen = datagen.flow_from_directory(
        dataset_b_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
    )
    return train_gen, val_gen


class ResNetClassifier:
    """
    Inference wrapper for the trained ResNet-50 pair classifier.

    Parameters
    ----------
    model_path   : path to the saved .keras model.
    class_names  : ordered class names; defaults to KARYOTYPE_ORDER.
    """

    def __init__(
        self,
        model_path: str | Path,
        class_names: Optional[list[str]] = None,
    ):
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.applications.resnet50 import preprocess_input

        self.model = load_model(str(model_path))
        self._preprocess = preprocess_input
        self.class_names = class_names if class_names else KARYOTYPE_ORDER

    def _prepare(self, image_rgb: np.ndarray) -> np.ndarray:
        import cv2
        resized = cv2.resize(image_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
        arr = resized.astype(np.float32)
        return self._preprocess(arr[np.newaxis])  # (1, 224, 224, 3)

    def predict(self, image_rgb: np.ndarray) -> tuple[str, float]:
        """Return (class_name, confidence) for one RGB crop."""
        probs = self.model.predict(self._prepare(image_rgb), verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.class_names[idx], float(probs[idx])

    def predict_batch(self, images_rgb: list[np.ndarray]) -> list[tuple[str, float]]:
        if not images_rgb:
            return []
        batch = np.concatenate([self._prepare(img) for img in images_rgb], axis=0)
        probs_all = self.model.predict(batch, verbose=0)
        return [
            (self.class_names[int(np.argmax(p))], float(np.max(p)))
            for p in probs_all
        ]

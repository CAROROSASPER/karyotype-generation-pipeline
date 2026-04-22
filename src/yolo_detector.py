"""
yolo_detector.py
----------------
YOLOv8 detection wrapper for chromosome detection.

Supports two modes:
  - multiclass : detects and classifies all 24 chromosome pairs in one pass (Pipeline A)
  - single     : detects chromosomes as a single class; crops returned for ResNet (Pipeline B)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Single bounding box prediction."""
    label: str
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def crop(self, image_rgb: np.ndarray, pad: int = 0) -> Optional[np.ndarray]:
        """Return the cropped region from *image_rgb*, with optional padding."""
        h, w = image_rgb.shape[:2]
        x1 = max(0, int(round(self.x1)) - pad)
        y1 = max(0, int(round(self.y1)) - pad)
        x2 = min(w, int(round(self.x2)) + pad)
        y2 = min(h, int(round(self.y2)) + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return image_rgb[y1:y2, x1:x2].copy()


class YOLODetector:
    """
    Thin wrapper around an Ultralytics YOLOv8 model for chromosome detection.

    Parameters
    ----------
    model_path : str | Path
        Path to the .pt weights file.
    conf_thr : float
        Confidence threshold (default: 0.25).
    iou_thr : float
        IoU threshold for NMS (default: 0.70).
    device : int | str
        Torch device (default: 0 for first GPU).
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_thr: float = 0.25,
        iou_thr: float = 0.70,
        device: int | str = 0,
    ):
        self.model = YOLO(str(model_path))
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.device = device

        names = self.model.names
        self.id_to_label: dict[int, str] = (
            names if isinstance(names, dict)
            else {i: n for i, n in enumerate(names)}
        )

    def predict(self, image_path: str | Path) -> list[Detection]:
        """Run inference on a single image and return a list of detections."""
        result = self.model(
            str(image_path),
            verbose=False,
            device=self.device,
            conf=self.conf_thr,
            iou=self.iou_thr,
        )[0]

        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    label=self.id_to_label.get(cls_id, str(cls_id)),
                    class_id=cls_id,
                    confidence=float(box.conf[0]),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                )
            )
        return detections

    def predict_and_crop(
        self,
        image_path: str | Path,
        pad: int = 0,
    ) -> list[tuple[Detection, np.ndarray]]:
        """
        Predict detections and return (detection, crop_rgb) pairs.
        Crops with zero area are skipped.
        """
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        detections = self.predict(image_path)
        result = []
        for det in detections:
            crop = det.crop(img_rgb, pad=pad)
            if crop is not None:
                result.append((det, crop))
        return result

    def count_by_label(self, detections: list[Detection]) -> dict[str, int]:
        """Return a {label: count} dict from a list of detections."""
        counts: dict[str, int] = {}
        for det in detections:
            counts[det.label] = counts.get(det.label, 0) + 1
        return counts

"""
overlap_filter.py
-----------------
Overlapped metaphase gate.

Runs a YOLOv8 model trained to detect the 'overlapped' class.
Images where the class is detected above the confidence threshold are
discarded; the rest are forwarded to the chromosome detection stage.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import cv2
from tqdm import tqdm
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(directory: str | Path) -> list[Path]:
    """Return sorted list of image paths found recursively under *directory*."""
    root = Path(directory)
    files: list[Path] = []
    for ext in IMG_EXTS:
        files += list(root.rglob(f"*{ext}"))
        files += list(root.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


class OverlapFilter:
    """
    Gate that removes metaphase images containing overlapping chromosomes.

    Parameters
    ----------
    model_path : str | Path
        Path to the YOLOv8 .pt weights trained on the 'overlapped' class.
    label_name : str
        Name of the overlapped class as stored in the model (default: "overlapped").
    conf_thr : float
        Confidence threshold for detection (default: 0.25).
    iou_thr : float
        IoU threshold for NMS (default: 0.70).
    device : int | str
        Torch device — 0 for first GPU, "cpu" for CPU (default: 0).
    """

    def __init__(
        self,
        model_path: str | Path,
        label_name: str = "overlapped",
        conf_thr: float = 0.25,
        iou_thr: float = 0.70,
        device: int | str = 0,
    ):
        self.model = YOLO(str(model_path))
        self.conf_thr = conf_thr
        self.iou_thr = iou_thr
        self.device = device

        # Resolve class id for the overlapped label
        names = self.model.names
        id_to_name = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
        name_to_id = {str(v).strip().lower(): int(k) for k, v in id_to_name.items()}
        key = label_name.strip().lower()
        if key not in name_to_id:
            raise ValueError(
                f"Label '{label_name}' not found in model.names={id_to_name}. "
                f"Check label_name in config."
            )
        self.overlap_id: int = name_to_id[key]

    def is_overlapped(self, image_path: str | Path) -> bool:
        """Return True if the image contains the overlapped class."""
        result = self.model(
            str(image_path),
            verbose=False,
            device=self.device,
            conf=self.conf_thr,
            iou=self.iou_thr,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return False
        detected_ids = [int(c) for c in result.boxes.cls]
        return self.overlap_id in detected_ids

    def filter_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        batch_size: int = 10,
    ) -> tuple[list[Path], list[Path]]:
        """
        Copy non-overlapped images from *input_dir* to *output_dir*.

        Returns
        -------
        kept : list[Path]
            Images that passed the gate.
        discarded : list[Path]
            Images that were rejected.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = list_images(input_dir)
        kept: list[Path] = []
        discarded: list[Path] = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            results = self.model(
                [str(p) for p in batch],
                verbose=False,
                device=self.device,
                conf=self.conf_thr,
                iou=self.iou_thr,
            )
            for img_path, result in zip(batch, results):
                detected_ids = (
                    [int(c) for c in result.boxes.cls]
                    if result.boxes is not None and len(result.boxes)
                    else []
                )
                if self.overlap_id in detected_ids:
                    discarded.append(img_path)
                else:
                    dst = output_dir / img_path.name
                    shutil.copy(str(img_path), str(dst))
                    kept.append(img_path)

        return kept, discarded

    def filter_images(
        self,
        image_paths: Iterable[str | Path],
    ) -> tuple[list[Path], list[Path]]:
        """
        Filter an arbitrary list of image paths without copying files.

        Returns (kept, discarded) path lists.
        """
        kept: list[Path] = []
        discarded: list[Path] = []
        for p in tqdm(list(image_paths), desc="Overlap gate"):
            p = Path(p)
            if self.is_overlapped(p):
                discarded.append(p)
            else:
                kept.append(p)
        return kept, discarded

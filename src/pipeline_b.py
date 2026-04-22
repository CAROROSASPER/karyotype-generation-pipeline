"""
pipeline_b.py
-------------
Pipeline B — Detect-then-Classify.

Workflow
--------
  overlap filter → YOLO single-class → ResNet50 classification
  → count validation → karyotype assembly → autoencoder anomaly detection

Usage (standalone)
------------------
  python src/pipeline_b.py --config configs/config.json \\
                            --images /path/to/metaphase/images \\
                            --out    /path/to/output/dir
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from overlap_filter import OverlapFilter, list_images
from yolo_detector import YOLODetector
from resnet_classifier import ResNetClassifier
from karyotype_assembler import KaryotypeAssembler, numerical_alert
from anomaly_detector import AnomalyDetector


def run_pipeline_b(
    images_dir: str | Path,
    out_dir: str | Path,
    overlap_model: str | Path,
    yolo_single_model: str | Path,
    resnet_model: str | Path,
    ae_model: Optional[str | Path] = None,
    overlap_conf: float = 0.25,
    yolo_conf: float = 0.25,
    yolo_iou: float = 0.70,
    resnet_class_names: Optional[list[str]] = None,
    ae_threshold: Optional[float] = None,
    ae_percentile: float = 95.0,
    device: int | str = 0,
    karyotype_order: Optional[list[str]] = None,
    cell_size: tuple[int, int] = (220, 220),
    cols: int = 4,
    pad: int = 10,
) -> list[dict]:
    """
    Run Pipeline B end-to-end on a directory of metaphase images.

    Parameters
    ----------
    images_dir       : directory containing metaphase images.
    out_dir          : output directory.
    overlap_model    : path to overlapped-gate YOLOv8 weights.
    yolo_single_model: path to single-class chromosome YOLOv8 weights.
    resnet_model     : path to ResNet50 Keras weights.
    ae_model         : (optional) path to autoencoder Keras weights.
    overlap_conf     : confidence threshold for overlap gate.
    yolo_conf        : confidence threshold for single-class detection.
    yolo_iou         : IoU threshold for NMS.
    resnet_class_names: ordered class list for the ResNet model.
    ae_threshold     : fixed AE threshold; if None, fitted at ae_percentile.
    ae_percentile    : percentile for AE threshold fitting.
    device           : torch device.
    karyotype_order, cell_size, cols, pad : grid layout parameters.

    Returns
    -------
    results : list of per-image dicts with keys:
        image_name, kept, counts, numerical_alerts,
        karyotype_path, ae_scores (if AE provided).
    """
    out_dir = Path(out_dir)
    karyotype_dir = out_dir / "karyotypes_B"
    karyotype_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Overlap gate ───────────────────────────────────────────────
    print("[Pipeline B] Stage 1/5 — Overlap gate …")
    gate = OverlapFilter(
        model_path=overlap_model,
        conf_thr=overlap_conf,
        device=device,
    )
    all_images = list_images(images_dir)
    kept_paths, _ = gate.filter_images(all_images)
    print(f"  Kept {len(kept_paths)} / {len(all_images)} images after gate.")

    # ── 2. Load models ────────────────────────────────────────────────
    print("[Pipeline B] Loading YOLO single + ResNet50 …")
    detector = YOLODetector(
        model_path=yolo_single_model,
        conf_thr=yolo_conf,
        iou_thr=yolo_iou,
        device=device,
    )
    classifier = ResNetClassifier(
        model_path=resnet_model,
        class_names=resnet_class_names,
    )
    assembler = KaryotypeAssembler(
        order=karyotype_order,
        cell_size=cell_size,
        cols=cols,
        pad=pad,
    )

    ae: Optional[AnomalyDetector] = None
    if ae_model and Path(ae_model).exists():
        print("[Pipeline B] Loading autoencoder …")
        ae = AnomalyDetector(model_path=ae_model, threshold=ae_threshold)

    # ── 3. Process each image ─────────────────────────────────────────
    print("[Pipeline B] Stages 2-5/5 — Detect → classify → assemble → score …")
    results: list[dict] = []
    all_crops_for_ae_fit: list[np.ndarray] = []

    for img_path in tqdm(kept_paths, desc="Pipeline B"):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Single-class detection → crops
        det_and_crops = detector.predict_and_crop(img_path)

        crops_rgb = [crop for _, crop in det_and_crops]
        if not crops_rgb:
            results.append({
                "image_name": img_path.name,
                "kept": True,
                "counts": {},
                "numerical_alerts": {},
                "karyotype_path": None,
                "ae_scores": None,
            })
            continue

        # ResNet50 classification
        classifications = classifier.predict_batch(crops_rgb)

        counts: dict[str, int] = {}
        crops_by_label: dict[str, list[np.ndarray]] = {}
        for (det, crop), (label, conf) in zip(det_and_crops, classifications):
            counts[label] = counts.get(label, 0) + 1
            crops_by_label.setdefault(label, []).append(crop)
            all_crops_for_ae_fit.append(crop)

        # Numerical count validation
        alerts = numerical_alert(counts)

        # Karyotype assembly
        karyo_path = karyotype_dir / f"{img_path.stem}_karyo_B.png"
        assembler.build(crops_by_label, output_path=karyo_path)

        entry: dict = {
            "image_name": img_path.name,
            "kept": True,
            "counts": counts,
            "numerical_alerts": alerts,
            "karyotype_path": str(karyo_path),
            "ae_scores": None,
        }

        if ae is not None and ae.threshold is not None:
            entry["ae_scores"] = ae.score_karyotype(crops_by_label)

        results.append(entry)

    # Fit AE threshold on collected crops if not pre-set
    if ae is not None and ae.threshold is None and all_crops_for_ae_fit:
        print(f"[Pipeline B] Fitting AE threshold at p{ae_percentile} …")
        ae.fit_threshold(all_crops_for_ae_fit, percentile=ae_percentile)
        # Re-score
        for entry in results:
            if entry["karyotype_path"] is None:
                continue
            img_path = Path(images_dir) / entry["image_name"]
            det_and_crops = detector.predict_and_crop(img_path)
            crops_rgb = [crop for _, crop in det_and_crops]
            if not crops_rgb:
                continue
            classifications = classifier.predict_batch(crops_rgb)
            crops_by_label = {}
            for (_, crop), (label, _) in zip(det_and_crops, classifications):
                crops_by_label.setdefault(label, []).append(crop)
            entry["ae_scores"] = ae.score_karyotype(crops_by_label)

    # Persist results
    results_path = out_dir / "pipeline_b_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"[Pipeline B] Done. Results saved to {results_path}")

    return results


# ── CLI entry point ────────────────────────────────────────────────────
def _cli():
    parser = argparse.ArgumentParser(description="Run Pipeline B")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--images", required=True, help="Directory with metaphase images")
    parser.add_argument("--out",    required=True, help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"

    run_pipeline_b(
        images_dir=args.images,
        out_dir=args.out,
        overlap_model=cfg["models"]["overlap_model"],
        yolo_single_model=cfg["models"]["yolo_single"],
        resnet_model=cfg["models"]["resnet_model"],
        ae_model=cfg["models"].get("autoencoder_model"),
        overlap_conf=cfg["inference"]["overlap_conf_thr"],
        yolo_conf=cfg["inference"]["yolo_conf_thr"],
        yolo_iou=cfg["inference"]["yolo_iou_thr"],
        ae_threshold=None,
        ae_percentile=95.0,
        device=device,
        karyotype_order=cfg["karyotype"]["order"],
        cell_size=tuple(cfg["karyotype"]["cell_size"]),
        cols=cfg["karyotype"]["cols"],
        pad=cfg["karyotype"]["pad"],
    )


if __name__ == "__main__":
    _cli()
